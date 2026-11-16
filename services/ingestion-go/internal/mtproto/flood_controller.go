package mtproto

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/gotd/contrib/middleware/floodwait"
	"github.com/gotd/td/bin"
	"github.com/gotd/td/telegram"
	"github.com/gotd/td/tg"
	"github.com/gotd/td/tgerr"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

const defaultLocalMaxWait = 60 * time.Second

// DynamicRateTracker computes adapted method throughput r = N / (elapsed + X) when FLOOD_WAIT_X occurs.
type DynamicRateTracker struct {
	mu          sync.Mutex
	windowStart time.Time
	count       int
	currentRate float64
}

// NewDynamicRateTracker initializes a DynamicRateTracker.
func NewDynamicRateTracker() *DynamicRateTracker {
	return &DynamicRateTracker{
		windowStart: time.Now(),
	}
}

// RecordCall increments the request counter in the current measurement window.
func (t *DynamicRateTracker) RecordCall() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.count++
}

// AdjustForFloodWait recalculates r = N / (elapsed + X) upon FLOOD_WAIT_X and resets the window.
func (t *DynamicRateTracker) AdjustForFloodWait(waitDuration time.Duration) float64 {
	t.mu.Lock()
	defer t.mu.Unlock()

	elapsed := time.Since(t.windowStart).Seconds()
	if elapsed < 0.001 {
		elapsed = 0.001
	}
	n := t.count
	if n <= 0 {
		n = 1
	}
	rate := float64(n) / (elapsed + waitDuration.Seconds())
	t.currentRate = rate
	t.windowStart = time.Now()
	t.count = 0
	return rate
}

// CurrentRate returns the latest computed sustainable RPS after flood wait adaptation.
func (t *DynamicRateTracker) CurrentRate() float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.currentRate
}

// FloodController combines gotd/contrib floodwait.Waiter (local pauses <= 60s) with persistent
// DB quarantine (> 60s) and MTProto risk/FSM handling (SLOWMODE_WAIT, PEER_FLOOD, FROZEN_*, 401).
type FloodController struct {
	accountID int
	maxWait   time.Duration
	waiter    *floodwait.Waiter
	scheduler domain.SessionScheduler
	tracker   *DynamicRateTracker
	logger    *slog.Logger
}

// NewFloodController creates a FloodController bound to accountID.
func NewFloodController(
	accountID int,
	scheduler domain.SessionScheduler,
	logger *slog.Logger,
) *FloodController {
	if logger == nil {
		logger = slog.Default()
	}

	fc := &FloodController{
		accountID: accountID,
		maxWait:   defaultLocalMaxWait,
		scheduler: scheduler,
		tracker:   NewDynamicRateTracker(),
		logger:    logger,
	}

	fc.waiter = floodwait.NewWaiter().
		WithMaxWait(defaultLocalMaxWait).
		WithServerErrorRetries(2).
		WithAdaptivePacing(100*time.Millisecond, 5*time.Second).
		WithCallback(fc.onWait)

	return fc
}

// Waiter returns the underlying floodwait.Waiter (its Run loop must wrap client.Run).
func (fc *FloodController) Waiter() *floodwait.Waiter {
	return fc.waiter
}

// RateTracker exposes the dynamic r = N/(elapsed + X) tracker.
func (fc *FloodController) RateTracker() *DynamicRateTracker {
	return fc.tracker
}

func (fc *FloodController) onWait(_ context.Context, wait floodwait.FloodWait) {
	rate := fc.tracker.AdjustForFloodWait(wait.Duration)
	fc.logger.Warn("MTProto wait triggered",
		"account_id", fc.accountID,
		"method", wait.Name,
		"duration", wait.Duration.String(),
		"per_type", wait.PerType,
		"adapted_rps", rate,
		"error", wait.Err,
	)
}

// ErrorClassifierMiddleware sits outside floodwait.Waiter so any unretried wait (> 60s),
// PEER_FLOOD, FROZEN_*, or 401 error transitions PostgreSQL session state cleanly.
func (fc *FloodController) ErrorClassifierMiddleware() telegram.Middleware {
	return floodClassifierMiddleware{fc: fc}
}

type floodClassifierMiddleware struct {
	fc *FloodController
}

func (m floodClassifierMiddleware) Handle(next tg.Invoker) telegram.InvokeFunc {
	return func(ctx context.Context, input bin.Encoder, output bin.Decoder) error {
		m.fc.tracker.RecordCall()
		err := next.Invoke(ctx, input, output)
		if err == nil {
			return nil
		}
		return m.fc.ClassifyError(ctx, err)
	}
}

// ClassifyError inspects RPC errors after floodwait.Waiter and enforces Section 7.1.4 policy.
func (fc *FloodController) ClassifyError(ctx context.Context, err error) error {
	rpcErr, ok := tgerr.As(err)
	if !ok {
		return err
	}

	switch {
	case tgerr.Is(err, "PEER_FLOOD"):
		if fc.scheduler != nil {
			if rErr := fc.scheduler.ReportPeerFlood(ctx, fc.accountID); rErr != nil {
				fc.logger.Error("ReportPeerFlood failed", "account_id", fc.accountID, "err", rErr)
			}
		}
		return fmt.Errorf("%w: %v", domain.ErrSessionQuarantined, err)

	case tgerr.Is(err, "FROZEN_METHOD_INVALID", "FROZEN_PARTICIPANT_MISSING"):
		if fc.scheduler != nil {
			if rErr := fc.scheduler.ReportFrozen(ctx, fc.accountID); rErr != nil {
				fc.logger.Error("ReportFrozen failed", "account_id", fc.accountID, "err", rErr)
			}
		}
		return fmt.Errorf("session %d frozen (%s): %w", fc.accountID, rpcErr.Type, err)

	case rpcErr.Code == 401 || tgerr.Is(err, "AUTH_KEY_UNREGISTERED", "SESSION_REVOKED", "USER_DEACTIVATED_BAN"):
		if fc.scheduler != nil {
			if rErr := fc.scheduler.ReportDeadSession(ctx, fc.accountID); rErr != nil {
				fc.logger.Error("ReportDeadSession failed", "account_id", fc.accountID, "err", rErr)
			}
		}
		return fmt.Errorf("session %d terminated (%s): %w", fc.accountID, rpcErr.Type, err)

	case tgerr.Is(err, "SLOWMODE_WAIT"):
		// SLOWMODE_WAIT is channel/supergroup-scoped; never quarantine the whole session for it.
		waitDur := time.Duration(rpcErr.Argument) * time.Second
		fc.logger.Warn("channel SLOWMODE_WAIT encountered without global session quarantine",
			"account_id", fc.accountID,
			"wait_duration", waitDur.String(),
		)
		return err

	case tgerr.Is(err, tgerr.ErrPremiumFloodWait):
		fc.logger.Warn("FLOOD_PREMIUM_WAIT encountered (non-fatal media throttle / downgrade)",
			"account_id", fc.accountID,
			"wait_sec", rpcErr.Argument,
		)
		return fmt.Errorf("%w: %v", domain.ErrFloodPremiumWait, err)

	case tgerr.Is(err, tgerr.ErrFloodWait):
		waitDur := time.Duration(rpcErr.Argument) * time.Second
		if waitDur <= 0 {
			if d, ok := tgerr.AsFloodWait(err); ok {
				waitDur = d
			}
		}
		rate := fc.tracker.AdjustForFloodWait(waitDur)
		if waitDur > fc.maxWait && fc.scheduler != nil {
			if rErr := fc.scheduler.ReportFloodWait(ctx, fc.accountID, waitDur, rate); rErr != nil {
				fc.logger.Error("ReportFloodWait failed", "account_id", fc.accountID, "err", rErr)
			}
			return fmt.Errorf("%w (duration=%v): %v", domain.ErrSessionQuarantined, waitDur, err)
		}
		return err
	}

	return err
}
