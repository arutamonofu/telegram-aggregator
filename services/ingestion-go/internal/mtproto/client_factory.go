package mtproto

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/gotd/td/telegram"
	"github.com/gotd/td/telegram/updates"
	updhook "github.com/gotd/td/telegram/updates/hook"
	"github.com/gotd/td/tg"
	"golang.org/x/sync/errgroup"

	"github.com/tg-agg/ingestion-go/internal/config"
	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/storage/bbolt"
)

// ManagedClient bundles a configured gotd/td client, its persistent bbolt store,
// update gap-recovery manager, and FloodWait/FSM controllers for one MTProto account.
type ManagedClient struct {
	AccountID  int
	SessionKey string
	Client     *telegram.Client
	Updates    *updates.Manager
	Flood      *FloodController
	Store      *bbolt.Store
	scheduler  domain.SessionScheduler
	logger     *slog.Logger
}

// ManagedClientParams wires dependencies for one MTProto account worker.
type ManagedClientParams struct {
	Config        config.Config
	AccountID     int
	SessionKey    string
	Store         *bbolt.Store
	StateStorage  updates.StateStorage
	UpdateHandler telegram.UpdateHandler
	Scheduler     domain.SessionScheduler
	Watermarks    domain.WatermarkStore
	AccessHashes  domain.AccessHashStore
	GapSink       RecoveredBatchSink
	Logger        *slog.Logger
}

// NewManagedClient initializes an MTProto client adhering to Section 7.1.1-7.1.4:
// - bbolt persistent session and peer access_hash storage,
// - SOCKS5 resolver via dcs.Plain,
// - updates.Manager with StopOnApplyError=true and updhook.UpdateHook/AffectedHook,
// - ChannelTooLongPatch middleware (bypassing gotd/td#1711 before pts reset),
// - FloodController (local <=60s Waiter + >60s DB quarantine / PEER_FLOOD / FROZEN / 401).
func NewManagedClient(p ManagedClientParams) (*ManagedClient, error) {
	if p.Store == nil {
		return nil, fmt.Errorf("nil bbolt Store for account %d", p.AccountID)
	}
	logger := p.Logger
	if logger == nil {
		logger = slog.Default()
	}

	resolver, err := NewSOCKS5Resolver(p.Config.Socks5ProxyURL)
	if err != nil {
		return nil, fmt.Errorf("init SOCKS5 resolver: %w", err)
	}

	stateStor := p.StateStorage
	if stateStor == nil {
		stateStor = p.Store.StateStorage()
	}

	mc := &ManagedClient{
		AccountID:  p.AccountID,
		SessionKey: p.SessionKey,
		Store:      p.Store,
		Flood:      NewFloodController(p.AccountID, p.Scheduler, logger),
		scheduler:  p.Scheduler,
		logger:     logger,
	}

	var gapSvc *ChannelGapRecoveryService
	tooLongHandler := TooLongHandlerFunc(func(ctx context.Context, payload ChannelTooLongPayload) error {
		if gapSvc == nil && mc.Client != nil {
			gapSvc = NewChannelGapRecoveryService(
				p.AccountID,
				mc.Client.API(),
				p.Watermarks,
				p.AccessHashes,
				p.GapSink,
			)
		}
		if gapSvc != nil {
			return gapSvc.HandleTooLong(ctx, payload)
		}
		return nil
	})

	mc.Updates = updates.New(updates.Config{
		Handler:          p.UpdateHandler,
		Storage:          stateStor,
		AccessHasher:     p.Store.AccessHasher(),
		StopOnApplyError: true,
		OnChannelTooLong: func(channelID int64) {
			logger.Warn("updates.Manager reported channel too long",
				"account_id", p.AccountID,
				"channel_id", channelID,
			)
		},
		OnChannelGap: func(gap updates.ChannelGap) {
			logger.Warn("updates.Manager reported ChannelGap fallback",
				"account_id", p.AccountID,
				"channel_id", gap.ChannelID,
				"local_pts", gap.LocalPts,
				"remote_pts", gap.RemotePts,
				"top_message", gap.TopMessage,
				"embedded_messages", len(gap.Messages),
			)
			// Note: primary gap recovery already runs synchronously inside NewChannelTooLongPatchMiddleware
			// before SetChannelPts is reached in state_channel.go.
		},
	})

	opts := telegram.Options{
		Resolver:       resolver,
		SessionStorage: p.Store.SessionStorage(p.SessionKey),
		UpdateHandler:  mc.Updates,
		Middlewares: []telegram.Middleware{
			updhook.UpdateHook(mc.Updates.Handle),
			updhook.AffectedHook(mc.Updates),
			NewChannelTooLongPatchMiddleware(tooLongHandler),
			mc.Flood.ErrorClassifierMiddleware(),
			mc.Flood.Waiter(),
		},
		OnAuthLost: func(ctx context.Context, authErr error) error {
			logger.Error("MTProto auth lost", "account_id", p.AccountID, "err", authErr)
			if p.Scheduler != nil {
				return p.Scheduler.ReportDeadSession(ctx, p.AccountID)
			}
			return nil
		},
		OnFrozen: func(ctx context.Context, state telegram.FreezeState) {
			logger.Error("MTProto account frozen",
				"account_id", p.AccountID,
				"since", state.Since,
				"until", state.Until,
			)
			if p.Scheduler != nil {
				if fErr := p.Scheduler.ReportFrozen(ctx, p.AccountID); fErr != nil {
					logger.Error("ReportFrozen failed", "account_id", p.AccountID, "err", fErr)
				}
			}
		},
	}

	mc.Client = telegram.NewClient(p.Config.TGAppID, p.Config.TGAppHash, opts)
	return mc, nil
}

// API returns the wrapped tg.Client RPC caller.
func (m *ManagedClient) API() *tg.Client {
	return m.Client.API()
}

// Run starts the FloodWait Waiter loop and MTProto client lifecycle. If the session is authorized,
// it starts updates.Manager and invokes onReady(ctx, selfUserID); otherwise it logs standby status
// and waits for context cancellation so HTTP health/debug endpoints remain responsive.
func (m *ManagedClient) Run(ctx context.Context, onReady func(ctx context.Context, selfUserID int64) error) error {
	return m.Flood.Waiter().Run(ctx, func(ctx context.Context) error {
		return m.Client.Run(ctx, func(ctx context.Context) error {
			status, err := m.Client.Auth().Status(ctx)
			if err != nil {
				return fmt.Errorf("check auth status for account %d: %w", m.AccountID, err)
			}
			if !status.Authorized || status.User == nil {
				m.logger.Warn("MTProto session is not authorized yet; standing by for seeded bbolt auth",
					"account_id", m.AccountID,
					"session_key", m.SessionKey,
				)
				<-ctx.Done()
				return nil
			}

			selfID := status.User.ID
			g, gCtx := errgroup.WithContext(ctx)
			g.Go(func() error {
				return m.Updates.Run(gCtx, m.Client.API(), selfID, updates.AuthOptions{
					IsBot:  false,
					Forget: false,
					OnStart: func(startedCtx context.Context) {
						m.logger.Info("MTProto updates.Manager started",
							"account_id", m.AccountID,
							"self_id", selfID,
						)
					},
				})
			})

			if onReady != nil {
				g.Go(func() error {
					return onReady(gCtx, selfID)
				})
			}

			return g.Wait()
		})
	})
}
