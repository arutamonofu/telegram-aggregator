package mtproto

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

type accountRuntimeState struct {
	accountID      int
	fsmState       domain.SessionFSMState
	riskScore      int
	floodWaitUntil time.Time
}

// StaticSingleTierScheduler implements domain.SessionScheduler for Stage 1 (Dev),
// delegating quarantine/frozen/dead transitions to PostgreSQL mtproto_sessions while
// remaining cleanly swappable with MVP's TieredFSMOrchestrator.
type StaticSingleTierScheduler struct {
	mu           sync.RWMutex
	accounts     map[int]*accountRuntimeState
	sessionStore domain.SessionStateStore
	accessHashes domain.AccessHashStore
	logger       *slog.Logger
}

var _ domain.SessionScheduler = (*StaticSingleTierScheduler)(nil)

// NewStaticSingleTierScheduler initializes the Stage 1 session scheduler.
func NewStaticSingleTierScheduler(
	sessionStore domain.SessionStateStore,
	accessHashes domain.AccessHashStore,
	logger *slog.Logger,
) *StaticSingleTierScheduler {
	if logger == nil {
		logger = slog.Default()
	}
	return &StaticSingleTierScheduler{
		accounts:     make(map[int]*accountRuntimeState),
		sessionStore: sessionStore,
		accessHashes: accessHashes,
		logger:       logger,
	}
}

// RegisterAccount tracks a registered DB account_id inside the runtime scheduler.
func (s *StaticSingleTierScheduler) RegisterAccount(rec domain.SessionRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var waitUntil time.Time
	if rec.FloodWaitUntil != nil {
		waitUntil = *rec.FloodWaitUntil
	}
	state := rec.FSMState
	if state == "" {
		state = domain.SessionActivePolling
	}
	s.accounts[rec.AccountID] = &accountRuntimeState{
		accountID:      rec.AccountID,
		fsmState:       state,
		riskScore:      rec.RiskScore,
		floodWaitUntil: waitUntil,
	}
}

// HealthyAccountIDs returns active non-quarantined account IDs whose flood_wait_until has expired.
func (s *StaticSingleTierScheduler) HealthyAccountIDs(_ context.Context) ([]int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	now := time.Now().UTC()
	var ids []int
	for id, st := range s.accounts {
		switch st.fsmState {
		case domain.SessionFrozenAppeal, domain.SessionTerminatedDead:
			continue
		case domain.SessionDegradedQuarantine:
			if !st.floodWaitUntil.IsZero() && now.Before(st.floodWaitUntil) {
				continue
			}
			// Permanent quarantine (e.g. PEER_FLOOD without deadline) stays excluded until manually cleared.
			if st.floodWaitUntil.IsZero() {
				continue
			}
		}
		ids = append(ids, id)
	}
	return ids, nil
}

// IsAccountQuarantined reports whether accountID is currently excluded from dispatch.
func (s *StaticSingleTierScheduler) IsAccountQuarantined(accountID int) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	st, ok := s.accounts[accountID]
	if !ok {
		return true
	}
	if st.fsmState == domain.SessionFrozenAppeal || st.fsmState == domain.SessionTerminatedDead {
		return true
	}
	if st.fsmState == domain.SessionDegradedQuarantine {
		if st.floodWaitUntil.IsZero() {
			return true
		}
		return time.Now().UTC().Before(st.floodWaitUntil)
	}
	return false
}

// ReportFloodWait transitions accountID into DEGRADED_QUARANTINE in memory and PostgreSQL.
func (s *StaticSingleTierScheduler) ReportFloodWait(
	ctx context.Context,
	accountID int,
	waitDuration time.Duration,
	methodRate float64,
) error {
	until := time.Now().UTC().Add(waitDuration)

	s.mu.Lock()
	if st, ok := s.accounts[accountID]; ok {
		st.fsmState = domain.SessionDegradedQuarantine
		st.floodWaitUntil = until
		st.riskScore = min(100, st.riskScore+15)
	}
	s.mu.Unlock()

	s.logger.Warn("account quarantined on FLOOD_WAIT > 60s",
		"account_id", accountID,
		"until", until,
		"adapted_rps", methodRate,
	)

	if s.sessionStore != nil {
		return s.sessionStore.SetSessionFloodQuarantine(ctx, accountID, until, 15)
	}
	return nil
}

// ReportPeerFlood increases risk_score by 50 and quarantines accountID.
func (s *StaticSingleTierScheduler) ReportPeerFlood(ctx context.Context, accountID int) error {
	newScore := 50
	s.mu.Lock()
	if st, ok := s.accounts[accountID]; ok {
		st.fsmState = domain.SessionDegradedQuarantine
		st.riskScore = min(100, st.riskScore+50)
		newScore = st.riskScore
	}
	s.mu.Unlock()

	s.logger.Error("account hit PEER_FLOOD; moved to DEGRADED_QUARANTINE",
		"account_id", accountID,
		"risk_score", newScore,
	)

	if s.sessionStore != nil {
		return s.sessionStore.SetSessionFSMState(ctx, accountID, domain.SessionDegradedQuarantine, newScore)
	}
	return nil
}

// ReportFrozen marks accountID as FROZEN_APPEAL (24h review / appeal).
func (s *StaticSingleTierScheduler) ReportFrozen(ctx context.Context, accountID int) error {
	s.mu.Lock()
	if st, ok := s.accounts[accountID]; ok {
		st.fsmState = domain.SessionFrozenAppeal
	}
	s.mu.Unlock()

	s.logger.Error("account reported FROZEN_*; moved to FROZEN_APPEAL", "account_id", accountID)

	if s.sessionStore != nil {
		return s.sessionStore.SetSessionFSMState(ctx, accountID, domain.SessionFrozenAppeal, 100)
	}
	return nil
}

// ReportDeadSession marks accountID TERMINATED_DEAD and purges per-account access hashes.
func (s *StaticSingleTierScheduler) ReportDeadSession(ctx context.Context, accountID int) error {
	s.mu.Lock()
	if st, ok := s.accounts[accountID]; ok {
		st.fsmState = domain.SessionTerminatedDead
	}
	s.mu.Unlock()

	s.logger.Error("account 401/dead; moved to TERMINATED_DEAD and purging access hashes", "account_id", accountID)

	if s.sessionStore != nil {
		if err := s.sessionStore.SetSessionFSMState(ctx, accountID, domain.SessionTerminatedDead, 100); err != nil {
			return err
		}
	}
	if s.accessHashes != nil {
		return s.accessHashes.DeleteAccountAccessHashes(ctx, accountID)
	}
	return nil
}
