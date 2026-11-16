package domain

import (
	"context"
	"errors"
	"time"
)

var (
	// ErrMediaPermanentlyGone signals that all file_sources for a file_id are deleted (-> HTTP 410 Gone).
	ErrMediaPermanentlyGone = errors.New("media permanently gone across all sources")
	// ErrMediaNotFound signals that file_id is absent from file_references.
	ErrMediaNotFound = errors.New("media file_id not found")
	// ErrSessionQuarantined signals that the session entered FLOOD_WAIT > 60s or PEER_FLOOD quarantine.
	ErrSessionQuarantined = errors.New("session quarantined")
	// ErrFloodPremiumWait signals FLOOD_PREMIUM_WAIT_X requiring reroute to a Premium session.
	ErrFloodPremiumWait = errors.New("flood premium wait")
	// ErrNoAvailableSession signals that all sessions are quarantined or terminated.
	ErrNoAvailableSession = errors.New("no healthy MTProto session available")
)

// EventPublisher publishes canonical events synchronously to Redis Streams and returns the Stream Entry ID.
type EventPublisher interface {
	PublishPost(ctx context.Context, event IngestPostEventV1) (string, error)
	PublishCommentsDelta(ctx context.Context, event IngestCommentsDeltaEventV1) (string, error)
	PublishDeletion(ctx context.Context, event IngestDeletionEventV1) (string, error)
}

// WatermarkStore manages global channel watermarks, file references, and post sync metadata in PostgreSQL.
type WatermarkStore interface {
	GetChannelState(ctx context.Context, channelID int64) (ChannelStateRecord, bool, error)
	GetChannelByUsername(ctx context.Context, username string) (ChannelStateRecord, bool, error)
	UpsertChannel(ctx context.Context, ch ChannelStateRecord) error
	GetActiveChannelsDueForPoll(ctx context.Context, limit int) ([]ChannelStateRecord, error)
	UpdateChannelNextPoll(ctx context.Context, channelID int64, nextPollAt time.Time) error
	SetChannelLinkedChatID(ctx context.Context, channelID int64, linkedChatID int64) error
	CommitPostWatermarkAndFiles(
		ctx context.Context,
		channelID int64,
		maxMsgID int,
		maxEditDate *time.Time,
		refs []FileReferenceRecord,
		sources []FileSourceRecord,
	) error
	GetPostSyncMeta(ctx context.Context, channelID int64, msgID int) (PostSyncMeta, bool, error)
	UpdatePostRepliesSynced(ctx context.Context, channelID int64, msgID int, repliesCount, repliesMaxID, syncedCount int) error
	GetRecentPostIDsForEditRescan(ctx context.Context, channelID int64, windowHours int, limit int) ([]int, error)
}

// AccessHashStore persists per-account MTProto access hashes indefinitely in PostgreSQL.
type AccessHashStore interface {
	GetAccessHash(ctx context.Context, accountID int, peerID int64) (int64, bool, error)
	SaveAccessHash(ctx context.Context, accountID int, peerID int64, peerType string, accessHash int64) error
	DeleteAccountAccessHashes(ctx context.Context, accountID int) error
}

// SessionStateStore persists MTProto session FSM state and quarantine deadlines.
type SessionStateStore interface {
	UpsertSession(ctx context.Context, rec SessionRecord) (int, error)
	SetSessionFloodQuarantine(ctx context.Context, accountID int, until time.Time, riskDelta int) error
	SetSessionFSMState(ctx context.Context, accountID int, state SessionFSMState, riskScore int) error
}

// FileRefRepository persists file_references and file_sources for media streaming and refresh.
type FileRefRepository interface {
	GetFileReference(ctx context.Context, fileID string) (FileReferenceRecord, bool, error)
	GetActiveFileSources(ctx context.Context, fileID string) ([]FileSourceRecord, error)
	UpdateFileReferenceBytes(ctx context.Context, fileID string, newBytes []byte, fileSize int64) error
	MarkFileSourceDeleted(ctx context.Context, sourceID int64) error
	MarkFilePermanentlyGone(ctx context.Context, fileID string) error
	UpdateFileLocalCache(ctx context.Context, fileID string, cachePath string, expiresAt time.Time) error
}

// MediaStorageProvider abstracts local 24h disk cache (Dev) vs Hybrid S3 Edge Proxy (MVP).
type MediaStorageProvider interface {
	GetChunk(ctx context.Context, fileID string, alignedOffset int64) ([]byte, bool, error)
	PutChunk(ctx context.Context, fileID string, alignedOffset int64, data []byte) (string, time.Time, error)
	Invalidate(ctx context.Context, fileID string) error
}

// SessionScheduler abstracts Dev StaticSingleTierScheduler vs MVP TieredFSMOrchestrator.
type SessionScheduler interface {
	HealthyAccountIDs(ctx context.Context) ([]int, error)
	IsAccountQuarantined(accountID int) bool
	ReportFloodWait(ctx context.Context, accountID int, waitDuration time.Duration, methodRate float64) error
	ReportPeerFlood(ctx context.Context, accountID int) error
	ReportFrozen(ctx context.Context, accountID int) error
	ReportDeadSession(ctx context.Context, accountID int) error
}
