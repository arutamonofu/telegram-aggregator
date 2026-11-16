package postgres

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

var (
	_ domain.WatermarkStore    = (*Repository)(nil)
	_ domain.AccessHashStore   = (*Repository)(nil)
	_ domain.SessionStateStore = (*Repository)(nil)
	_ domain.FileRefRepository = (*Repository)(nil)
)

// Repository implements PostgreSQL persistence against the core_public schema.
type Repository struct {
	pool *pgxpool.Pool
}

// NewRepository wraps an initialized pgxpool.Pool.
func NewRepository(pool *pgxpool.Pool) *Repository {
	return &Repository{pool: pool}
}

// GetChannelState loads a channel's global watermark state by channel_id.
func (r *Repository) GetChannelState(ctx context.Context, channelID int64) (domain.ChannelStateRecord, bool, error) {
	const q = `
SELECT
	channel_id,
	username,
	title,
	tier::text,
	last_msg_id,
	max_edit_date_checked,
	linked_chat_id,
	pinned_msg_ids,
	poll_interval_sec,
	next_poll_at,
	is_active
FROM core_public.channels
WHERE channel_id = $1`

	var rec domain.ChannelStateRecord
	err := r.pool.QueryRow(ctx, q, channelID).Scan(
		&rec.ChannelID,
		&rec.Username,
		&rec.Title,
		&rec.Tier,
		&rec.LastMsgID,
		&rec.MaxEditDateChecked,
		&rec.LinkedChatID,
		&rec.PinnedMsgIDs,
		&rec.PollIntervalSec,
		&rec.NextPollAt,
		&rec.IsActive,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return domain.ChannelStateRecord{}, false, nil
	}
	if err != nil {
		return domain.ChannelStateRecord{}, false, fmt.Errorf("get channel state %d: %w", channelID, err)
	}
	return rec, true, nil
}

// GetChannelByUsername looks up a tracked channel by normalized username (without '@').
func (r *Repository) GetChannelByUsername(ctx context.Context, username string) (domain.ChannelStateRecord, bool, error) {
	const q = `
SELECT
	channel_id,
	username,
	title,
	tier::text,
	last_msg_id,
	max_edit_date_checked,
	linked_chat_id,
	pinned_msg_ids,
	poll_interval_sec,
	next_poll_at,
	is_active
FROM core_public.channels
WHERE lower(username) = lower($1)`

	var rec domain.ChannelStateRecord
	err := r.pool.QueryRow(ctx, q, username).Scan(
		&rec.ChannelID,
		&rec.Username,
		&rec.Title,
		&rec.Tier,
		&rec.LastMsgID,
		&rec.MaxEditDateChecked,
		&rec.LinkedChatID,
		&rec.PinnedMsgIDs,
		&rec.PollIntervalSec,
		&rec.NextPollAt,
		&rec.IsActive,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return domain.ChannelStateRecord{}, false, nil
	}
	if err != nil {
		return domain.ChannelStateRecord{}, false, fmt.Errorf("get channel by username %q: %w", username, err)
	}
	return rec, true, nil
}

// UpsertChannel inserts or updates channel registration metadata without regressing global watermarks.
func (r *Repository) UpsertChannel(ctx context.Context, ch domain.ChannelStateRecord) error {
	tier := ch.Tier
	if tier == "" {
		tier = "TIER_2"
	}
	pollSec := ch.PollIntervalSec
	if pollSec <= 0 {
		pollSec = 900
	}
	nextPoll := ch.NextPollAt
	if nextPoll.IsZero() {
		nextPoll = time.Now().UTC()
	}
	pinned := ch.PinnedMsgIDs
	if pinned == nil {
		pinned = []int{}
	}

	const q = `
INSERT INTO core_public.channels (
	channel_id,
	username,
	title,
	tier,
	last_msg_id,
	max_edit_date_checked,
	linked_chat_id,
	pinned_msg_ids,
	poll_interval_sec,
	next_poll_at,
	is_active
) VALUES (
	$1, $2, $3, $4::core_public.channel_tier_enum, $5, $6, $7, $8, $9, $10, $11
)
ON CONFLICT (channel_id) DO UPDATE SET
	username              = COALESCE(EXCLUDED.username, core_public.channels.username),
	title                 = CASE WHEN EXCLUDED.title <> '' THEN EXCLUDED.title ELSE core_public.channels.title END,
	tier                  = EXCLUDED.tier,
	last_msg_id           = GREATEST(core_public.channels.last_msg_id, EXCLUDED.last_msg_id),
	linked_chat_id        = COALESCE(EXCLUDED.linked_chat_id, core_public.channels.linked_chat_id),
	poll_interval_sec     = EXCLUDED.poll_interval_sec,
	is_active             = EXCLUDED.is_active`

	_, err := r.pool.Exec(ctx, q,
		ch.ChannelID,
		ch.Username,
		ch.Title,
		tier,
		ch.LastMsgID,
		ch.MaxEditDateChecked,
		ch.LinkedChatID,
		pinned,
		pollSec,
		nextPoll,
		ch.IsActive,
	)
	if err != nil {
		return fmt.Errorf("upsert channel %d: %w", ch.ChannelID, err)
	}
	return nil
}

// GetActiveChannelsDueForPoll returns active channels whose next_poll_at has elapsed.
func (r *Repository) GetActiveChannelsDueForPoll(ctx context.Context, limit int) ([]domain.ChannelStateRecord, error) {
	if limit <= 0 {
		limit = 50
	}

	const q = `
SELECT
	channel_id,
	username,
	title,
	tier::text,
	last_msg_id,
	max_edit_date_checked,
	linked_chat_id,
	pinned_msg_ids,
	poll_interval_sec,
	next_poll_at,
	is_active
FROM core_public.channels
WHERE is_active = TRUE AND next_poll_at <= NOW()
ORDER BY next_poll_at ASC
LIMIT $1`

	rows, err := r.pool.Query(ctx, q, limit)
	if err != nil {
		return nil, fmt.Errorf("query channels due for poll: %w", err)
	}
	defer rows.Close()

	var result []domain.ChannelStateRecord
	for rows.Next() {
		var rec domain.ChannelStateRecord
		if err := rows.Scan(
			&rec.ChannelID,
			&rec.Username,
			&rec.Title,
			&rec.Tier,
			&rec.LastMsgID,
			&rec.MaxEditDateChecked,
			&rec.LinkedChatID,
			&rec.PinnedMsgIDs,
			&rec.PollIntervalSec,
			&rec.NextPollAt,
			&rec.IsActive,
		); err != nil {
			return nil, fmt.Errorf("scan channel poll row: %w", err)
		}
		result = append(result, rec)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate channel poll rows: %w", err)
	}
	return result, nil
}

// UpdateChannelNextPoll schedules the next fallback polling timestamp for a channel.
func (r *Repository) UpdateChannelNextPoll(ctx context.Context, channelID int64, nextPollAt time.Time) error {
	const q = `UPDATE core_public.channels SET next_poll_at = $2 WHERE channel_id = $1`
	if _, err := r.pool.Exec(ctx, q, channelID, nextPollAt.UTC()); err != nil {
		return fmt.Errorf("update next_poll_at for channel %d: %w", channelID, err)
	}
	return nil
}

// SetChannelLinkedChatID caches the discussion supergroup ID linked to channelID.
func (r *Repository) SetChannelLinkedChatID(ctx context.Context, channelID int64, linkedChatID int64) error {
	const q = `UPDATE core_public.channels SET linked_chat_id = $2 WHERE channel_id = $1`
	if _, err := r.pool.Exec(ctx, q, channelID, linkedChatID); err != nil {
		return fmt.Errorf("set linked_chat_id for channel %d: %w", channelID, err)
	}
	return nil
}

// CommitPostWatermarkAndFiles atomically upserts file_references, file_sources, and advances
// channels.last_msg_id / max_edit_date_checked inside a single PostgreSQL transaction.
func (r *Repository) CommitPostWatermarkAndFiles(
	ctx context.Context,
	channelID int64,
	maxMsgID int,
	maxEditDate *time.Time,
	refs []domain.FileReferenceRecord,
	sources []domain.FileSourceRecord,
) error {
	tx, err := r.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin watermark commit tx: %w", err)
	}
	defer func() {
		_ = tx.Rollback(ctx)
	}()

	const upsertRefSQL = `
INSERT INTO core_public.file_references (
	file_id,
	tg_id,
	access_hash,
	dc_id,
	media_kind,
	thumb_size,
	file_reference,
	file_size,
	mime_type,
	refreshed_at,
	is_permanently_gone
) VALUES (
	$1, $2, $3, $4, $5::core_public.media_kind_enum, $6, $7, $8, $9, NOW(), FALSE
)
ON CONFLICT (file_id) DO UPDATE SET
	access_hash         = EXCLUDED.access_hash,
	dc_id               = EXCLUDED.dc_id,
	file_reference      = EXCLUDED.file_reference,
	file_size           = CASE WHEN EXCLUDED.file_size > 0 THEN EXCLUDED.file_size ELSE core_public.file_references.file_size END,
	mime_type           = COALESCE(EXCLUDED.mime_type, core_public.file_references.mime_type),
	refreshed_at        = NOW(),
	is_permanently_gone = FALSE`

	for _, ref := range refs {
		if _, err := tx.Exec(ctx, upsertRefSQL,
			ref.FileID,
			ref.TGID,
			ref.AccessHash,
			ref.DCID,
			string(ref.MediaKind),
			ref.ThumbSize,
			ref.FileReference,
			ref.FileSize,
			ref.MimeType,
		); err != nil {
			return fmt.Errorf("upsert file_reference %q: %w", ref.FileID, err)
		}
	}

	const upsertSourceSQL = `
INSERT INTO core_public.file_sources (
	file_id,
	channel_id,
	msg_id,
	grouped_id,
	album_msg_ids,
	is_source_deleted
) VALUES (
	$1, $2, $3, $4, $5, FALSE
)
ON CONFLICT (file_id, channel_id, msg_id) DO UPDATE SET
	grouped_id        = COALESCE(EXCLUDED.grouped_id, core_public.file_sources.grouped_id),
	album_msg_ids     = CASE
		WHEN COALESCE(array_length(EXCLUDED.album_msg_ids, 1), 0) > 0 THEN EXCLUDED.album_msg_ids
		ELSE core_public.file_sources.album_msg_ids
	END,
	is_source_deleted = FALSE`

	for _, src := range sources {
		albumIDs := src.AlbumMsgIDs
		if albumIDs == nil {
			albumIDs = []int{}
		}
		if _, err := tx.Exec(ctx, upsertSourceSQL,
			src.FileID,
			src.ChannelID,
			src.MsgID,
			src.GroupedID,
			albumIDs,
		); err != nil {
			return fmt.Errorf("upsert file_source (%s, %d, %d): %w", src.FileID, src.ChannelID, src.MsgID, err)
		}
	}

	const advanceWatermarkSQL = `
UPDATE core_public.channels
SET
	last_msg_id = GREATEST(last_msg_id, $2),
	max_edit_date_checked = CASE
		WHEN $3::timestamptz IS NULL THEN max_edit_date_checked
		WHEN max_edit_date_checked IS NULL THEN $3::timestamptz
		ELSE GREATEST(max_edit_date_checked, $3::timestamptz)
	END
WHERE channel_id = $1`

	if _, err := tx.Exec(ctx, advanceWatermarkSQL, channelID, maxMsgID, maxEditDate); err != nil {
		return fmt.Errorf("advance watermark for channel %d: %w", channelID, err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit watermark tx for channel %d: %w", channelID, err)
	}
	return nil
}

// GetPostSyncMeta loads existing edit_date, content_sha256, and reply sync counters from core_public.posts.
func (r *Repository) GetPostSyncMeta(ctx context.Context, channelID int64, msgID int) (domain.PostSyncMeta, bool, error) {
	const q = `
SELECT
	channel_id,
	msg_id,
	edit_date,
	content_sha256,
	replies_count,
	replies_max_id,
	replies_synced_count
FROM core_public.posts
WHERE channel_id = $1 AND msg_id = $2`

	var meta domain.PostSyncMeta
	err := r.pool.QueryRow(ctx, q, channelID, msgID).Scan(
		&meta.ChannelID,
		&meta.MsgID,
		&meta.EditDate,
		&meta.ContentSHA256,
		&meta.RepliesCount,
		&meta.RepliesMaxID,
		&meta.RepliesSyncedCount,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return domain.PostSyncMeta{}, false, nil
	}
	if err != nil {
		return domain.PostSyncMeta{}, false, fmt.Errorf("get post sync meta (%d, %d): %w", channelID, msgID, err)
	}
	return meta, true, nil
}

// UpdatePostRepliesSynced records updated comment counters and high-watermark comment ID on core_public.posts.
func (r *Repository) UpdatePostRepliesSynced(
	ctx context.Context,
	channelID int64,
	msgID int,
	repliesCount, repliesMaxID, syncedCount int,
) error {
	const q = `
UPDATE core_public.posts
SET
	replies_count        = GREATEST(replies_count, $3),
	replies_max_id       = GREATEST(replies_max_id, $4),
	replies_synced_count = GREATEST(replies_synced_count, $5)
WHERE channel_id = $1 AND msg_id = $2`

	if _, err := r.pool.Exec(ctx, q, channelID, msgID, repliesCount, repliesMaxID, syncedCount); err != nil {
		return fmt.Errorf("update post replies synced (%d, %d): %w", channelID, msgID, err)
	}
	return nil
}

// GetRecentPostIDsForEditRescan fetches message IDs within the active rescan window (e.g. 24h)
// plus pinned channel messages so edits made after publication are detected during polling.
func (r *Repository) GetRecentPostIDsForEditRescan(
	ctx context.Context,
	channelID int64,
	windowHours int,
	limit int,
) ([]int, error) {
	if windowHours <= 0 {
		windowHours = 24
	}
	if limit <= 0 {
		limit = 100
	}

	const q = `
WITH candidates AS (
	SELECT msg_id
	FROM core_public.posts
	WHERE channel_id = $1
		AND is_deleted = FALSE
		AND published_at >= NOW() - make_interval(hours => $2)
	UNION
	SELECT unnest(pinned_msg_ids) AS msg_id
	FROM core_public.channels
	WHERE channel_id = $1
)
SELECT msg_id
FROM candidates
ORDER BY msg_id DESC
LIMIT $3`

	rows, err := r.pool.Query(ctx, q, channelID, windowHours, limit)
	if err != nil {
		return nil, fmt.Errorf("query recent posts for edit rescan %d: %w", channelID, err)
	}
	defer rows.Close()

	var ids []int
	for rows.Next() {
		var id int
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan post id for rescan: %w", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate rescan post ids: %w", err)
	}
	return ids, nil
}

// GetAccessHash retrieves the eternal per-account access_hash for a peer_id.
func (r *Repository) GetAccessHash(ctx context.Context, accountID int, peerID int64) (int64, bool, error) {
	const q = `
SELECT access_hash
FROM core_public.channel_access_hashes
WHERE account_id = $1 AND peer_id = $2`

	var hash int64
	err := r.pool.QueryRow(ctx, q, accountID, peerID).Scan(&hash)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, fmt.Errorf("get access_hash (account=%d, peer=%d): %w", accountID, peerID, err)
	}
	return hash, true, nil
}

// SaveAccessHash indefinitely caches an access_hash bound to a specific MTProto account_id.
func (r *Repository) SaveAccessHash(ctx context.Context, accountID int, peerID int64, peerType string, accessHash int64) error {
	if peerType != "channel" && peerType != "discussion" {
		return fmt.Errorf("invalid peer_type %q", peerType)
	}
	const q = `
INSERT INTO core_public.channel_access_hashes (account_id, peer_id, peer_type, access_hash, resolved_at)
VALUES ($1, $2, $3, $4, NOW())
ON CONFLICT (account_id, peer_id) DO UPDATE SET
	peer_type   = EXCLUDED.peer_type,
	access_hash = EXCLUDED.access_hash,
	resolved_at = NOW()`

	if _, err := r.pool.Exec(ctx, q, accountID, peerID, peerType, accessHash); err != nil {
		return fmt.Errorf("save access_hash (account=%d, peer=%d): %w", accountID, peerID, err)
	}
	return nil
}

// DeleteAccountAccessHashes invalidates all cached access_hashes when an account's auth_key changes or dies.
func (r *Repository) DeleteAccountAccessHashes(ctx context.Context, accountID int) error {
	const q = `DELETE FROM core_public.channel_access_hashes WHERE account_id = $1`
	if _, err := r.pool.Exec(ctx, q, accountID); err != nil {
		return fmt.Errorf("delete access hashes for account %d: %w", accountID, err)
	}
	return nil
}

// UpsertSession registers or loads an MTProto session row by unique phone_hash.
func (r *Repository) UpsertSession(ctx context.Context, rec domain.SessionRecord) (int, error) {
	state := rec.FSMState
	if state == "" {
		state = domain.SessionActivePolling
	}

	const q = `
INSERT INTO core_public.mtproto_sessions (
	phone_hash,
	is_premium,
	fsm_state,
	risk_score,
	proxy_ipv4,
	flood_wait_until,
	bbolt_file_path
) VALUES (
	$1, $2, $3::core_public.session_state_enum, $4, $5::inet, $6, $7
)
ON CONFLICT (phone_hash) DO UPDATE SET
	is_premium       = EXCLUDED.is_premium,
	bbolt_file_path  = EXCLUDED.bbolt_file_path,
	proxy_ipv4       = COALESCE(EXCLUDED.proxy_ipv4, core_public.mtproto_sessions.proxy_ipv4)
RETURNING account_id`

	var accountID int
	err := r.pool.QueryRow(ctx, q,
		rec.PhoneHash,
		rec.IsPremium,
		string(state),
		rec.RiskScore,
		rec.ProxyIPv4,
		rec.FloodWaitUntil,
		rec.BBoltFilePath,
	).Scan(&accountID)
	if err != nil {
		return 0, fmt.Errorf("upsert mtproto session %q: %w", rec.PhoneHash, err)
	}
	return accountID, nil
}

// SetSessionFloodQuarantine persists a FLOOD_WAIT > 60s quarantine deadline in PostgreSQL.
func (r *Repository) SetSessionFloodQuarantine(ctx context.Context, accountID int, until time.Time, riskDelta int) error {
	const q = `
UPDATE core_public.mtproto_sessions
SET
	fsm_state        = 'DEGRADED_QUARANTINE'::core_public.session_state_enum,
	flood_wait_until = $2,
	risk_score       = LEAST(100, GREATEST(0, risk_score + $3))
WHERE account_id = $1`

	if _, err := r.pool.Exec(ctx, q, accountID, until.UTC(), riskDelta); err != nil {
		return fmt.Errorf("set flood quarantine for account %d: %w", accountID, err)
	}
	return nil
}

// SetSessionFSMState updates session FSM state and risk_score.
func (r *Repository) SetSessionFSMState(ctx context.Context, accountID int, state domain.SessionFSMState, riskScore int) error {
	const q = `
UPDATE core_public.mtproto_sessions
SET
	fsm_state  = $2::core_public.session_state_enum,
	risk_score = LEAST(100, GREATEST(0, $3))
WHERE account_id = $1`

	if _, err := r.pool.Exec(ctx, q, accountID, string(state), riskScore); err != nil {
		return fmt.Errorf("set fsm_state %s for account %d: %w", state, accountID, err)
	}
	return nil
}

// GetFileReference fetches file metadata and current file_reference bytes for streaming.
func (r *Repository) GetFileReference(ctx context.Context, fileID string) (domain.FileReferenceRecord, bool, error) {
	const q = `
SELECT
	file_id,
	tg_id,
	access_hash,
	dc_id,
	media_kind::text,
	thumb_size,
	file_reference,
	file_size,
	mime_type,
	local_cache_path,
	local_cache_expires_at,
	refreshed_at,
	is_permanently_gone
FROM core_public.file_references
WHERE file_id = $1`

	var (
		rec       domain.FileReferenceRecord
		mediaKind string
	)
	err := r.pool.QueryRow(ctx, q, fileID).Scan(
		&rec.FileID,
		&rec.TGID,
		&rec.AccessHash,
		&rec.DCID,
		&mediaKind,
		&rec.ThumbSize,
		&rec.FileReference,
		&rec.FileSize,
		&rec.MimeType,
		&rec.LocalCachePath,
		&rec.LocalCacheExpires,
		&rec.RefreshedAt,
		&rec.IsPermanentlyGone,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return domain.FileReferenceRecord{}, false, nil
	}
	if err != nil {
		return domain.FileReferenceRecord{}, false, fmt.Errorf("get file_reference %q: %w", fileID, err)
	}
	rec.MediaKind = domain.MediaKind(mediaKind)
	return rec, true, nil
}

// GetActiveFileSources lists non-deleted channel messages hosting fileID, newest first.
func (r *Repository) GetActiveFileSources(ctx context.Context, fileID string) ([]domain.FileSourceRecord, error) {
	const q = `
SELECT
	source_id,
	file_id,
	channel_id,
	msg_id,
	grouped_id,
	album_msg_ids,
	is_source_deleted,
	created_at
FROM core_public.file_sources
WHERE file_id = $1 AND is_source_deleted = FALSE
ORDER BY created_at DESC`

	rows, err := r.pool.Query(ctx, q, fileID)
	if err != nil {
		return nil, fmt.Errorf("query active file_sources for %q: %w", fileID, err)
	}
	defer rows.Close()

	var sources []domain.FileSourceRecord
	for rows.Next() {
		var src domain.FileSourceRecord
		if err := rows.Scan(
			&src.SourceID,
			&src.FileID,
			&src.ChannelID,
			&src.MsgID,
			&src.GroupedID,
			&src.AlbumMsgIDs,
			&src.IsSourceDeleted,
			&src.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan file_source for %q: %w", fileID, err)
		}
		sources = append(sources, src)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate file_sources for %q: %w", fileID, err)
	}
	return sources, nil
}

// UpdateFileReferenceBytes persists a refreshed cryptographic file_reference token.
func (r *Repository) UpdateFileReferenceBytes(ctx context.Context, fileID string, newBytes []byte, fileSize int64) error {
	const q = `
UPDATE core_public.file_references
SET
	file_reference = $2,
	file_size      = CASE WHEN $3 > 0 THEN $3 ELSE file_size END,
	refreshed_at   = NOW()
WHERE file_id = $1`

	if _, err := r.pool.Exec(ctx, q, fileID, newBytes, fileSize); err != nil {
		return fmt.Errorf("update file_reference bytes for %q: %w", fileID, err)
	}
	return nil
}

// MarkFileSourceDeleted flags a single file source message as deleted in Telegram.
func (r *Repository) MarkFileSourceDeleted(ctx context.Context, sourceID int64) error {
	const q = `
UPDATE core_public.file_sources
SET
	is_source_deleted       = TRUE,
	last_refresh_attempt_at = NOW()
WHERE source_id = $1`

	if _, err := r.pool.Exec(ctx, q, sourceID); err != nil {
		return fmt.Errorf("mark file_source deleted %d: %w", sourceID, err)
	}
	return nil
}

// MarkFilePermanentlyGone marks a file_id as unrecoverable (triggers HTTP 410 Gone).
func (r *Repository) MarkFilePermanentlyGone(ctx context.Context, fileID string) error {
	const q = `
UPDATE core_public.file_references
SET
	is_permanently_gone = TRUE,
	refreshed_at        = NOW()
WHERE file_id = $1`

	if _, err := r.pool.Exec(ctx, q, fileID); err != nil {
		return fmt.Errorf("mark file permanently gone %q: %w", fileID, err)
	}
	return nil
}

// UpdateFileLocalCache updates the local disk cache location and expiration timestamp.
func (r *Repository) UpdateFileLocalCache(ctx context.Context, fileID string, cachePath string, expiresAt time.Time) error {
	const q = `
UPDATE core_public.file_references
SET
	local_cache_path       = $2,
	local_cache_expires_at = $3
WHERE file_id = $1`

	if _, err := r.pool.Exec(ctx, q, fileID, cachePath, expiresAt.UTC()); err != nil {
		return fmt.Errorf("update local cache path for %q: %w", fileID, err)
	}
	return nil
}
