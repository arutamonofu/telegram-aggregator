package updates

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/album"
	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/mtproto"
)

// CommentSyncTrigger requests delta comment synchronization when replies.replies exceeds DB state.
type CommentSyncTrigger func(ctx context.Context, channelID int64, postMsgID int, replies domain.RepliesInfo)

// ChannelPipeline routes live MTProto updates and polled history through the album assembler,
// enforces Watermark Commit Invariant (Redis XADD ACK -> DB TX commit -> durable pts release),
// deduplicates dummy edit_date mutations, and implements mtproto.RecoveredBatchSink.
type ChannelPipeline struct {
	accountID    int
	watermarks   domain.WatermarkStore
	accessHashes domain.AccessHashStore
	publisher    domain.EventPublisher
	stateStorage *WatermarkSafeStateStorage
	assembler    *album.Assembler
	onComments   CommentSyncTrigger
	logger       *slog.Logger
}

var _ mtproto.RecoveredBatchSink = (*ChannelPipeline)(nil)

// NewChannelPipeline wires live dispatching, album buffering, and durable watermark commits.
func NewChannelPipeline(
	accountID int,
	albumWindow time.Duration,
	watermarks domain.WatermarkStore,
	accessHashes domain.AccessHashStore,
	publisher domain.EventPublisher,
	stateStorage *WatermarkSafeStateStorage,
	onComments CommentSyncTrigger,
	logger *slog.Logger,
) *ChannelPipeline {
	if logger == nil {
		logger = slog.Default()
	}

	p := &ChannelPipeline{
		accountID:    accountID,
		watermarks:   watermarks,
		accessHashes: accessHashes,
		publisher:    publisher,
		stateStorage: stateStorage,
		onComments:   onComments,
		logger:       logger,
	}

	hooks := album.HoldHooks{
		OnAcquire: func(channelID int64) {
			if stateStorage != nil {
				stateStorage.AcquireChannelHold(channelID)
			}
		},
	}

	p.assembler = album.NewAssembler(albumWindow, p.commitFlushedBatch, hooks)
	return p
}

// Assembler exposes the underlying album.Assembler.
func (p *ChannelPipeline) Assembler() *album.Assembler {
	return p.assembler
}

// RegisterHandlers wires tg.UpdateDispatcher handlers for channel posts, edits, and deletions.
func (p *ChannelPipeline) RegisterHandlers(disp tg.UpdateDispatcher) {
	disp.OnNewChannelMessage(func(ctx context.Context, e tg.Entities, update *tg.UpdateNewChannelMessage) error {
		msg, ok := update.GetMessage().(*tg.Message)
		if !ok || msg == nil {
			return nil
		}
		return p.handleChannelMessage(ctx, e.Channels, msg, domain.EventTypePostNew)
	})

	disp.OnEditChannelMessage(func(ctx context.Context, e tg.Entities, update *tg.UpdateEditChannelMessage) error {
		msg, ok := update.GetMessage().(*tg.Message)
		if !ok || msg == nil {
			return nil
		}
		return p.handleChannelMessage(ctx, e.Channels, msg, domain.EventTypePostEdit)
	})

	disp.OnDeleteChannelMessages(func(ctx context.Context, _ tg.Entities, update *tg.UpdateDeleteChannelMessages) error {
		if update == nil || len(update.Messages) == 0 {
			return nil
		}
		return p.handleDeletions(ctx, update.ChannelID, update.Messages)
	})
}

// IngestRecoveredMessages processes messages extracted from updates.channelDifferenceTooLong + messages.getHistory.
func (p *ChannelPipeline) IngestRecoveredMessages(
	ctx context.Context,
	channelID int64,
	chats []tg.ChatClass,
	_ []tg.UserClass,
	messages []*tg.Message,
) error {
	chMap := BuildChannelEntitiesMap(chats)
	for _, msg := range messages {
		if err := p.handleChannelMessage(ctx, chMap, msg, domain.EventTypePostNew); err != nil {
			return err
		}
	}
	if err := p.assembler.FlushChannel(ctx, channelID); err != nil {
		return fmt.Errorf("flush recovered albums for channel %d: %w", channelID, err)
	}
	return nil
}

// ProcessPolledMessage processes a post fetched during fallback polling / edit rescaning.
func (p *ChannelPipeline) ProcessPolledMessage(
	ctx context.Context,
	chMap map[int64]*tg.Channel,
	msg *tg.Message,
	eventType domain.EventType,
) error {
	return p.handleChannelMessage(ctx, chMap, msg, eventType)
}

func (p *ChannelPipeline) handleChannelMessage(
	ctx context.Context,
	chEntities map[int64]*tg.Channel,
	msg *tg.Message,
	eventType domain.EventType,
) error {
	chCtx := p.resolveChannelContext(ctx, chEntities, msg)
	canonical, err := CanonicalizeMessage(msg, chCtx, eventType)
	if err != nil {
		return err
	}

	// Edit deduplication & delta comment check against PostgreSQL post metadata.
	if eventType == domain.EventTypePostEdit && p.watermarks != nil {
		meta, found, err := p.watermarks.GetPostSyncMeta(ctx, chCtx.ChannelID, msg.ID)
		if err != nil {
			return err
		}
		if found {
			if p.onComments != nil && canonical.Event.Replies.CommentsEnabled &&
				canonical.Event.Replies.RepliesCount > meta.RepliesSyncedCount {
				p.onComments(ctx, chCtx.ChannelID, msg.ID, canonical.Event.Replies)
			}

			sameHash := bytes.Equal(meta.ContentSHA256, canonical.ContentHash)
			sameEditDate := (meta.EditDate == nil && canonical.EditDateTime == nil) ||
				(meta.EditDate != nil && canonical.EditDateTime != nil && canonical.EditDateTime.Equal(*meta.EditDate))
			if sameHash && len(canonical.Event.MediaItems) == 0 {
				// Ignore noisy edit_date / view-counter updates when content hash is unchanged.
				p.logger.Debug("ignoring no-op edit_date mutation",
					"channel_id", chCtx.ChannelID,
					"msg_id", msg.ID,
					"same_edit_date", sameEditDate,
				)
				return nil
			}
		}
	}

	if err := p.assembler.Push(ctx, canonical.ToBufferedMessage()); err != nil {
		return err
	}

	if eventType == domain.EventTypePostNew && p.onComments != nil &&
		canonical.Event.Replies.CommentsEnabled && canonical.Event.Replies.RepliesCount > 0 {
		p.onComments(ctx, chCtx.ChannelID, msg.ID, canonical.Event.Replies)
	}
	return nil
}

func (p *ChannelPipeline) commitFlushedBatch(ctx context.Context, batch album.FlushedBatch) error {
	isAlbum := batch.Event.GroupedID != nil && *batch.Event.GroupedID != 0

	// Watermark Commit Invariant:
	// 1. Synchronous Redis XADD + wait for non-empty Entry ID ACK.
	if p.publisher != nil {
		entryID, err := p.publisher.PublishPost(ctx, batch.Event)
		if err != nil || entryID == "" {
			if isAlbum && p.stateStorage != nil {
				p.stateStorage.AbortChannelHold(batch.ChannelID)
			}
			if err != nil {
				return fmt.Errorf("publish post batch (%d, %d): %w", batch.ChannelID, batch.Event.MsgID, err)
			}
			return fmt.Errorf("publish post batch (%d, %d): empty redis stream entry id", batch.ChannelID, batch.Event.MsgID)
		}
	}

	// 2. PostgreSQL TX commit: upsert file_references, file_sources, and advance channel watermarks.
	if p.watermarks != nil {
		if err := p.watermarks.CommitPostWatermarkAndFiles(
			ctx,
			batch.ChannelID,
			batch.MaxMsgID,
			batch.MaxEditDate,
			batch.FileRefs,
			batch.FileSources,
		); err != nil {
			if isAlbum && p.stateStorage != nil {
				p.stateStorage.AbortChannelHold(batch.ChannelID)
			}
			return fmt.Errorf("commit watermark (%d, %d): %w", batch.ChannelID, batch.MaxMsgID, err)
		}
	}

	// 3. Only after Redis ACK + PG commit succeeded, release durable channel_pts hold in bbolt.
	if isAlbum && p.stateStorage != nil {
		if err := p.stateStorage.CommitAndReleaseChannelHold(ctx, batch.ChannelID); err != nil {
			return fmt.Errorf("release channel pts hold %d: %w", batch.ChannelID, err)
		}
	}
	return nil
}

func (p *ChannelPipeline) handleDeletions(ctx context.Context, channelID int64, msgIDs []int) error {
	if p.publisher == nil {
		return nil
	}
	evt := domain.IngestDeletionEventV1{
		SchemaVersion: "ingest.deletion.v1",
		EventID:       uuid.NewString(),
		ChannelID:     channelID,
		DeletedMsgIDs: msgIDs,
		OccurredAt:    time.Now().UTC().Format(time.RFC3339),
	}
	if _, err := p.publisher.PublishDeletion(ctx, evt); err != nil {
		return fmt.Errorf("publish channel deletions (%d): %w", channelID, err)
	}
	return nil
}

func (p *ChannelPipeline) resolveChannelContext(
	ctx context.Context,
	chEntities map[int64]*tg.Channel,
	msg *tg.Message,
) ChannelContext {
	var chCtx ChannelContext
	if peerCh, ok := msg.PeerID.(*tg.PeerChannel); ok && peerCh != nil {
		chCtx.ChannelID = peerCh.ChannelID
	}

	if ent, ok := chEntities[chCtx.ChannelID]; ok && ent != nil {
		chCtx.Title = ent.Title
		if u, hasU := ent.GetUsername(); hasU && u != "" {
			chCtx.Username = &u
		}
		if h, hasHash := ent.GetAccessHash(); hasHash {
			chCtx.AccessHash = h
			if p.accessHashes != nil && p.accountID > 0 {
				_ = p.accessHashes.SaveAccessHash(ctx, p.accountID, chCtx.ChannelID, "channel", h)
			}
		}
	}

	if (chCtx.Title == "" || chCtx.Username == nil) && p.watermarks != nil && chCtx.ChannelID != 0 {
		if st, found, err := p.watermarks.GetChannelState(ctx, chCtx.ChannelID); err == nil && found {
			if chCtx.Title == "" {
				chCtx.Title = st.Title
			}
			if chCtx.Username == nil && st.Username != nil {
				u := *st.Username
				chCtx.Username = &u
			}
		}
	}
	return chCtx
}
