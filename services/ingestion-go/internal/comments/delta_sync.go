package comments

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"github.com/google/uuid"
	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/updates"
)

// CommentsAPI abstracts the Telegram RPCs needed for discussion resolution and reply fetching.
type CommentsAPI interface {
	MessagesGetDiscussionMessage(ctx context.Context, request *tg.MessagesGetDiscussionMessageRequest) (*tg.MessagesDiscussionMessage, error)
	MessagesGetReplies(ctx context.Context, request *tg.MessagesGetRepliesRequest) (tg.MessagesMessagesClass, error)
}

// DeltaSyncService fetches comment replies without joining discussion supergroups:
//  1. Resolves and caches linked_chat_id once via messages.getDiscussionMessage.
//  2. Strictly invokes messages.getReplies ONLY when replies.replies > replies_synced_count in DB.
//  3. Publishes anonymized delta comments synchronously to stream:ingest:comments before updating DB cursors.
type DeltaSyncService struct {
	accountID    int
	api          CommentsAPI
	watermarks   domain.WatermarkStore
	accessHashes domain.AccessHashStore
	publisher    domain.EventPublisher
	authorSalt   string
	pageLimit    int
	logger       *slog.Logger
}

// NewDeltaSyncService creates a comment delta synchronization service.
func NewDeltaSyncService(
	accountID int,
	api CommentsAPI,
	watermarks domain.WatermarkStore,
	accessHashes domain.AccessHashStore,
	publisher domain.EventPublisher,
	logger *slog.Logger,
) *DeltaSyncService {
	if logger == nil {
		logger = slog.Default()
	}
	return &DeltaSyncService{
		accountID:    accountID,
		api:          api,
		watermarks:   watermarks,
		accessHashes: accessHashes,
		publisher:    publisher,
		authorSalt:   "tg-agg-anon-commenter-v1",
		pageLimit:    100,
		logger:       logger,
	}
}

// SyncPostComments checks whether telegramReplies.RepliesCount exceeds the already-synced count
// in PostgreSQL and, ONLY if strictly greater, paginates messages.getReplies from replies_max_id.
func (s *DeltaSyncService) SyncPostComments(
	ctx context.Context,
	channelID int64,
	postMsgID int,
	telegramReplies domain.RepliesInfo,
) (int, error) {
	if s.api == nil {
		return 0, errors.New("comments API is nil")
	}
	if !telegramReplies.CommentsEnabled {
		return 0, nil
	}

	var (
		syncedCount  int
		repliesMaxID int
	)
	if s.watermarks != nil {
		meta, found, err := s.watermarks.GetPostSyncMeta(ctx, channelID, postMsgID)
		if err != nil {
			return 0, fmt.Errorf("load post sync meta (%d, %d): %w", channelID, postMsgID, err)
		}
		if found {
			syncedCount = meta.RepliesSyncedCount
			repliesMaxID = meta.RepliesMaxID
		}
	}

	// Delta invariant: never call messages.getReplies unless replies.replies grew relative to DB.
	if telegramReplies.RepliesCount <= syncedCount {
		return 0, nil
	}

	chAccessHash := int64(0)
	if s.accessHashes != nil {
		if h, ok, err := s.accessHashes.GetAccessHash(ctx, s.accountID, channelID); err == nil && ok {
			chAccessHash = h
		}
	}

	linkedChatID, err := s.ensureLinkedChatID(ctx, channelID, chAccessHash, postMsgID, telegramReplies.LinkedChatID)
	if err != nil {
		return 0, err
	}

	rawComments, err := s.fetchDeltaReplies(ctx, channelID, chAccessHash, postMsgID, repliesMaxID)
	if err != nil {
		return 0, err
	}
	if len(rawComments) == 0 {
		return 0, nil
	}

	items := make([]domain.IngestCommentItem, 0, len(rawComments))
	newMaxID := repliesMaxID
	for _, msg := range rawComments {
		if msg.ID <= repliesMaxID {
			continue
		}
		if msg.ID > newMaxID {
			newMaxID = msg.ID
		}
		items = append(items, s.toCommentItem(msg))
	}
	if len(items) == 0 {
		return 0, nil
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].CommentMsgID < items[j].CommentMsgID
	})

	if s.publisher != nil {
		evt := domain.IngestCommentsDeltaEventV1{
			SchemaVersion:     "ingest.comments.v1",
			EventID:           uuid.NewString(),
			ChannelID:         channelID,
			PostMsgID:         postMsgID,
			LinkedChatID:      linkedChatID,
			TotalRepliesCount: telegramReplies.RepliesCount,
			Comments:          items,
		}
		entryID, err := s.publisher.PublishCommentsDelta(ctx, evt)
		if err != nil {
			return 0, fmt.Errorf("publish delta comments (%d, %d): %w", channelID, postMsgID, err)
		}
		if entryID == "" {
			return 0, fmt.Errorf("publish delta comments (%d, %d): empty redis stream entry id", channelID, postMsgID)
		}
	}

	newSyncedCount := syncedCount + len(items)
	if newSyncedCount < telegramReplies.RepliesCount {
		// Record exact count fetched so another delta triggers if server reports even more.
	} else if newSyncedCount > telegramReplies.RepliesCount {
		newSyncedCount = telegramReplies.RepliesCount
	}

	if s.watermarks != nil {
		if err := s.watermarks.UpdatePostRepliesSynced(
			ctx,
			channelID,
			postMsgID,
			telegramReplies.RepliesCount,
			newMaxID,
			newSyncedCount,
		); err != nil {
			return 0, fmt.Errorf("update replies watermark (%d, %d): %w", channelID, postMsgID, err)
		}
	}

	return len(items), nil
}

func (s *DeltaSyncService) ensureLinkedChatID(
	ctx context.Context,
	channelID int64,
	chAccessHash int64,
	postMsgID int,
	hintLinkedID *int64,
) (int64, error) {
	if s.watermarks != nil {
		ch, found, err := s.watermarks.GetChannelState(ctx, channelID)
		if err != nil {
			return 0, fmt.Errorf("load channel %d for linked_chat_id: %w", channelID, err)
		}
		if found && ch.LinkedChatID != nil && *ch.LinkedChatID != 0 {
			return *ch.LinkedChatID, nil
		}
	}

	if hintLinkedID != nil && *hintLinkedID != 0 && s.accessHashes != nil {
		if _, ok, err := s.accessHashes.GetAccessHash(ctx, s.accountID, *hintLinkedID); err == nil && ok {
			if s.watermarks != nil {
				_ = s.watermarks.SetChannelLinkedChatID(ctx, channelID, *hintLinkedID)
			}
			return *hintLinkedID, nil
		}
	}

	disc, err := s.api.MessagesGetDiscussionMessage(ctx, &tg.MessagesGetDiscussionMessageRequest{
		Peer: &tg.InputPeerChannel{
			ChannelID:  channelID,
			AccessHash: chAccessHash,
		},
		MsgID: postMsgID,
	})
	if err != nil {
		return 0, fmt.Errorf("messages.getDiscussionMessage (%d, %d): %w", channelID, postMsgID, err)
	}

	linkedID, linkedHash, err := extractDiscussionChat(disc)
	if err != nil {
		return 0, err
	}

	if s.watermarks != nil {
		if err := s.watermarks.SetChannelLinkedChatID(ctx, channelID, linkedID); err != nil {
			return 0, fmt.Errorf("persist linked_chat_id %d for channel %d: %w", linkedID, channelID, err)
		}
	}
	if s.accessHashes != nil && linkedHash != 0 {
		if err := s.accessHashes.SaveAccessHash(ctx, s.accountID, linkedID, "discussion", linkedHash); err != nil {
			return 0, fmt.Errorf("persist discussion access_hash %d: %w", linkedID, err)
		}
	}
	return linkedID, nil
}

func (s *DeltaSyncService) fetchDeltaReplies(
	ctx context.Context,
	channelID int64,
	chAccessHash int64,
	postMsgID int,
	minID int,
) ([]*tg.Message, error) {
	const maxPages = 20
	limit := s.pageLimit
	if limit <= 0 {
		limit = 100
	}

	seen := make(map[int]*tg.Message)
	offsetID := 0

	for page := 0; page < maxPages; page++ {
		res, err := s.api.MessagesGetReplies(ctx, &tg.MessagesGetRepliesRequest{
			Peer: &tg.InputPeerChannel{
				ChannelID:  channelID,
				AccessHash: chAccessHash,
			},
			MsgID:    postMsgID,
			OffsetID: offsetID,
			Limit:    limit,
			MinID:    minID,
		})
		if err != nil {
			return nil, fmt.Errorf("messages.getReplies (%d, %d, offset=%d): %w", channelID, postMsgID, offsetID, err)
		}

		pageMsgs := extractCommentMessages(res)
		if len(pageMsgs) == 0 {
			break
		}

		minPageID := 0
		for _, rm := range pageMsgs {
			msg, ok := rm.(*tg.Message)
			if !ok || msg == nil {
				continue
			}
			if minPageID == 0 || msg.ID < minPageID {
				minPageID = msg.ID
			}
			if msg.ID > minID {
				seen[msg.ID] = msg
			}
		}

		if len(pageMsgs) < limit || minPageID <= minID+1 || minPageID == 0 || minPageID == offsetID {
			break
		}
		offsetID = minPageID
	}

	out := make([]*tg.Message, 0, len(seen))
	for _, msg := range seen {
		out = append(out, msg)
	}
	return out, nil
}

func (s *DeltaSyncService) toCommentItem(msg *tg.Message) domain.IngestCommentItem {
	trimmed := updates.TrimRightSpaces(msg.Message)
	entities := updates.MapEntities(msg.Entities, updates.UTF16CodeUnitLen(trimmed))

	var replyToPtr *int
	if rh, ok := msg.GetReplyTo(); ok && rh != nil {
		if header, ok := rh.(*tg.MessageReplyHeader); ok && header != nil {
			if targetID, hasTarget := header.GetReplyToMsgID(); hasTarget && targetID > 0 {
				v := targetID
				replyToPtr = &v
			}
		}
	}

	var editPtr *string
	if editSec, ok := msg.GetEditDate(); ok && editSec > 0 {
		str := time.Unix(int64(editSec), 0).UTC().Format(time.RFC3339)
		editPtr = &str
	}

	return domain.IngestCommentItem{
		CommentMsgID:   msg.ID,
		ReplyToMsgID:   replyToPtr,
		AuthorPeerHash: s.anonymizeAuthor(msg),
		PublishedAt:    time.Unix(int64(msg.Date), 0).UTC().Format(time.RFC3339),
		EditDate:       editPtr,
		TextRaw:        trimmed,
		Entities:       entities,
	}
}

func (s *DeltaSyncService) anonymizeAuthor(msg *tg.Message) string {
	var raw string
	if from, ok := msg.GetFromID(); ok && from != nil {
		switch p := from.(type) {
		case *tg.PeerUser:
			raw = fmt.Sprintf("%s:user:%d", s.authorSalt, p.UserID)
		case *tg.PeerChannel:
			raw = fmt.Sprintf("%s:channel:%d", s.authorSalt, p.ChannelID)
		case *tg.PeerChat:
			raw = fmt.Sprintf("%s:chat:%d", s.authorSalt, p.ChatID)
		default:
			raw = fmt.Sprintf("%s:anon:%d", s.authorSalt, msg.ID)
		}
	} else {
		raw = fmt.Sprintf("%s:hidden:%d", s.authorSalt, msg.ID)
	}
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:8])
}

func extractDiscussionChat(disc *tg.MessagesDiscussionMessage) (int64, int64, error) {
	if disc == nil {
		return 0, 0, errors.New("nil MessagesDiscussionMessage response")
	}

	var targetID int64
	for _, rm := range disc.GetMessages() {
		if msg, ok := rm.(*tg.Message); ok && msg != nil {
			if peerCh, ok := msg.PeerID.(*tg.PeerChannel); ok && peerCh != nil {
				targetID = peerCh.ChannelID
				break
			}
		}
	}

	for _, rc := range disc.GetChats() {
		ch, ok := rc.(*tg.Channel)
		if !ok || ch == nil {
			continue
		}
		if targetID != 0 && ch.ID == targetID {
			hash, _ := ch.GetAccessHash()
			return ch.ID, hash, nil
		}
	}

	// Fallback if messages did not include PeerChannel directly.
	for _, rc := range disc.GetChats() {
		ch, ok := rc.(*tg.Channel)
		if !ok || ch == nil {
			continue
		}
		if ch.Megagroup {
			hash, _ := ch.GetAccessHash()
			return ch.ID, hash, nil
		}
	}

	return 0, 0, errors.New("linked discussion supergroup missing in MessagesDiscussionMessage")
}

func extractCommentMessages(res tg.MessagesMessagesClass) []tg.MessageClass {
	switch v := res.(type) {
	case *tg.MessagesChannelMessages:
		return v.GetMessages()
	case *tg.MessagesMessagesSlice:
		return v.GetMessages()
	case *tg.MessagesMessages:
		return v.GetMessages()
	default:
		return nil
	}
}
