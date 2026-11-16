package mtproto

import (
	"context"
	"fmt"
	"sort"

	"github.com/gotd/td/bin"
	"github.com/gotd/td/telegram"
	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

// HistoryAPI abstracts messages.getHistory for channel gap backfilling and tests.
type HistoryAPI interface {
	MessagesGetHistory(ctx context.Context, request *tg.MessagesGetHistoryRequest) (tg.MessagesMessagesClass, error)
}

// ChannelTooLongPayload holds extracted updates.channelDifferenceTooLong fields before pts reset.
type ChannelTooLongPayload struct {
	ChannelID    int64
	AccessHash   int64
	RequestedPts int
	DialogPts    int
	TopMessage   int
	Final        bool
	Messages     []tg.MessageClass
	Chats        []tg.ChatClass
	Users        []tg.UserClass
}

// TooLongHandler processes updates.channelDifferenceTooLong BEFORE updates.Manager resets local pts.
// Returning an error aborts the RPC middleware result so local pts is NOT advanced until recovery succeeds.
type TooLongHandler interface {
	HandleTooLong(ctx context.Context, payload ChannelTooLongPayload) error
}

// TooLongHandlerFunc adapts a plain function to TooLongHandler.
type TooLongHandlerFunc func(ctx context.Context, payload ChannelTooLongPayload) error

// HandleTooLong implements TooLongHandler.
func (f TooLongHandlerFunc) HandleTooLong(ctx context.Context, payload ChannelTooLongPayload) error {
	return f(ctx, payload)
}

// RecoveredBatchSink receives extracted & backfilled channel messages in ascending msg_id order
// and must persist/publish them with synchronous ACK before returning nil.
type RecoveredBatchSink interface {
	IngestRecoveredMessages(
		ctx context.Context,
		channelID int64,
		chats []tg.ChatClass,
		users []tg.UserClass,
		messages []*tg.Message,
	) error
}

// NewChannelTooLongPatchMiddleware builds a telegram.Middleware that works around gotd/td#1711
// and preserves the Watermark Commit Invariant:
//  1. Intercepts updates.getChannelDifference returning *tg.UpdatesChannelDifferenceTooLong.
//  2. Extracts dialog PTS/TopMessage and nested messages, chats, users BEFORE updates.Manager
//     can advance or reset local pts.
//  3. Runs handler synchronously (including messages.getHistory backfill and synchronous Redis/PG commit);
//     if handler returns an error, the middleware returns that error so pts does NOT advance.
func NewChannelTooLongPatchMiddleware(handler TooLongHandler) telegram.Middleware {
	return tooLongMiddleware{handler: handler}
}

type tooLongMiddleware struct {
	handler TooLongHandler
}

func (m tooLongMiddleware) Handle(next tg.Invoker) telegram.InvokeFunc {
	return func(ctx context.Context, input bin.Encoder, output bin.Decoder) error {
		if err := next.Invoke(ctx, input, output); err != nil {
			return err
		}

		if m.handler == nil {
			return nil
		}

		tooLong, ok := extractTooLong(output)
		if !ok {
			return nil
		}

		payload := ChannelTooLongPayload{
			Final:    tooLong.Final,
			Messages: tooLong.GetMessages(),
			Chats:    tooLong.GetChats(),
			Users:    tooLong.GetUsers(),
		}

		if req, ok := input.(*tg.UpdatesGetChannelDifferenceRequest); ok && req != nil {
			payload.RequestedPts = req.Pts
			if ch, ok := req.GetChannel().(*tg.InputChannel); ok && ch != nil {
				payload.ChannelID = ch.ChannelID
				payload.AccessHash = ch.AccessHash
			} else if chWithID, ok := req.GetChannel().(interface{ GetChannelID() int64 }); ok {
				payload.ChannelID = chWithID.GetChannelID()
			}
		}

		if dlg, ok := tooLong.GetDialog().(*tg.Dialog); ok && dlg != nil {
			payload.TopMessage = dlg.TopMessage
			if pts, hasPts := dlg.GetPts(); hasPts {
				payload.DialogPts = pts
			}
			if payload.ChannelID == 0 {
				if peerCh, ok := dlg.Peer.(*tg.PeerChannel); ok {
					payload.ChannelID = peerCh.ChannelID
				}
			}
		}

		if err := m.handler.HandleTooLong(ctx, payload); err != nil {
			return fmt.Errorf("channelDifferenceTooLong recovery (channel=%d): %w", payload.ChannelID, err)
		}
		return nil
	}
}

func extractTooLong(output bin.Decoder) (*tg.UpdatesChannelDifferenceTooLong, bool) {
	switch v := output.(type) {
	case *tg.UpdatesChannelDifferenceBox:
		if v == nil {
			return nil, false
		}
		tl, ok := v.ChannelDifference.(*tg.UpdatesChannelDifferenceTooLong)
		return tl, ok
	case *tg.UpdatesChannelDifferenceTooLong:
		return v, v != nil
	default:
		return nil, false
	}
}

// ChannelGapRecoveryService recovers channel gaps on updates.channelDifferenceTooLong by:
// - caching peer access hashes from Dialog/Chats,
// - extracting embedded tail messages,
// - paginating messages.getHistory back to last_msg_id,
// - flushing them in ascending msg_id order into RecoveredBatchSink before pts advances.
type ChannelGapRecoveryService struct {
	accountID    int
	historyLimit int
	maxPages     int
	historyAPI   HistoryAPI
	watermarks   domain.WatermarkStore
	accessHashes domain.AccessHashStore
	sink         RecoveredBatchSink
}

// NewChannelGapRecoveryService creates a ChannelGapRecoveryService.
func NewChannelGapRecoveryService(
	accountID int,
	historyAPI HistoryAPI,
	watermarks domain.WatermarkStore,
	accessHashes domain.AccessHashStore,
	sink RecoveredBatchSink,
) *ChannelGapRecoveryService {
	return &ChannelGapRecoveryService{
		accountID:    accountID,
		historyLimit: 100,
		maxPages:     50,
		historyAPI:   historyAPI,
		watermarks:   watermarks,
		accessHashes: accessHashes,
		sink:         sink,
	}
}

// WithPagination configures page size and page cap (useful for testing).
func (s *ChannelGapRecoveryService) WithPagination(limit, maxPages int) *ChannelGapRecoveryService {
	if limit > 0 {
		s.historyLimit = limit
	}
	if maxPages > 0 {
		s.maxPages = maxPages
	}
	return s
}

// HandleTooLong implements TooLongHandler.
func (s *ChannelGapRecoveryService) HandleTooLong(ctx context.Context, payload ChannelTooLongPayload) error {
	if err := s.cacheEntities(ctx, payload.Chats); err != nil {
		return err
	}

	var lastMsgID int
	if s.watermarks != nil && payload.ChannelID != 0 {
		st, found, err := s.watermarks.GetChannelState(ctx, payload.ChannelID)
		if err != nil {
			return fmt.Errorf("load channel watermark %d: %w", payload.ChannelID, err)
		}
		if found {
			lastMsgID = st.LastMsgID
		}
	}

	accessHash := payload.AccessHash
	if accessHash == 0 {
		accessHash = findChannelAccessHash(payload.Chats, payload.ChannelID)
	}
	if accessHash == 0 && s.accessHashes != nil && payload.ChannelID != 0 {
		if h, ok, err := s.accessHashes.GetAccessHash(ctx, s.accountID, payload.ChannelID); err == nil && ok {
			accessHash = h
		}
	}

	msgByID := make(map[int]*tg.Message, len(payload.Messages))
	chats := append([]tg.ChatClass{}, payload.Chats...)
	users := append([]tg.UserClass{}, payload.Users...)

	addMessages(msgByID, payload.Messages, lastMsgID)

	if s.historyAPI != nil && payload.ChannelID != 0 {
		offsetID := 0
		for page := 0; page < s.maxPages; page++ {
			resp, err := s.historyAPI.MessagesGetHistory(ctx, &tg.MessagesGetHistoryRequest{
				Peer: &tg.InputPeerChannel{
					ChannelID:  payload.ChannelID,
					AccessHash: accessHash,
				},
				OffsetID: offsetID,
				Limit:    s.historyLimit,
				MinID:    lastMsgID,
			})
			if err != nil {
				return fmt.Errorf("messages.getHistory gap backfill (channel=%d, offset_id=%d): %w", payload.ChannelID, offsetID, err)
			}

			pageMsgs, pageChats, pageUsers := unpackMessagesResult(resp)
			if len(pageChats) > 0 {
				if err := s.cacheEntities(ctx, pageChats); err != nil {
					return err
				}
				chats = append(chats, pageChats...)
			}
			users = append(users, pageUsers...)

			if len(pageMsgs) == 0 {
				break
			}

			minPageID := 0
			for _, m := range pageMsgs {
				msg, ok := m.(*tg.Message)
				if !ok {
					continue
				}
				if minPageID == 0 || msg.ID < minPageID {
					minPageID = msg.ID
				}
				if msg.ID > lastMsgID {
					msgByID[msg.ID] = msg
				}
			}

			if minPageID <= lastMsgID+1 || len(pageMsgs) < s.historyLimit || minPageID == 0 || minPageID == offsetID {
				break
			}
			offsetID = minPageID
		}
	}

	if len(msgByID) == 0 || s.sink == nil {
		return nil
	}

	sortedMsgs := make([]*tg.Message, 0, len(msgByID))
	for _, msg := range msgByID {
		sortedMsgs = append(sortedMsgs, msg)
	}
	sort.Slice(sortedMsgs, func(i, j int) bool {
		return sortedMsgs[i].ID < sortedMsgs[j].ID
	})

	if err := s.sink.IngestRecoveredMessages(ctx, payload.ChannelID, chats, users, sortedMsgs); err != nil {
		return fmt.Errorf("ingest recovered gap messages (channel=%d): %w", payload.ChannelID, err)
	}
	return nil
}

func (s *ChannelGapRecoveryService) cacheEntities(ctx context.Context, chats []tg.ChatClass) error {
	if s.accessHashes == nil {
		return nil
	}
	for _, c := range chats {
		ch, ok := c.(*tg.Channel)
		if !ok || ch == nil {
			continue
		}
		hash, hasHash := ch.GetAccessHash()
		if !hasHash {
			continue
		}
		peerType := "channel"
		if ch.Megagroup && !ch.Broadcast {
			peerType = "discussion"
		}
		if err := s.accessHashes.SaveAccessHash(ctx, s.accountID, ch.ID, peerType, hash); err != nil {
			return fmt.Errorf("cache entity access_hash (channel=%d): %w", ch.ID, err)
		}
	}
	return nil
}

func addMessages(dst map[int]*tg.Message, raw []tg.MessageClass, minExclusiveID int) {
	for _, m := range raw {
		msg, ok := m.(*tg.Message)
		if !ok || msg == nil {
			continue
		}
		if msg.ID > minExclusiveID {
			dst[msg.ID] = msg
		}
	}
}

func findChannelAccessHash(chats []tg.ChatClass, channelID int64) int64 {
	for _, c := range chats {
		if ch, ok := c.(*tg.Channel); ok && ch != nil && ch.ID == channelID {
			if h, has := ch.GetAccessHash(); has {
				return h
			}
		}
	}
	return 0
}

func unpackMessagesResult(res tg.MessagesMessagesClass) ([]tg.MessageClass, []tg.ChatClass, []tg.UserClass) {
	if res == nil {
		return nil, nil, nil
	}
	switch v := res.(type) {
	case *tg.MessagesChannelMessages:
		return v.GetMessages(), v.GetChats(), v.GetUsers()
	case *tg.MessagesMessagesSlice:
		return v.GetMessages(), v.GetChats(), v.GetUsers()
	case *tg.MessagesMessages:
		return v.GetMessages(), v.GetChats(), v.GetUsers()
	default:
		return nil, nil, nil
	}
}
