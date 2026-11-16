package updates

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

// PollAPI abstracts the Telegram RPCs used by fallback polling and edit_date rescaning.
type PollAPI interface {
	MessagesGetHistory(ctx context.Context, request *tg.MessagesGetHistoryRequest) (tg.MessagesMessagesClass, error)
	ChannelsGetMessages(ctx context.Context, request *tg.ChannelsGetMessagesRequest) (tg.MessagesMessagesClass, error)
}

// Poller performs adaptive fallback polling (Section 7.1.2 / 7.1.4):
// 1. Fetches new posts above channels.last_msg_id via messages.getHistory.
// 2. Rescans recent posts (24h window + pinned_msg_ids) for edit_date > max_edit_date_checked.
// 3. Schedules adaptive backoff next_poll_at when channels have zero new posts.
type Poller struct {
	accountID    int
	api          PollAPI
	watermarks   domain.WatermarkStore
	accessHashes domain.AccessHashStore
	pipeline     *ChannelPipeline
	baseInterval time.Duration
	maxBackoff   time.Duration
	logger       *slog.Logger
}

// NewPoller initializes a fallback history/edit poller.
func NewPoller(
	accountID int,
	api PollAPI,
	watermarks domain.WatermarkStore,
	accessHashes domain.AccessHashStore,
	pipeline *ChannelPipeline,
	baseInterval time.Duration,
	logger *slog.Logger,
) *Poller {
	if baseInterval <= 0 {
		baseInterval = 900 * time.Second
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Poller{
		accountID:    accountID,
		api:          api,
		watermarks:   watermarks,
		accessHashes: accessHashes,
		pipeline:     pipeline,
		baseInterval: baseInterval,
		maxBackoff:   2 * time.Hour,
		logger:       logger,
	}
}

// Run executes periodic polling ticks until ctx is cancelled.
func (p *Poller) Run(ctx context.Context, tickInterval time.Duration) error {
	if tickInterval <= 0 {
		tickInterval = 15 * time.Second
	}

	ticker := time.NewTicker(tickInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := p.PollDueOnce(ctx); err != nil {
				p.logger.Warn("fallback poll iteration error", "err", err)
			}
		}
	}
}

// PollDueOnce processes one batch of due active channels.
func (p *Poller) PollDueOnce(ctx context.Context) error {
	if p.watermarks == nil || p.api == nil || p.pipeline == nil {
		return nil
	}

	dueChannels, err := p.watermarks.GetActiveChannelsDueForPoll(ctx, 25)
	if err != nil {
		return fmt.Errorf("fetch due channels for poll: %w", err)
	}

	for _, ch := range dueChannels {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err := p.PollChannel(ctx, ch); err != nil {
			p.logger.Warn("poll channel failed", "channel_id", ch.ChannelID, "err", err)
		}
	}
	return nil
}

// PollChannel fetches missed messages and edits for one channel and schedules adaptive next_poll_at.
func (p *Poller) PollChannel(ctx context.Context, ch domain.ChannelStateRecord) error {
	accessHash, ok, err := p.accessHashes.GetAccessHash(ctx, p.accountID, ch.ChannelID)
	if err != nil {
		return fmt.Errorf("load access_hash for poll (channel=%d): %w", ch.ChannelID, err)
	}
	if !ok {
		return nil
	}

	resp, err := p.api.MessagesGetHistory(ctx, &tg.MessagesGetHistoryRequest{
		Peer: &tg.InputPeerChannel{
			ChannelID:  ch.ChannelID,
			AccessHash: accessHash,
		},
		Limit: 100,
		MinID: ch.LastMsgID,
	})
	if err != nil {
		return fmt.Errorf("messages.getHistory poll (channel=%d): %w", ch.ChannelID, err)
	}

	rawMsgs, rawChats := extractPollMessagesAndChats(resp)
	chMap := BuildChannelEntitiesMap(rawChats)

	var newMsgs []*tg.Message
	for _, rm := range rawMsgs {
		m, ok := rm.(*tg.Message)
		if !ok || m == nil {
			continue
		}
		if m.ID > ch.LastMsgID {
			newMsgs = append(newMsgs, m)
		}
	}

	sort.Slice(newMsgs, func(i, j int) bool {
		return newMsgs[i].ID < newMsgs[j].ID
	})

	for _, m := range newMsgs {
		if err := p.pipeline.ProcessPolledMessage(ctx, chMap, m, domain.EventTypePostNew); err != nil {
			return err
		}
	}

	// Rescan recent posts (24h window + pinned messages) to catch edits missed during disconnections.
	if err := p.rescanEdits(ctx, ch, accessHash); err != nil {
		p.logger.Warn("edit rescan failed", "channel_id", ch.ChannelID, "err", err)
	}

	if err := p.pipeline.Assembler().FlushChannel(ctx, ch.ChannelID); err != nil {
		return err
	}

	nextInterval := p.baseInterval
	if ch.PollIntervalSec > 0 {
		nextInterval = time.Duration(ch.PollIntervalSec) * time.Second
	}
	if len(newMsgs) == 0 {
		// Adaptive backoff when channel is silent (doubled up to maxBackoff).
		nextInterval *= 2
		if nextInterval > p.maxBackoff {
			nextInterval = p.maxBackoff
		}
	}
	return p.watermarks.UpdateChannelNextPoll(ctx, ch.ChannelID, time.Now().UTC().Add(nextInterval))
}

func (p *Poller) rescanEdits(ctx context.Context, ch domain.ChannelStateRecord, accessHash int64) error {
	postIDs, err := p.watermarks.GetRecentPostIDsForEditRescan(ctx, ch.ChannelID, 24, 50)
	if err != nil || len(postIDs) == 0 {
		return err
	}

	inputIDs := make([]tg.InputMessageClass, 0, len(postIDs))
	for _, id := range postIDs {
		inputIDs = append(inputIDs, &tg.InputMessageID{ID: id})
	}

	res, err := p.api.ChannelsGetMessages(ctx, &tg.ChannelsGetMessagesRequest{
		Channel: &tg.InputChannel{
			ChannelID:  ch.ChannelID,
			AccessHash: accessHash,
		},
		ID: inputIDs,
	})
	if err != nil {
		return err
	}

	rawMsgs, rawChats := extractPollMessagesAndChats(res)
	chMap := BuildChannelEntitiesMap(rawChats)

	for _, rm := range rawMsgs {
		m, ok := rm.(*tg.Message)
		if !ok || m == nil {
			continue
		}
		editSec, hasEdit := m.GetEditDate()
		if !hasEdit || editSec <= 0 {
			continue
		}
		editTime := time.Unix(int64(editSec), 0).UTC()
		if ch.MaxEditDateChecked != nil && !editTime.After(*ch.MaxEditDateChecked) {
			continue
		}
		if err := p.pipeline.ProcessPolledMessage(ctx, chMap, m, domain.EventTypePostEdit); err != nil {
			return err
		}
	}
	return nil
}

func extractPollMessagesAndChats(res tg.MessagesMessagesClass) ([]tg.MessageClass, []tg.ChatClass) {
	switch v := res.(type) {
	case *tg.MessagesChannelMessages:
		return v.GetMessages(), v.GetChats()
	case *tg.MessagesMessagesSlice:
		return v.GetMessages(), v.GetChats()
	case *tg.MessagesMessages:
		return v.GetMessages(), v.GetChats()
	default:
		return nil, nil
	}
}
