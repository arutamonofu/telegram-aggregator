package comments_test

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/comments"
	"github.com/tg-agg/ingestion-go/internal/domain"
)

type spyCommentsAPI struct {
	discussionCalls int
	repliesCalls    int
	lastMinID       int
	replies         []tg.MessageClass
}

func (s *spyCommentsAPI) MessagesGetDiscussionMessage(_ context.Context, _ *tg.MessagesGetDiscussionMessageRequest) (*tg.MessagesDiscussionMessage, error) {
	s.discussionCalls++
	ch := &tg.Channel{ID: 9009, AccessHash: 7777, Megagroup: true, Title: "Discussion"}
	ch.SetFlags()
	return &tg.MessagesDiscussionMessage{
		Messages: []tg.MessageClass{
			&tg.Message{ID: 500, PeerID: &tg.PeerChannel{ChannelID: 9009}},
		},
		Chats: []tg.ChatClass{ch},
	}, nil
}

func (s *spyCommentsAPI) MessagesGetReplies(_ context.Context, req *tg.MessagesGetRepliesRequest) (tg.MessagesMessagesClass, error) {
	s.repliesCalls++
	s.lastMinID = req.MinID
	return &tg.MessagesChannelMessages{Messages: s.replies}, nil
}

type fakeCommentsStore struct {
	ch   domain.ChannelStateRecord
	meta domain.PostSyncMeta
}

func (f *fakeCommentsStore) GetChannelState(_ context.Context, _ int64) (domain.ChannelStateRecord, bool, error) {
	return f.ch, true, nil
}
func (f *fakeCommentsStore) GetChannelByUsername(_ context.Context, _ string) (domain.ChannelStateRecord, bool, error) {
	return domain.ChannelStateRecord{}, false, nil
}
func (f *fakeCommentsStore) UpsertChannel(_ context.Context, _ domain.ChannelStateRecord) error {
	return nil
}
func (f *fakeCommentsStore) GetActiveChannelsDueForPoll(_ context.Context, _ int) ([]domain.ChannelStateRecord, error) {
	return nil, nil
}
func (f *fakeCommentsStore) UpdateChannelNextPoll(_ context.Context, _ int64, _ time.Time) error {
	return nil
}
func (f *fakeCommentsStore) SetChannelLinkedChatID(_ context.Context, _ int64, linkedChatID int64) error {
	f.ch.LinkedChatID = &linkedChatID
	return nil
}
func (f *fakeCommentsStore) CommitPostWatermarkAndFiles(_ context.Context, _ int64, _ int, _ *time.Time, _ []domain.FileReferenceRecord, _ []domain.FileSourceRecord) error {
	return nil
}
func (f *fakeCommentsStore) GetPostSyncMeta(_ context.Context, _ int64, _ int) (domain.PostSyncMeta, bool, error) {
	return f.meta, true, nil
}
func (f *fakeCommentsStore) UpdatePostRepliesSynced(_ context.Context, _ int64, _ int, repliesCount, repliesMaxID, syncedCount int) error {
	f.meta.RepliesCount = repliesCount
	f.meta.RepliesMaxID = repliesMaxID
	f.meta.RepliesSyncedCount = syncedCount
	return nil
}
func (f *fakeCommentsStore) GetRecentPostIDsForEditRescan(_ context.Context, _ int64, _, _ int) ([]int, error) {
	return nil, nil
}

type spyPublisher struct {
	events []domain.IngestCommentsDeltaEventV1
}

func (s *spyPublisher) PublishPost(_ context.Context, _ domain.IngestPostEventV1) (string, error) {
	return "1-0", nil
}
func (s *spyPublisher) PublishCommentsDelta(_ context.Context, evt domain.IngestCommentsDeltaEventV1) (string, error) {
	s.events = append(s.events, evt)
	return "2-0", nil
}
func (s *spyPublisher) PublishDeletion(_ context.Context, _ domain.IngestDeletionEventV1) (string, error) {
	return "3-0", nil
}

type memAH struct {
	hashes map[int64]int64
}

func (m *memAH) GetAccessHash(_ context.Context, _ int, peerID int64) (int64, bool, error) {
	v, ok := m.hashes[peerID]
	return v, ok, nil
}
func (m *memAH) SaveAccessHash(_ context.Context, _ int, peerID int64, _ string, accessHash int64) error {
	m.hashes[peerID] = accessHash
	return nil
}
func (m *memAH) DeleteAccountAccessHashes(_ context.Context, _ int) error { return nil }

func TestDeltaSyncService_SkipsWhenNoReplyGrowth(t *testing.T) {
	t.Parallel()

	api := &spyCommentsAPI{}
	wm := &fakeCommentsStore{
		ch:   domain.ChannelStateRecord{ChannelID: 1001},
		meta: domain.PostSyncMeta{ChannelID: 1001, MsgID: 42, RepliesSyncedCount: 15, RepliesMaxID: 215},
	}
	svc := comments.NewDeltaSyncService(1, api, wm, &memAH{hashes: map[int64]int64{1001: 111}}, &spyPublisher{}, slog.Default())

	n, err := svc.SyncPostComments(context.Background(), 1001, 42, domain.RepliesInfo{
		CommentsEnabled: true,
		RepliesCount:    15,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 0 || api.discussionCalls != 0 || api.repliesCalls != 0 {
		t.Fatalf("expected zero RPC calls when count unchanged, got synced=%d disc=%d replies=%d",
			n, api.discussionCalls, api.repliesCalls)
	}
}

func makeCommentMsg(id, date int, text string, userID int64) *tg.Message {
	m := &tg.Message{ID: id, Date: date, Message: text, FromID: &tg.PeerUser{UserID: userID}}
	m.SetFlags()
	return m
}

func TestDeltaSyncService_CachesLinkedChatIDOnceAndFetchesDelta(t *testing.T) {
	t.Parallel()

	api := &spyCommentsAPI{
		replies: []tg.MessageClass{
			makeCommentMsg(201, 1700000000, "first   ", 500),
			makeCommentMsg(202, 1700000010, "second", 501),
		},
	}
	wm := &fakeCommentsStore{
		ch:   domain.ChannelStateRecord{ChannelID: 1001},
		meta: domain.PostSyncMeta{ChannelID: 1001, MsgID: 42, RepliesSyncedCount: 0, RepliesMaxID: 0},
	}
	ah := &memAH{hashes: map[int64]int64{1001: 111}}
	pub := &spyPublisher{}

	svc := comments.NewDeltaSyncService(1, api, wm, ah, pub, slog.Default())

	// First delta: resolves and caches linked_chat_id=9009 once.
	n1, err := svc.SyncPostComments(context.Background(), 1001, 42, domain.RepliesInfo{
		CommentsEnabled: true,
		RepliesCount:    2,
	})
	if err != nil || n1 != 2 {
		t.Fatalf("first sync failed: n=%d err=%v", n1, err)
	}
	if api.discussionCalls != 1 || wm.ch.LinkedChatID == nil || *wm.ch.LinkedChatID != 9009 {
		t.Fatalf("expected linked_chat_id=9009 cached on first call, discCalls=%d linked=%v", api.discussionCalls, wm.ch.LinkedChatID)
	}
	if ah.hashes[9009] != 7777 {
		t.Fatalf("expected discussion access_hash 7777 saved, got %d", ah.hashes[9009])
	}
	if len(pub.events) != 1 || len(pub.events[0].Comments[0].AuthorPeerHash) != 16 {
		t.Fatalf("expected published anonymized comments, got %+v", pub.events)
	}

	// Second delta: growth from 2 -> 3 must NOT re-call getDiscussionMessage and must pass MinID=202.
	api.replies = []tg.MessageClass{
		makeCommentMsg(202, 1700000010, "old duplicate", 501),
		makeCommentMsg(203, 1700000020, "new third", 502),
	}
	n2, err := svc.SyncPostComments(context.Background(), 1001, 42, domain.RepliesInfo{
		CommentsEnabled: true,
		RepliesCount:    3,
	})
	if err != nil || n2 != 1 {
		t.Fatalf("second sync failed: n=%d err=%v", n2, err)
	}
	if api.discussionCalls != 1 {
		t.Fatalf("expected cached linked_chat_id to avoid extra getDiscussionMessage call, got %d", api.discussionCalls)
	}
	if api.lastMinID != 202 || wm.meta.RepliesMaxID != 203 || wm.meta.RepliesSyncedCount != 3 {
		t.Fatalf("unexpected delta state: minID=%d maxID=%d synced=%d", api.lastMinID, wm.meta.RepliesMaxID, wm.meta.RepliesSyncedCount)
	}
}
