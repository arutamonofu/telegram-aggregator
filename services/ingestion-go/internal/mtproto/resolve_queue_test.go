package mtproto_test

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/mtproto"
)

type stubResolveAPI struct {
	calls atomic.Int32
}

func (s *stubResolveAPI) ContactsResolveUsername(_ context.Context, req *tg.ContactsResolveUsernameRequest) (*tg.ContactsResolvedPeer, error) {
	n := int64(s.calls.Add(1))
	ch := &tg.Channel{
		ID:         100 + n,
		AccessHash: 9000 + n,
		Username:   req.Username,
		Title:      "Chan " + req.Username,
	}
	ch.SetFlags()
	return &tg.ContactsResolvedPeer{
		Peer:  &tg.PeerChannel{ChannelID: 100 + n},
		Chats: []tg.ChatClass{ch},
	}, nil
}

type memResolveStore struct {
	byUser map[string]domain.ChannelStateRecord
	hashes map[int64]int64
}

func newMemResolveStore() *memResolveStore {
	return &memResolveStore{
		byUser: make(map[string]domain.ChannelStateRecord),
		hashes: make(map[int64]int64),
	}
}

func (m *memResolveStore) GetChannelState(_ context.Context, _ int64) (domain.ChannelStateRecord, bool, error) {
	return domain.ChannelStateRecord{}, false, nil
}
func (m *memResolveStore) GetChannelByUsername(_ context.Context, username string) (domain.ChannelStateRecord, bool, error) {
	ch, ok := m.byUser[username]
	return ch, ok, nil
}
func (m *memResolveStore) UpsertChannel(_ context.Context, ch domain.ChannelStateRecord) error {
	if ch.Username != nil {
		m.byUser[*ch.Username] = ch
	}
	return nil
}
func (m *memResolveStore) GetActiveChannelsDueForPoll(_ context.Context, _ int) ([]domain.ChannelStateRecord, error) {
	return nil, nil
}
func (m *memResolveStore) UpdateChannelNextPoll(_ context.Context, _ int64, _ time.Time) error {
	return nil
}
func (m *memResolveStore) SetChannelLinkedChatID(_ context.Context, _ int64, _ int64) error {
	return nil
}
func (m *memResolveStore) CommitPostWatermarkAndFiles(_ context.Context, _ int64, _ int, _ *time.Time, _ []domain.FileReferenceRecord, _ []domain.FileSourceRecord) error {
	return nil
}
func (m *memResolveStore) GetPostSyncMeta(_ context.Context, _ int64, _ int) (domain.PostSyncMeta, bool, error) {
	return domain.PostSyncMeta{}, false, nil
}
func (m *memResolveStore) UpdatePostRepliesSynced(_ context.Context, _ int64, _ int, _, _, _ int) error {
	return nil
}
func (m *memResolveStore) GetRecentPostIDsForEditRescan(_ context.Context, _ int64, _, _ int) ([]int, error) {
	return nil, nil
}
func (m *memResolveStore) GetAccessHash(_ context.Context, _ int, peerID int64) (int64, bool, error) {
	h, ok := m.hashes[peerID]
	return h, ok, nil
}
func (m *memResolveStore) SaveAccessHash(_ context.Context, _ int, peerID int64, _ string, accessHash int64) error {
	m.hashes[peerID] = accessHash
	return nil
}
func (m *memResolveStore) DeleteAccountAccessHashes(_ context.Context, _ int) error {
	return nil
}

func TestUsernameResolver_EternalCacheAndGlobalRateLimit(t *testing.T) {
	t.Parallel()

	store := newMemResolveStore()
	api := &stubResolveAPI{}
	resolver := mtproto.NewUsernameResolver(60*time.Millisecond, store, store)

	res1, err := resolver.ResolveChannel(context.Background(), 1, "@BreakingNews", api)
	if err != nil {
		t.Fatalf("first resolve failed: %v", err)
	}
	if res1.FromCache || res1.ChannelID != 101 || res1.AccessHash != 9001 {
		t.Fatalf("unexpected first resolve result: %+v", res1)
	}

	// Subsequent resolution (normalized username) must hit eternal cache with 0 RPC calls.
	res2, err := resolver.ResolveChannel(context.Background(), 1, "breakingnews", api)
	if err != nil {
		t.Fatalf("cached resolve failed: %v", err)
	}
	if !res2.FromCache || res2.ChannelID != 101 || api.calls.Load() != 1 {
		t.Fatalf("expected eternal cache hit, got res=%+v calls=%d", res2, api.calls.Load())
	}

	// Next uncached username must be rate-limited (>= ~50ms after first resolve).
	start := time.Now()
	res3, err := resolver.ResolveChannel(context.Background(), 1, "@TechDigest", api)
	if err != nil {
		t.Fatalf("second uncached resolve failed: %v", err)
	}
	if time.Since(start) < 40*time.Millisecond {
		t.Fatalf("expected rate-limited spacing for second uncached resolve, elapsed=%v", time.Since(start))
	}
	if res3.FromCache || res3.ChannelID != 102 {
		t.Fatalf("unexpected third resolve result: %+v", res3)
	}
}
