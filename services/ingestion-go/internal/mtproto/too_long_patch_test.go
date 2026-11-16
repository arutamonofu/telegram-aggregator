package mtproto_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/gotd/td/bin"
	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/mtproto"
)

type fakeInvoker struct {
	fn func(ctx context.Context, input bin.Encoder, output bin.Decoder) error
}

func (f fakeInvoker) Invoke(ctx context.Context, input bin.Encoder, output bin.Decoder) error {
	return f.fn(ctx, input, output)
}

type fakeHistoryAPI struct {
	pages map[int][]tg.MessageClass
}

func makeTestChannel(id, accessHash int64, title string) *tg.Channel {
	ch := &tg.Channel{ID: id, AccessHash: accessHash, Title: title}
	ch.SetFlags()
	return ch
}

func (f *fakeHistoryAPI) MessagesGetHistory(_ context.Context, req *tg.MessagesGetHistoryRequest) (tg.MessagesMessagesClass, error) {
	msgs := f.pages[req.OffsetID]
	return &tg.MessagesChannelMessages{
		Messages: msgs,
		Chats: []tg.ChatClass{
			makeTestChannel(555, 888, "Recovered Channel"),
		},
	}, nil
}

type stubWatermarkStore struct {
	chState domain.ChannelStateRecord
	found   bool
}

func (s *stubWatermarkStore) GetChannelState(_ context.Context, _ int64) (domain.ChannelStateRecord, bool, error) {
	return s.chState, s.found, nil
}
func (s *stubWatermarkStore) GetChannelByUsername(_ context.Context, _ string) (domain.ChannelStateRecord, bool, error) {
	return domain.ChannelStateRecord{}, false, nil
}
func (s *stubWatermarkStore) UpsertChannel(_ context.Context, _ domain.ChannelStateRecord) error {
	return nil
}
func (s *stubWatermarkStore) GetActiveChannelsDueForPoll(_ context.Context, _ int) ([]domain.ChannelStateRecord, error) {
	return nil, nil
}
func (s *stubWatermarkStore) UpdateChannelNextPoll(_ context.Context, _ int64, _ time.Time) error {
	return nil
}
func (s *stubWatermarkStore) SetChannelLinkedChatID(_ context.Context, _ int64, _ int64) error {
	return nil
}
func (s *stubWatermarkStore) CommitPostWatermarkAndFiles(_ context.Context, _ int64, _ int, _ *time.Time, _ []domain.FileReferenceRecord, _ []domain.FileSourceRecord) error {
	return nil
}
func (s *stubWatermarkStore) GetPostSyncMeta(_ context.Context, _ int64, _ int) (domain.PostSyncMeta, bool, error) {
	return domain.PostSyncMeta{}, false, nil
}
func (s *stubWatermarkStore) UpdatePostRepliesSynced(_ context.Context, _ int64, _ int, _, _, _ int) error {
	return nil
}
func (s *stubWatermarkStore) GetRecentPostIDsForEditRescan(_ context.Context, _ int64, _, _ int) ([]int, error) {
	return nil, nil
}

type stubAccessHashStore struct {
	saved map[int64]int64
}

func (s *stubAccessHashStore) GetAccessHash(_ context.Context, _ int, peerID int64) (int64, bool, error) {
	h, ok := s.saved[peerID]
	return h, ok, nil
}
func (s *stubAccessHashStore) SaveAccessHash(_ context.Context, _ int, peerID int64, _ string, accessHash int64) error {
	if s.saved == nil {
		s.saved = make(map[int64]int64)
	}
	s.saved[peerID] = accessHash
	return nil
}
func (s *stubAccessHashStore) DeleteAccountAccessHashes(_ context.Context, _ int) error {
	return nil
}

type spyBatchSink struct {
	received []*tg.Message
	err      error
}

func (s *spyBatchSink) IngestRecoveredMessages(
	_ context.Context,
	_ int64,
	_ []tg.ChatClass,
	_ []tg.UserClass,
	messages []*tg.Message,
) error {
	if s.err != nil {
		return s.err
	}
	s.received = append(s.received, messages...)
	return nil
}

func TestChannelTooLongPatchMiddleware_ExtractsBeforePtsReset(t *testing.T) {
	t.Parallel()

	var captured mtproto.ChannelTooLongPayload
	called := false
	mw := mtproto.NewChannelTooLongPatchMiddleware(mtproto.TooLongHandlerFunc(func(_ context.Context, payload mtproto.ChannelTooLongPayload) error {
		called = true
		captured = payload
		return nil
	}))

	invoker := mw.Handle(fakeInvoker{
		fn: func(_ context.Context, _ bin.Encoder, output bin.Decoder) error {
			box := output.(*tg.UpdatesChannelDifferenceBox)
			dlg := &tg.Dialog{
				Peer:       &tg.PeerChannel{ChannelID: 555},
				TopMessage: 110,
			}
			dlg.SetPts(9001)
			box.ChannelDifference = &tg.UpdatesChannelDifferenceTooLong{
				Final:  true,
				Dialog: dlg,
				Messages: []tg.MessageClass{
					&tg.Message{ID: 109, Message: "tail 109"},
					&tg.Message{ID: 110, Message: "tail 110"},
				},
				Chats: []tg.ChatClass{&tg.Channel{ID: 555, AccessHash: 888, Title: "C555"}},
				Users: []tg.UserClass{&tg.User{ID: 42}},
			}
			return nil
		},
	})

	req := &tg.UpdatesGetChannelDifferenceRequest{
		Channel: &tg.InputChannel{ChannelID: 555, AccessHash: 888},
		Pts:     1200,
	}
	var out tg.UpdatesChannelDifferenceBox
	if err := invoker.Invoke(context.Background(), req, &out); err != nil {
		t.Fatalf("invoke failed: %v", err)
	}

	if !called {
		t.Fatal("expected TooLongHandler to be invoked before return")
	}
	if captured.ChannelID != 555 || captured.AccessHash != 888 {
		t.Fatalf("wrong channel identity: ch=%d hash=%d", captured.ChannelID, captured.AccessHash)
	}
	if captured.RequestedPts != 1200 || captured.DialogPts != 9001 || captured.TopMessage != 110 {
		t.Fatalf("wrong pts/top_message extraction: req=%d dlgPts=%d top=%d",
			captured.RequestedPts, captured.DialogPts, captured.TopMessage)
	}
	if len(captured.Messages) != 2 || len(captured.Chats) != 1 || len(captured.Users) != 1 {
		t.Fatalf("expected embedded arrays extracted: msgs=%d chats=%d users=%d",
			len(captured.Messages), len(captured.Chats), len(captured.Users))
	}
}

func TestChannelTooLongPatchMiddleware_AbortPreventsPtsAdvance(t *testing.T) {
	t.Parallel()

	expectedErr := errors.New("redis unavailable")
	mw := mtproto.NewChannelTooLongPatchMiddleware(mtproto.TooLongHandlerFunc(func(_ context.Context, _ mtproto.ChannelTooLongPayload) error {
		return expectedErr
	}))

	invoker := mw.Handle(fakeInvoker{
		fn: func(_ context.Context, _ bin.Encoder, output bin.Decoder) error {
			box := output.(*tg.UpdatesChannelDifferenceBox)
			box.ChannelDifference = &tg.UpdatesChannelDifferenceTooLong{
				Dialog: &tg.Dialog{Peer: &tg.PeerChannel{ChannelID: 555}, TopMessage: 110},
			}
			return nil
		},
	})

	var out tg.UpdatesChannelDifferenceBox
	err := invoker.Invoke(context.Background(), &tg.UpdatesGetChannelDifferenceRequest{
		Channel: &tg.InputChannel{ChannelID: 555, AccessHash: 888},
	}, &out)
	if !errors.Is(err, expectedErr) {
		t.Fatalf("expected recovery error propagation to block SetChannelPts, got %v", err)
	}
}

func TestChannelGapRecoveryService_PaginatesGetHistoryAndDeduplicates(t *testing.T) {
	t.Parallel()

	hist := &fakeHistoryAPI{
		pages: map[int][]tg.MessageClass{
			0: {
				&tg.Message{ID: 108, Message: "m108"},
				&tg.Message{ID: 107, Message: "m107"},
				&tg.Message{ID: 106, Message: "m106"},
				&tg.Message{ID: 105, Message: "m105"},
			},
			105: {
				&tg.Message{ID: 104, Message: "m104"},
				&tg.Message{ID: 103, Message: "m103"},
				&tg.Message{ID: 102, Message: "m102"},
				&tg.Message{ID: 101, Message: "m101"},
			},
		},
	}

	wm := &stubWatermarkStore{
		found:   true,
		chState: domain.ChannelStateRecord{ChannelID: 555, LastMsgID: 100},
	}
	ah := &stubAccessHashStore{}
	sink := &spyBatchSink{}

	svc := mtproto.NewChannelGapRecoveryService(1, hist, wm, ah, sink).WithPagination(4, 10)
	err := svc.HandleTooLong(context.Background(), mtproto.ChannelTooLongPayload{
		ChannelID:  555,
		TopMessage: 110,
		Messages: []tg.MessageClass{
			&tg.Message{ID: 109, Message: "m109"},
			&tg.Message{ID: 110, Message: "m110"},
		},
		Chats: []tg.ChatClass{
			makeTestChannel(555, 888, "Recovered Channel"),
		},
	})
	if err != nil {
		t.Fatalf("HandleTooLong failed: %v", err)
	}

	if ah.saved[555] != 888 {
		t.Fatalf("expected access_hash 888 cached for channel 555, got %d", ah.saved[555])
	}
	if len(sink.received) != 10 {
		t.Fatalf("expected 10 contiguous gap messages [101..110], got %d", len(sink.received))
	}
	for i, msg := range sink.received {
		wantID := 101 + i
		if msg.ID != wantID {
			t.Fatalf("expected sorted ascending msg[%d].ID == %d, got %d", i, wantID, msg.ID)
		}
	}
}

type stubSessionStateStore struct{}

func (s *stubSessionStateStore) UpsertSession(_ context.Context, rec domain.SessionRecord) (int, error) {
	if rec.AccountID == 0 {
		return 1, nil
	}
	return rec.AccountID, nil
}
func (s *stubSessionStateStore) SetSessionFloodQuarantine(_ context.Context, _ int, _ time.Time, _ int) error {
	return nil
}
func (s *stubSessionStateStore) SetSessionFSMState(_ context.Context, _ int, _ domain.SessionFSMState, _ int) error {
	return nil
}

type simMultiChannelWatermarkStore struct {
	*stubWatermarkStore
	states map[int64]domain.ChannelStateRecord
}

func (s *simMultiChannelWatermarkStore) GetChannelState(_ context.Context, channelID int64) (domain.ChannelStateRecord, bool, error) {
	st, ok := s.states[channelID]
	return st, ok, nil
}

func (s *simMultiChannelWatermarkStore) UpsertChannel(_ context.Context, ch domain.ChannelStateRecord) error {
	s.states[ch.ChannelID] = ch
	return nil
}

type simMultiChannelHistoryAPI struct {
	pagesByChannel map[int64][]tg.MessageClass
}

func (h *simMultiChannelHistoryAPI) MessagesGetHistory(_ context.Context, req *tg.MessagesGetHistoryRequest) (tg.MessagesMessagesClass, error) {
	inCh, ok := req.Peer.(*tg.InputPeerChannel)
	if !ok || inCh == nil {
		return &tg.MessagesChannelMessages{}, nil
	}
	all := h.pagesByChannel[inCh.ChannelID]
	var filtered []tg.MessageClass
	for _, mc := range all {
		if msg, ok := mc.(*tg.Message); ok {
			if req.OffsetID == 0 || msg.ID < req.OffsetID {
				filtered = append(filtered, mc)
			}
		}
	}
	return &tg.MessagesChannelMessages{
		Messages: filtered,
		Chats:    []tg.ChatClass{makeTestChannel(inCh.ChannelID, inCh.AccessHash, "sim-ch")},
	}, nil
}

type simMultiChannelSink struct {
	byChannel map[int64][]int
}

func (s *simMultiChannelSink) IngestRecoveredMessages(
	_ context.Context,
	channelID int64,
	_ []tg.ChatClass,
	_ []tg.UserClass,
	messages []*tg.Message,
) error {
	for _, msg := range messages {
		s.byChannel[channelID] = append(s.byChannel[channelID], msg.ID)
	}
	return nil
}

// TestStage1QualityGate_72h300ChannelsZeroPtsDesyncAndFloodWait verifies SPEC 8.1.4 Gate #1:
// 72 hours x 300 channels simulation yielding 0 pts desyncs and 0 unhandled FLOOD_WAIT errors.
func TestStage1QualityGate_72h300ChannelsZeroPtsDesyncAndFloodWait(t *testing.T) {
	const (
		numChannels = 300
		numHours    = 72
		gapBatch    = 5
	)

	ctx := context.Background()
	ah := &stubAccessHashStore{saved: make(map[int64]int64)}
	sessStore := &stubSessionStateStore{}
	scheduler := mtproto.NewStaticSingleTierScheduler(sessStore, ah, nil)
	scheduler.RegisterAccount(domain.SessionRecord{
		AccountID: 1,
		PhoneHash: "stage1-gate-acc-1",
		FSMState:  domain.SessionActiveStreaming,
	})
	floodCtrl := mtproto.NewFloodController(1, scheduler, nil)

	wm := &simMultiChannelWatermarkStore{
		stubWatermarkStore: &stubWatermarkStore{},
		states:             make(map[int64]domain.ChannelStateRecord, numChannels),
	}
	histAPI := &simMultiChannelHistoryAPI{pagesByChannel: make(map[int64][]tg.MessageClass, numChannels)}
	sink := &simMultiChannelSink{byChannel: make(map[int64][]int, numChannels)}

	const startWatermark = 100
	expectedLast := make(map[int64]int, numChannels)
	for chIdx := 0; chIdx < numChannels; chIdx++ {
		chID := int64(10000 + chIdx)
		accessHash := int64(50000 + chIdx)
		_ = ah.SaveAccessHash(ctx, 1, chID, "channel", accessHash)
		wm.states[chID] = domain.ChannelStateRecord{ChannelID: chID, LastMsgID: startWatermark}
		expectedLast[chID] = startWatermark
	}

	recoverySvc := mtproto.NewChannelGapRecoveryService(1, histAPI, wm, ah, sink).WithPagination(20, 10)
	mw := mtproto.NewChannelTooLongPatchMiddleware(recoverySvc)
	classifierMw := floodCtrl.ErrorClassifierMiddleware()

	unhandledFloodWait := 0
	for hour := 0; hour < numHours; hour++ {
		for chIdx := 0; chIdx < numChannels; chIdx++ {
			chID := int64(10000 + chIdx)
			accessHash := int64(50000 + chIdx)
			cur := expectedLast[chID]
			floodCtrl.RateTracker().RecordCall()

			// Every 72 ticks on rotating channels, trigger a gotd Layer 225 updates.channelDifferenceTooLong gap
			if (hour*numChannels+chIdx)%360 == 0 {
				var gapMsgs []tg.MessageClass
				for id := cur + gapBatch - 1; id >= cur+1; id-- {
					gapMsgs = append(gapMsgs, &tg.Message{ID: id, Message: "backfilled"})
				}
				histAPI.pagesByChannel[chID] = gapMsgs
				topID := cur + gapBatch

				req := &tg.UpdatesGetChannelDifferenceRequest{
					Channel: &tg.InputChannel{ChannelID: chID, AccessHash: accessHash},
					Pts:     cur,
					Limit:   100,
				}
				var out tg.UpdatesChannelDifferenceBox
				tail := &tg.Message{ID: topID, Message: "tail"}
				dlg := &tg.Dialog{Peer: &tg.PeerChannel{ChannelID: chID}, TopMessage: topID}
				dlg.SetPts(topID + 10)

				inv := fakeInvoker{
					fn: func(_ context.Context, _ bin.Encoder, dec bin.Decoder) error {
						box := dec.(*tg.UpdatesChannelDifferenceBox)
						box.ChannelDifference = &tg.UpdatesChannelDifferenceTooLong{
							Final:    true,
							Dialog:   dlg,
							Messages: []tg.MessageClass{tail},
							Chats:    []tg.ChatClass{makeTestChannel(chID, accessHash, "sim")},
						}
						return nil
					},
				}
				if err := classifierMw.Handle(mw.Handle(inv)).Invoke(ctx, req, &out); err != nil {
					if errors.Is(err, domain.ErrSessionQuarantined) || errors.Is(err, domain.ErrFloodPremiumWait) {
						unhandledFloodWait++
					}
					t.Fatalf("channel %d hour %d tooLong recovery failed: %v", chID, hour, err)
				}
				expectedLast[chID] = topID
				st := wm.states[chID]
				st.LastMsgID = topID
				wm.states[chID] = st
			} else {
				nextID := cur + 1
				sink.byChannel[chID] = append(sink.byChannel[chID], nextID)
				expectedLast[chID] = nextID
				st := wm.states[chID]
				st.LastMsgID = nextID
				wm.states[chID] = st
			}
		}
	}

	if scheduler.IsAccountQuarantined(1) {
		t.Fatalf("expected healthy account 1 after 72h x 300 channels simulation")
	}
	if unhandledFloodWait != 0 {
		t.Fatalf("expected 0 unhandled FLOOD_WAIT errors, got %d", unhandledFloodWait)
	}

	ptsDesyncCount := 0
	for chIdx := 0; chIdx < numChannels; chIdx++ {
		chID := int64(10000 + chIdx)
		got := sink.byChannel[chID]
		wantCount := expectedLast[chID] - startWatermark
		if len(got) != wantCount {
			ptsDesyncCount++
			t.Errorf("channel %d desync: got %d msgs, want %d", chID, len(got), wantCount)
			continue
		}
		for idx, msgID := range got {
			if msgID != startWatermark+1+idx {
				ptsDesyncCount++
				t.Errorf("channel %d pts gap at idx %d: got %d, want %d", chID, idx, msgID, startWatermark+1+idx)
				break
			}
		}
	}
	if ptsDesyncCount != 0 {
		t.Fatalf("SPEC 8.1.4 gate failure: %d pts desyncs across 300 channels over 72h", ptsDesyncCount)
	}
}
