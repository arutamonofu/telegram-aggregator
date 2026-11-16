package album_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/tg-agg/ingestion-go/internal/album"
	"github.com/tg-agg/ingestion-go/internal/domain"
)

func strPtr(s string) *string { return &s }
func i64Ptr(v int64) *int64   { return &v }

func makeAlbumMsg(chID int64, msgID int, groupedID *int64, text string, entities []domain.EntityItem) album.BufferedMessage {
	fileID := "photo:2:" + string(rune('a'+msgID%26)) + ":y"
	if entities == nil {
		entities = []domain.EntityItem{}
	}
	return album.BufferedMessage{
		Event: domain.IngestPostEventV1{
			SchemaVersion: "ingest.post.v1",
			EventID:       "evt",
			EventType:     domain.EventTypePostNew,
			ChannelID:     chID,
			ChannelTitle:  "TestChan",
			MsgID:         msgID,
			GroupedID:     groupedID,
			PublishedAt:   "2026-11-15T12:00:00Z",
			TextRaw:       text,
			Entities:      entities,
			MediaItems: []domain.MediaItem{
				{
					FileID:           fileID,
					TGID:             int64(msgID * 10),
					AccessHash:       99,
					DCID:             2,
					MediaKind:        domain.MediaKindPhoto,
					ThumbSize:        "y",
					FileReferenceB64: "AQID",
					FileSize:         1024,
				},
			},
		},
		FileRefs: []domain.FileReferenceRecord{
			{FileID: fileID, TGID: int64(msgID * 10), AccessHash: 99, DCID: 2, MediaKind: domain.MediaKindPhoto, ThumbSize: "y"},
		},
		FileSources: []domain.FileSourceRecord{
			{FileID: fileID, ChannelID: chID, MsgID: msgID, GroupedID: groupedID},
		},
	}
}

func TestAssembler_Default1500msWindowGroupingAndCaption(t *testing.T) {
	t.Parallel()

	var (
		mu      sync.Mutex
		flushed []album.FlushedBatch
		holds   int
		doneCh  = make(chan struct{}, 4)
	)

	asm := album.NewAssembler(1500*time.Millisecond, func(_ context.Context, batch album.FlushedBatch) error {
		mu.Lock()
		flushed = append(flushed, batch)
		mu.Unlock()
		doneCh <- struct{}{}
		return nil
	}, album.HoldHooks{
		OnAcquire: func(_ int64) {
			mu.Lock()
			holds++
			mu.Unlock()
		},
	})

	gid := int64(777001)
	start := time.Now()
	if err := asm.Push(context.Background(), makeAlbumMsg(1001, 105, &gid, "", nil)); err != nil {
		t.Fatalf("push 105: %v", err)
	}
	if err := asm.Push(context.Background(), makeAlbumMsg(1001, 103, &gid, "Headline text", []domain.EntityItem{
		{Type: "messageEntityBold", Offset: 0, Length: 8},
	})); err != nil {
		t.Fatalf("push 103: %v", err)
	}
	if err := asm.Push(context.Background(), makeAlbumMsg(1001, 104, &gid, "", nil)); err != nil {
		t.Fatalf("push 104: %v", err)
	}

	mu.Lock()
	if len(flushed) != 0 {
		t.Fatalf("expected 0 flushed albums before 1500ms window, got %d", len(flushed))
	}
	if holds != 1 {
		t.Fatalf("expected exactly 1 channel hold acquire for one album, got %d", holds)
	}
	mu.Unlock()

	select {
	case <-doneCh:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for 1500ms album flush")
	}
	if elapsed := time.Since(start); elapsed < 1400*time.Millisecond {
		t.Fatalf("album flushed too early after %v (< 1400ms)", elapsed)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(flushed) != 1 {
		t.Fatalf("expected 1 flushed album, got %d", len(flushed))
	}
	got := flushed[0]
	if got.Event.MsgID != 103 || got.MaxMsgID != 105 {
		t.Fatalf("unexpected msg_id bounds: min=%d max=%d", got.Event.MsgID, got.MaxMsgID)
	}
	if len(got.Event.AlbumMsgIDs) != 3 || got.Event.AlbumMsgIDs[0] != 103 || got.Event.AlbumMsgIDs[1] != 104 || got.Event.AlbumMsgIDs[2] != 105 {
		t.Fatalf("expected sorted album_msg_ids [103 104 105], got %v", got.Event.AlbumMsgIDs)
	}
	if got.Event.TextRaw != "Headline text" || len(got.Event.Entities) != 1 {
		t.Fatalf("unexpected merged caption/entities: %q %v", got.Event.TextRaw, got.Event.Entities)
	}
	for i, src := range got.FileSources {
		if len(src.AlbumMsgIDs) != 3 {
			t.Fatalf("file source %d missing full album_msg_ids: %v", i, src.AlbumMsgIDs)
		}
	}
	for i, m := range got.Event.MediaItems {
		if m.Caption != nil {
			t.Fatalf("single-caption album should clear per-item media caption on item %d", i)
		}
	}
}

func TestAssembler_MultiCaptionUTF16OffsetShift(t *testing.T) {
	t.Parallel()

	var flushed []album.FlushedBatch
	asm := album.NewAssembler(50*time.Millisecond, func(_ context.Context, batch album.FlushedBatch) error {
		flushed = append(flushed, batch)
		return nil
	}, album.HoldHooks{})

	gid := int64(888)
	// "🚀 A" has 4 UTF-16 code units ('🚀' is surrogate pair len 2, ' ' len 1, 'A' len 1).
	// Joining with "\n\n" (len 2) shifts second caption entity offsets by 4 + 2 = 6.
	_ = asm.Push(context.Background(), makeAlbumMsg(2002, 10, &gid, "🚀 A", []domain.EntityItem{
		{Type: "messageEntityBold", Offset: 0, Length: 4},
	}))
	_ = asm.Push(context.Background(), makeAlbumMsg(2002, 11, &gid, "Second", []domain.EntityItem{
		{Type: "messageEntityItalic", Offset: 0, Length: 6},
	}))

	if err := asm.FlushChannel(context.Background(), 2002); err != nil {
		t.Fatalf("flush channel: %v", err)
	}
	if len(flushed) != 1 {
		t.Fatalf("expected 1 album, got %d", len(flushed))
	}

	ev := flushed[0].Event
	if ev.TextRaw != "🚀 A\n\nSecond" {
		t.Fatalf("unexpected merged multi-caption text: %q", ev.TextRaw)
	}
	if len(ev.Entities) != 2 {
		t.Fatalf("expected 2 entities, got %d", len(ev.Entities))
	}
	if ev.Entities[1].Offset != 6 || ev.Entities[1].Length != 6 {
		t.Fatalf("expected shifted UTF-16 offset=6 len=6, got offset=%d len=%d", ev.Entities[1].Offset, ev.Entities[1].Length)
	}
	if ev.MediaItems[0].Caption == nil || *ev.MediaItems[0].Caption != "🚀 A" {
		t.Fatalf("expected per-item caption preserved on item 0, got %v", ev.MediaItems[0].Caption)
	}
	if ev.MediaItems[1].Caption == nil || *ev.MediaItems[1].Caption != "Second" {
		t.Fatalf("expected per-item caption preserved on item 1, got %v", ev.MediaItems[1].Caption)
	}
}

func TestAssembler_EarlyFlushOnDifferentGroupAndSingleMessage(t *testing.T) {
	t.Parallel()

	var flushed []int
	asm := album.NewAssembler(10*time.Second, func(_ context.Context, batch album.FlushedBatch) error {
		flushed = append(flushed, batch.Event.MsgID)
		return nil
	}, album.HoldHooks{})

	_ = asm.Push(context.Background(), makeAlbumMsg(3003, 1, i64Ptr(111), "album 1", nil))
	// Different grouped_id on same channel must flush album 1 immediately without waiting 10s.
	_ = asm.Push(context.Background(), makeAlbumMsg(3003, 2, i64Ptr(222), "album 2", nil))
	if len(flushed) != 1 || flushed[0] != 1 {
		t.Fatalf("expected album 1 early-flushed, got %v", flushed)
	}

	// Non-album message (grouped_id == nil) must early-flush album 2 before emitting standalone msg 3.
	_ = asm.Push(context.Background(), makeAlbumMsg(3003, 3, nil, "single post", nil))
	if len(flushed) != 3 || flushed[1] != 2 || flushed[2] != 3 {
		t.Fatalf("expected flush order [1 2 3], got %v", flushed)
	}
}

func TestAssembler_FlushAtTenItems(t *testing.T) {
	t.Parallel()

	var flushed []album.FlushedBatch
	asm := album.NewAssembler(10*time.Second, func(_ context.Context, batch album.FlushedBatch) error {
		flushed = append(flushed, batch)
		return nil
	}, album.HoldHooks{})

	gid := int64(555)
	for i := 1; i <= 10; i++ {
		if err := asm.Push(context.Background(), makeAlbumMsg(4004, i, &gid, "", nil)); err != nil {
			t.Fatalf("push %d: %v", i, err)
		}
	}
	if len(flushed) != 1 {
		t.Fatalf("expected immediate flush on 10th album message, got %d", len(flushed))
	}
	if len(flushed[0].Event.AlbumMsgIDs) != 10 {
		t.Fatalf("expected 10 album_msg_ids, got %d", len(flushed[0].Event.AlbumMsgIDs))
	}
}
