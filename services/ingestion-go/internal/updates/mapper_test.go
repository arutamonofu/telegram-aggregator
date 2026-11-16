package updates_test

import (
	"context"
	"encoding/hex"
	"testing"

	gotdupdates "github.com/gotd/td/telegram/updates"
	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/updates"
)

func TestCanonicalizeMessage_RtrimUTF16AndEntitiesAndPhotoThumbSize(t *testing.T) {
	t.Parallel()

	dateEntity := &tg.MessageEntityFormattedDate{
		Relative:  true,
		ShortTime: true,
		Offset:    0,
		Length:    2,
		Date:      1763210000,
	}
	dateEntity.SetFlags()
	blockquote := &tg.MessageEntityBlockquote{
		Collapsed: true,
		Offset:    3,
		Length:    99, // Intentionally extends past trimmed text; must clamp cleanly
	}
	blockquote.SetFlags()
	emoji := &tg.MessageEntityCustomEmoji{
		Offset:     0,
		Length:     2,
		DocumentID: 555666777,
	}

	photo := &tg.Photo{
		ID:            123456,
		AccessHash:    987654,
		DCID:          2,
		FileReference: []byte("ref-bytes"),
		Sizes: []tg.PhotoSizeClass{
			&tg.PhotoStrippedSize{Type: "i", Bytes: []byte{1, 2, 3}},
			&tg.PhotoSize{Type: "m", W: 320, H: 180, Size: 12000},
			&tg.PhotoSize{Type: "y", W: 1280, H: 720, Size: 85000},
		},
	}
	photo.SetFlags()
	mediaPhoto := &tg.MessageMediaPhoto{Photo: photo}
	mediaPhoto.SetFlags()

	msg := &tg.Message{
		ID:          42,
		Date:        1763210000,
		InvertMedia: true,
		Message:     "🔥Quote   \n\t ",
		Entities: []tg.MessageEntityClass{
			dateEntity,
			blockquote,
			emoji,
		},
		Media: mediaPhoto,
	}
	msg.SetFlags()

	uname := "news"
	post, err := updates.CanonicalizeMessage(msg, updates.ChannelContext{
		ChannelID: 1001,
		Username:  &uname,
		Title:     "News",
	}, domain.EventTypePostNew)
	if err != nil {
		t.Fatalf("CanonicalizeMessage failed: %v", err)
	}

	if post.Event.TextRaw != "🔥Quote" {
		t.Fatalf("expected rtrimmed text %q, got %q", "🔥Quote", post.Event.TextRaw)
	}
	if !post.Event.ShowCaptionAboveMedia {
		t.Fatal("expected ShowCaptionAboveMedia=true")
	}
	// UTF-16 length of "🔥Quote" is 2 ('🔥') + 5 ('Quote') = 7; blockquote at offset 3 clamps to len 4.
	if len(post.Event.Entities) != 3 {
		t.Fatalf("expected 3 entities, got %d", len(post.Event.Entities))
	}
	bq := post.Event.Entities[1]
	if bq.Type != "messageEntityBlockquote" || bq.Offset != 3 || bq.Length != 4 || bq.Collapsed == nil || !*bq.Collapsed {
		t.Fatalf("unexpected clamped blockquote entity: %+v", bq)
	}
	if post.Event.Entities[0].UnixTime == nil || *post.Event.Entities[0].UnixTime != 1763210000 {
		t.Fatalf("unexpected FormattedDate unix time: %+v", post.Event.Entities[0])
	}
	if len(post.Event.MediaItems) != 1 || post.Event.MediaItems[0].ThumbSize != "y" || post.Event.MediaItems[0].FileID != "photo:2:123456:y" {
		t.Fatalf("expected largest PhotoSize.Type 'y' in FileID/ThumbSize, got %+v", post.Event.MediaItems)
	}
	if post.Event.MediaItems[0].StrippedThumbB64 == nil {
		t.Fatal("expected PhotoStrippedSize preserved as StrippedThumbB64")
	}
	if hex.EncodeToString(post.ContentHash) != hex.EncodeToString(updates.ContentSHA256("🔥Quote")) {
		t.Fatal("content sha256 mismatch with rtrimmed text")
	}
}

type stubInnerState struct {
	pts map[int64]int
}

func (s *stubInnerState) GetState(_ context.Context, _ int64) (gotdupdates.State, bool, error) {
	return gotdupdates.State{}, false, nil
}
func (s *stubInnerState) SetState(_ context.Context, _ int64, _ gotdupdates.State) error {
	return nil
}
func (s *stubInnerState) SetPts(_ context.Context, _ int64, _ int) error     { return nil }
func (s *stubInnerState) SetQts(_ context.Context, _ int64, _ int) error     { return nil }
func (s *stubInnerState) SetDate(_ context.Context, _ int64, _ int) error    { return nil }
func (s *stubInnerState) SetSeq(_ context.Context, _ int64, _ int) error     { return nil }
func (s *stubInnerState) SetDateSeq(_ context.Context, _ int64, _, _ int) error { return nil }
func (s *stubInnerState) GetChannelPts(_ context.Context, _, channelID int64) (int, bool, error) {
	v, ok := s.pts[channelID]
	return v, ok, nil
}
func (s *stubInnerState) SetChannelPts(_ context.Context, _, channelID int64, pts int) error {
	s.pts[channelID] = pts
	return nil
}
func (s *stubInnerState) ForEachChannels(_ context.Context, _ int64, _ func(context.Context, int64, int) error) error {
	return nil
}

func TestWatermarkSafeStateStorage_HoldUntilCommitOrAbort(t *testing.T) {
	t.Parallel()

	inner := &stubInnerState{pts: map[int64]int{1001: 50}}
	safe := updates.NewWatermarkSafeStateStorage(inner)
	ctx := context.Background()

	// While hold is active, durable bbolt pts must NOT advance beyond 50.
	safe.AcquireChannelHold(1001)
	if err := safe.SetChannelPts(ctx, 7, 1001, 55); err != nil {
		t.Fatalf("SetChannelPts while held failed: %v", err)
	}
	if inner.pts[1001] != 50 {
		t.Fatalf("Watermark Commit Invariant violated: inner pts advanced to %d while held", inner.pts[1001])
	}

	// Abort must discard deferred pts.
	safe.AbortChannelHold(1001)
	if inner.pts[1001] != 50 {
		t.Fatalf("expected inner pts to remain 50 after abort, got %d", inner.pts[1001])
	}

	// Acquire again, defer pts=60, then release via CommitAndReleaseChannelHold.
	safe.AcquireChannelHold(1001)
	_ = safe.SetChannelPts(ctx, 7, 1001, 60)
	if err := safe.CommitAndReleaseChannelHold(ctx, 1001); err != nil {
		t.Fatalf("CommitAndReleaseChannelHold failed: %v", err)
	}
	if inner.pts[1001] != 60 {
		t.Fatalf("expected inner pts committed to 60 after ACK release, got %d", inner.pts[1001])
	}
}
