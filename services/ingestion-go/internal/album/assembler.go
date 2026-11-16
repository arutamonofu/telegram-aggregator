package album

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf16"

	"github.com/google/uuid"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

// BufferedMessage represents an individual Telegram message staged for album assembly.
type BufferedMessage struct {
	Event       domain.IngestPostEventV1
	FileRefs    []domain.FileReferenceRecord
	FileSources []domain.FileSourceRecord
	ContentHash []byte
	EditDate    *time.Time
}

// FlushedBatch is the merged canonical document and persistence metadata emitted when an album
// window closes (or immediately for single non-album messages).
type FlushedBatch struct {
	ChannelID   int64
	Event       domain.IngestPostEventV1
	FileRefs    []domain.FileReferenceRecord
	FileSources []domain.FileSourceRecord
	MaxMsgID    int
	MaxEditDate *time.Time
	ContentHash []byte
}

// FlushFunc delivers a merged album or standalone post downstream (Redis XADD ACK -> DB TX commit -> pts commit).
type FlushFunc func(ctx context.Context, batch FlushedBatch) error

// HoldHooks notifies the watermark hold controller when an album buffer opens/closes on a channel.
type HoldHooks struct {
	OnAcquire func(channelID int64)
}

type albumKey struct {
	channelID int64
	groupedID int64
}

type pendingAlbum struct {
	key        albumKey
	generation uint64
	messages   map[int]BufferedMessage
	timer      *time.Timer
	firstAt    time.Time
}

// Assembler groups media album messages by (channel_id, grouped_id) in a configurable window
// (default 1500ms) and flushes early if a different grouped_id or non-album message arrives
// on the same channel.
type Assembler struct {
	window time.Duration
	onFlush FlushFunc
	hooks   HoldHooks

	mu      sync.Mutex
	gen     uint64
	buffers map[albumKey]*pendingAlbum
	byChan  map[int64][]albumKey
}

// NewAssembler creates an album Assembler with buffer window d (defaults to 1500ms if <= 0).
func NewAssembler(window time.Duration, onFlush FlushFunc, hooks HoldHooks) *Assembler {
	if window <= 0 {
		window = 1500 * time.Millisecond
	}
	return &Assembler{
		window:  window,
		onFlush: onFlush,
		hooks:   hooks,
		buffers: make(map[albumKey]*pendingAlbum),
		byChan:  make(map[int64][]albumKey),
	}
}

// Push routes msg into either:
//  1. An immediate flush (after flushing any earlier pending albums on this channel) when grouped_id == nil/0, OR
//  2. A (channel_id, grouped_id) album buffer timed at Assembler.window (flushing any earlier different grouped_id first).
func (a *Assembler) Push(ctx context.Context, msg BufferedMessage) error {
	chID := msg.Event.ChannelID
	var groupedID int64
	if msg.Event.GroupedID != nil {
		groupedID = *msg.Event.GroupedID
	}

	// Early-flush any other open album(s) on the same channel with a different grouped_id.
	otherBatches := a.popOtherAlbumsForChannel(chID, groupedID)
	for _, b := range otherBatches {
		if err := a.emit(ctx, b); err != nil {
			return err
		}
	}

	if groupedID == 0 {
		batch := FlushedBatch{
			ChannelID:   chID,
			Event:       msg.Event,
			FileRefs:    msg.FileRefs,
			FileSources: msg.FileSources,
			MaxMsgID:    msg.Event.MsgID,
			MaxEditDate: msg.EditDate,
			ContentHash: msg.ContentHash,
		}
		if batch.Event.Entities == nil {
			batch.Event.Entities = []domain.EntityItem{}
		}
		return a.emit(ctx, batch)
	}

	key := albumKey{channelID: chID, groupedID: groupedID}

	a.mu.Lock()
	buf, exists := a.buffers[key]
	if !exists {
		a.gen++
		gen := a.gen
		buf = &pendingAlbum{
			key:        key,
			generation: gen,
			messages:   make(map[int]BufferedMessage),
			firstAt:    time.Now(),
		}
		a.buffers[key] = buf
		a.byChan[chID] = append(a.byChan[chID], key)
		if a.hooks.OnAcquire != nil {
			a.hooks.OnAcquire(chID)
		}
		buf.timer = time.AfterFunc(a.window, func() {
			a.flushByTimer(key, gen)
		})
	} else if buf.timer != nil {
		buf.timer.Reset(a.window)
	}

	buf.messages[msg.Event.MsgID] = msg
	msgCount := len(buf.messages)
	a.mu.Unlock()

	// Telegram limits albums to 10 items; flush immediately once 10 items are present.
	if msgCount >= 10 {
		if batch, ok := a.popAlbum(key); ok {
			return a.emit(ctx, batch)
		}
	}
	return nil
}

// FlushChannel forces immediate emission of all buffered albums belonging to channelID.
func (a *Assembler) FlushChannel(ctx context.Context, channelID int64) error {
	batches := a.popOtherAlbumsForChannel(channelID, 0)
	for _, b := range batches {
		if err := a.emit(ctx, b); err != nil {
			return err
		}
	}
	return nil
}

// FlushAll forces emission of all pending albums (used during graceful shutdown).
func (a *Assembler) FlushAll(ctx context.Context) error {
	a.mu.Lock()
	keys := make([]albumKey, 0, len(a.buffers))
	for k := range a.buffers {
		keys = append(keys, k)
	}
	a.mu.Unlock()

	for _, k := range keys {
		if batch, ok := a.popAlbum(k); ok {
			if err := a.emit(ctx, batch); err != nil {
				return err
			}
		}
	}
	return nil
}

func (a *Assembler) flushByTimer(key albumKey, gen uint64) {
	a.mu.Lock()
	buf, ok := a.buffers[key]
	if !ok || buf.generation != gen {
		a.mu.Unlock()
		return
	}
	a.removeKeyLocked(key)
	batch := buildMergedBatch(buf)
	a.mu.Unlock()

	flushCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = a.emit(flushCtx, batch)
}

func (a *Assembler) popAlbum(key albumKey) (FlushedBatch, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	buf, ok := a.buffers[key]
	if !ok {
		return FlushedBatch{}, false
	}
	if buf.timer != nil {
		buf.timer.Stop()
	}
	a.removeKeyLocked(key)
	return buildMergedBatch(buf), true
}

func (a *Assembler) popOtherAlbumsForChannel(channelID, keepGroupedID int64) []FlushedBatch {
	a.mu.Lock()
	defer a.mu.Unlock()

	keys := a.byChan[channelID]
	if len(keys) == 0 {
		return nil
	}

	var popped []FlushedBatch
	remaining := keys[:0]
	for _, k := range keys {
		if keepGroupedID != 0 && k.groupedID == keepGroupedID {
			remaining = append(remaining, k)
			continue
		}
		buf, ok := a.buffers[k]
		if !ok {
			continue
		}
		if buf.timer != nil {
			buf.timer.Stop()
		}
		delete(a.buffers, k)
		popped = append(popped, buildMergedBatch(buf))
	}

	if len(remaining) == 0 {
		delete(a.byChan, channelID)
	} else {
		a.byChan[channelID] = remaining
	}

	sort.Slice(popped, func(i, j int) bool {
		return popped[i].Event.MsgID < popped[j].Event.MsgID
	})
	return popped
}

func (a *Assembler) removeKeyLocked(key albumKey) {
	delete(a.buffers, key)
	keys := a.byChan[key.channelID]
	if len(keys) == 0 {
		return
	}
	out := keys[:0]
	for _, k := range keys {
		if k != key {
			out = append(out, k)
		}
	}
	if len(out) == 0 {
		delete(a.byChan, key.channelID)
	} else {
		a.byChan[key.channelID] = out
	}
}

func (a *Assembler) emit(ctx context.Context, batch FlushedBatch) error {
	if a.onFlush == nil {
		return nil
	}
	return a.onFlush(ctx, batch)
}

func buildMergedBatch(buf *pendingAlbum) FlushedBatch {
	items := make([]BufferedMessage, 0, len(buf.messages))
	for _, m := range buf.messages {
		items = append(items, m)
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].Event.MsgID < items[j].Event.MsgID
	})

	first := items[0]
	albumIDs := make([]int, len(items))
	var (
		mediaItems   []domain.MediaItem
		fileRefs     []domain.FileReferenceRecord
		fileSources  []domain.FileSourceRecord
		maxMsgID     = first.Event.MsgID
		maxEditDate  *time.Time
		maxViews     = first.Event.Views
		maxForwards  = first.Event.Forwards
		replies      = first.Event.Replies
		captionAbove = first.Event.ShowCaptionAboveMedia
		eventType    = first.Event.EventType
	)

	var captions []struct {
		text     string
		entities []domain.EntityItem
	}

	for i, it := range items {
		albumIDs[i] = it.Event.MsgID
		if it.Event.MsgID > maxMsgID {
			maxMsgID = it.Event.MsgID
		}
		if it.Event.EventType == domain.EventTypePostEdit {
			eventType = domain.EventTypePostEdit
		}
		if it.EditDate != nil && (maxEditDate == nil || it.EditDate.After(*maxEditDate)) {
			t := *it.EditDate
			maxEditDate = &t
		}
		if it.Event.Views > maxViews {
			maxViews = it.Event.Views
		}
		if it.Event.Forwards > maxForwards {
			maxForwards = it.Event.Forwards
		}
		if it.Event.Replies.CommentsEnabled && (!replies.CommentsEnabled || it.Event.Replies.RepliesCount > replies.RepliesCount) {
			replies = it.Event.Replies
		}
		if it.Event.ShowCaptionAboveMedia {
			captionAbove = true
		}

		itemText := it.Event.TextRaw
		if itemText != "" {
			captions = append(captions, struct {
				text     string
				entities []domain.EntityItem
			}{text: itemText, entities: it.Event.Entities})
		}

		for _, mi := range it.Event.MediaItems {
			// When an album has multiple captions, attach each media's own caption on its MediaItem.
			if itemText != "" {
				c := itemText
				mi.Caption = &c
			}
			mediaItems = append(mediaItems, mi)
		}
		fileRefs = append(fileRefs, it.FileRefs...)
		fileSources = append(fileSources, it.FileSources...)
	}

	// Single-caption album clears redundant per-item MediaItem.Caption (unless multiple captions exist)
	// and places the caption & entities at top-level text_raw / entities.
	var (
		mergedText     string
		mergedEntities = []domain.EntityItem{}
	)
	switch len(captions) {
	case 0:
		mergedText = ""
	case 1:
		mergedText = captions[0].text
		mergedEntities = append(mergedEntities, captions[0].entities...)
		for i := range mediaItems {
			mediaItems[i].Caption = nil
		}
	default:
		var sb strings.Builder
		utf16Offset := 0
		for idx, c := range captions {
			if idx > 0 {
				sb.WriteString("\n\n")
				utf16Offset += 2
			}
			for _, ent := range c.entities {
				shifted := ent
				shifted.Offset += utf16Offset
				mergedEntities = append(mergedEntities, shifted)
			}
			sb.WriteString(c.text)
			utf16Offset += utf16Len(c.text)
		}
		mergedText = sb.String()
	}

	// Ensure every FileSourceRecord points to the full album_msg_ids list for batch channels.getMessages refresh.
	gID := buf.key.groupedID
	for i := range fileSources {
		fileSources[i].GroupedID = &gID
		fileSources[i].AlbumMsgIDs = append([]int(nil), albumIDs...)
	}

	var maxEditStr *string
	if maxEditDate != nil {
		s := maxEditDate.UTC().Format(time.RFC3339)
		maxEditStr = &s
	}

	mergedEvent := domain.IngestPostEventV1{
		SchemaVersion:         "ingest.post.v1",
		EventID:               uuid.NewString(),
		EventType:             eventType,
		ChannelID:             buf.key.channelID,
		ChannelUsername:       first.Event.ChannelUsername,
		ChannelTitle:          first.Event.ChannelTitle,
		MsgID:                 first.Event.MsgID,
		GroupedID:             &gID,
		AlbumMsgIDs:           albumIDs,
		PublishedAt:           first.Event.PublishedAt,
		EditDate:              maxEditStr,
		TextRaw:               mergedText,
		ShowCaptionAboveMedia: captionAbove,
		Views:                 maxViews,
		Forwards:              maxForwards,
		Entities:              mergedEntities,
		Replies:               replies,
		MediaItems:            mediaItems,
	}

	sum := sha256.Sum256([]byte(mergedText))
	return FlushedBatch{
		ChannelID:   buf.key.channelID,
		Event:       mergedEvent,
		FileRefs:    fileRefs,
		FileSources: fileSources,
		MaxMsgID:    maxMsgID,
		MaxEditDate: maxEditDate,
		ContentHash: sum[:],
	}
}

func utf16Len(s string) int {
	n := 0
	for _, r := range s {
		n += utf16.RuneLen(r)
	}
	return n
}

// KeyString returns a stable string key for debugging.
func (b FlushedBatch) KeyString() string {
	return fmt.Sprintf("ch:%d:msg:%d", b.ChannelID, b.Event.MsgID)
}
