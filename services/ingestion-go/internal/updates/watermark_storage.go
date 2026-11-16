package updates

import (
	"context"
	"sync"

	gotdupdates "github.com/gotd/td/telegram/updates"
)

// WatermarkSafeStateStorage wraps an updates.StateStorage (bbolt-backed) so that durable
// channel_pts checkpoints can be temporarily held while grouped_id albums are buffered
// in the 1500ms window and committed strictly AFTER Redis XADD ACK + PostgreSQL TX commit.
type WatermarkSafeStateStorage struct {
	inner gotdupdates.StateStorage

	mu          sync.Mutex
	lastUserID  int64
	holdCount   map[int64]int
	deferredPts map[int64]int
}

var _ gotdupdates.StateStorage = (*WatermarkSafeStateStorage)(nil)

// NewWatermarkSafeStateStorage wraps inner with hold/commit barrier semantics.
func NewWatermarkSafeStateStorage(inner gotdupdates.StateStorage) *WatermarkSafeStateStorage {
	return &WatermarkSafeStateStorage{
		inner:       inner,
		holdCount:   make(map[int64]int),
		deferredPts: make(map[int64]int),
	}
}

// AcquireChannelHold increments an active hold barrier for channelID. While held > 0,
// SetChannelPts buffers pts in memory instead of writing durable storage immediately.
func (w *WatermarkSafeStateStorage) AcquireChannelHold(channelID int64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.holdCount[channelID]++
}

// CommitAndReleaseChannelHold decrements the hold barrier after successful Redis ACK + DB TX commit.
// When hold count reaches 0, any deferred channel_pts is persisted to the durable bbolt store.
func (w *WatermarkSafeStateStorage) CommitAndReleaseChannelHold(ctx context.Context, channelID int64) error {
	w.mu.Lock()
	count := w.holdCount[channelID]
	if count > 1 {
		w.holdCount[channelID] = count - 1
		w.mu.Unlock()
		return nil
	}
	delete(w.holdCount, channelID)
	pts, hasDeferred := w.deferredPts[channelID]
	if hasDeferred {
		delete(w.deferredPts, channelID)
	}
	userID := w.lastUserID
	w.mu.Unlock()

	if hasDeferred && userID != 0 && w.inner != nil {
		return w.inner.SetChannelPts(ctx, userID, channelID, pts)
	}
	return nil
}

// AbortChannelHold releases a hold barrier WITHOUT flushing deferred pts (used on publish/DB error
// so Watermark Commit Invariant preserves the pre-batch pts on crash or retry).
func (w *WatermarkSafeStateStorage) AbortChannelHold(channelID int64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	count := w.holdCount[channelID]
	if count > 1 {
		w.holdCount[channelID] = count - 1
		return
	}
	delete(w.holdCount, channelID)
	delete(w.deferredPts, channelID)
}

func (w *WatermarkSafeStateStorage) GetState(ctx context.Context, userID int64) (gotdupdates.State, bool, error) {
	w.mu.Lock()
	w.lastUserID = userID
	w.mu.Unlock()
	return w.inner.GetState(ctx, userID)
}

func (w *WatermarkSafeStateStorage) SetState(ctx context.Context, userID int64, state gotdupdates.State) error {
	w.mu.Lock()
	w.lastUserID = userID
	w.mu.Unlock()
	return w.inner.SetState(ctx, userID, state)
}

func (w *WatermarkSafeStateStorage) SetPts(ctx context.Context, userID int64, pts int) error {
	return w.inner.SetPts(ctx, userID, pts)
}

func (w *WatermarkSafeStateStorage) SetQts(ctx context.Context, userID int64, qts int) error {
	return w.inner.SetQts(ctx, userID, qts)
}

func (w *WatermarkSafeStateStorage) SetDate(ctx context.Context, userID int64, date int) error {
	return w.inner.SetDate(ctx, userID, date)
}

func (w *WatermarkSafeStateStorage) SetSeq(ctx context.Context, userID int64, seq int) error {
	return w.inner.SetSeq(ctx, userID, seq)
}

func (w *WatermarkSafeStateStorage) SetDateSeq(ctx context.Context, userID int64, date, seq int) error {
	return w.inner.SetDateSeq(ctx, userID, date, seq)
}

func (w *WatermarkSafeStateStorage) GetChannelPts(ctx context.Context, userID, channelID int64) (int, bool, error) {
	w.mu.Lock()
	w.lastUserID = userID
	w.mu.Unlock()
	return w.inner.GetChannelPts(ctx, userID, channelID)
}

func (w *WatermarkSafeStateStorage) SetChannelPts(ctx context.Context, userID, channelID int64, pts int) error {
	w.mu.Lock()
	w.lastUserID = userID
	if w.holdCount[channelID] > 0 {
		if pts > w.deferredPts[channelID] {
			w.deferredPts[channelID] = pts
		}
		w.mu.Unlock()
		return nil
	}
	w.mu.Unlock()
	return w.inner.SetChannelPts(ctx, userID, channelID, pts)
}

func (w *WatermarkSafeStateStorage) ForEachChannels(
	ctx context.Context,
	userID int64,
	f func(ctx context.Context, channelID int64, pts int) error,
) error {
	w.mu.Lock()
	w.lastUserID = userID
	w.mu.Unlock()
	return w.inner.ForEachChannels(ctx, userID, f)
}
