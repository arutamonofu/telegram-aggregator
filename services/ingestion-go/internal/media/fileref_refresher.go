package media

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

// ChannelMessagesAPI abstracts channels.getMessages for file_reference renewal and tests.
type ChannelMessagesAPI interface {
	ChannelsGetMessages(ctx context.Context, request *tg.ChannelsGetMessagesRequest) (tg.MessagesMessagesClass, error)
}

type refreshFlight struct {
	done chan struct{}
	rec  domain.FileReferenceRecord
	err  error
}

// FileRefRefresher renews expired Telegram file_reference tokens via File Source Table lookup
// and batch channels.getMessages requests, collapsing concurrent refreshes per file_id.
type FileRefRefresher struct {
	accountID    int
	api          ChannelMessagesAPI
	fileRepo     domain.FileRefRepository
	accessHashes domain.AccessHashStore
	logger       *slog.Logger

	mu       sync.Mutex
	inFlight map[string]*refreshFlight
}

// NewFileRefRefresher initializes a singleflight FileRefRefresher.
func NewFileRefRefresher(
	accountID int,
	api ChannelMessagesAPI,
	fileRepo domain.FileRefRepository,
	accessHashes domain.AccessHashStore,
	logger *slog.Logger,
) *FileRefRefresher {
	if logger == nil {
		logger = slog.Default()
	}
	return &FileRefRefresher{
		accountID:    accountID,
		api:          api,
		fileRepo:     fileRepo,
		accessHashes: accessHashes,
		logger:       logger,
		inFlight:     make(map[string]*refreshFlight),
	}
}

// Refresh reloads fresh file_reference bytes for rec from active File Sources, or marks the file
// permanently gone (ErrMediaPermanentlyGone -> HTTP 410 Gone) if every source post was deleted.
func (r *FileRefRefresher) Refresh(ctx context.Context, rec domain.FileReferenceRecord) (domain.FileReferenceRecord, error) {
	if r.fileRepo == nil || r.api == nil {
		return domain.FileReferenceRecord{}, fmt.Errorf("file refresher missing dependencies")
	}

	r.mu.Lock()
	if fl, ok := r.inFlight[rec.FileID]; ok {
		r.mu.Unlock()
		select {
		case <-ctx.Done():
			return domain.FileReferenceRecord{}, ctx.Err()
		case <-fl.done:
			return fl.rec, fl.err
		}
	}
	fl := &refreshFlight{done: make(chan struct{})}
	r.inFlight[rec.FileID] = fl
	r.mu.Unlock()

	defer func() {
		close(fl.done)
		r.mu.Lock()
		delete(r.inFlight, rec.FileID)
		r.mu.Unlock()
	}()

	updated, err := r.doRefresh(ctx, rec)
	fl.rec = updated
	fl.err = err
	return updated, err
}

func (r *FileRefRefresher) doRefresh(ctx context.Context, rec domain.FileReferenceRecord) (domain.FileReferenceRecord, error) {
	sources, err := r.fileRepo.GetActiveFileSources(ctx, rec.FileID)
	if err != nil {
		return domain.FileReferenceRecord{}, fmt.Errorf("query file sources for %q: %w", rec.FileID, err)
	}
	if len(sources) == 0 {
		_ = r.fileRepo.MarkFilePermanentlyGone(ctx, rec.FileID)
		return domain.FileReferenceRecord{}, domain.ErrMediaPermanentlyGone
	}

	for _, src := range sources {
		var chHash int64
		if r.accessHashes != nil {
			if h, ok, hErr := r.accessHashes.GetAccessHash(ctx, r.accountID, src.ChannelID); hErr == nil && ok {
				chHash = h
			}
		}

		msgIDs := collectSourceMsgIDs(src)
		resp, err := r.api.ChannelsGetMessages(ctx, &tg.ChannelsGetMessagesRequest{
			Channel: &tg.InputChannel{
				ChannelID:  src.ChannelID,
				AccessHash: chHash,
			},
			ID: msgIDs,
		})
		if err != nil {
			r.logger.Warn("channels.getMessages failed during file_reference refresh",
				"file_id", rec.FileID,
				"channel_id", src.ChannelID,
				"msg_id", src.MsgID,
				"err", err,
			)
			continue
		}

		newRef, newSize, found := findFreshFileReference(resp, rec)
		if found {
			if err := r.fileRepo.UpdateFileReferenceBytes(ctx, rec.FileID, newRef, newSize); err != nil {
				return domain.FileReferenceRecord{}, fmt.Errorf("save refreshed file_reference %q: %w", rec.FileID, err)
			}
			updated := rec
			updated.FileReference = newRef
			if newSize > 0 {
				updated.FileSize = newSize
			}
			return updated, nil
		}

		// Source message(s) no longer carry this file -> mark deleted so we move to next source.
		if err := r.fileRepo.MarkFileSourceDeleted(ctx, src.SourceID); err != nil {
			r.logger.Warn("failed marking file source deleted", "source_id", src.SourceID, "err", err)
		}
	}

	if err := r.fileRepo.MarkFilePermanentlyGone(ctx, rec.FileID); err != nil {
		r.logger.Warn("failed marking file permanently gone", "file_id", rec.FileID, "err", err)
	}
	return domain.FileReferenceRecord{}, domain.ErrMediaPermanentlyGone
}

func collectSourceMsgIDs(src domain.FileSourceRecord) []tg.InputMessageClass {
	seen := make(map[int]struct{}, len(src.AlbumMsgIDs)+1)
	var out []tg.InputMessageClass
	if src.MsgID > 0 {
		seen[src.MsgID] = struct{}{}
		out = append(out, &tg.InputMessageID{ID: src.MsgID})
	}
	for _, id := range src.AlbumMsgIDs {
		if id <= 0 {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, &tg.InputMessageID{ID: id})
	}
	return out
}

func findFreshFileReference(res tg.MessagesMessagesClass, target domain.FileReferenceRecord) ([]byte, int64, bool) {
	var rawMsgs []tg.MessageClass
	switch v := res.(type) {
	case *tg.MessagesChannelMessages:
		rawMsgs = v.GetMessages()
	case *tg.MessagesMessagesSlice:
		rawMsgs = v.GetMessages()
	case *tg.MessagesMessages:
		rawMsgs = v.GetMessages()
	default:
		return nil, 0, false
	}

	for _, rm := range rawMsgs {
		msg, ok := rm.(*tg.Message)
		if !ok || msg == nil {
			continue
		}
		media, ok := msg.GetMedia()
		if !ok || media == nil {
			continue
		}

		switch m := media.(type) {
		case *tg.MessageMediaPhoto:
			if target.MediaKind != domain.MediaKindPhoto {
				continue
			}
			rp, hasP := m.GetPhoto()
			if !hasP {
				continue
			}
			p, ok := rp.(*tg.Photo)
			if !ok || p == nil || p.ID != target.TGID || len(p.FileReference) == 0 {
				continue
			}
			return append([]byte(nil), p.FileReference...), target.FileSize, true

		case *tg.MessageMediaDocument:
			if target.MediaKind == domain.MediaKindPhoto {
				continue
			}
			rd, hasD := m.GetDocument()
			if !hasD {
				continue
			}
			d, ok := rd.(*tg.Document)
			if !ok || d == nil || d.ID != target.TGID || len(d.FileReference) == 0 {
				continue
			}
			return append([]byte(nil), d.FileReference...), d.Size, true
		}
	}

	return nil, 0, false
}
