package media

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gotd/td/telegram"
	"github.com/gotd/td/tg"
	"github.com/gotd/td/tgerr"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

// UploadClient downloads 512 KB chunks from the target Telegram file DC.
type UploadClient interface {
	UploadGetFileForDC(ctx context.Context, dcID int, req *tg.UploadGetFileRequest) (tg.UploadFileClass, error)
}

// GotdUploadClient adapts *telegram.Client to UploadClient using FileDC when dcID differs.
type GotdUploadClient struct {
	Client *telegram.Client
}

// UploadGetFileForDC executes upload.getFile on the file's home DC (falling back to primary API on error/DC 0).
func (g GotdUploadClient) UploadGetFileForDC(
	ctx context.Context,
	dcID int,
	req *tg.UploadGetFileRequest,
) (tg.UploadFileClass, error) {
	if g.Client == nil {
		return nil, errors.New("telegram client is nil")
	}
	if dcID > 0 {
		inv, err := g.Client.FileDC(ctx, dcID)
		if err == nil && inv != nil {
			res, callErr := tg.NewClient(inv).UploadGetFile(ctx, req)
			if callErr == nil {
				return res, nil
			}
			// Fall back or propagate RPC error so FILE_REFERENCE_EXPIRED / FILE_MIGRATE are handled.
			if _, isRPC := tgerr.As(callErr); isRPC {
				return nil, callErr
			}
		}
	}
	return g.Client.API().UploadGetFile(ctx, req)
}

// Streamer serves HTTP 206 Partial Content media responses backed by 512 KB aligned upload.getFile
// chunks, a 24h local disk cache (MediaStorageProvider), and automatic FILE_REFERENCE_EXPIRED renewal.
type Streamer struct {
	upload    UploadClient
	fileRepo  domain.FileRefRepository
	cache     domain.MediaStorageProvider
	refresher *FileRefRefresher
	logger    *slog.Logger
}

// NewStreamer constructs a media Streamer.
func NewStreamer(
	upload UploadClient,
	fileRepo domain.FileRefRepository,
	cache domain.MediaStorageProvider,
	refresher *FileRefRefresher,
	logger *slog.Logger,
) *Streamer {
	if logger == nil {
		logger = slog.Default()
	}
	return &Streamer{
		upload:    upload,
		fileRepo:  fileRepo,
		cache:     cache,
		refresher: refresher,
		logger:    logger,
	}
}

type byteRange struct {
	start   int64
	end     int64
	isRange bool
	unknown bool
}

// ServeMedia handles GET /internal/v1/media/{file_id} with HTTP 206 Range streaming.
func (s *Streamer) ServeMedia(w http.ResponseWriter, r *http.Request, fileID string) {
	if s.fileRepo == nil {
		http.Error(w, "file repository unavailable", http.StatusServiceUnavailable)
		return
	}

	ctx := r.Context()
	rec, found, err := s.fileRepo.GetFileReference(ctx, fileID)
	if err != nil {
		s.logger.Error("lookup file reference failed", "file_id", fileID, "err", err)
		http.Error(w, "failed to load media metadata", http.StatusInternalServerError)
		return
	}
	if !found {
		http.Error(w, "media not found", http.StatusNotFound)
		return
	}
	if rec.IsPermanentlyGone {
		http.Error(w, "media permanently deleted", http.StatusGone)
		return
	}

	br, ok := parseRangeHeader(r.Header.Get("Range"), rec.FileSize)
	if !ok {
		if rec.FileSize > 0 {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", rec.FileSize))
		}
		http.Error(w, "invalid or unsatisfiable byte range", http.StatusRequestedRangeNotSatisfiable)
		return
	}

	mimeType := "application/octet-stream"
	if rec.MimeType != nil && *rec.MimeType != "" {
		mimeType = *rec.MimeType
	}

	// Sequential unknown-size streaming path (when FileSize == 0 and no finite range requested).
	if br.unknown {
		s.streamSequentialUnknownSize(ctx, w, &rec, mimeType)
		return
	}

	chunks, err := AlignChunkRange(br.start, br.end, rec.FileSize)
	if err != nil {
		http.Error(w, err.Error(), http.StatusRequestedRangeNotSatisfiable)
		return
	}

	// Prefetch the first chunk BEFORE writing HTTP 200/206 status so FILE_REFERENCE_EXPIRED -> 410 Gone
	// can still return an accurate 410 status code cleanly.
	firstBytes, err := s.getChunkWithRefresh(ctx, &rec, chunks[0])
	if err != nil {
		s.writeStreamError(w, fileID, err)
		return
	}

	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("Accept-Ranges", "bytes")
	w.Header().Set("Cache-Control", "private, max-age=86400")

	if br.isRange {
		totalLabel := "*"
		if rec.FileSize > 0 {
			totalLabel = strconv.FormatInt(rec.FileSize, 10)
			w.Header().Set("Content-Length", strconv.FormatInt(br.end-br.start+1, 10))
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%s", br.start, br.end, totalLabel))
		w.WriteHeader(http.StatusPartialContent)
	} else {
		if rec.FileSize > 0 {
			w.Header().Set("Content-Length", strconv.FormatInt(rec.FileSize, 10))
		}
		w.WriteHeader(http.StatusOK)
	}

	flusher, _ := w.(http.Flusher)
	for idx, ch := range chunks {
		var data []byte
		if idx == 0 {
			data = firstBytes
		} else {
			data, err = s.getChunkWithRefresh(ctx, &rec, ch)
			if err != nil {
				s.logger.Warn("mid-stream chunk fetch failed", "file_id", fileID, "offset", ch.Offset, "err", err)
				return
			}
		}

		if ch.SliceStart >= len(data) {
			break
		}
		endIdx := ch.SliceEnd
		if endIdx > len(data) {
			endIdx = len(data)
		}
		if _, writeErr := w.Write(data[ch.SliceStart:endIdx]); writeErr != nil {
			return
		}
		if flusher != nil {
			flusher.Flush()
		}
		if len(data) < ch.Limit {
			break
		}
	}
}

func (s *Streamer) streamSequentialUnknownSize(
	ctx context.Context,
	w http.ResponseWriter,
	rec *domain.FileReferenceRecord,
	mimeType string,
) {
	offset := int64(0)
	limit := int(AlignedChunkSize)
	headerWritten := false
	flusher, _ := w.(http.Flusher)

	for {
		chunk := AlignedChunk{
			Offset:     offset,
			Limit:      limit,
			SliceStart: 0,
			SliceEnd:   limit,
		}
		data, err := s.getChunkWithRefresh(ctx, rec, chunk)
		if err != nil {
			if !headerWritten {
				s.writeStreamError(w, rec.FileID, err)
			}
			return
		}
		if !headerWritten {
			w.Header().Set("Content-Type", mimeType)
			w.Header().Set("Accept-Ranges", "bytes")
			w.Header().Set("Cache-Control", "private, max-age=86400")
			w.WriteHeader(http.StatusOK)
			headerWritten = true
		}
		if len(data) > 0 {
			if _, wErr := w.Write(data); wErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if len(data) < limit {
			break
		}
		offset += int64(limit)
	}
}

func (s *Streamer) getChunkWithRefresh(
	ctx context.Context,
	rec *domain.FileReferenceRecord,
	chunk AlignedChunk,
) ([]byte, error) {
	if err := ValidateUploadPartParams(chunk.Offset, chunk.Limit); err != nil {
		return nil, err
	}

	if s.cache != nil {
		cached, hit, err := s.cache.GetChunk(ctx, rec.FileID, chunk.Offset)
		if err == nil && hit {
			return cached, nil
		}
	}

	if s.upload == nil {
		return nil, errors.New("upload client unavailable")
	}

	refreshedOnce := false
	targetDC := rec.DCID

	for attempt := 0; attempt < 3; attempt++ {
		loc := buildInputLocation(*rec)
		resp, err := s.upload.UploadGetFileForDC(ctx, targetDC, &tg.UploadGetFileRequest{
			Location: loc,
			Offset:   chunk.Offset,
			Limit:    chunk.Limit,
		})
		if err != nil {
			if isFileReferenceError(err) && !refreshedOnce && s.refresher != nil {
				refreshedOnce = true
				updated, refErr := s.refresher.Refresh(ctx, *rec)
				if refErr != nil {
					return nil, refErr
				}
				*rec = updated
				targetDC = updated.DCID
				continue
			}

			if rpcErr, ok := tgerr.As(err); ok && strings.HasPrefix(rpcErr.Type, "FILE_MIGRATE") && rpcErr.Argument > 0 {
				targetDC = rpcErr.Argument
				continue
			}

			return nil, fmt.Errorf("upload.getFile (%s, offset=%d): %w", rec.FileID, chunk.Offset, err)
		}

		fileObj, ok := resp.(*tg.UploadFile)
		if !ok || fileObj == nil {
			return nil, fmt.Errorf("unexpected upload.getFile response %T", resp)
		}

		data := fileObj.GetBytes()
		if s.cache != nil && len(data) > 0 {
			if dir, expiresAt, cErr := s.cache.PutChunk(ctx, rec.FileID, chunk.Offset, data); cErr == nil && s.fileRepo != nil {
				_ = s.fileRepo.UpdateFileLocalCache(ctx, rec.FileID, dir, expiresAt)
			}
		}
		return data, nil
	}

	return nil, fmt.Errorf("exhausted upload.getFile retries for %q", rec.FileID)
}

func (s *Streamer) writeStreamError(w http.ResponseWriter, fileID string, err error) {
	switch {
	case errors.Is(err, domain.ErrMediaPermanentlyGone):
		http.Error(w, "media permanently gone across all sources", http.StatusGone)
	case errors.Is(err, domain.ErrFloodPremiumWait):
		w.Header().Set("Retry-After", "5")
		http.Error(w, "upstream media rate limit (premium wait)", http.StatusServiceUnavailable)
	case errors.Is(err, domain.ErrSessionQuarantined):
		http.Error(w, "upstream session temporarily quarantined", http.StatusServiceUnavailable)
	default:
		s.logger.Error("media stream error", "file_id", fileID, "err", err)
		http.Error(w, "failed to stream media from MTProto", http.StatusBadGateway)
	}
}

func buildInputLocation(rec domain.FileReferenceRecord) tg.InputFileLocationClass {
	if rec.MediaKind == domain.MediaKindPhoto {
		return &tg.InputPhotoFileLocation{
			ID:            rec.TGID,
			AccessHash:    rec.AccessHash,
			FileReference: rec.FileReference,
			ThumbSize:     rec.ThumbSize,
		}
	}
	return &tg.InputDocumentFileLocation{
		ID:            rec.TGID,
		AccessHash:    rec.AccessHash,
		FileReference: rec.FileReference,
		ThumbSize:     rec.ThumbSize,
	}
}

func isFileReferenceError(err error) bool {
	if tgerr.Is(err, "FILE_REFERENCE_EXPIRED", "FILE_REFERENCE_INVALID", "FILE_REFERENCE_EMPTY") {
		return true
	}
	if rpcErr, ok := tgerr.As(err); ok {
		return strings.Contains(rpcErr.Type, "FILE_REFERENCE")
	}
	return false
}

func parseRangeHeader(header string, totalSize int64) (byteRange, bool) {
	if header == "" {
		if totalSize <= 0 {
			return byteRange{unknown: true}, true
		}
		return byteRange{start: 0, end: totalSize - 1, isRange: false}, true
	}

	if !strings.HasPrefix(header, "bytes=") {
		return byteRange{}, false
	}
	spec := strings.TrimPrefix(header, "bytes=")
	if strings.Contains(spec, ",") {
		spec = strings.SplitN(spec, ",", 2)[0]
	}
	spec = strings.TrimSpace(spec)
	parts := strings.SplitN(spec, "-", 2)
	if len(parts) != 2 {
		return byteRange{}, false
	}

	startStr := strings.TrimSpace(parts[0])
	endStr := strings.TrimSpace(parts[1])

	// Suffix range: bytes=-<N>
	if startStr == "" {
		if totalSize <= 0 || endStr == "" {
			return byteRange{}, false
		}
		suffix, err := strconv.ParseInt(endStr, 10, 64)
		if err != nil || suffix <= 0 {
			return byteRange{}, false
		}
		start := totalSize - suffix
		if start < 0 {
			start = 0
		}
		return byteRange{start: start, end: totalSize - 1, isRange: true}, true
	}

	start, err := strconv.ParseInt(startStr, 10, 64)
	if err != nil || start < 0 {
		return byteRange{}, false
	}
	if totalSize > 0 && start >= totalSize {
		return byteRange{}, false
	}

	if endStr == "" {
		if totalSize <= 0 {
			// Cap unknown open range at one aligned 512 KB chunk for bounded 206 streaming.
			return byteRange{start: start, end: start + AlignedChunkSize - 1, isRange: true}, true
		}
		return byteRange{start: start, end: totalSize - 1, isRange: true}, true
	}

	end, err := strconv.ParseInt(endStr, 10, 64)
	if err != nil || end < start {
		return byteRange{}, false
	}
	if totalSize > 0 && end >= totalSize {
		end = totalSize - 1
	}
	return byteRange{start: start, end: end, isRange: true}, true
}

// Keep unused time import check satisfied if helper relies on time elsewhere.
var _ = time.Second
