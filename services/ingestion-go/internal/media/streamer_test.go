package media_test

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gotd/td/tg"
	"github.com/gotd/td/tgerr"

	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/media"
)

func TestAlignChunkRange_512KBInvariant(t *testing.T) {
	t.Parallel()

	chunks, err := media.AlignChunkRange(524280, 524295, 1_200_000)
	if err != nil {
		t.Fatalf("AlignChunkRange failed: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks across 512KB boundary, got %d", len(chunks))
	}

	if chunks[0].Offset != 0 || chunks[0].Limit != 524288 || chunks[0].SliceStart != 524280 || chunks[0].SliceEnd != 524288 {
		t.Fatalf("unexpected chunk[0]: %+v", chunks[0])
	}
	if chunks[1].Offset != 524288 || chunks[1].Limit != 524288 || chunks[1].SliceStart != 0 || chunks[1].SliceEnd != 8 {
		t.Fatalf("unexpected chunk[1]: %+v", chunks[1])
	}

	for i, ch := range chunks {
		if err := media.ValidateUploadPartParams(ch.Offset, ch.Limit); err != nil {
			t.Fatalf("chunk[%d] violated upload.getFile constraints: %v", i, err)
		}
		if ch.Limit%1024 != 0 || 524288%ch.Limit != 0 || ch.Offset%524288 != 0 {
			t.Fatalf("chunk[%d] alignment check failed: offset=%d limit=%d", i, ch.Offset, ch.Limit)
		}
	}

	// Verify invalid alignments are explicitly rejected.
	if err := media.ValidateUploadPartParams(0, 1500); err == nil {
		t.Fatal("expected ValidateUploadPartParams to reject non-1KB part_size 1500")
	}
	if err := media.ValidateUploadPartParams(0, 3072); err == nil {
		t.Fatal("expected ValidateUploadPartParams to reject part_size 3072 not dividing 524288")
	}
	if err := media.ValidateUploadPartParams(4096, 524288); err == nil {
		t.Fatal("expected ValidateUploadPartParams to reject unaligned offset 4096 for 524288 chunk")
	}
}

type fakeFileRepo struct {
	rec     domain.FileReferenceRecord
	sources []domain.FileSourceRecord
	gone    bool
}

func (f *fakeFileRepo) GetFileReference(_ context.Context, fileID string) (domain.FileReferenceRecord, bool, error) {
	if f.rec.FileID != fileID {
		return domain.FileReferenceRecord{}, false, nil
	}
	return f.rec, true, nil
}
func (f *fakeFileRepo) GetActiveFileSources(_ context.Context, _ string) ([]domain.FileSourceRecord, error) {
	var out []domain.FileSourceRecord
	for _, s := range f.sources {
		if !s.IsSourceDeleted {
			out = append(out, s)
		}
	}
	return out, nil
}
func (f *fakeFileRepo) UpdateFileReferenceBytes(_ context.Context, _ string, newBytes []byte, fileSize int64) error {
	f.rec.FileReference = append([]byte(nil), newBytes...)
	if fileSize > 0 {
		f.rec.FileSize = fileSize
	}
	return nil
}
func (f *fakeFileRepo) MarkFileSourceDeleted(_ context.Context, sourceID int64) error {
	for i := range f.sources {
		if f.sources[i].SourceID == sourceID {
			f.sources[i].IsSourceDeleted = true
		}
	}
	return nil
}
func (f *fakeFileRepo) MarkFilePermanentlyGone(_ context.Context, _ string) error {
	f.rec.IsPermanentlyGone = true
	f.gone = true
	return nil
}
func (f *fakeFileRepo) UpdateFileLocalCache(_ context.Context, _ string, path string, exp time.Time) error {
	f.rec.LocalCachePath = &path
	f.rec.LocalCacheExpires = &exp
	return nil
}

type fakeUploadClient struct {
	data  []byte
	calls atomic.Int32
	fn    func(attempt int, dcID int, req *tg.UploadGetFileRequest) (tg.UploadFileClass, error)
}

func (f *fakeUploadClient) UploadGetFileForDC(_ context.Context, dcID int, req *tg.UploadGetFileRequest) (tg.UploadFileClass, error) {
	n := int(f.calls.Add(1))
	if f.fn != nil {
		return f.fn(n, dcID, req)
	}
	start := int(req.Offset)
	end := min(len(f.data), start+req.Limit)
	return &tg.UploadFile{Bytes: f.data[start:end]}, nil
}

type fakeChannelMsgsAPI struct {
	resp tg.MessagesMessagesClass
}

func (f *fakeChannelMsgsAPI) ChannelsGetMessages(_ context.Context, _ *tg.ChannelsGetMessagesRequest) (tg.MessagesMessagesClass, error) {
	return f.resp, nil
}

type stubAH struct{}

func (stubAH) GetAccessHash(_ context.Context, _ int, _ int64) (int64, bool, error) { return 77, true, nil }
func (stubAH) SaveAccessHash(_ context.Context, _ int, _ int64, _ string, _ int64) error {
	return nil
}
func (stubAH) DeleteAccountAccessHashes(_ context.Context, _ int) error { return nil }

func TestStreamer_HTTP206PartialContentAndCacheHit(t *testing.T) {
	t.Parallel()

	payload := bytes.Repeat([]byte("0123456789ABCDEF"), 75_000) // 1,200,000 bytes
	mime := "video/mp4"
	repo := &fakeFileRepo{
		rec: domain.FileReferenceRecord{
			FileID:        "video:2:999:",
			TGID:          999,
			AccessHash:    1234,
			DCID:          2,
			MediaKind:     domain.MediaKindVideo,
			FileReference: []byte("ref-1"),
			FileSize:      int64(len(payload)),
			MimeType:      &mime,
		},
	}

	upload := &fakeUploadClient{data: payload}
	cache, err := media.NewLocalDiskCacheProvider(t.TempDir(), time.Hour)
	if err != nil {
		t.Fatalf("NewLocalDiskCacheProvider: %v", err)
	}
	streamer := media.NewStreamer(upload, repo, cache, nil, slog.Default())

	req := httptest.NewRequest(http.MethodGet, "/internal/v1/media/video:2:999:", nil)
	req.Header.Set("Range", "bytes=524280-524295")
	rr := httptest.NewRecorder()
	streamer.ServeMedia(rr, req, "video:2:999:")

	if rr.Code != http.StatusPartialContent {
		t.Fatalf("expected HTTP 206, got %d (%s)", rr.Code, rr.Body.String())
	}
	if cr := rr.Header().Get("Content-Range"); cr != "bytes 524280-524295/1200000" {
		t.Fatalf("unexpected Content-Range %q", cr)
	}
	if !bytes.Equal(rr.Body.Bytes(), payload[524280:524296]) {
		t.Fatalf("body mismatch: got %q want %q", rr.Body.Bytes(), payload[524280:524296])
	}
	if upload.calls.Load() != 2 {
		t.Fatalf("expected 2 512KB MTProto calls across boundary, got %d", upload.calls.Load())
	}

	// Second request over the same cached chunks must serve entirely from local disk cache.
	rr2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodGet, "/internal/v1/media/video:2:999:", nil)
	req2.Header.Set("Range", "bytes=524280-524295")
	streamer.ServeMedia(rr2, req2, "video:2:999:")

	if rr2.Code != http.StatusPartialContent || !bytes.Equal(rr2.Body.Bytes(), payload[524280:524296]) {
		t.Fatalf("cache hit serve mismatch: code=%d", rr2.Code)
	}
	if upload.calls.Load() != 2 {
		t.Fatalf("expected cache hit with 0 extra MTProto calls, got total=%d", upload.calls.Load())
	}
}

func TestStreamer_FileReferenceExpiredRefreshAndRetry(t *testing.T) {
	t.Parallel()

	payload := bytes.Repeat([]byte("X"), 4096)
	mime := "video/mp4"
	repo := &fakeFileRepo{
		rec: domain.FileReferenceRecord{
			FileID:        "video:2:777:",
			TGID:          777,
			AccessHash:    55,
			DCID:          2,
			MediaKind:     domain.MediaKindVideo,
			FileReference: []byte("expired-ref"),
			FileSize:      int64(len(payload)),
			MimeType:      &mime,
		},
		sources: []domain.FileSourceRecord{
			{SourceID: 1, FileID: "video:2:777:", ChannelID: 1001, MsgID: 50, AlbumMsgIDs: []int{49, 50}},
		},
	}

	upload := &fakeUploadClient{
		fn: func(_ int, _ int, req *tg.UploadGetFileRequest) (tg.UploadFileClass, error) {
			loc := req.Location.(*tg.InputDocumentFileLocation)
			if string(loc.FileReference) == "expired-ref" {
				return nil, tgerr.New(400, "FILE_REFERENCE_EXPIRED")
			}
			return &tg.UploadFile{Bytes: payload}, nil
		},
	}

	docMedia := &tg.MessageMediaDocument{
		Document: &tg.Document{
			ID:            777,
			AccessHash:    55,
			DCID:          2,
			FileReference: []byte("fresh-ref"),
			Size:          int64(len(payload)),
		},
	}
	docMedia.SetFlags()
	refreshedMsg := &tg.Message{ID: 50, Media: docMedia}
	refreshedMsg.SetFlags()

	chAPI := &fakeChannelMsgsAPI{
		resp: &tg.MessagesChannelMessages{
			Messages: []tg.MessageClass{refreshedMsg},
		},
	}

	refresher := media.NewFileRefRefresher(1, chAPI, repo, stubAH{}, slog.Default())
	streamer := media.NewStreamer(upload, repo, nil, refresher, slog.Default())

	req := httptest.NewRequest(http.MethodGet, "/internal/v1/media/video:2:777:", nil)
	req.Header.Set("Range", "bytes=0-1023")
	rr := httptest.NewRecorder()
	streamer.ServeMedia(rr, req, "video:2:777:")

	if rr.Code != http.StatusPartialContent {
		t.Fatalf("expected 206 after file_reference refresh, got %d (%s)", rr.Code, rr.Body.String())
	}
	if string(repo.rec.FileReference) != "fresh-ref" {
		t.Fatalf("expected refreshed file_reference saved in repo, got %q", string(repo.rec.FileReference))
	}
	if len(rr.Body.Bytes()) != 1024 {
		t.Fatalf("expected 1024 bytes, got %d", len(rr.Body.Bytes()))
	}
}

func TestStreamer_AllSourcesDeletedReturns410Gone(t *testing.T) {
	t.Parallel()

	repo := &fakeFileRepo{
		rec: domain.FileReferenceRecord{
			FileID:        "photo:2:888:y",
			TGID:          888,
			AccessHash:    11,
			DCID:          2,
			MediaKind:     domain.MediaKindPhoto,
			ThumbSize:     "y",
			FileReference: []byte("expired-ref"),
			FileSize:      2048,
		},
		sources: []domain.FileSourceRecord{
			{SourceID: 1, FileID: "photo:2:888:y", ChannelID: 1001, MsgID: 99},
		},
	}

	upload := &fakeUploadClient{
		fn: func(_ int, _ int, _ *tg.UploadGetFileRequest) (tg.UploadFileClass, error) {
			return nil, tgerr.New(400, "FILE_REFERENCE_EXPIRED")
		},
	}
	chAPI := &fakeChannelMsgsAPI{
		resp: &tg.MessagesChannelMessages{
			Messages: []tg.MessageClass{&tg.MessageEmpty{ID: 99}},
		},
	}

	refresher := media.NewFileRefRefresher(1, chAPI, repo, stubAH{}, slog.Default())
	streamer := media.NewStreamer(upload, repo, nil, refresher, slog.Default())

	req := httptest.NewRequest(http.MethodGet, "/internal/v1/media/photo:2:888:y", nil)
	req.Header.Set("Range", "bytes=0-100")
	rr := httptest.NewRecorder()
	streamer.ServeMedia(rr, req, "photo:2:888:y")

	if rr.Code != http.StatusGone {
		t.Fatalf("expected HTTP 410 Gone when all sources deleted, got %d", rr.Code)
	}
	if !repo.gone {
		t.Fatal("expected repo to mark is_permanently_gone = true")
	}
}
