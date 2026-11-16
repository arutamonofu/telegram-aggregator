package media

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

// LocalDiskCacheProvider caches aligned 512 KB media chunks on local disk with 24h TTL
// for Stage 1 (Dev), implementing domain.MediaStorageProvider.
type LocalDiskCacheProvider struct {
	baseDir string
	ttl     time.Duration
}

var _ domain.MediaStorageProvider = (*LocalDiskCacheProvider)(nil)

// NewLocalDiskCacheProvider creates a disk cache rooted at baseDir with chunk expiry ttl.
func NewLocalDiskCacheProvider(baseDir string, ttl time.Duration) (*LocalDiskCacheProvider, error) {
	if baseDir == "" {
		return nil, fmt.Errorf("empty media cache directory")
	}
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}
	if err := os.MkdirAll(baseDir, 0o750); err != nil {
		return nil, fmt.Errorf("create media cache dir %q: %w", baseDir, err)
	}
	return &LocalDiskCacheProvider{
		baseDir: baseDir,
		ttl:     ttl,
	}, nil
}

// GetChunk reads an aligned cached chunk if present and within TTL.
func (c *LocalDiskCacheProvider) GetChunk(_ context.Context, fileID string, alignedOffset int64) ([]byte, bool, error) {
	path := c.chunkPath(fileID, alignedOffset)
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}

	if time.Since(info.ModTime()) > c.ttl {
		_ = os.Remove(path)
		return nil, false, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false, err
	}
	return data, true, nil
}

// PutChunk writes an aligned chunk atomically to disk and returns the file directory and expiry.
func (c *LocalDiskCacheProvider) PutChunk(_ context.Context, fileID string, alignedOffset int64, data []byte) (string, time.Time, error) {
	dir := c.fileDir(fileID)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return "", time.Time{}, fmt.Errorf("create cache dir for %q: %w", fileID, err)
	}

	target := c.chunkPath(fileID, alignedOffset)
	tmp, err := os.CreateTemp(dir, "chunk-*.tmp")
	if err != nil {
		return "", time.Time{}, err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return "", time.Time{}, err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return "", time.Time{}, err
	}
	if err := os.Rename(tmpPath, target); err != nil {
		_ = os.Remove(tmpPath)
		return "", time.Time{}, err
	}

	return dir, time.Now().UTC().Add(c.ttl), nil
}

// Invalidate removes all cached chunks for fileID.
func (c *LocalDiskCacheProvider) Invalidate(_ context.Context, fileID string) error {
	return os.RemoveAll(c.fileDir(fileID))
}

// CleanExpired scans cache directories and removes chunks older than ttl.
func (c *LocalDiskCacheProvider) CleanExpired(ctx context.Context) error {
	return filepath.WalkDir(c.baseDir, func(path string, d os.DirEntry, walkErr error) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if walkErr != nil || d.IsDir() {
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".bin") {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if time.Since(info.ModTime()) > c.ttl {
			_ = os.Remove(path)
		}
		return nil
	})
}

func (c *LocalDiskCacheProvider) fileDir(fileID string) string {
	sum := sha256.Sum256([]byte(fileID))
	return filepath.Join(c.baseDir, hex.EncodeToString(sum[:16]))
}

func (c *LocalDiskCacheProvider) chunkPath(fileID string, alignedOffset int64) string {
	return filepath.Join(c.fileDir(fileID), strconv.FormatInt(alignedOffset, 10)+".bin")
}
