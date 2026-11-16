package bbolt

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	boltstor "github.com/gotd/contrib/bbolt"
	"github.com/gotd/td/telegram/updates"
	bolt "go.etcd.io/bbolt"
)

var (
	bucketSessions = []byte("mtproto_sessions")
	bucketPeers    = []byte("peers")
)

// Store manages the isolated local bbolt database for an MTProto session,
// update watermarks (pts/qts/date/seq), and channel/user access_hash caches.
type Store struct {
	db *bolt.DB
}

// Open opens (or creates) an isolated bbolt database file at dbPath.
func Open(dbPath string) (*Store, error) {
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("create bbolt directory %q: %w", dir, err)
	}

	db, err := bolt.Open(dbPath, 0o600, &bolt.Options{
		Timeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("open bbolt db %q: %w", dbPath, err)
	}

	return &Store{db: db}, nil
}

// Close closes the underlying bbolt database handle.
func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

// DB exposes the underlying *bbolt.DB instance.
func (s *Store) DB() *bolt.DB {
	return s.db
}

// SessionStorage returns a gotd-compatible persistent MTProto auth_key storage.
// Note: gotd/contrib/bbolt takes (db, key, bucket) in that exact order.
func (s *Store) SessionStorage(sessionKey string) boltstor.SessionStorage {
	return boltstor.NewSessionStorage(s.db, sessionKey, bucketSessions)
}

// StateStorage returns a persistent updates.StateStorage storing pts/qts/date/seq and channel pts.
func (s *Store) StateStorage() updates.StateStorage {
	return boltstor.NewStateStorage(s.db)
}

// AccessHasher returns persistent channel/user access_hash storage for updates.Manager.
func (s *Store) AccessHasher() *boltstor.AccessHasher {
	return boltstor.NewAccessHasher(s.db)
}

// PeerStorage returns a persistent peer storage backed by bbolt.
func (s *Store) PeerStorage() *boltstor.PeerStorage {
	return boltstor.NewPeerStorage(s.db, bucketPeers)
}

// GetChannelPts reads the local channel pts for selfUserID from bbolt.
func (s *Store) GetChannelPts(ctx context.Context, selfUserID, channelID int64) (int, bool, error) {
	return s.StateStorage().GetChannelPts(ctx, selfUserID, channelID)
}

// SetChannelPts persists the local channel pts for selfUserID in bbolt.
func (s *Store) SetChannelPts(ctx context.Context, selfUserID, channelID int64, pts int) error {
	return s.StateStorage().SetChannelPts(ctx, selfUserID, channelID, pts)
}
