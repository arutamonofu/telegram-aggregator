package mtproto

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/gotd/td/tg"
	"golang.org/x/time/rate"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

// ResolveAPI abstracts the Telegram contacts.resolveUsername RPC for testability.
type ResolveAPI interface {
	ContactsResolveUsername(ctx context.Context, request *tg.ContactsResolveUsernameRequest) (*tg.ContactsResolvedPeer, error)
}

// ResolvedChannel holds the channel identity and access hash resolved for an MTProto account.
type ResolvedChannel struct {
	ChannelID  int64
	AccessHash int64
	Username   string
	Title      string
	FromCache  bool
}

type resolveFlight struct {
	done chan struct{}
	res  ResolvedChannel
	err  error
}

// UsernameResolver enforces eternal per-account caching of access_hash values and serializes
// cache misses through a globally paced contacts.resolveUsername queue (~1 req / 5-15s).
type UsernameResolver struct {
	limiter      *rate.Limiter
	accessHashes domain.AccessHashStore
	watermarks   domain.WatermarkStore

	mu       sync.Mutex
	inFlight map[string]*resolveFlight
}

// NewUsernameResolver creates a globally rate-limited username resolver.
func NewUsernameResolver(
	minInterval time.Duration,
	accessHashes domain.AccessHashStore,
	watermarks domain.WatermarkStore,
) *UsernameResolver {
	if minInterval <= 0 {
		minInterval = 5 * time.Second
	}
	return &UsernameResolver{
		limiter:      rate.NewLimiter(rate.Every(minInterval), 1),
		accessHashes: accessHashes,
		watermarks:   watermarks,
		inFlight:     make(map[string]*resolveFlight),
	}
}

// NormalizeUsername trims spaces and leading '@', and lowercases the handle.
func NormalizeUsername(raw string) string {
	s := strings.TrimSpace(raw)
	s = strings.TrimPrefix(s, "@")
	return strings.ToLower(s)
}

// ResolveChannel returns channel identity and account-bound access_hash, consulting the eternal DB
// cache before hitting contacts.resolveUsername via the global rate limiter.
func (r *UsernameResolver) ResolveChannel(
	ctx context.Context,
	accountID int,
	username string,
	api ResolveAPI,
) (ResolvedChannel, error) {
	norm := NormalizeUsername(username)
	if norm == "" {
		return ResolvedChannel{}, errors.New("empty channel username")
	}

	if cached, ok, err := r.tryCache(ctx, accountID, norm); err != nil {
		return ResolvedChannel{}, err
	} else if ok {
		return cached, nil
	}

	flightKey := fmt.Sprintf("%d:%s", accountID, norm)
	r.mu.Lock()
	if existing, ok := r.inFlight[flightKey]; ok {
		r.mu.Unlock()
		select {
		case <-ctx.Done():
			return ResolvedChannel{}, ctx.Err()
		case <-existing.done:
			return existing.res, existing.err
		}
	}
	flight := &resolveFlight{done: make(chan struct{})}
	r.inFlight[flightKey] = flight
	r.mu.Unlock()

	defer func() {
		close(flight.done)
		r.mu.Lock()
		delete(r.inFlight, flightKey)
		r.mu.Unlock()
	}()

	// Re-check cache after winning singleflight in case another worker populated it just before.
	if cached, ok, err := r.tryCache(ctx, accountID, norm); err != nil {
		flight.err = err
		return ResolvedChannel{}, err
	} else if ok {
		flight.res = cached
		return cached, nil
	}

	if api == nil {
		flight.err = errors.New("resolve API client is nil")
		return ResolvedChannel{}, flight.err
	}

	if err := r.limiter.Wait(ctx); err != nil {
		flight.err = fmt.Errorf("wait resolve rate limiter for @%s: %w", norm, err)
		return ResolvedChannel{}, flight.err
	}

	resolved, err := api.ContactsResolveUsername(ctx, &tg.ContactsResolveUsernameRequest{
		Username: norm,
	})
	if err != nil {
		flight.err = fmt.Errorf("contacts.resolveUsername(@%s): %w", norm, err)
		return ResolvedChannel{}, flight.err
	}

	res, err := extractResolvedChannel(resolved, norm)
	if err != nil {
		flight.err = err
		return ResolvedChannel{}, err
	}

	if r.watermarks != nil {
		u := norm
		if err := r.watermarks.UpsertChannel(ctx, domain.ChannelStateRecord{
			ChannelID:       res.ChannelID,
			Username:        &u,
			Title:           res.Title,
			Tier:            "TIER_2",
			PollIntervalSec: 900,
			NextPollAt:      time.Now().UTC(),
			IsActive:        true,
		}); err != nil {
			flight.err = fmt.Errorf("persist resolved channel %d: %w", res.ChannelID, err)
			return ResolvedChannel{}, flight.err
		}
	}

	if r.accessHashes != nil {
		if err := r.accessHashes.SaveAccessHash(ctx, accountID, res.ChannelID, "channel", res.AccessHash); err != nil {
			flight.err = fmt.Errorf("persist eternal access_hash for channel %d: %w", res.ChannelID, err)
			return ResolvedChannel{}, flight.err
		}
	}

	flight.res = res
	return res, nil
}

func (r *UsernameResolver) tryCache(ctx context.Context, accountID int, norm string) (ResolvedChannel, bool, error) {
	if r.watermarks == nil || r.accessHashes == nil {
		return ResolvedChannel{}, false, nil
	}

	ch, foundCh, err := r.watermarks.GetChannelByUsername(ctx, norm)
	if err != nil {
		return ResolvedChannel{}, false, fmt.Errorf("lookup channel cache @%s: %w", norm, err)
	}
	if !foundCh {
		return ResolvedChannel{}, false, nil
	}

	hash, foundHash, err := r.accessHashes.GetAccessHash(ctx, accountID, ch.ChannelID)
	if err != nil {
		return ResolvedChannel{}, false, fmt.Errorf("lookup access_hash cache (%d, %d): %w", accountID, ch.ChannelID, err)
	}
	if !foundHash {
		return ResolvedChannel{}, false, nil
	}

	return ResolvedChannel{
		ChannelID:  ch.ChannelID,
		AccessHash: hash,
		Username:   norm,
		Title:      ch.Title,
		FromCache:  true,
	}, true, nil
}

func extractResolvedChannel(resolved *tg.ContactsResolvedPeer, fallbackUsername string) (ResolvedChannel, error) {
	if resolved == nil {
		return ResolvedChannel{}, errors.New("nil ContactsResolvedPeer response")
	}

	peerChannel, ok := resolved.Peer.(*tg.PeerChannel)
	if !ok {
		return ResolvedChannel{}, fmt.Errorf("username @%s did not resolve to a channel (got %T)", fallbackUsername, resolved.Peer)
	}

	for _, c := range resolved.Chats {
		ch, ok := c.(*tg.Channel)
		if !ok || ch.ID != peerChannel.ChannelID {
			continue
		}
		accessHash, hasHash := ch.GetAccessHash()
		if !hasHash {
			return ResolvedChannel{}, fmt.Errorf("resolved channel %d has no access_hash", ch.ID)
		}
		uname := fallbackUsername
		if u, ok := ch.GetUsername(); ok && u != "" {
			uname = strings.ToLower(u)
		}
		return ResolvedChannel{
			ChannelID:  ch.ID,
			AccessHash: accessHash,
			Username:   uname,
			Title:      ch.Title,
			FromCache:  false,
		}, nil
	}

	return ResolvedChannel{}, fmt.Errorf("channel %d missing in ContactsResolvedPeer.Chats", peerChannel.ChannelID)
}
