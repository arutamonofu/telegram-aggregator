package publisher

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/tg-agg/ingestion-go/internal/domain"
)

const (
	StreamIngestPosts     = "stream:ingest:posts"
	StreamIngestComments  = "stream:ingest:comments"
	StreamIngestDeletions = "stream:ingest:deletions"
)

var _ domain.EventPublisher = (*RedisStreamsPublisher)(nil)

// RedisStreamsPublisher publishes ingestion events synchronously via Redis XADD
// and verifies non-empty Stream Entry ID acknowledgements before returning.
type RedisStreamsPublisher struct {
	client redis.UniversalClient
	maxLen int64
}

// NewRedisStreamsPublisher creates a publisher backed by go-redis UniversalClient.
func NewRedisStreamsPublisher(client redis.UniversalClient) *RedisStreamsPublisher {
	return &RedisStreamsPublisher{
		client: client,
		maxLen: 100_000,
	}
}

// PublishPost synchronously publishes IngestPostEventV1 to stream:ingest:posts.
func (p *RedisStreamsPublisher) PublishPost(ctx context.Context, event domain.IngestPostEventV1) (string, error) {
	if event.SchemaVersion == "" {
		event.SchemaVersion = "ingest.post.v1"
	}
	return p.publishJSON(ctx, StreamIngestPosts, event.EventID, event.SchemaVersion, event)
}

// PublishCommentsDelta synchronously publishes IngestCommentsDeltaEventV1 to stream:ingest:comments.
func (p *RedisStreamsPublisher) PublishCommentsDelta(ctx context.Context, event domain.IngestCommentsDeltaEventV1) (string, error) {
	if event.SchemaVersion == "" {
		event.SchemaVersion = "ingest.comments.v1"
	}
	return p.publishJSON(ctx, StreamIngestComments, event.EventID, event.SchemaVersion, event)
}

// PublishDeletion synchronously publishes IngestDeletionEventV1 to stream:ingest:deletions.
func (p *RedisStreamsPublisher) PublishDeletion(ctx context.Context, event domain.IngestDeletionEventV1) (string, error) {
	if event.SchemaVersion == "" {
		event.SchemaVersion = "ingest.deletion.v1"
	}
	return p.publishJSON(ctx, StreamIngestDeletions, event.EventID, event.SchemaVersion, event)
}

func (p *RedisStreamsPublisher) publishJSON(
	ctx context.Context,
	stream string,
	eventID string,
	schemaVersion string,
	payload any,
) (string, error) {
	if p.client == nil {
		return "", errors.New("redis client is nil")
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal %s payload: %w", schemaVersion, err)
	}

	args := &redis.XAddArgs{
		Stream: stream,
		ID:     "*",
		Values: map[string]any{
			"event_id":       eventID,
			"schema_version": schemaVersion,
			"payload":        string(raw),
		},
	}
	if p.maxLen > 0 {
		args.MaxLen = p.maxLen
		args.Approx = true
	}

	entryID, err := p.client.XAdd(ctx, args).Result()
	if err != nil {
		return "", fmt.Errorf("redis XADD %s: %w", stream, err)
	}
	if entryID == "" {
		return "", fmt.Errorf("redis XADD %s returned empty entry ID", stream)
	}
	return entryID, nil
}
