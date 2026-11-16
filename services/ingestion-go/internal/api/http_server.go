package api

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"regexp"
	"strings"

	"github.com/tg-agg/ingestion-go/internal/comments"
	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/media"
	"github.com/tg-agg/ingestion-go/internal/mtproto"
)

var validFileIDRe = regexp.MustCompile(`^[a-zA-Z0-9:_-]+$`)

// CommentSyncer triggers on-demand delta comment sync.
type CommentSyncer interface {
	SyncPostComments(ctx context.Context, channelID int64, postMsgID int, telegramReplies domain.RepliesInfo) (int, error)
}

var _ CommentSyncer = (*comments.DeltaSyncService)(nil)

// ChannelRegistrar resolves @username safely and starts live channel tracking.
type ChannelRegistrar func(ctx context.Context, username, tier string) (mtproto.ResolvedChannel, error)

// HTTPServer exposes ingestion-go's internal HTTP endpoints on :8081:
//   - GET /healthz
//   - GET /internal/v1/media/{file_id}
//   - POST /internal/v1/comments/sync
//   - POST /internal/v1/channels/register
type HTTPServer struct {
	streamer         *media.Streamer
	commentSyncer    CommentSyncer
	scheduler        domain.SessionScheduler
	channelRegistrar ChannelRegistrar
	logger           *slog.Logger
	mux              *http.ServeMux
}

// NewHTTPServer builds and configures the internal HTTP router.
func NewHTTPServer(
	streamer *media.Streamer,
	commentSyncer CommentSyncer,
	scheduler domain.SessionScheduler,
	channelRegistrar ChannelRegistrar,
	logger *slog.Logger,
) *HTTPServer {
	if logger == nil {
		logger = slog.Default()
	}

	s := &HTTPServer{
		streamer:         streamer,
		commentSyncer:    commentSyncer,
		scheduler:        scheduler,
		channelRegistrar: channelRegistrar,
		logger:           logger,
		mux:              http.NewServeMux(),
	}

	s.mux.HandleFunc("GET /healthz", s.handleHealthz)
	s.mux.HandleFunc("GET /internal/v1/media/{file_id}", s.handleMedia)
	s.mux.HandleFunc("POST /internal/v1/comments/sync", s.handleCommentSync)
	s.mux.HandleFunc("POST /internal/v1/channels/register", s.handleRegisterChannel)
	return s
}

// Handler returns the configured http.Handler.
func (s *HTTPServer) Handler() http.Handler {
	return s.mux
}

func (s *HTTPServer) handleHealthz(w http.ResponseWriter, r *http.Request) {
	var healthyIDs []int
	if s.scheduler != nil {
		healthyIDs, _ = s.scheduler.HealthyAccountIDs(r.Context())
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":           "ok",
		"service":          "ingestion-go",
		"healthy_accounts": len(healthyIDs),
	})
}

func (s *HTTPServer) handleMedia(w http.ResponseWriter, r *http.Request) {
	if s.streamer == nil {
		http.Error(w, "media streamer disabled", http.StatusServiceUnavailable)
		return
	}

	fileID := r.PathValue("file_id")
	if fileID == "" {
		fileID = strings.TrimPrefix(r.URL.Path, "/internal/v1/media/")
	}
	if fileID == "" || !validFileIDRe.MatchString(fileID) {
		http.Error(w, "invalid file_id", http.StatusBadRequest)
		return
	}

	s.streamer.ServeMedia(w, r, fileID)
}

type commentSyncRequest struct {
	ChannelID       int64  `json:"channel_id"`
	PostMsgID       int    `json:"post_msg_id"`
	CommentsEnabled bool   `json:"comments_enabled"`
	LinkedChatID    *int64 `json:"linked_chat_id,omitempty"`
	RepliesCount    int    `json:"replies_count"`
	RepliesMaxID    int    `json:"replies_max_id"`
}

func (s *HTTPServer) handleCommentSync(w http.ResponseWriter, r *http.Request) {
	if s.commentSyncer == nil {
		http.Error(w, "comment syncer disabled", http.StatusServiceUnavailable)
		return
	}

	var req commentSyncRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	if req.ChannelID == 0 || req.PostMsgID <= 0 {
		http.Error(w, "channel_id and post_msg_id are required", http.StatusBadRequest)
		return
	}

	synced, err := s.commentSyncer.SyncPostComments(r.Context(), req.ChannelID, req.PostMsgID, domain.RepliesInfo{
		CommentsEnabled: req.CommentsEnabled,
		LinkedChatID:    req.LinkedChatID,
		RepliesCount:    req.RepliesCount,
		RepliesMaxID:    req.RepliesMaxID,
	})
	if err != nil {
		if errors.Is(err, domain.ErrSessionQuarantined) {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		s.logger.Error("comment sync failed", "channel_id", req.ChannelID, "msg_id", req.PostMsgID, "err", err)
		http.Error(w, "comment sync error", http.StatusBadGateway)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":         "ok",
		"synced_count":   synced,
		"channel_id":     req.ChannelID,
		"post_msg_id":    req.PostMsgID,
		"replies_target": req.RepliesCount,
	})
}

type registerChannelRequest struct {
	Username string `json:"username"`
	Tier     string `json:"tier,omitempty"`
}

func (s *HTTPServer) handleRegisterChannel(w http.ResponseWriter, r *http.Request) {
	if s.channelRegistrar == nil {
		http.Error(w, "channel registration unavailable", http.StatusServiceUnavailable)
		return
	}

	var req registerChannelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Username) == "" {
		http.Error(w, "username is required", http.StatusBadRequest)
		return
	}

	res, err := s.channelRegistrar(r.Context(), req.Username, req.Tier)
	if err != nil {
		s.logger.Error("register channel failed", "username", req.Username, "err", err)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"channel_id": res.ChannelID,
		"username":   res.Username,
		"title":      res.Title,
		"from_cache": res.FromCache,
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
