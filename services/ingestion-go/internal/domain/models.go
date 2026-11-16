package domain

import "time"

// EventType identifies post lifecycle events on stream:ingest:posts.
type EventType string

const (
	EventTypePostNew  EventType = "post_new"
	EventTypePostEdit EventType = "post_edit"
)

// MediaKind mirrors core_public.media_kind_enum and IngestPostEventV1.
type MediaKind string

const (
	MediaKindPhoto    MediaKind = "photo"
	MediaKindDocument MediaKind = "document"
	MediaKindVideo    MediaKind = "video"
	MediaKindVoice    MediaKind = "voice"
	MediaKindSticker  MediaKind = "sticker"
)

// SessionFSMState mirrors core_public.session_state_enum.
type SessionFSMState string

const (
	SessionWarmingUp          SessionFSMState = "WARMING_UP"
	SessionStandbyReady       SessionFSMState = "STANDBY_READY"
	SessionActiveStreaming    SessionFSMState = "ACTIVE_STREAMING"
	SessionActivePolling      SessionFSMState = "ACTIVE_POLLING"
	SessionDegradedQuarantine SessionFSMState = "DEGRADED_QUARANTINE"
	SessionFrozenAppeal       SessionFSMState = "FROZEN_APPEAL"
	SessionTerminatedDead     SessionFSMState = "TERMINATED_DEAD"
)

// IngestPostEventV1 matches packages/contracts/events/ingest_post_v1.schema.json.
type IngestPostEventV1 struct {
	SchemaVersion         string       `json:"schema_version"`
	EventID               string       `json:"event_id"`
	EventType             EventType    `json:"event_type"`
	ChannelID             int64        `json:"channel_id"`
	ChannelUsername       *string      `json:"channel_username,omitempty"`
	ChannelTitle          string       `json:"channel_title"`
	MsgID                 int          `json:"msg_id"`
	GroupedID             *int64       `json:"grouped_id,omitempty"`
	AlbumMsgIDs           []int        `json:"album_msg_ids,omitempty"`
	PublishedAt           string       `json:"published_at"`
	EditDate              *string      `json:"edit_date"`
	TextRaw               string       `json:"text_raw"`
	ShowCaptionAboveMedia bool         `json:"show_caption_above_media"`
	Views                 int          `json:"views"`
	Forwards              int          `json:"forwards"`
	Entities              []EntityItem `json:"entities"`
	Replies               RepliesInfo  `json:"replies"`
	MediaItems            []MediaItem  `json:"media_items"`
}

// EntityItem describes a Telegram UTF-16 message entity (Layer 225+).
type EntityItem struct {
	Type       string  `json:"type"`
	Offset     int     `json:"offset"`
	Length     int     `json:"length"`
	Language   *string `json:"language,omitempty"`
	URL        *string `json:"url,omitempty"`
	Collapsed  *bool   `json:"collapsed,omitempty"`
	DocumentID *string `json:"document_id,omitempty"`
	UnixTime   *int64  `json:"unix_time,omitempty"`
	DateFlags  *int    `json:"date_flags,omitempty"`
	OldText    *string `json:"old_text,omitempty"`
}

// RepliesInfo carries discussion thread metadata extracted from tg.MessageReplies.
type RepliesInfo struct {
	CommentsEnabled bool   `json:"comments_enabled"`
	LinkedChatID    *int64 `json:"linked_chat_id,omitempty"`
	RepliesCount    int    `json:"replies_count"`
	RepliesMaxID    int    `json:"replies_max_id"`
}

// MediaItem represents an individual photo, video, voice, sticker, or document attachment.
type MediaItem struct {
	FileID           string    `json:"file_id"`
	TGID             int64     `json:"tg_id"`
	AccessHash       int64     `json:"access_hash"`
	DCID             int       `json:"dc_id"`
	MediaKind        MediaKind `json:"media_kind"`
	ThumbSize        string    `json:"thumb_size"`
	FileReferenceB64 string    `json:"file_reference_b64"`
	MimeType         *string   `json:"mime_type,omitempty"`
	FileSize         int64     `json:"file_size"`
	Width            *int      `json:"width,omitempty"`
	Height           *int      `json:"height,omitempty"`
	DurationSec      *int      `json:"duration_sec,omitempty"`
	Caption          *string   `json:"caption,omitempty"`
	StrippedThumbB64 *string   `json:"stripped_thumb_b64,omitempty"`
}

// IngestCommentsDeltaEventV1 matches packages/contracts/events/ingest_comments_v1.schema.json.
type IngestCommentsDeltaEventV1 struct {
	SchemaVersion     string              `json:"schema_version"`
	EventID           string              `json:"event_id"`
	ChannelID         int64               `json:"channel_id"`
	PostMsgID         int                 `json:"post_msg_id"`
	LinkedChatID      int64               `json:"linked_chat_id"`
	TotalRepliesCount int                 `json:"total_replies_count"`
	Comments          []IngestCommentItem `json:"comments"`
}

// IngestCommentItem represents a single comment message from a discussion supergroup.
type IngestCommentItem struct {
	CommentMsgID   int          `json:"comment_msg_id"`
	ReplyToMsgID   *int         `json:"reply_to_msg_id,omitempty"`
	AuthorPeerHash string       `json:"author_peer_hash"`
	PublishedAt    string       `json:"published_at"`
	EditDate       *string      `json:"edit_date,omitempty"`
	TextRaw        string       `json:"text_raw"`
	Entities       []EntityItem `json:"entities"`
}

// IngestDeletionEventV1 matches packages/contracts/events/ingest_deletion_v1.schema.json.
type IngestDeletionEventV1 struct {
	SchemaVersion string `json:"schema_version"`
	EventID       string `json:"event_id"`
	ChannelID     int64  `json:"channel_id"`
	DeletedMsgIDs []int  `json:"deleted_msg_ids"`
	OccurredAt    string `json:"occurred_at"`
}

// ChannelStateRecord mirrors global channel cursors in core_public.channels.
type ChannelStateRecord struct {
	ChannelID          int64
	Username           *string
	Title              string
	Tier               string
	LastMsgID          int
	MaxEditDateChecked *time.Time
	LinkedChatID       *int64
	PinnedMsgIDs       []int
	PollIntervalSec    int
	NextPollAt         time.Time
	IsActive           bool
}

// PostSyncMeta holds existing post reply and edit metadata for delta checks.
type PostSyncMeta struct {
	ChannelID          int64
	MsgID              int
	EditDate           *time.Time
	ContentSHA256      []byte
	RepliesCount       int
	RepliesMaxID       int
	RepliesSyncedCount int
}

// FileReferenceRecord mirrors core_public.file_references.
type FileReferenceRecord struct {
	FileID            string
	TGID              int64
	AccessHash        int64
	DCID              int
	MediaKind         MediaKind
	ThumbSize         string
	FileReference     []byte
	FileSize          int64
	MimeType          *string
	LocalCachePath    *string
	LocalCacheExpires *time.Time
	RefreshedAt       time.Time
	IsPermanentlyGone bool
}

// FileSourceRecord mirrors core_public.file_sources.
type FileSourceRecord struct {
	SourceID        int64
	FileID          string
	ChannelID       int64
	MsgID           int
	GroupedID       *int64
	AlbumMsgIDs     []int
	IsSourceDeleted bool
	CreatedAt       time.Time
}

// SessionRecord mirrors core_public.mtproto_sessions.
type SessionRecord struct {
	AccountID      int
	PhoneHash      string
	IsPremium      bool
	FSMState       SessionFSMState
	RiskScore      int
	ProxyIPv4      *string
	FloodWaitUntil *time.Time
	BBoltFilePath  string
}
