package updates

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf16"

	"github.com/google/uuid"
	"github.com/gotd/td/tg"

	"github.com/tg-agg/ingestion-go/internal/album"
	"github.com/tg-agg/ingestion-go/internal/domain"
)

// ChannelContext provides cached channel metadata for canonicalizing updates.
type ChannelContext struct {
	ChannelID  int64
	AccessHash int64
	Username   *string
	Title      string
}

// CanonicalPost holds a normalized Telegram channel message plus storage metadata.
type CanonicalPost struct {
	Event        domain.IngestPostEventV1
	FileRefs     []domain.FileReferenceRecord
	FileSources  []domain.FileSourceRecord
	ContentHash  []byte
	MediaIDs     []string
	PublishedAt  time.Time
	EditDateTime *time.Time
	MsgID        int
	GroupedID    int64
}

// TrimRightSpaces strips trailing Unicode whitespace before hashing and UTF-16 entity bounds checking.
func TrimRightSpaces(s string) string {
	return strings.TrimRightFunc(s, unicode.IsSpace)
}

// UTF16CodeUnitLen returns the exact UTF-16 code unit length matching Telegram entity offsets.
func UTF16CodeUnitLen(s string) int {
	n := 0
	for _, r := range s {
		n += utf16.RuneLen(r)
	}
	return n
}

// ContentSHA256 computes the canonical SHA-256 digest of rtrimmed text_raw.
func ContentSHA256(trimmedText string) []byte {
	sum := sha256.Sum256([]byte(trimmedText))
	return sum[:]
}

// CanonicalizeMessage converts a *tg.Message into our normalized event, file_references, and file_sources.
func CanonicalizeMessage(
	msg *tg.Message,
	ch ChannelContext,
	eventType domain.EventType,
) (CanonicalPost, error) {
	if msg == nil {
		return CanonicalPost{}, fmt.Errorf("nil tg.Message")
	}

	channelID := ch.ChannelID
	if channelID == 0 {
		if peerCh, ok := msg.PeerID.(*tg.PeerChannel); ok && peerCh != nil {
			channelID = peerCh.ChannelID
		}
	}

	trimmed := TrimRightSpaces(msg.Message)
	utf16Len := UTF16CodeUnitLen(trimmed)
	entities := MapEntities(msg.Entities, utf16Len)

	pubAt := time.Unix(int64(msg.Date), 0).UTC()
	var (
		editTimePtr *time.Time
		editStrPtr  *string
	)
	if editSec, ok := msg.GetEditDate(); ok && editSec > 0 {
		t := time.Unix(int64(editSec), 0).UTC()
		editTimePtr = &t
		s := t.Format(time.RFC3339)
		editStrPtr = &s
	}

	var groupedPtr *int64
	groupedVal, hasGrouped := msg.GetGroupedID()
	if hasGrouped && groupedVal != 0 {
		g := groupedVal
		groupedPtr = &g
	}

	views, _ := msg.GetViews()
	forwards, _ := msg.GetForwards()

	repliesInfo := domain.RepliesInfo{}
	if rep, ok := msg.GetReplies(); ok {
		repliesInfo.CommentsEnabled = rep.GetComments()
		repliesInfo.RepliesCount = rep.GetReplies()
		if maxID, hasMax := rep.GetMaxID(); hasMax {
			repliesInfo.RepliesMaxID = maxID
		}
		if linkedID, hasLinked := rep.GetChannelID(); hasLinked && linkedID != 0 {
			l := linkedID
			repliesInfo.LinkedChatID = &l
		}
	}

	var (
		mediaItems  []domain.MediaItem
		fileRefs    []domain.FileReferenceRecord
		fileSources []domain.FileSourceRecord
		mediaIDs    []string
	)
	if rawMedia, ok := msg.GetMedia(); ok && rawMedia != nil {
		item, ref, extracted := extractMedia(rawMedia)
		if extracted {
			mediaItems = append(mediaItems, item)
			fileRefs = append(fileRefs, ref)
			mediaIDs = append(mediaIDs, item.FileID)
			fileSources = append(fileSources, domain.FileSourceRecord{
				FileID:    item.FileID,
				ChannelID: channelID,
				MsgID:     msg.ID,
				GroupedID: groupedPtr,
			})
		}
	}

	title := ch.Title
	if title == "" {
		title = fmt.Sprintf("channel:%d", channelID)
	}

	event := domain.IngestPostEventV1{
		SchemaVersion:         "ingest.post.v1",
		EventID:               uuid.NewString(),
		EventType:             eventType,
		ChannelID:             channelID,
		ChannelUsername:       ch.Username,
		ChannelTitle:          title,
		MsgID:                 msg.ID,
		GroupedID:             groupedPtr,
		PublishedAt:           pubAt.Format(time.RFC3339),
		EditDate:              editStrPtr,
		TextRaw:               trimmed,
		ShowCaptionAboveMedia: msg.GetInvertMedia(),
		Views:                 views,
		Forwards:              forwards,
		Entities:              entities,
		Replies:               repliesInfo,
		MediaItems:            mediaItems,
	}

	return CanonicalPost{
		Event:        event,
		FileRefs:     fileRefs,
		FileSources:  fileSources,
		ContentHash:  ContentSHA256(trimmed),
		MediaIDs:     mediaIDs,
		PublishedAt:  pubAt,
		EditDateTime: editTimePtr,
		MsgID:        msg.ID,
		GroupedID:    groupedVal,
	}, nil
}

// ToBufferedMessage adapts CanonicalPost into an album.BufferedMessage.
func (c CanonicalPost) ToBufferedMessage() album.BufferedMessage {
	return album.BufferedMessage{
		Event:       c.Event,
		FileRefs:    c.FileRefs,
		FileSources: c.FileSources,
		ContentHash: c.ContentHash,
		EditDate:    c.EditDateTime,
	}
}

// MapEntities converts Telegram Layer 225+ entities into canonical UTF-16 AST items,
// clamping each offset/length pair within [0, maxUTF16Len].
func MapEntities(raw []tg.MessageEntityClass, maxUTF16Len int) []domain.EntityItem {
	if len(raw) == 0 {
		return []domain.EntityItem{}
	}

	out := make([]domain.EntityItem, 0, len(raw))
	for _, ent := range raw {
		item, ok := mapSingleEntity(ent)
		if !ok || item.Length <= 0 || item.Offset >= maxUTF16Len {
			continue
		}
		if item.Offset < 0 {
			item.Length += item.Offset
			item.Offset = 0
		}
		if item.Offset+item.Length > maxUTF16Len {
			item.Length = maxUTF16Len - item.Offset
		}
		if item.Length <= 0 {
			continue
		}
		out = append(out, item)
	}
	return out
}

func mapSingleEntity(ent tg.MessageEntityClass) (domain.EntityItem, bool) {
	switch e := ent.(type) {
	case *tg.MessageEntityUnknown:
		return domain.EntityItem{Type: "messageEntityUnknown", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityMention:
		return domain.EntityItem{Type: "messageEntityMention", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityHashtag:
		return domain.EntityItem{Type: "messageEntityHashtag", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityBotCommand:
		return domain.EntityItem{Type: "messageEntityBotCommand", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityURL:
		return domain.EntityItem{Type: "messageEntityUrl", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityEmail:
		return domain.EntityItem{Type: "messageEntityEmail", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityBold:
		return domain.EntityItem{Type: "messageEntityBold", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityItalic:
		return domain.EntityItem{Type: "messageEntityItalic", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityCode:
		return domain.EntityItem{Type: "messageEntityCode", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityPre:
		var lang *string
		if e.Language != "" {
			l := e.Language
			lang = &l
		}
		return domain.EntityItem{Type: "messageEntityPre", Offset: e.Offset, Length: e.Length, Language: lang}, true
	case *tg.MessageEntityTextURL:
		u := e.URL
		return domain.EntityItem{Type: "messageEntityTextUrl", Offset: e.Offset, Length: e.Length, URL: &u}, true
	case *tg.MessageEntityMentionName:
		return domain.EntityItem{Type: "messageEntityMentionName", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityPhone:
		return domain.EntityItem{Type: "messageEntityPhone", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityCashtag:
		return domain.EntityItem{Type: "messageEntityCashtag", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityUnderline:
		return domain.EntityItem{Type: "messageEntityUnderline", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityStrike:
		return domain.EntityItem{Type: "messageEntityStrike", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityBankCard:
		return domain.EntityItem{Type: "messageEntityBankCard", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntitySpoiler:
		return domain.EntityItem{Type: "messageEntitySpoiler", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityCustomEmoji:
		docID := strconv.FormatInt(e.DocumentID, 10)
		return domain.EntityItem{Type: "messageEntityCustomEmoji", Offset: e.Offset, Length: e.Length, DocumentID: &docID}, true
	case *tg.MessageEntityBlockquote:
		c := e.Collapsed
		return domain.EntityItem{Type: "messageEntityBlockquote", Offset: e.Offset, Length: e.Length, Collapsed: &c}, true
	case *tg.MessageEntityFormattedDate:
		unix := int64(e.Date)
		flags := 0
		if e.Relative {
			flags |= 1 << 0
		}
		if e.ShortTime {
			flags |= 1 << 1
		}
		if e.LongTime {
			flags |= 1 << 2
		}
		if e.ShortDate {
			flags |= 1 << 3
		}
		if e.LongDate {
			flags |= 1 << 4
		}
		if e.DayOfWeek {
			flags |= 1 << 5
		}
		return domain.EntityItem{
			Type:      "messageEntityFormattedDate",
			Offset:    e.Offset,
			Length:    e.Length,
			UnixTime:  &unix,
			DateFlags: &flags,
		}, true
	case *tg.MessageEntityDiffInsert:
		return domain.EntityItem{Type: "messageEntityDiffInsert", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityDiffReplace:
		oldText := e.OldText
		return domain.EntityItem{Type: "messageEntityDiffReplace", Offset: e.Offset, Length: e.Length, OldText: &oldText}, true
	case *tg.MessageEntityDiffDelete:
		return domain.EntityItem{Type: "messageEntityDiffDelete", Offset: e.Offset, Length: e.Length}, true
	case *tg.MessageEntityTonAddress:
		return domain.EntityItem{Type: "messageEntityTonAddress", Offset: e.Offset, Length: e.Length}, true
	default:
		return domain.EntityItem{}, false
	}
}

func extractMedia(raw tg.MessageMediaClass) (domain.MediaItem, domain.FileReferenceRecord, bool) {
	switch m := raw.(type) {
	case *tg.MessageMediaPhoto:
		rawPhoto, ok := m.GetPhoto()
		if !ok {
			return domain.MediaItem{}, domain.FileReferenceRecord{}, false
		}
		photo, ok := rawPhoto.(*tg.Photo)
		if !ok || photo == nil || photo.ID == 0 {
			return domain.MediaItem{}, domain.FileReferenceRecord{}, false
		}
		return ExtractPhotoMedia(photo)

	case *tg.MessageMediaDocument:
		rawDoc, ok := m.GetDocument()
		if !ok {
			return domain.MediaItem{}, domain.FileReferenceRecord{}, false
		}
		doc, ok := rawDoc.(*tg.Document)
		if !ok || doc == nil || doc.ID == 0 {
			return domain.MediaItem{}, domain.FileReferenceRecord{}, false
		}
		return ExtractDocumentMedia(doc)

	default:
		return domain.MediaItem{}, domain.FileReferenceRecord{}, false
	}
}

// ExtractPhotoMedia extracts file_id, largest PhotoSize.Type as thumb_size, stripped thumb, and FileReferenceRecord.
func ExtractPhotoMedia(photo *tg.Photo) (domain.MediaItem, domain.FileReferenceRecord, bool) {
	if photo == nil || photo.ID == 0 {
		return domain.MediaItem{}, domain.FileReferenceRecord{}, false
	}

	var (
		bestType   = "y"
		bestSize   int64
		bestW      int
		bestH      int
		strippedB64 *string
	)

	for _, ps := range photo.Sizes {
		switch s := ps.(type) {
		case *tg.PhotoStrippedSize:
			if len(s.Bytes) > 0 {
				enc := base64.StdEncoding.EncodeToString(s.Bytes)
				strippedB64 = &enc
			}
		case *tg.PhotoSize:
			sz := int64(s.Size)
			if sz == 0 {
				sz = int64(s.W) * int64(s.H)
			}
			if sz >= bestSize && s.Type != "" {
				bestSize = int64(s.Size)
				bestType = s.Type
				bestW = s.W
				bestH = s.H
			}
		case *tg.PhotoSizeProgressive:
			var progMax int64
			if len(s.Sizes) > 0 {
				progMax = int64(s.Sizes[len(s.Sizes)-1])
			}
			if progMax >= bestSize && s.Type != "" {
				bestSize = progMax
				bestType = s.Type
				bestW = s.W
				bestH = s.H
			}
		case *tg.PhotoCachedSize:
			sz := int64(len(s.Bytes))
			if sz > bestSize && s.Type != "" {
				bestSize = sz
				bestType = s.Type
				bestW = s.W
				bestH = s.H
			}
		}
	}

	fileID := fmt.Sprintf("%s:%d:%d:%s", domain.MediaKindPhoto, photo.DCID, photo.ID, bestType)
	mime := "image/jpeg"
	refBytes := append([]byte(nil), photo.FileReference...)
	refB64 := base64.StdEncoding.EncodeToString(refBytes)

	var wPtr, hPtr *int
	if bestW > 0 {
		w := bestW
		wPtr = &w
	}
	if bestH > 0 {
		h := bestH
		hPtr = &h
	}

	item := domain.MediaItem{
		FileID:           fileID,
		TGID:             photo.ID,
		AccessHash:       photo.AccessHash,
		DCID:             photo.DCID,
		MediaKind:        domain.MediaKindPhoto,
		ThumbSize:        bestType,
		FileReferenceB64: refB64,
		MimeType:         &mime,
		FileSize:         bestSize,
		Width:            wPtr,
		Height:           hPtr,
		StrippedThumbB64: strippedB64,
	}

	rec := domain.FileReferenceRecord{
		FileID:        fileID,
		TGID:          photo.ID,
		AccessHash:    photo.AccessHash,
		DCID:          photo.DCID,
		MediaKind:     domain.MediaKindPhoto,
		ThumbSize:     bestType,
		FileReference: refBytes,
		FileSize:      bestSize,
		MimeType:      &mime,
		RefreshedAt:   time.Now().UTC(),
	}
	return item, rec, true
}

// ExtractDocumentMedia classifies document/video/voice/sticker attributes and builds file_id.
func ExtractDocumentMedia(doc *tg.Document) (domain.MediaItem, domain.FileReferenceRecord, bool) {
	if doc == nil || doc.ID == 0 {
		return domain.MediaItem{}, domain.FileReferenceRecord{}, false
	}

	kind := domain.MediaKindDocument
	var wPtr, hPtr, durPtr *int

	for _, attr := range doc.Attributes {
		switch a := attr.(type) {
		case *tg.DocumentAttributeVideo:
			kind = domain.MediaKindVideo
			w, h := a.W, a.H
			dur := int(a.Duration)
			wPtr, hPtr, durPtr = &w, &h, &dur
		case *tg.DocumentAttributeAudio:
			if a.Voice {
				kind = domain.MediaKindVoice
			}
			dur := a.Duration
			durPtr = &dur
		case *tg.DocumentAttributeSticker:
			kind = domain.MediaKindSticker
		case *tg.DocumentAttributeImageSize:
			w, h := a.W, a.H
			if wPtr == nil {
				wPtr = &w
			}
			if hPtr == nil {
				hPtr = &h
			}
		}
	}

	fileID := fmt.Sprintf("%s:%d:%d:", kind, doc.DCID, doc.ID)
	var mimePtr *string
	if doc.MimeType != "" {
		m := doc.MimeType
		mimePtr = &m
	}
	refBytes := append([]byte(nil), doc.FileReference...)
	refB64 := base64.StdEncoding.EncodeToString(refBytes)

	item := domain.MediaItem{
		FileID:           fileID,
		TGID:             doc.ID,
		AccessHash:       doc.AccessHash,
		DCID:             doc.DCID,
		MediaKind:        kind,
		ThumbSize:        "",
		FileReferenceB64: refB64,
		MimeType:         mimePtr,
		FileSize:         doc.Size,
		Width:            wPtr,
		Height:           hPtr,
		DurationSec:      durPtr,
	}

	rec := domain.FileReferenceRecord{
		FileID:        fileID,
		TGID:          doc.ID,
		AccessHash:    doc.AccessHash,
		DCID:          doc.DCID,
		MediaKind:     kind,
		ThumbSize:     "",
		FileReference: refBytes,
		FileSize:      doc.Size,
		MimeType:      mimePtr,
		RefreshedAt:   time.Now().UTC(),
	}
	return item, rec, true
}

// BuildChannelEntitiesMap indexes *tg.Channel metadata from updates/differences by channel ID.
func BuildChannelEntitiesMap(chats []tg.ChatClass) map[int64]*tg.Channel {
	res := make(map[int64]*tg.Channel, len(chats))
	for _, c := range chats {
		if ch, ok := c.(*tg.Channel); ok && ch != nil {
			res[ch.ID] = ch
		}
	}
	return res
}

// MediaIDSignature returns a deterministic slice of file_ids for edit deduplication comparison.
func MediaIDSignature(items []domain.MediaItem) []string {
	ids := make([]string, len(items))
	for i, it := range items {
		ids[i] = it.FileID
	}
	sort.Strings(ids)
	return ids
}
