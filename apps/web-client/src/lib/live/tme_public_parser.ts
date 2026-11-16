import type { RawTelegramEntity } from '@/lib/ast/types';
import type {
  CleanPostItem,
  DigestCard,
  MediaAttachmentItem,
  SourcePlatform,
} from '@/lib/types/feed';

const HTML_NAMED_ENTITIES: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
  mdash: '—',
  ndash: '–',
  laquo: '«',
  raquo: '»',
};

export function decodeHtmlEntities(input: string): string {
  return input.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (full, entity: string) => {
    if (entity.startsWith('#x') || entity.startsWith('#X')) {
      const code = Number.parseInt(entity.slice(2), 16);
      return Number.isFinite(code) ? String.fromCodePoint(code) : full;
    }
    if (entity.startsWith('#')) {
      const code = Number.parseInt(entity.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : full;
    }
    return HTML_NAMED_ENTITIES[entity] ?? full;
  });
}

function stripTags(html: string): string {
  return decodeHtmlEntities(html.replace(/<[^>]+>/g, '')).trim();
}

function deriveHeadlineFromText(plainText: string, fallbackTitle: string): {
  neutralHeadline: string;
  originalHeadline?: string;
} {
  const firstLine =
    plainText
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => line.length > 0) ?? fallbackTitle;

  const cleaned = firstLine
    .replace(/^[🔥⚡🚨❗️❗🔴🟢📢🚀✨💼🎯]+\s*/u, '')
    .replace(/\s+/g, ' ')
    .trim();

  const clipped =
    cleaned.length > 105 ? `${cleaned.slice(0, 102).trimEnd()}…` : cleaned;

  const neutralHeadline = clipped || fallbackTitle;
  return { neutralHeadline, originalHeadline: undefined };
}

/**
 * Converts a Telegram message HTML fragment (`.tgme_widget_message_text`)
 * into plain text + UTF-16 Telegram entities (`text_link`, `bold`, `code`, `mention`, `hashtag`).
 */
export function extractTextAndEntitiesFromTelegramHtml(fragmentHtml: string): {
  text: string;
  entities: RawTelegramEntity[];
} {
  const normalizedHtml = fragmentHtml
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>\s*<p[^>]*>/gi, '\n\n');

  const entities: RawTelegramEntity[] = [];
  let text = '';

  const tokenRe = /<(\/?)([a-zA-Z0-9]+)([^>]*)>|([^<]+)/g;
  const openStack: Array<{ tag: string; offset: number; href?: string }> = [];

  let match: RegExpExecArray | null;
  while ((match = tokenRe.exec(normalizedHtml)) !== null) {
    const [, closingSlash, rawTag, attrs = '', rawText] = match;
    if (rawText !== undefined) {
      text += decodeHtmlEntities(rawText);
      continue;
    }

    const tag = rawTag.toLowerCase();
    if (!closingSlash) {
      if (tag === 'a') {
        const hrefMatch = attrs.match(/href\s*=\s*["']([^"']+)["']/i);
        openStack.push({
          tag,
          offset: text.length,
          href: hrefMatch ? decodeHtmlEntities(hrefMatch[1]) : undefined,
        });
      } else if (tag === 'b' || tag === 'strong' || tag === 'i' || tag === 'em' || tag === 'code' || tag === 'pre') {
        openStack.push({ tag, offset: text.length });
      }
    } else {
      for (let i = openStack.length - 1; i >= 0; i -= 1) {
        const opener = openStack[i];
        if (opener.tag === tag) {
          openStack.splice(i, 1);
          const length = text.length - opener.offset;
          if (length > 0) {
            if (tag === 'a' && opener.href && /^https?:\/\//i.test(opener.href)) {
              entities.push({
                type: 'text_link',
                offset: opener.offset,
                length,
                url: opener.href,
              });
            } else if (tag === 'b' || tag === 'strong') {
              entities.push({ type: 'bold', offset: opener.offset, length });
            } else if (tag === 'i' || tag === 'em') {
              entities.push({ type: 'italic', offset: opener.offset, length });
            } else if (tag === 'code') {
              entities.push({ type: 'code', offset: opener.offset, length });
            } else if (tag === 'pre') {
              entities.push({ type: 'pre', offset: opener.offset, length });
            }
          }
          break;
        }
      }
    }
  }

  const trimmedStart = text.trimStart();
  const leadingTrim = text.length - trimmedStart.length;
  const finalText = trimmedStart.trimEnd();

  const shiftedEntities = entities
    .map((ent) => ({
      ...ent,
      offset: ent.offset - leadingTrim,
    }))
    .filter((ent) => ent.offset >= 0 && ent.offset + ent.length <= finalText.length);

  // Extract inline @mentions and #hashtags if not already covered by links
  const mentionOrTagRe = /(^|[\s(])([@#][a-zA-Zа-яА-ЯёЁ0-9_]{2,64})/g;
  let m: RegExpExecArray | null;
  while ((m = mentionOrTagRe.exec(finalText)) !== null) {
    const token = m[2];
    const start = m.index + m[1].length;
    shiftedEntities.push({
      type: token.startsWith('@') ? 'mention' : 'hashtag',
      offset: start,
      length: token.length,
    });
  }

  return { text: finalText, entities: shiftedEntities };
}

function extractBalancedDivInner(html: string, startIdx: number): string {
  let depth = 0;
  const tagRe = /<\/?div\b[^>]*>/gi;
  tagRe.lastIndex = startIdx;
  let contentStart = -1;
  let m: RegExpExecArray | null;
  while ((m = tagRe.exec(html)) !== null) {
    const isClose = m[0].startsWith('</');
    if (!isClose) {
      if (depth === 0) {
        contentStart = m.index + m[0].length;
      }
      depth += 1;
    } else {
      depth -= 1;
      if (depth === 0 && contentStart !== -1) {
        return html.slice(contentStart, m.index);
      }
    }
  }
  return '';
}

/**
 * Extracts the real public Telegram channel title from `https://t.me/s/<username>` HTML.
 */
export function extractTelegramChannelTitle(html: string): string | null {
  const headerMatch = html.match(
    /<div[^>]*class="[^"]*tgme_channel_info_header_title[^"]*"[^>]*>([\s\S]*?)<\/div>/i
  );
  const title = headerMatch ? stripTags(headerMatch[1]) : '';
  return title || null;
}

/**
 * Parses public Telegram channel web preview HTML (`https://t.me/s/<username>`)
 * into normalized `CleanPostItem[]` records without requiring MTProto credentials.
 */
export function parseTelegramPublicHtml(
  html: string,
  channelMeta: { channelId: string; handle: string; title?: string },
  maxPosts = 20,
): CleanPostItem[] {
  const cleanHandle = channelMeta.handle.startsWith('@')
    ? channelMeta.handle
    : `@${channelMeta.handle}`;
  const username = cleanHandle.slice(1);

  const resolvedTitle =
    extractTelegramChannelTitle(html) ||
    channelMeta.title ||
    cleanHandle;

  const msgMarkers = [...html.matchAll(/data-post="([^"]+)"/g)];
  const items: CleanPostItem[] = [];

  for (let i = 0; i < msgMarkers.length; i += 1) {
    const marker = msgMarkers[i];
    const dataPost = marker[1];
    const blockStart = marker.index ?? 0;
    const blockEnd =
      i + 1 < msgMarkers.length
        ? (msgMarkers[i + 1].index ?? html.length)
        : html.length;
    const block = html.slice(blockStart, blockEnd);

    const textDivMatch = block.match(
      /<div[^>]*class="[^"]*tgme_widget_message_text[^"]*"[^>]*>/i
    );
    const rawTextHtml = textDivMatch && textDivMatch.index !== undefined
      ? extractBalancedDivInner(block, textDivMatch.index)
      : '';

    const { text, entities } = extractTextAndEntitiesFromTelegramHtml(rawTextHtml);
    if (!text) {
      continue;
    }

    const timeMatch = block.match(/<time[^>]*datetime="([^"]+)"/i);
    const publishedAt = timeMatch ? timeMatch[1] : new Date().toISOString();

    const postParts = dataPost.split('/');
    const numericPostId = postParts[1] ?? `${i + 1}`;
    const postId = `tg-${username}-${numericPostId}`;

    // Extract optional photo preview
    const media: MediaAttachmentItem[] = [];
    const photoMatches = [
      ...block.matchAll(/tgme_widget_message_photo_wrap[^>]*style="[^"]*background-image:url\('([^']+)'\)/gi),
    ];
    photoMatches.forEach((pm, mIdx) => {
      const url = pm[1];
      if (url && /^https?:\/\//i.test(url)) {
        media.push({
          mediaId: `${postId}-m-${mIdx}`,
          fileId: `${postId}-f-${mIdx}`,
          mediaType: 'photo',
          mimeType: 'image/jpeg',
          width: 800,
          height: 500,
          thumbnailDataUrl: url,
          streamUrl: url,
          caption: `${resolvedTitle} • #${numericPostId}`,
        });
      }
    });

    const { neutralHeadline, originalHeadline } = deriveHeadlineFromText(
      text,
      `${resolvedTitle} — публикация #${numericPostId}`
    );

    items.push({
      id: postId,
      channelId: channelMeta.channelId,
      channelTitle: resolvedTitle,
      channelHandle: cleanHandle,
      channelAvatarInitials: resolvedTitle.slice(0, 2).toUpperCase(),
      postId: numericPostId,
      publishedAt,
      neutralHeadline,
      originalHeadline,
      sourcePlatform: 'TG',
      clusterId: `cluster-${username}-${numericPostId}`,
      text,
      entities,
      media,
      originScore: 0.9,
      duplicateCount: 0,
      commentCount: 0,
      rubric: 'Подписки',
    });
  }

  // Most recent posts first
  items.sort((a, b) => (a.publishedAt < b.publishedAt ? 1 : -1));
  return items.slice(0, maxPosts);
}

/**
 * Parses RSS 2.0 / Atom XML feeds into `CleanPostItem[]`.
 */
export function parseRssOrAtomXml(
  xml: string,
  channelMeta: { channelId: string; handle: string; title?: string },
  maxPosts = 20,
): CleanPostItem[] {
  const feedTitleMatch = xml.match(/<channel>[\s\S]*?<title[^>]*>([\s\S]*?)<\/title>/i)
    ?? xml.match(/<feed[\s\S]*?<title[^>]*>([\s\S]*?)<\/title>/i);
  const resolvedTitle =
    (feedTitleMatch ? stripTags(feedTitleMatch[1].replace(/<!\[CDATA\[|\]\]>/g, '')) : '') ||
    channelMeta.title ||
    channelMeta.handle;

  const entryMatches = [
    ...xml.matchAll(/<item\b[^>]*>([\s\S]*?)<\/item>/gi),
    ...xml.matchAll(/<entry\b[^>]*>([\s\S]*?)<\/entry>/gi),
  ];

  const items: CleanPostItem[] = [];
  entryMatches.forEach((m, idx) => {
    const body = m[1];
    const cleanCdata = (val: string) => val.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1');

    const titleRaw = body.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] ?? '';
    const descRaw =
      body.match(/<description[^>]*>([\s\S]*?)<\/description>/i)?.[1] ??
      body.match(/<summary[^>]*>([\s\S]*?)<\/summary>/i)?.[1] ??
      body.match(/<content[^>]*>([\s\S]*?)<\/content>/i)?.[1] ??
      '';
    const linkRaw =
      body.match(/<link[^>]*href="([^"]+)"/i)?.[1] ??
      body.match(/<link[^>]*>([\s\S]*?)<\/link>/i)?.[1] ??
      '';
    const pubDateRaw =
      body.match(/<pubDate[^>]*>([\s\S]*?)<\/pubDate>/i)?.[1] ??
      body.match(/<updated[^>]*>([\s\S]*?)<\/updated>/i)?.[1] ??
      body.match(/<published[^>]*>([\s\S]*?)<\/published>/i)?.[1] ??
      '';

    const title = stripTags(cleanCdata(titleRaw));
    const summary = stripTags(cleanCdata(descRaw));
    const link = stripTags(cleanCdata(linkRaw));

    const combinedText = [title, summary, link ? `Источник: ${link}` : '']
      .filter(Boolean)
      .join('\n\n');

    if (!combinedText) return;

    const parsedTime = pubDateRaw ? new Date(stripTags(pubDateRaw)) : new Date();
    const publishedAt = Number.isNaN(parsedTime.getTime())
      ? new Date().toISOString()
      : parsedTime.toISOString();

    const entities: RawTelegramEntity[] = [];
    if (link && /^https?:\/\//i.test(link)) {
      const linkPos = combinedText.lastIndexOf(link);
      if (linkPos >= 0) {
        entities.push({
          type: 'text_link',
          offset: linkPos,
          length: link.length,
          url: link,
        });
      }
    }

    const postId = `rss-${channelMeta.channelId}-${idx + 1}`;
    items.push({
      id: postId,
      channelId: channelMeta.channelId,
      channelTitle: resolvedTitle,
      channelHandle: channelMeta.handle,
      channelAvatarInitials: resolvedTitle.slice(0, 2).toUpperCase(),
      postId: `${idx + 1}`,
      publishedAt,
      neutralHeadline: title || `${resolvedTitle} #${idx + 1}`,
      sourcePlatform: 'RSS' as SourcePlatform,
      clusterId: `cluster-${postId}`,
      text: combinedText,
      entities,
      media: [],
      originScore: 0.9,
      duplicateCount: 0,
      commentCount: 0,
      rubric: 'RSS-ленты',
    });
  });

  items.sort((a, b) => (a.publishedAt < b.publishedAt ? 1 : -1));
  return items.slice(0, maxPosts);
}

/**
 * Groups live `CleanPostItem[]` by channel into concise `DigestCard[]` items
 * so newly added channels immediately appear in both "ИИ-Сводка" and "Очищенная лента".
 */
export function buildDigestCardsFromCleanPosts(posts: CleanPostItem[]): DigestCard[] {
  const byChannel = new Map<string, CleanPostItem[]>();
  for (const post of posts) {
    const list = byChannel.get(post.channelId) ?? [];
    list.push(post);
    byChannel.set(post.channelId, list);
  }

  const cards: DigestCard[] = [];
  for (const [channelId, chPosts] of byChannel.entries()) {
    const sorted = [...chPosts].sort((a, b) => (a.publishedAt < b.publishedAt ? 1 : -1));
    const top = sorted.slice(0, 3);
    const primary = top[0];
    if (!primary) continue;

    const bullets = top.map((p, idx) => {
      const snippet = p.text.replace(/\s+/g, ' ').trim();
      const bulletText = snippet.length > 180 ? `${snippet.slice(0, 177).trimEnd()}…` : snippet;
      const quoteEnd = Math.min(p.text.length, 90);
      return {
        text: bulletText,
        citations: [
          {
            claimId: `live-${channelId}-${idx}`,
            channelId: p.channelId,
            channelTitle: p.channelTitle,
            channelHandle: p.channelHandle,
            postId: p.id,
            quoteText: p.text.slice(0, quoteEnd),
            publishedAt: p.publishedAt,
            originScore: p.originScore ?? 0.9,
            startUtf16: 0,
            endUtf16: quoteEnd,
          },
        ],
      };
    });

    cards.push({
      clusterId: primary.clusterId ?? `cluster-${channelId}`,
      rubric: primary.rubric ?? 'Подписки',
      neutralTitle: `${primary.channelTitle}: ${primary.neutralHeadline}`,
      summaryBullets: bullets,
      primarySource: {
        channelId: primary.channelId,
        channelTitle: primary.channelTitle,
        channelHandle: primary.channelHandle,
        originScore: primary.originScore ?? 0.9,
        postId: String(primary.postId),
        publishedAt: primary.publishedAt,
      },
      otherChannelsCount: 0,
      otherChannels: [],
      importanceScore: 0.85,
      estimatedReadSec: Math.max(30, top.length * 25),
      groupingMode: 'channel_latest',
    });
  }

  return cards;
}
