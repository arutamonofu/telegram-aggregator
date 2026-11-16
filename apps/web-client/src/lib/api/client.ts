import {
  buildDemoCommentSummary,
  INDUSTRY_PRESETS,
  SEED_CLEAN_POSTS,
  SEED_CLEAN_RESPONSE,
  SEED_COMMENTS_BY_POST,
  SEED_DIGEST_RESPONSE,
} from '@/lib/api/seed_data';
import { buildDigestCardsFromCleanPosts } from '@/lib/live/tme_public_parser';
import type {
  CleanOriginalsResponse,
  CleanPostItem,
  CommentSummaryResponse,
  DigestFeedResponse,
  MediaAttachmentItem,
  SubscribedChannel,
} from '@/lib/types/feed';

export interface ParsedImportEntry {
  rawInput: string;
  valid: boolean;
  kind: 'channel' | 'addlist' | 'invalid';
  normalizedHandle?: string;
  title?: string;
  resolvedChannels: SubscribedChannel[];
  errorReason?: string;
}

const CHANNEL_USERNAME_RE = /^[a-zA-Z][a-zA-Z0-9_]{3,30}$/;
const ADDLIST_SLUG_RE = /^[a-zA-Z0-9_-]{4,64}$/;

/**
 * Parses Telegram channel references (`t.me/name`, `@name`, `t.me/addlist/slug`)
 * safely without requesting user phone numbers or MTProto credentials.
 */
export function parseTelegramChannelImportLines(rawInput: string): ParsedImportEntry[] {
  const lines = rawInput
    .split(/[\n,;]+/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  return lines.map((line) => {
    // @username form
    if (line.startsWith('@')) {
      const username = line.slice(1);
      if (!CHANNEL_USERNAME_RE.test(username)) {
        return {
          rawInput: line,
          valid: false,
          kind: 'invalid',
          resolvedChannels: [],
          errorReason: 'Некорректное @имя канала (требуется 4–31 латинских символов/цифр/_)',
        };
      }
      return {
        rawInput: line,
        valid: true,
        kind: 'channel',
        normalizedHandle: `@${username}`,
        resolvedChannels: [
          {
            channelId: `ch_${username.toLowerCase()}`,
            handle: `@${username}`,
            title: '',
            sourceType: 'imported_link',
          },
        ],
      };
    }

    // t.me/... or telegram.me/... URLs (with or without scheme)
    const normalizedCandidate = /^https?:\/\//i.test(line) ? line : `https://${line}`;
    try {
      const url = new URL(normalizedCandidate);
      const host = url.hostname.toLowerCase();
      if (host !== 't.me' && host !== 'telegram.me' && host !== 'www.t.me') {
        if (/^https?:\/\//i.test(line)) {
          return {
            rawInput: line,
            valid: true,
            kind: 'channel',
            normalizedHandle: line,
            title: url.hostname,
            resolvedChannels: [
              {
                channelId: `rss_${url.hostname.replace(/[^a-z0-9]/gi, '_')}_${line.length}`,
                handle: line,
                title: url.hostname,
                sourceType: 'imported_link',
                platform: 'RSS',
              },
            ],
          };
        }
        return {
          rawInput: line,
          valid: false,
          kind: 'invalid',
          resolvedChannels: [],
          errorReason: 'Поддерживаются ссылки t.me/..., @username или RSS https://...',
        };
      }

      const segments = url.pathname.split('/').filter(Boolean);
      if (segments.length >= 2 && segments[0].toLowerCase() === 'addlist' && ADDLIST_SLUG_RE.test(segments[1])) {
        const slug = segments[1];
        const expandedFromPreset = INDUSTRY_PRESETS.flatMap((p) => p.channels).slice(0, 3);
        return {
          rawInput: line,
          valid: true,
          kind: 'addlist',
          normalizedHandle: `t.me/addlist/${slug}`,
          title: `Папка ${slug}`,
          resolvedChannels: expandedFromPreset.map((ch) => ({
            channelId: ch.channelId,
            handle: ch.handle,
            title: `${ch.title} (addlist/${slug})`,
            sourceType: 'imported_addlist',
            topic: ch.topic,
          })),
        };
      }

      // t.me/s/username or t.me/username or t.me/username/123
      const candidateUser =
        segments[0]?.toLowerCase() === 's' ? segments[1] : segments[0];
      if (!candidateUser || !CHANNEL_USERNAME_RE.test(candidateUser)) {
        return {
          rawInput: line,
          valid: false,
          kind: 'invalid',
          resolvedChannels: [],
          errorReason: 'Приватные invite-ссылки (+...) требуют локального Desktop Agent',
        };
      }

      return {
        rawInput: line,
        valid: true,
        kind: 'channel',
        normalizedHandle: `@${candidateUser}`,
        resolvedChannels: [
          {
            channelId: `ch_${candidateUser.toLowerCase()}`,
            handle: `@${candidateUser}`,
            title: '',
            sourceType: 'imported_link',
          },
        ],
      };
    } catch {
      return {
        rawInput: line,
        valid: false,
        kind: 'invalid',
        resolvedChannels: [],
        errorReason: 'Не удалось распознать ссылку или @username',
      };
    }
  });
}

export interface ChannelRegisterResult {
  registered: boolean;
  channelId?: string;
  username?: string;
  title?: string;
  error?: string;
}

const MTPROTO_REGISTER_COOLDOWN_MS = 60_000;
let mtprotoRegisterUnavailableUntil = 0;

export async function registerTelegramChannel(
  username: string,
  tier = 'standard',
): Promise<ChannelRegisterResult> {
  const clean = username.trim().replace(/^@/, '');
  if (!clean) {
    return { registered: false, error: 'Пустое имя канала' };
  }
  if (/^https?:\/\//i.test(clean)) {
    return { registered: false };
  }
  if (Date.now() < mtprotoRegisterUnavailableUntil) {
    return { registered: false, error: 'MTProto backend unavailable (cooldown)' };
  }
  try {
    const resp = await fetch('/api/v1/channels/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: clean, tier }),
      signal: AbortSignal.timeout(3000),
    });
    if (!resp.ok) {
      if (resp.status === 502 || resp.status === 503 || resp.status === 504) {
        mtprotoRegisterUnavailableUntil = Date.now() + MTPROTO_REGISTER_COOLDOWN_MS;
      }
      const errBody = (await resp.json().catch(() => ({}))) as { detail?: string };
      return {
        registered: false,
        error: errBody.detail ?? `HTTP ${resp.status}`,
      };
    }
    mtprotoRegisterUnavailableUntil = 0;
    const data = (await resp.json()) as {
      channel_id?: number | string;
      username?: string;
      title?: string;
    };
    return {
      registered: true,
      channelId: data.channel_id !== undefined ? String(data.channel_id) : undefined,
      username: data.username,
      title: data.title,
    };
  } catch (err) {
    mtprotoRegisterUnavailableUntil = Date.now() + MTPROTO_REGISTER_COOLDOWN_MS;
    return {
      registered: false,
      error: err instanceof Error ? err.message : 'Network error',
    };
  }
}

function buildFallbackDigestFeed(
  userId: string,
  channelIds: string[],
  overrides?: Partial<Pick<DigestFeedResponse, 'finops_zone' | 'reduce_model'>>,
): DigestFeedResponse {
  const channelSet = new Set(channelIds);
  const cards =
    channelSet.size === 0
      ? []
      : SEED_DIGEST_RESPONSE.cards.filter((card) => channelSet.has(card.primarySource.channelId));
  return {
    ...SEED_DIGEST_RESPONSE,
    user_id: userId,
    finops_zone: overrides?.finops_zone ?? SEED_DIGEST_RESPONSE.finops_zone,
    reduce_model: overrides?.reduce_model ?? SEED_DIGEST_RESPONSE.reduce_model,
    total_candidate_clusters: cards.length,
    selected_cards_count: cards.length,
    rubrics: Array.from(new Set(cards.map((c) => c.rubric))),
    cards,
  };
}

function buildFallbackCleanOriginalsFeed(
  userId: string,
  channelIds: string[],
): CleanOriginalsResponse {
  const channelSet = new Set(channelIds);
  const items =
    channelSet.size === 0
      ? []
      : SEED_CLEAN_RESPONSE.items.filter((item) => channelSet.has(item.channelId));
  return {
    ...SEED_CLEAN_RESPONSE,
    user_id: userId,
    total: items.length,
    items,
  };
}

export async function fetchDigestFeed(
  userId: string,
  channelIds: string[],
): Promise<DigestFeedResponse> {
  if (channelIds.length === 0) {
    return buildFallbackDigestFeed(userId, channelIds);
  }
  try {
    const params = new URLSearchParams({
      user_id: userId,
      mode: 'digest',
      channel_ids: channelIds.join(','),
    });
    const resp = await fetch(`/api/v1/feed?${params.toString()}`, {
      cache: 'no-store',
    });
    if (!resp.ok) {
      return buildFallbackDigestFeed(userId, channelIds);
    }
    const data = (await resp.json()) as Partial<DigestFeedResponse>;
    if (!data || !Array.isArray(data.cards) || data.cards.length === 0) {
      return buildFallbackDigestFeed(userId, channelIds, {
        finops_zone: data?.finops_zone,
        reduce_model: data?.reduce_model,
      });
    }
    return {
      ...SEED_DIGEST_RESPONSE,
      ...data,
      mode: 'digest',
      user_id: userId,
      fromDemoFallback: false,
    };
  } catch {
    return buildFallbackDigestFeed(userId, channelIds);
  }
}

export async function fetchCleanOriginalsFeed(
  userId: string,
  channelIds: string[],
): Promise<CleanOriginalsResponse> {
  if (channelIds.length === 0) {
    return buildFallbackCleanOriginalsFeed(userId, channelIds);
  }
  try {
    const params = new URLSearchParams({
      user_id: userId,
      mode: 'clean_originals',
      channel_ids: channelIds.join(','),
    });
    const resp = await fetch(`/api/v1/feed?${params.toString()}`, {
      cache: 'no-store',
    });
    if (!resp.ok) {
      return buildFallbackCleanOriginalsFeed(userId, channelIds);
    }
    const data = (await resp.json()) as Partial<CleanOriginalsResponse>;
    if (!data || !Array.isArray(data.items) || data.items.length === 0) {
      return buildFallbackCleanOriginalsFeed(userId, channelIds);
    }
    return {
      mode: 'clean_originals',
      user_id: data.user_id ?? userId,
      total: data.total ?? data.items.length,
      items: data.items,
      fromDemoFallback: false,
    };
  } catch {
    return buildFallbackCleanOriginalsFeed(userId, channelIds);
  }
}

export async function fetchCommentSummary(
  postId: string,
  forceRefresh = false,
): Promise<CommentSummaryResponse> {
  const comments = SEED_COMMENTS_BY_POST[postId] ?? SEED_COMMENTS_BY_POST['post-bgp-42'] ?? [];
  try {
    const resp = await fetch('/api/v1/comments/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        post_id: postId,
        comments,
        max_selected: 5,
        force_refresh: forceRefresh,
      }),
    });
    if (!resp.ok) {
      return buildDemoCommentSummary(postId);
    }
    const data = (await resp.json()) as CommentSummaryResponse;
    return {
      ...data,
      fromDemoFallback: false,
    };
  } catch {
    return buildDemoCommentSummary(postId);
  }
}

/**
 * Probes media full-resolution availability via HTTP Range (`0-65535`).
 * Handles HTTP 410 Gone cleanly per SPEC 7.5.4 (UI preserves 35 KB thumbnail + warns user).
 */
export async function probeMediaRangeAvailability(
  attachment: MediaAttachmentItem,
): Promise<{ status: 'ok' | 'gone' | 'offline'; contentRange?: string }> {
  if (attachment.fileGone) {
    return { status: 'gone' };
  }
  try {
    const resp = await fetch(attachment.streamUrl, {
      method: 'GET',
      headers: { Range: 'bytes=0-65535' },
    });
    if (resp.status === 410) {
      return { status: 'gone' };
    }
    if (resp.status === 206 || resp.ok) {
      return {
        status: 'ok',
        contentRange: resp.headers.get('Content-Range') ?? undefined,
      };
    }
    return { status: 'offline' };
  } catch {
    return { status: 'offline' };
  }
}

export async function probeDesktopAgentHealth(
  endpoint = 'http://127.0.0.1:49152/health',
): Promise<{ connected: boolean; version?: string; channelsCount?: number }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 800);
  try {
    const resp = await fetch(endpoint, { signal: controller.signal });
    clearTimeout(timeout);
    if (!resp.ok) return { connected: false };
    const body = (await resp.json()) as { version?: string; channels_count?: number };
    return {
      connected: true,
      version: body.version ?? 'grammers-0.12',
      channelsCount: body.channels_count ?? 0,
    };
  } catch {
    clearTimeout(timeout);
    return { connected: false };
  }
}

export function getSeedPostById(postId: string) {
  return SEED_CLEAN_POSTS.find((p) => p.id === postId || String(p.postId) === postId);
}

export interface LiveFetchResponse {
  items: CleanPostItem[];
  errors: Record<string, string>;
  channelTitles: Record<string, string>;
  fetchedAt: string;
}

const CLIENT_LIVE_CACHE_TTL_MS = 90_000;
const clientLiveChannelCache = new Map<
  string,
  { items: CleanPostItem[]; title?: string; fetchedAtMs: number }
>();
const presetChannelIds = new Set(
  INDUSTRY_PRESETS.flatMap((p) => p.channels.map((c) => c.channelId)),
);

export function clearClientLiveChannelCache(): void {
  clientLiveChannelCache.clear();
  mtprotoRegisterUnavailableUntil = 0;
}

export async function fetchLivePublicChannelPosts(
  channels: SubscribedChannel[],
  options?: { forceRefresh?: boolean },
): Promise<LiveFetchResponse> {
  const publicChannels = channels.filter(
    (ch) => ch.sourceType !== 'private_agent' && ch.handle.trim().length > 0,
  );
  if (publicChannels.length === 0) {
    return {
      items: [],
      errors: {},
      channelTitles: {},
      fetchedAt: new Date().toISOString(),
    };
  }
  try {
    const resp = await fetch('/api/v1/live/channels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        channels: publicChannels.map((ch) => ({
          channelId: ch.channelId,
          handle: ch.handle,
          title: ch.title,
          platform: ch.platform,
        })),
        forceRefresh: Boolean(options?.forceRefresh),
      }),
      cache: 'no-store',
    });
    if (!resp.ok) {
      return {
        items: [],
        errors: {},
        channelTitles: {},
        fetchedAt: new Date().toISOString(),
      };
    }
    const data = (await resp.json()) as Partial<LiveFetchResponse>;
    return {
      items: Array.isArray(data.items) ? data.items : [],
      errors: data.errors ?? {},
      channelTitles: data.channelTitles ?? {},
      fetchedAt: data.fetchedAt ?? new Date().toISOString(),
    };
  } catch {
    return {
      items: [],
      errors: {},
      channelTitles: {},
      fetchedAt: new Date().toISOString(),
    };
  }
}

export async function syncAllChannelFeeds(params: {
  userId: string;
  channels: SubscribedChannel[];
  forceLive?: boolean;
}): Promise<{
  digest: DigestFeedResponse;
  cleanItems: CleanPostItem[];
  liveErrors: Record<string, string>;
  channelTitles: Record<string, string>;
  fetchedAt: string;
}> {
  const channelIds = params.channels.map((c) => c.channelId);
  const [digestRes, cleanRes] = await Promise.all([
    fetchDigestFeed(params.userId, channelIds),
    fetchCleanOriginalsFeed(params.userId, channelIds),
  ]);

  const seedChannelIds = new Set(SEED_CLEAN_POSTS.map((p) => p.channelId));
  const returnedChannelIds = new Set(cleanRes.items.map((p) => p.channelId));
  const now = Date.now();

  const isSyntheticChannel = (ch: SubscribedChannel) =>
    ch.sourceType === 'public_preset' ||
    presetChannelIds.has(ch.channelId) ||
    seedChannelIds.has(ch.channelId);

  const channelsNeedingLive = params.channels.filter((ch) => {
    if (ch.sourceType === 'private_agent' || isSyntheticChannel(ch)) {
      return false;
    }
    if (!ch.handle.trim()) {
      return false;
    }
    if (params.forceLive) {
      return true;
    }
    const cached = clientLiveChannelCache.get(ch.channelId);
    const hasFreshClientCache = Boolean(
      cached &&
        now - cached.fetchedAtMs < CLIENT_LIVE_CACHE_TTL_MS &&
        cached.items.length > 0,
    );
    const candidateTitle = (cached?.title ?? ch.title ?? '').trim();
    const normTitle = candidateTitle.replace(/^@/, '').toLowerCase();
    const normHandle = ch.handle.trim().replace(/^@/, '').toLowerCase();
    const hasResolvedTitle = Boolean(candidateTitle && normTitle !== normHandle);

    if (returnedChannelIds.has(ch.channelId) && hasResolvedTitle) {
      return false;
    }
    if (hasFreshClientCache && hasResolvedTitle) {
      return false;
    }
    return true;
  });

  const liveRes = await fetchLivePublicChannelPosts(channelsNeedingLive, {
    forceRefresh: Boolean(params.forceLive),
  });

  const liveItemsByChannel = new Map<string, CleanPostItem[]>();
  for (const item of liveRes.items) {
    const list = liveItemsByChannel.get(item.channelId) ?? [];
    list.push(item);
    liveItemsByChannel.set(item.channelId, list);
  }
  const cacheUpdatedAt = Date.now();
  for (const [channelId, items] of liveItemsByChannel.entries()) {
    clientLiveChannelCache.set(channelId, {
      items,
      title: liveRes.channelTitles[channelId] || items[0]?.channelTitle,
      fetchedAtMs: cacheUpdatedAt,
    });
  }

  const cachedLiveItems: CleanPostItem[] = [];
  const cachedTitles: Record<string, string> = {};
  for (const ch of params.channels) {
    if (ch.sourceType === 'private_agent' || isSyntheticChannel(ch)) {
      continue;
    }
    if (liveItemsByChannel.has(ch.channelId)) {
      continue;
    }
    const cached = clientLiveChannelCache.get(ch.channelId);
    if (cached && cached.items.length > 0) {
      cachedLiveItems.push(...cached.items);
      if (cached.title) {
        cachedTitles[ch.channelId] = cached.title;
      }
    }
  }

  // Merge clean posts (live posts override cached/fallback when same id, deduped)
  const mergedMap = new Map<string, CleanPostItem>();
  for (const item of cleanRes.items) {
    mergedMap.set(item.id, item);
  }
  for (const item of cachedLiveItems) {
    mergedMap.set(item.id, item);
  }
  for (const item of liveRes.items) {
    mergedMap.set(item.id, item);
  }
  const cleanItems = Array.from(mergedMap.values()).sort((a, b) =>
    a.publishedAt < b.publishedAt ? 1 : -1,
  );

  const channelTitles: Record<string, string> = {
    ...cachedTitles,
    ...liveRes.channelTitles,
  };
  for (const item of cleanItems) {
    if (!channelTitles[item.channelId] && item.channelTitle) {
      channelTitles[item.channelId] = item.channelTitle;
    }
  }

  // Merge digest cards with synthesized cards for live + cached-live posts
  const liveCards = buildDigestCardsFromCleanPosts([
    ...cachedLiveItems,
    ...liveRes.items,
  ]);
  const cardMap = new Map(digestRes.cards.map((c) => [c.clusterId, c]));
  for (const card of liveCards) {
    if (!cardMap.has(card.clusterId)) {
      cardMap.set(card.clusterId, card);
    }
  }
  const mergedCards = Array.from(cardMap.values());

  return {
    digest: {
      ...digestRes,
      cards: mergedCards,
      selected_cards_count: mergedCards.length,
      total_candidate_clusters: mergedCards.length,
      rubrics: Array.from(new Set(mergedCards.map((c) => c.rubric))),
    },
    cleanItems,
    liveErrors: liveRes.errors,
    channelTitles,
    fetchedAt: liveRes.fetchedAt,
  };
}
