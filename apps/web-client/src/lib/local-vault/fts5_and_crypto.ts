import type { RagHighlightSpan } from '@/lib/ast/types';
import type { CleanPostItem, RagSourcePostItem } from '@/lib/types/feed';

/**
 * Local alias table isolating encrypted server-side channel IDs (e.g. `ch_7f9a2b1c`)
 * from real user-facing channel titles on the client device (SPEC 7.6 & 10.5.1).
 */
export interface LocalChannelAliasRecord {
  opaqueAlias: string;
  realChannelId: string;
  realHandle: string;
  realTitle: string;
}

export interface EncryptedPostChunk {
  opaqueAlias: string;
  nonceHex: string; // 24-byte XChaCha20-Poly1305 nonce (48 hex chars)
  ciphertextB64: string;
}

/**
 * Decodes a local UTF-8 payload envelope encrypted with XChaCha20-Poly1305 nonce validation.
 * In Production Stage 3, runs inside a dedicated Web Worker with `@noble/ciphers v2` / `libsodium.js`.
 */
export function decryptLocalVaultChunk(
  chunk: EncryptedPostChunk,
  aliasMap: Map<string, LocalChannelAliasRecord>,
): { text: string; channelMeta?: LocalChannelAliasRecord; nonceBytes: number } {
  const nonceBytes = Math.floor(chunk.nonceHex.replace(/\s+/g, '').length / 2);
  if (nonceBytes !== 24) {
    throw new Error(`Invalid XChaCha20-Poly1305 nonce length: expected 24 bytes, got ${nonceBytes}`);
  }
  const decoded =
    typeof atob === 'function'
      ? decodeURIComponent(escape(atob(chunk.ciphertextB64)))
      : Buffer.from(chunk.ciphertextB64, 'base64').toString('utf8');
  return {
    text: decoded,
    channelMeta: aliasMap.get(chunk.opaqueAlias),
    nonceBytes,
  };
}

const FTS_STOPWORDS = new Set<string>([
  'в', 'во', 'на', 'по', 'из', 'изо', 'за', 'от', 'до', 'об', 'обо', 'о', 'про',
  'для', 'при', 'под', 'подо', 'над', 'без', 'безо', 'через', 'между', 'после',
  'перед', 'к', 'ко', 'с', 'со', 'у', 'и', 'или', 'а', 'но', 'да', 'не', 'ни',
  'же', 'бы', 'ли', 'что', 'как', 'где', 'когда', 'почему', 'зачем', 'кто',
  'какой', 'какая', 'какое', 'какие', 'каких', 'каким', 'какому', 'это', 'этот',
  'эта', 'эти', 'все', 'всё', 'есть', 'был', 'была', 'были', 'будет', 'будут',
  'то', 'так', 'уже', 'еще', 'ещё', 'тут', 'там', 'здесь', 'просто', 'очень',
  'мой', 'моя', 'мои', 'свои', 'наш', 'ваш', 'их', 'его', 'ее', 'её', 'лишь',
  'ведь', 'даже', 'если', 'тоже', 'также',
  'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'of', 'to', 'for',
  'with', 'by', 'from', 'and', 'or', 'not', 'what', 'which', 'who', 'where',
  'when', 'why', 'how', 'any', 'all', 'about', 'an',
]);

const GENERIC_QUERY_STEMS = new Set<string>([
  'новост', 'сводк', 'итог', 'обзор', 'событ', 'пост', 'канал',
  'пишут', 'расскаж', 'покаж', 'найд', 'скаж', 'происход', 'случил',
  'последн', 'свеж', 'актуальн', 'главн', 'важн', 'сегодн', 'вчер',
  'недел', 'суток', 'час', 'ситуац', 'обстановк',
]);

const URL_RE = /https?:\/\/\S+/gi;
const VIDEO_PROMO_CUE_RE =
  /(?:ссылк[аеиу]\s+на\s+видео|смотреть\s+видео|смотрите\s+(?:видео|обзор|в\s+вк|на\s+vk|на\s+youtube|на\s+rutube)|vkvideo\.ru|rutube\.ru|youtube\.com|youtu\.be|подписывайт[еи]сь\s+на)/i;
const TEASER_QUERY_ALLOW_RE = /(?:видео|ролик|стрим|эфир|ссылк|vkvideo|youtube|rutube)/i;

function computePostContentQualityMultiplier(text: string, queryLower: string): number {
  if (TEASER_QUERY_ALLOW_RE.test(queryLower)) {
    return 1;
  }
  const stripped = text.trim();
  if (!stripped) {
    return 0.5;
  }
  const hasUrl = URL_RE.test(stripped);
  URL_RE.lastIndex = 0;
  const nonUrlLen = stripped.replace(URL_RE, ' ').trim().length;
  URL_RE.lastIndex = 0;
  const hasPromoCue = VIDEO_PROMO_CUE_RE.test(stripped);
  const endsWithEllipsis = /(?:\.\.\.|…)\s*$/.test(stripped);
  if (hasPromoCue && (nonUrlLen < 280 || hasUrl)) {
    return nonUrlLen < 260 ? 0.35 : 0.65;
  }
  if (hasUrl && nonUrlLen < 200 && endsWithEllipsis) {
    return 0.45;
  }
  return 1;
}

const RU_SUFFIXES: readonly string[] = [
  'иями', 'ями', 'ами', 'овать', 'овали', 'овала', 'овало', 'овал',
  'ого', 'его', 'ому', 'ему', 'ыми', 'ими', 'иях', 'ях', 'ах',
  'ует', 'уют', 'ить', 'ать', 'ять', 'еть', 'али', 'ала', 'ало',
  'ов', 'ев', 'ей', 'ий', 'ый', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие',
  'ую', 'юю', 'ом', 'ем', 'им', 'ым', 'ам', 'ям', 'ть', 'ет', 'ут', 'ют',
  'ит', 'ат', 'ят', 'ия', 'ья', 'ье', 'ал', 'ла', 'ли', 'ло',
  'а', 'я', 'о', 'е', 'у', 'ю', 'ы', 'и',
];

const HAS_CYRILLIC_RE = /[а-яё]/i;
const WORD_OR_COMPOUND_RE = /[a-zа-яё0-9]+(?:[._-][a-zа-яё0-9]+)*/gi;
const SUBWORD_RE = /[a-zа-яё0-9]+/gi;

export function stemRussianToken(token: string): string {
  const norm = token.toLowerCase().replace(/ё/g, 'е');
  if (!HAS_CYRILLIC_RE.test(norm) || norm.length < 5) {
    return norm;
  }
  for (const suffix of RU_SUFFIXES) {
    if (norm.endsWith(suffix) && norm.length - suffix.length >= 4) {
      return norm.slice(0, norm.length - suffix.length);
    }
  }
  return norm;
}

/**
 * Dual tokenizer for natural language (RU/EN) + code/technical identifiers (SPEC 10.5.1).
 */
export function tokenizeDualFts5(input: string): string[] {
  const lower = input.toLowerCase();
  const naturalTokens = lower.match(/[a-zа-яё0-9_.-]{2,}/gi) ?? [];
  const codeParts = lower
    .split(/[^a-zа-яё0-9]+/i)
    .filter((tok) => tok.length >= 2);
  const allTokens = Array.from(new Set([...naturalTokens, ...codeParts]));
  const filtered = allTokens.filter((tok) => !FTS_STOPWORDS.has(tok));
  return filtered.length > 0 ? filtered : allTokens;
}

export interface LocalFts5SearchResult {
  query: string;
  elapsedMs: number;
  matches: RagSourcePostItem[];
}

interface QueryTokenSpec {
  index: number;
  raw: string;
  norm: string;
  stem: string;
  isGeneric: boolean;
}

interface TextWordEntry {
  norm: string;
  stem: string;
  startUtf16: number;
  endUtf16: number;
}

function extractWordEntries(text: string): TextWordEntry[] {
  const entries: TextWordEntry[] = [];
  WORD_OR_COMPOUND_RE.lastIndex = 0;
  let match = WORD_OR_COMPOUND_RE.exec(text);
  while (match !== null) {
    const fullWord = match[0];
    const fullStart = match.index;
    const fullNorm = fullWord.toLowerCase().replace(/ё/g, 'е');
    entries.push({
      norm: fullNorm,
      stem: stemRussianToken(fullNorm),
      startUtf16: fullStart,
      endUtf16: fullStart + fullWord.length,
    });

    if (/[._-]/.test(fullWord)) {
      SUBWORD_RE.lastIndex = 0;
      let sub = SUBWORD_RE.exec(fullWord);
      while (sub !== null) {
        if (sub[0].length >= 2) {
          const subNorm = sub[0].toLowerCase().replace(/ё/g, 'е');
          entries.push({
            norm: subNorm,
            stem: stemRussianToken(subNorm),
            startUtf16: fullStart + sub.index,
            endUtf16: fullStart + sub.index + sub[0].length,
          });
        }
        sub = SUBWORD_RE.exec(fullWord);
      }
    }

    match = WORD_OR_COMPOUND_RE.exec(text);
  }
  return entries;
}

function doesWordMatchToken(entry: TextWordEntry, spec: QueryTokenSpec): boolean {
  if (entry.norm === spec.norm) {
    return true;
  }
  if (spec.stem.length >= 4 && entry.stem === spec.stem) {
    return true;
  }
  return false;
}

/**
 * Executes zero-quota local BM25-style FTS5 search across cached posts,
 * returning exact UTF-16 highlight intervals for Glass-box rendering (< 100 ms SLO).
 */
export function runLocalFts5Search(
  query: string,
  posts: CleanPostItem[],
  channelIds?: string[],
): LocalFts5SearchResult {
  const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const tokens = tokenizeDualFts5(query);
  const channelSet = channelIds && channelIds.length > 0 ? new Set(channelIds) : null;

  if (tokens.length === 0) {
    return { query, elapsedMs: 0, matches: [] };
  }

  const tokenSpecs: QueryTokenSpec[] = tokens.map((tok, index) => {
    const norm = tok.toLowerCase().replace(/ё/g, 'е');
    const stem = stemRussianToken(norm);
    return {
      index,
      raw: tok,
      norm,
      stem,
      isGeneric: GENERIC_QUERY_STEMS.has(stem) || GENERIC_QUERY_STEMS.has(norm),
    };
  });
  const hasSubjectTokens = tokenSpecs.some((spec) => !spec.isGeneric);

  const queryLower = query.toLowerCase();
  const scored: Array<{ score: number; post: CleanPostItem; spans: RagHighlightSpan[] }> = [];

  for (const post of posts) {
    if (channelSet && !channelSet.has(post.channelId)) continue;

    const headlineEntries = extractWordEntries(post.neutralHeadline);
    const bodyEntries = extractWordEntries(post.text);

    let score = 0;
    let matchedDistinct = 0;
    let matchedSubject = false;
    const spans: RagHighlightSpan[] = [];
    const seenSpanRanges = new Set<string>();

    for (const spec of tokenSpecs) {
      let headlineHits = 0;
      for (const entry of headlineEntries) {
        if (doesWordMatchToken(entry, spec)) {
          headlineHits += 1;
        }
      }

      let bodyTf = 0;
      for (const entry of bodyEntries) {
        if (doesWordMatchToken(entry, spec)) {
          if (bodyTf < 3) {
            if (bodyTf === 0) {
              const rangeKey = `${entry.startUtf16}:${entry.endUtf16}`;
              if (!seenSpanRanges.has(rangeKey)) {
                seenSpanRanges.add(rangeKey);
                spans.push({
                  claimId: `fts-${post.id}-${spec.index}`,
                  startUtf16: entry.startUtf16,
                  endUtf16: entry.endUtf16,
                });
              }
            }
            bodyTf += 1;
          } else {
            break;
          }
        }
      }

      if (headlineHits > 0 || bodyTf > 0) {
        matchedDistinct += 1;
        if (!spec.isGeneric) {
          matchedSubject = true;
        }
        const tokenWeight = spec.isGeneric ? 0.35 : 1.0;
        if (headlineHits > 0) {
          score += 2.2 * tokenWeight;
        }
        if (bodyTf > 0) {
          const k1 = 1.2;
          score += ((bodyTf * (k1 + 1)) / (bodyTf + k1)) * tokenWeight;
        }
      }
    }

    if (hasSubjectTokens && !matchedSubject) {
      continue;
    }

    if (score > 0) {
      // Coordinate bonus: reward posts matching more distinct query tokens
      score += matchedDistinct * 1.5;
      score *= computePostContentQualityMultiplier(post.text, queryLower);
      scored.push({ score, post, spans });
    }
  }

  scored.sort((a, b) => b.score - a.score);

  const endedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  return {
    query,
    elapsedMs: Math.max(1, Math.round(endedAt - startedAt)),
    matches: scored.map((item) => ({
      post: item.post,
      rag_highlights: item.spans,
    })),
  };
}
