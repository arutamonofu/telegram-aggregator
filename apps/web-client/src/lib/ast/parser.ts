import { escapers, serialiseWith, serialisers } from '@telegraf/entity';
import type { Message } from '@telegraf/types';
import { isFormattedDateLengthValid } from './formatted_date';
import {
  highlightCodeWithLowlightSync,
  highlightCodeWithShikiAsync,
} from './shiki_highlighter';
import {
  type CanonicalEntityType,
  type CodeBlockAstNode,
  type InlineContainerTag,
  type NormalizedTelegramEntity,
  type ParsedTelegramAst,
  type RawTelegramEntity,
  type RichMessageContainer,
  type RichWidgetPayload,
  type TelegramAstNode,
  normalizeEntityType,
} from './types';

const TYPE_PRIORITY: Record<CanonicalEntityType, number> = {
  blockquote: 10,
  expandable_blockquote: 10,
  pre: 15,
  bold: 20,
  italic: 25,
  underline: 30,
  strikethrough: 35,
  spoiler: 40,
  text_link: 45,
  text_mention: 45,
  url: 48,
  mention: 50,
  hashtag: 52,
  cashtag: 54,
  bot_command: 56,
  email: 58,
  phone_number: 60,
  code: 65,
  diff_insert: 70,
  diff_delete: 70,
  diff_replace: 70,
  formatted_date: 80,
  custom_emoji: 90,
  ton_address: 92,
  bank_card_number: 92,
  widget: 95,
  rich_message: 95,
};

const baseTelegrafSerialiseHtml = serialiseWith(serialisers.HTML, escapers.HTML);

/**
 * Wrapper around `@telegraf/entity`'s `serialiseWith(serialisers.HTML, escapers.HTML)`
 * that safely normalizes entity sorting and guarantees escaping even when `entities` is empty
 * (working around `@telegraf/entity@0.7.0`'s unescaped fast-path on entity-free strings).
 */
export function serialiseBotApiMessageWithTelegraf(
  text: string,
  entities: Message.TextMessage['entities'] = [],
): string {
  const trimmed = rtrimUtf16(text).trimmedText;
  if (!entities || entities.length === 0) {
    return escapers.HTML(trimmed);
  }
  const sorted = [...entities]
    .filter((e) => e.offset >= 0 && e.length > 0 && e.offset < trimmed.length)
    .map((e) => ({
      ...e,
      length: Math.min(e.length, trimmed.length - e.offset),
    }))
    .sort((a, b) => a.offset - b.offset || b.length - a.length);

  if (sorted.length === 0) {
    return escapers.HTML(trimmed);
  }

  return baseTelegrafSerialiseHtml({
    text: trimmed,
    entities: sorted,
  } as Message.TextMessage);
}

/**
 * Strips trailing whitespace and newlines strictly preserving valid leading UTF-16 indices.
 */
export function rtrimUtf16(input: string): {
  trimmedText: string;
  trimmedTrailingCount: number;
} {
  let end = input.length;
  while (end > 0) {
    const code = input.charCodeAt(end - 1);
    if (
      code === 0x20 || // space
      code === 0x09 || // \t
      code === 0x0a || // \n
      code === 0x0d || // \r
      code === 0x0b || // \v
      code === 0x0c || // \f
      code === 0x00a0 || // nbsp
      code === 0x200b || // zwsp
      code === 0xfeff
    ) {
      end -= 1;
    } else {
      break;
    }
  }
  return {
    trimmedText: input.slice(0, end),
    trimmedTrailingCount: input.length - end,
  };
}

/**
 * Compute Twemoji lowercase dash-joined hex codepoints for an emoji string.
 */
export function computeTwemojiCodepoints(emojiText: string): string {
  const rawCodepoints: string[] = [];
  for (const symbol of emojiText) {
    const cp = symbol.codePointAt(0);
    if (cp !== undefined) {
      rawCodepoints.push(cp.toString(16));
    }
  }
  // Twemoji strips U+FE0F unless ZWJ (U+200D) is present
  const hasZwj = rawCodepoints.includes('200d');
  const filtered = hasZwj ? rawCodepoints : rawCodepoints.filter((cp) => cp !== 'fe0f');
  return (filtered.length > 0 ? filtered : rawCodepoints).join('-');
}

export function normalizeTelegramEntities(
  rawEntities: RawTelegramEntity[] | undefined,
  textLengthUtf16: number,
  warnings: string[],
): {
  entities: NormalizedTelegramEntity[];
  richContainers: RichMessageContainer[];
} {
  if (!rawEntities || rawEntities.length === 0) {
    return { entities: [], richContainers: [] };
  }

  const normalized: NormalizedTelegramEntity[] = [];
  const richContainers: RichMessageContainer[] = [];

  for (const raw of rawEntities) {
    const canonical = normalizeEntityType(raw.type);
    if (!canonical) {
      warnings.push(`Unknown entity type skipped: ${raw.type}`);
      continue;
    }

    if (canonical === 'rich_message') {
      if (raw.widget_payload) {
        richContainers.push({
          containerType: 'rich_message',
          title: raw.widget_payload.title,
          widgets: [raw.widget_payload],
        });
      }
      continue;
    }

    const offset = Number.isFinite(raw.offset) ? Math.trunc(raw.offset) : 0;
    const rawLength = Number.isFinite(raw.length) ? Math.trunc(raw.length) : 0;
    const start = Math.max(0, Math.min(textLengthUtf16, offset));
    const end = Math.max(start, Math.min(textLengthUtf16, offset + rawLength));
    const length = end - start;

    if (length <= 0) {
      if (canonical === 'widget' && raw.widget_payload) {
        richContainers.push({
          containerType: 'widget',
          title: raw.widget_payload.title,
          widgets: [raw.widget_payload],
        });
      }
      continue;
    }

    normalized.push({
      type: canonical,
      rawType: raw.type,
      offset: start,
      length,
      url: raw.url ?? undefined,
      language: raw.language ?? undefined,
      collapsed:
        canonical === 'expandable_blockquote'
          ? true
          : raw.collapsed !== undefined && raw.collapsed !== null
            ? Boolean(raw.collapsed)
            : undefined,
      documentId: raw.document_id ?? raw.custom_emoji_id ?? undefined,
      unixTime: raw.unix_time ?? undefined,
      dateFlags: raw.date_flags ?? undefined,
      dateTimeFormat: raw.date_time_format ?? undefined,
      oldText: raw.old_text ?? undefined,
      userId:
        raw.user_id !== undefined && raw.user_id !== null
          ? String(raw.user_id)
          : raw.user?.id !== undefined
            ? String(raw.user.id)
            : undefined,
      widgetPayload: raw.widget_payload ?? undefined,
    });
  }

  normalized.sort((a, b) => {
    if (a.offset !== b.offset) return a.offset - b.offset;
    if (a.length !== b.length) return b.length - a.length;
    return TYPE_PRIORITY[a.type] - TYPE_PRIORITY[b.type];
  });

  return { entities: normalized, richContainers };
}

function makeTextLeaf(sourceText: string, startUtf16: number, endUtf16: number): TelegramAstNode {
  const slice = sourceText.slice(startUtf16, endUtf16);
  return {
    kind: 'text',
    text: slice,
    escapedText: escapers.HTML(slice),
    startUtf16,
    endUtf16,
    citeIds: [],
  };
}

function mapInlineContainerTag(entityType: CanonicalEntityType): InlineContainerTag {
  switch (entityType) {
    case 'expandable_blockquote':
      return 'blockquote';
    case 'bold':
    case 'italic':
    case 'underline':
    case 'strikethrough':
    case 'code':
    case 'text_link':
    case 'text_mention':
    case 'url':
    case 'mention':
    case 'hashtag':
    case 'cashtag':
    case 'bot_command':
    case 'email':
    case 'phone_number':
    case 'blockquote':
    case 'spoiler':
    case 'ton_address':
    case 'bank_card_number':
      return entityType;
    default:
      return 'spoiler';
  }
}

function buildChildrenForRange(
  sourceText: string,
  rangeStart: number,
  rangeEnd: number,
  entities: NormalizedTelegramEntity[],
  startEntityIdx: number,
  warnings: string[],
): { nodes: TelegramAstNode[]; nextEntityIdx: number } {
  const nodes: TelegramAstNode[] = [];
  let cursor = rangeStart;
  let idx = startEntityIdx;

  while (idx < entities.length) {
    const ent = entities[idx];
    const entStart = ent.offset;
    const entEnd = ent.offset + ent.length;

    if (entStart >= rangeEnd) {
      break;
    }

    if (entEnd <= cursor) {
      idx += 1;
      continue;
    }

    const effectiveStart = Math.max(cursor, entStart);
    if (effectiveStart > cursor) {
      nodes.push(makeTextLeaf(sourceText, cursor, effectiveStart));
      cursor = effectiveStart;
    }

    const effectiveEnd = Math.min(rangeEnd, entEnd);
    const isTruncatedAtParentEnd = entEnd > rangeEnd;

    // Collect nested entities overlapping [effectiveStart, effectiveEnd)
    let childScanIdx = idx + 1;
    const childEntities: NormalizedTelegramEntity[] = [];
    while (childScanIdx < entities.length) {
      const candidate = entities[childScanIdx];
      const candStart = candidate.offset;
      const candEnd = candidate.offset + candidate.length;
      if (candStart >= effectiveEnd) {
        break;
      }
      if (candEnd > effectiveStart) {
        childEntities.push(candidate);
      }
      childScanIdx += 1;
    }

    const node = instantiateEntityAstNode(
      sourceText,
      ent,
      effectiveStart,
      effectiveEnd,
      childEntities,
      warnings,
    );
    nodes.push(node);
    cursor = effectiveEnd;

    if (!isTruncatedAtParentEnd) {
      // Advance past entities completely enclosed within [effectiveStart, effectiveEnd)
      let jump = idx + 1;
      while (jump < entities.length) {
        const inner = entities[jump];
        if (inner.offset + inner.length <= effectiveEnd) {
          jump += 1;
        } else {
          break;
        }
      }
      idx = jump;
    } else {
      // Parent ended before this entity finished: let parent resume this entity tail
      break;
    }
  }

  if (cursor < rangeEnd) {
    nodes.push(makeTextLeaf(sourceText, cursor, rangeEnd));
  }

  return { nodes, nextEntityIdx: idx };
}

function instantiateEntityAstNode(
  sourceText: string,
  ent: NormalizedTelegramEntity,
  startUtf16: number,
  endUtf16: number,
  candidateChildren: NormalizedTelegramEntity[],
  warnings: string[],
): TelegramAstNode {
  const slice = sourceText.slice(startUtf16, endUtf16);

  if (ent.type === 'pre') {
    if (candidateChildren.length > 0) {
      warnings.push(
        `Nested entities (${candidateChildren.length}) ignored inside messageEntityPre [${startUtf16}, ${endUtf16})`,
      );
    }
    const highlighted = highlightCodeWithLowlightSync(slice, ent.language);
    return {
      kind: 'code_block',
      code: slice,
      language: highlighted.resolvedLanguage,
      startUtf16,
      endUtf16,
      hast: highlighted.hast,
      highlighterEngine: highlighted.engine,
      citeIds: [],
    };
  }

  if (ent.type === 'formatted_date') {
    if (candidateChildren.length > 0) {
      warnings.push(
        `Nested entities (${candidateChildren.length}) ignored inside formatted_date [${startUtf16}, ${endUtf16})`,
      );
    }
    const lenValid = isFormattedDateLengthValid(
      endUtf16 - startUtf16,
      ent.dateFlags,
      ent.dateTimeFormat,
    );
    if (!lenValid) {
      warnings.push(
        `formatted_date UTF-16 length ${endUtf16 - startUtf16} exceeds spec limit at [${startUtf16}, ${endUtf16})`,
      );
    }
    return {
      kind: 'formatted_date',
      startUtf16,
      endUtf16,
      fallbackText: slice,
      unixTime: ent.unixTime,
      dateFlags: ent.dateFlags,
      dateTimeFormat: ent.dateTimeFormat,
      exceedsLimit: !lenValid,
      citeIds: [],
    };
  }

  const children = buildChildrenForRange(
    sourceText,
    startUtf16,
    endUtf16,
    candidateChildren,
    0,
    warnings,
  ).nodes;

  if (ent.type === 'custom_emoji') {
    const codepoints = computeTwemojiCodepoints(slice);
    return {
      kind: 'custom_emoji',
      startUtf16,
      endUtf16,
      alt: slice,
      documentId: ent.documentId,
      lottieUrl: ent.documentId
        ? `/api/v1/media/custom-emoji/${encodeURIComponent(ent.documentId)}.tgs`
        : undefined,
      twemojiUrl: `/api/v1/media/twemoji/${codepoints || '2753'}.svg`,
      children,
      citeIds: [],
    };
  }

  if (ent.type === 'diff_insert' || ent.type === 'diff_delete' || ent.type === 'diff_replace') {
    return {
      kind: 'diff',
      variant: ent.type,
      startUtf16,
      endUtf16,
      oldText: ent.oldText,
      newText: slice,
      children,
      citeIds: [],
    };
  }

  if (ent.type === 'widget') {
    const payload: RichWidgetPayload = ent.widgetPayload ?? {
      kind: 'tool_card',
      title: slice || 'Widget',
    };
    return {
      kind: 'widget',
      startUtf16,
      endUtf16,
      fallbackText: slice,
      payload,
      citeIds: [],
    };
  }

  let href: string | undefined;
  if (ent.type === 'text_link') {
    href = ent.url ?? slice;
  } else if (ent.type === 'url') {
    href = slice;
  } else if (ent.type === 'email') {
    href = `mailto:${slice}`;
  } else if (ent.type === 'phone_number') {
    href = `tel:${slice}`;
  } else if (ent.type === 'mention') {
    const username = slice.startsWith('@') ? slice.slice(1) : slice;
    href = `https://t.me/${username}`;
  } else if (ent.type === 'text_mention' && ent.userId) {
    href = `tg://user?id=${ent.userId}`;
  }

  return {
    kind: 'element',
    tag: mapInlineContainerTag(ent.type),
    startUtf16,
    endUtf16,
    href,
    collapsed: ent.type === 'blockquote' || ent.type === 'expandable_blockquote' ? Boolean(ent.collapsed) : undefined,
    children,
  };
}

/**
 * Synchronous deterministic AST parser for Telegram text + entities.
 * Applies `rtrim` in UTF-16 space, preserves exact `[startUtf16, endUtf16)` coordinates on every `TextLeafNode`,
 * and uses synchronous `lowlight` HAST generation for `messageEntityPre` code blocks.
 */
export function parseTelegramPostAstSync(
  rawText: string,
  rawEntities: RawTelegramEntity[] = [],
): ParsedTelegramAst {
  const { trimmedText, trimmedTrailingCount } = rtrimUtf16(rawText);
  const warnings: string[] = [];
  const { entities, richContainers } = normalizeTelegramEntities(
    rawEntities,
    trimmedText.length,
    warnings,
  );

  const { nodes } = buildChildrenForRange(
    trimmedText,
    0,
    trimmedText.length,
    entities,
    0,
    warnings,
  );

  return {
    trimmedText,
    trimmedTrailingCount,
    nodes,
    richContainers,
    warnings,
  };
}

async function upgradeCodeBlocksToShiki(
  nodes: TelegramAstNode[],
  theme: string,
): Promise<TelegramAstNode[]> {
  return Promise.all(
    nodes.map(async (node): Promise<TelegramAstNode> => {
      if (node.kind === 'code_block') {
        const res = await highlightCodeWithShikiAsync(node.code, node.language, theme);
        const updated: CodeBlockAstNode = {
          ...node,
          hast: res.hast,
          highlighterEngine: res.engine,
          language: res.resolvedLanguage,
        };
        return updated;
      }
      if (node.kind === 'element' || node.kind === 'custom_emoji' || node.kind === 'diff') {
        const upgradedChildren = await upgradeCodeBlocksToShiki(node.children, theme);
        return {
          ...node,
          children: upgradedChildren,
        };
      }
      return node;
    }),
  );
}

/**
 * Full async AST parser upgrading `messageEntityPre` blocks through Shiki (`codeToHast`)
 * while preserving the `lowlight` fallback if Shiki cannot highlight offline/unsupported languages.
 */
export async function parseTelegramPostAstAsync(
  rawText: string,
  rawEntities: RawTelegramEntity[] = [],
  shikiTheme: string = 'light-plus',
): Promise<ParsedTelegramAst> {
  const syncAst = parseTelegramPostAstSync(rawText, rawEntities);
  const nodes = await upgradeCodeBlocksToShiki(syncAst.nodes, shikiTheme);
  return {
    ...syncAst,
    nodes,
  };
}
