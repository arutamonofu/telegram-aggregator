import type { Root as HastRoot } from 'hast';

/**
 * Normalized entity types across MTProto Layer 225+, Bot API 9.5-10.4,
 * and platform ingest schema (`ingest_post_v1.schema.json`).
 */
export type CanonicalEntityType =
  | 'bold'
  | 'italic'
  | 'underline'
  | 'strikethrough'
  | 'code'
  | 'pre'
  | 'text_link'
  | 'text_mention'
  | 'url'
  | 'mention'
  | 'hashtag'
  | 'cashtag'
  | 'bot_command'
  | 'email'
  | 'phone_number'
  | 'blockquote'
  | 'expandable_blockquote'
  | 'spoiler'
  | 'custom_emoji'
  | 'formatted_date'
  | 'diff_insert'
  | 'diff_delete'
  | 'diff_replace'
  | 'rich_message'
  | 'widget'
  | 'ton_address'
  | 'bank_card_number';

/**
 * Raw Telegram entity accepted from MTProto, Bot API, or backend JSON payloads.
 * Offsets and lengths are strictly in UTF-16 code units.
 */
export interface RawTelegramEntity {
  type: string;
  offset: number;
  length: number;
  url?: string | null;
  language?: string | null;
  collapsed?: boolean | null;
  document_id?: string | null;
  custom_emoji_id?: string | null;
  unix_time?: number | null;
  date_flags?: number | null;
  date_time_format?: string | null;
  old_text?: string | null;
  user?: {
    id: number | string;
    first_name?: string;
    last_name?: string;
    username?: string;
  } | null;
  user_id?: number | string | null;
  widget_payload?: RichWidgetPayload | null;
}

export interface NormalizedTelegramEntity {
  type: CanonicalEntityType;
  rawType: string;
  offset: number;
  length: number;
  url?: string;
  language?: string;
  collapsed?: boolean;
  documentId?: string;
  unixTime?: number;
  dateFlags?: number;
  dateTimeFormat?: string;
  oldText?: string;
  userId?: string;
  widgetPayload?: RichWidgetPayload;
}

export interface RichWidgetPayload {
  kind: 'tool_card' | 'metric_badge' | 'action_row' | 'poll_summary' | 'unknown';
  title: string;
  subtitle?: string;
  items?: Array<{ label: string; value: string }>;
  actionUrl?: string;
}

export interface RichMessageContainer {
  containerType: 'rich_message' | 'widget';
  title?: string;
  widgets: RichWidgetPayload[];
}

export interface RagHighlightSpan {
  claimId: string;
  startUtf16: number;
  endUtf16: number;
}

/** Input shape variants returned by SSE or tests (`utf16_range` or `startUtf16`/`endUtf16`). */
export type RagHighlightInput =
  | RagHighlightSpan
  | {
      claim_id: string;
      utf16_range?: [number, number];
      start_utf16?: number;
      end_utf16?: number;
    };

export interface TextLeafNode {
  kind: 'text';
  text: string;
  escapedText: string;
  startUtf16: number;
  endUtf16: number;
  citeIds: string[];
}

export interface CodeBlockAstNode {
  kind: 'code_block';
  code: string;
  language: string;
  startUtf16: number;
  endUtf16: number;
  hast: HastRoot;
  highlighterEngine: 'shiki' | 'lowlight' | 'plain';
  citeIds: string[];
}

export interface FormattedDateAstNode {
  kind: 'formatted_date';
  startUtf16: number;
  endUtf16: number;
  fallbackText: string;
  unixTime?: number;
  dateFlags?: number;
  dateTimeFormat?: string;
  exceedsLimit: boolean;
  citeIds: string[];
}

export interface CustomEmojiAstNode {
  kind: 'custom_emoji';
  startUtf16: number;
  endUtf16: number;
  alt: string;
  documentId?: string;
  lottieUrl?: string;
  twemojiUrl: string;
  children: TelegramAstNode[];
  citeIds: string[];
}

export interface DiffAstNode {
  kind: 'diff';
  variant: 'diff_insert' | 'diff_delete' | 'diff_replace';
  startUtf16: number;
  endUtf16: number;
  oldText?: string;
  newText: string;
  children: TelegramAstNode[];
  citeIds: string[];
}

export interface WidgetAstNode {
  kind: 'widget';
  startUtf16: number;
  endUtf16: number;
  fallbackText: string;
  payload: RichWidgetPayload;
  citeIds: string[];
}

export type InlineContainerTag =
  | 'bold'
  | 'italic'
  | 'underline'
  | 'strikethrough'
  | 'code'
  | 'text_link'
  | 'text_mention'
  | 'url'
  | 'mention'
  | 'hashtag'
  | 'cashtag'
  | 'bot_command'
  | 'email'
  | 'phone_number'
  | 'blockquote'
  | 'spoiler'
  | 'ton_address'
  | 'bank_card_number';

export interface ElementAstNode {
  kind: 'element';
  tag: InlineContainerTag;
  startUtf16: number;
  endUtf16: number;
  href?: string;
  collapsed?: boolean;
  children: TelegramAstNode[];
}

export type TelegramAstNode =
  | TextLeafNode
  | ElementAstNode
  | CodeBlockAstNode
  | FormattedDateAstNode
  | CustomEmojiAstNode
  | DiffAstNode
  | WidgetAstNode;

export interface ParsedTelegramAst {
  /** UTF-16 trimmed source text used for AST offsets. */
  trimmedText: string;
  /** Number of trailing UTF-16 code units removed by `rtrim`. */
  trimmedTrailingCount: number;
  nodes: TelegramAstNode[];
  richContainers: RichMessageContainer[];
  warnings: string[];
}

const ENTITY_TYPE_ALIASES: Record<string, CanonicalEntityType> = {
  bold: 'bold',
  messageentitybold: 'bold',
  italic: 'italic',
  messageentityitalic: 'italic',
  underline: 'underline',
  messageentityunderline: 'underline',
  strikethrough: 'strikethrough',
  messageentitystrike: 'strikethrough',
  code: 'code',
  messageentitycode: 'code',
  pre: 'pre',
  messageentitypre: 'pre',
  text_link: 'text_link',
  messageentitytexturl: 'text_link',
  text_mention: 'text_mention',
  messageentitymentionname: 'text_mention',
  inputmessageentitymentionname: 'text_mention',
  url: 'url',
  messageentityurl: 'url',
  mention: 'mention',
  messageentitymention: 'mention',
  hashtag: 'hashtag',
  messageentityhashtag: 'hashtag',
  cashtag: 'cashtag',
  messageentitycashtag: 'cashtag',
  bankcard: 'bank_card_number',
  bank_card_number: 'bank_card_number',
  messageentitybankcard: 'bank_card_number',
  ton_address: 'ton_address',
  messageentitytonaddress: 'ton_address',
  bot_command: 'bot_command',
  messageentitybotcommand: 'bot_command',
  email: 'email',
  messageentityemail: 'email',
  phone_number: 'phone_number',
  messageentityphone: 'phone_number',
  blockquote: 'blockquote',
  messageentityblockquote: 'blockquote',
  expandable_blockquote: 'expandable_blockquote',
  spoiler: 'spoiler',
  messageentityspoiler: 'spoiler',
  custom_emoji: 'custom_emoji',
  messageentitycustomemoji: 'custom_emoji',
  formatted_date: 'formatted_date',
  date_time: 'formatted_date',
  messageentityformatteddate: 'formatted_date',
  diff_insert: 'diff_insert',
  messageentitydiffinsert: 'diff_insert',
  diff_delete: 'diff_delete',
  messageentitydiffdelete: 'diff_delete',
  diff_replace: 'diff_replace',
  messageentitydiffreplace: 'diff_replace',
  rich_message: 'rich_message',
  richmessage: 'rich_message',
  widget: 'widget',
};

export function normalizeEntityType(rawType: string): CanonicalEntityType | null {
  const key = rawType.trim().toLowerCase();
  return ENTITY_TYPE_ALIASES[key] ?? null;
}

export function normalizeRagHighlightSpan(input: RagHighlightInput): RagHighlightSpan {
  if ('claimId' in input) {
    return {
      claimId: input.claimId,
      startUtf16: input.startUtf16,
      endUtf16: input.endUtf16,
    };
  }
  const start = input.utf16_range?.[0] ?? input.start_utf16 ?? 0;
  const end = input.utf16_range?.[1] ?? input.end_utf16 ?? start;
  return {
    claimId: input.claim_id,
    startUtf16: start,
    endUtf16: end,
  };
}
