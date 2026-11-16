import { escapers } from '@telegraf/entity';
import {
  type RagHighlightInput,
  type RagHighlightSpan,
  type TelegramAstNode,
  type TextLeafNode,
  normalizeRagHighlightSpan,
} from './types';

function mergeCiteIds(existing: string[], extra: string[]): string[] {
  if (extra.length === 0) return existing;
  const set = new Set<string>(existing);
  for (const id of extra) {
    if (id) set.add(id);
  }
  return Array.from(set);
}

function normalizeSpans(rawSpans: RagHighlightInput[]): RagHighlightSpan[] {
  const seen = new Set<string>();
  const out: RagHighlightSpan[] = [];
  for (const item of rawSpans) {
    const span = normalizeRagHighlightSpan(item);
    if (!span.claimId || span.endUtf16 <= span.startUtf16) continue;
    const key = `${span.claimId}:${span.startUtf16}:${span.endUtf16}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(span);
  }
  out.sort((a, b) => a.startUtf16 - b.startUtf16 || a.endUtf16 - b.endUtf16);
  return out;
}

function splitTextLeaf(leaf: TextLeafNode, spans: RagHighlightSpan[]): TextLeafNode[] {
  const intersecting = spans.filter(
    (s) => s.endUtf16 > leaf.startUtf16 && s.startUtf16 < leaf.endUtf16,
  );
  if (intersecting.length === 0) {
    return [leaf];
  }

  const cutSet = new Set<number>([leaf.startUtf16, leaf.endUtf16]);
  for (const s of intersecting) {
    if (s.startUtf16 > leaf.startUtf16 && s.startUtf16 < leaf.endUtf16) {
      cutSet.add(s.startUtf16);
    }
    if (s.endUtf16 > leaf.startUtf16 && s.endUtf16 < leaf.endUtf16) {
      cutSet.add(s.endUtf16);
    }
  }

  const points = Array.from(cutSet).sort((a, b) => a - b);
  const result: TextLeafNode[] = [];

  for (let i = 0; i < points.length - 1; i += 1) {
    const segStart = points[i];
    const segEnd = points[i + 1];
    if (segEnd <= segStart) continue;

    const relStart = segStart - leaf.startUtf16;
    const relEnd = segEnd - leaf.startUtf16;
    const segText = leaf.text.slice(relStart, relEnd);
    const matchingCites = intersecting
      .filter((s) => s.startUtf16 <= segStart && s.endUtf16 >= segEnd)
      .map((s) => s.claimId);

    result.push({
      kind: 'text',
      text: segText,
      escapedText: escapers.HTML(segText),
      startUtf16: segStart,
      endUtf16: segEnd,
      citeIds: mergeCiteIds(leaf.citeIds, matchingCites),
    });
  }

  return result;
}

function splitSingleNode(node: TelegramAstNode, spans: RagHighlightSpan[]): TelegramAstNode[] {
  if (node.kind === 'text') {
    return splitTextLeaf(node, spans);
  }

  if (node.kind === 'code_block' || node.kind === 'formatted_date' || node.kind === 'widget') {
    const covering = spans
      .filter((s) => s.endUtf16 > node.startUtf16 && s.startUtf16 < node.endUtf16)
      .map((s) => s.claimId);
    if (covering.length === 0) {
      return [node];
    }
    return [
      {
        ...node,
        citeIds: mergeCiteIds(node.citeIds, covering),
      },
    ];
  }

  // Inline container nodes (`element`, `custom_emoji`, `diff`): recurse into children
  // without breaking parent hierarchy tags (`<strong>`, `<em>`, `<details>`, etc.).
  return [
    {
      ...node,
      children: splitAstLeavesWithNormalizedSpans(node.children, spans),
    },
  ];
}

function splitAstLeavesWithNormalizedSpans(
  nodes: TelegramAstNode[],
  spans: RagHighlightSpan[],
): TelegramAstNode[] {
  if (spans.length === 0) {
    return nodes;
  }
  const output: TelegramAstNode[] = [];
  for (const node of nodes) {
    output.push(...splitSingleNode(node, spans));
  }
  return output;
}

/**
 * Glass-box RAG Interval Split Pass (SPEC Section 7.5.5).
 *
 * Splits only `TextLeafNode` leaves overlapping verified RAG citation UTF-16 intervals
 * `[startUtf16, endUtf16)` while keeping parent formatting nodes intact.
 */
export function splitAstLeavesWithRagCitations(
  nodes: TelegramAstNode[],
  rawSpans: RagHighlightInput[],
): TelegramAstNode[] {
  const spans = normalizeSpans(rawSpans);
  return splitAstLeavesWithNormalizedSpans(nodes, spans);
}
