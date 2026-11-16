import type { RagHighlightInput, RagHighlightSpan } from '@/lib/ast/types';
import { normalizeRagHighlightSpan } from '@/lib/ast/types';

export interface DomUtf16NodeEntry {
  node: Text;
  startUtf16: number;
  endUtf16: number;
}

/**
 * Builds an ordered map of DOM Text nodes and their cumulative UTF-16 offsets
 * inside a post body container for native CSS Custom Highlight API registration.
 */
export function buildDomUtf16TextMap(rootEl: HTMLElement): DomUtf16NodeEntry[] {
  const entries: DomUtf16NodeEntry[] = [];
  if (typeof document === 'undefined' || !rootEl) return entries;

  const doc = rootEl.ownerDocument ?? document;
  const showTextMask = typeof NodeFilter !== 'undefined' ? NodeFilter.SHOW_TEXT : 4;
  const walker = doc.createTreeWalker(rootEl, showTextMask);
  let cursor = 0;
  let current = walker.nextNode();
  while (current) {
    const textNode = current as Text;
    const len = textNode.data.length;
    if (len > 0) {
      entries.push({
        node: textNode,
        startUtf16: cursor,
        endUtf16: cursor + len,
      });
      cursor += len;
    }
    current = walker.nextNode();
  }
  return entries;
}

export function resolveDomRangeForUtf16Span(
  entries: DomUtf16NodeEntry[],
  startUtf16: number,
  endUtf16: number,
  doc: Document,
): Range | null {
  if (endUtf16 <= startUtf16 || entries.length === 0) return null;

  let startNode: Text | null = null;
  let startOffset = 0;
  let endNode: Text | null = null;
  let endOffset = 0;

  for (const entry of entries) {
    if (!startNode && startUtf16 >= entry.startUtf16 && startUtf16 < entry.endUtf16) {
      startNode = entry.node;
      startOffset = startUtf16 - entry.startUtf16;
    }
    if (endUtf16 > entry.startUtf16 && endUtf16 <= entry.endUtf16) {
      endNode = entry.node;
      endOffset = endUtf16 - entry.startUtf16;
      break;
    }
  }

  if (!startNode || !endNode) return null;
  const targetDoc = doc ?? startNode.ownerDocument ?? (typeof document !== 'undefined' ? document : null);
  if (!targetDoc) return null;
  const range = targetDoc.createRange();
  range.setStart(startNode, startOffset);
  range.setEnd(endNode, endOffset);
  return range;
}

/**
 * Fuzzy re-anchoring when a post has been edited (`edit_date`), shifting UTF-16 coordinates.
 * Searches for exact quote or prefix/suffix match near the original hint window.
 */
export function fuzzyReanchorUtf16Span(
  currentText: string,
  expectedQuote: string,
  hintStartUtf16: number,
  hintEndUtf16: number,
): [number, number] | null {
  if (!currentText || !expectedQuote) return null;

  // 1. Exact slice match at hinted coordinates
  if (currentText.slice(hintStartUtf16, hintEndUtf16) === expectedQuote) {
    return [hintStartUtf16, hintEndUtf16];
  }

  // 2. Exact substring search nearest to hintStartUtf16
  let bestIdx = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  let idx = currentText.indexOf(expectedQuote);
  while (idx !== -1) {
    const dist = Math.abs(idx - hintStartUtf16);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = idx;
    }
    idx = currentText.indexOf(expectedQuote, idx + 1);
  }
  if (bestIdx !== -1) {
    return [bestIdx, bestIdx + expectedQuote.length];
  }

  // 3. Prefix + suffix anchor fallback (first 14 chars + last 14 chars)
  if (expectedQuote.length >= 24) {
    const prefix = expectedQuote.slice(0, 14);
    const suffix = expectedQuote.slice(-14);
    const pIdx = currentText.indexOf(prefix);
    if (pIdx !== -1) {
      const sIdx = currentText.indexOf(suffix, pIdx + prefix.length);
      if (sIdx !== -1 && sIdx - pIdx <= expectedQuote.length * 1.5) {
        return [pIdx, sIdx + suffix.length];
      }
    }
  }

  return null;
}

export function isCssCustomHighlightSupported(): boolean {
  return (
    typeof globalThis !== 'undefined' &&
    'CSS' in globalThis &&
    typeof (globalThis as unknown as { CSS?: { highlights?: unknown } }).CSS?.highlights === 'object' &&
    typeof Highlight === 'function'
  );
}

/**
 * Registers verified RAG UTF-16 ranges in native `CSS.highlights` (`::highlight(rag-evidence)`)
 * without mutating DOM nodes or breaking React reconciliation across formatting boundaries (SPEC 10.5.2).
 */
export function registerCssCustomHighlights(
  rootEl: HTMLElement,
  rawSpans: RagHighlightInput[],
  highlightName = 'rag-evidence',
): { registeredCount: number; ranges: Array<{ claimId: string; range: Range }> } {
  if (!isCssCustomHighlightSupported() || rawSpans.length === 0) {
    return { registeredCount: 0, ranges: [] };
  }

  const doc = rootEl.ownerDocument ?? document;
  const entries = buildDomUtf16TextMap(rootEl);
  const normalized: RagHighlightSpan[] = rawSpans
    .map(normalizeRagHighlightSpan)
    .filter((s) => s.endUtf16 > s.startUtf16);

  const createdRanges: Array<{ claimId: string; range: Range }> = [];
  for (const span of normalized) {
    const range = resolveDomRangeForUtf16Span(entries, span.startUtf16, span.endUtf16, doc);
    if (range) {
      createdRanges.push({ claimId: span.claimId, range });
    }
  }

  if (createdRanges.length > 0) {
    const cssHighlights = (globalThis as unknown as {
      CSS: { highlights: Map<string, Highlight> };
    }).CSS.highlights;
    const existing = cssHighlights.get(highlightName);
    const highlight = existing ?? new Highlight();
    for (const item of createdRanges) {
      highlight.add(item.range);
    }
    cssHighlights.set(highlightName, highlight);
  }

  return { registeredCount: createdRanges.length, ranges: createdRanges };
}

/**
 * Cross-browser click hit-testing for `CSS.highlights` ranges using
 * `document.caretPositionFromPoint` / `range.getClientRects()` (SPEC 10.5.2).
 */
export function hitTestHighlightAtPoint(
  clientX: number,
  clientY: number,
  ranges: Array<{ claimId: string; range: Range }>,
): string | null {
  for (const item of ranges) {
    const rects = typeof item.range.getClientRects === 'function' ? item.range.getClientRects() : [];
    for (let i = 0; i < rects.length; i += 1) {
      const r = rects[i];
      if (clientX >= r.left && clientX <= r.right && clientY >= r.top && clientY <= r.bottom) {
        return item.claimId;
      }
    }
  }
  return null;
}
