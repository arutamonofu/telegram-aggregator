/**
 * Virtualization scroll stabilizer (SPEC 10.5.3).
 * Prevents upward scroll jumps on dynamic-height cards (TanStack Virtual #659)
 * by gating DOM re-measurement on forward scroll direction only.
 */
export type ScrollDirection = 'forward' | 'backward' | null;

export function createDirectionAwareMeasurer(defaultEstimatePx = 220) {
  const heightCache = new Map<string, number>();

  return {
    measureElement(
      itemKey: string,
      element: { scrollHeight: number } | null,
      scrollDirection: ScrollDirection,
    ): number {
      const cached = heightCache.get(itemKey);
      if (scrollDirection === 'backward' && cached !== undefined) {
        return cached;
      }
      if (element && (scrollDirection === 'forward' || scrollDirection === null || cached === undefined)) {
        const measured = Math.max(1, element.scrollHeight);
        heightCache.set(itemKey, measured);
        return measured;
      }
      return cached ?? defaultEstimatePx;
    },
    getCachedHeight(itemKey: string): number | undefined {
      return heightCache.get(itemKey);
    },
  };
}

/**
 * Compensates scrollTop when new items are prepended above the viewport:
 * `newScrollTop = scrollHeight - previousScrollPositionFromBottom - clientHeight`
 */
export function computeCompensatedScrollTop(
  scrollHeight: number,
  previousScrollPositionFromBottom: number,
  clientHeight: number,
): number {
  return Math.max(0, scrollHeight - previousScrollPositionFromBottom - clientHeight);
}
