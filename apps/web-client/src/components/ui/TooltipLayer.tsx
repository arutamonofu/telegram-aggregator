'use client';

import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

interface ActiveTooltip {
  text: string;
  anchorRect: DOMRect;
  preferredPos: 'top' | 'bottom';
}

/**
 * Global viewport-clamped portal tooltip for all `[data-tooltip]` elements.
 * Renders directly in `document.body` so tooltips are never clipped by `overflow`
 * containers (like the left sidebar) or trapped under sticky/fixed headers.
 */
export function TooltipLayer() {
  const [mounted, setMounted] = useState(false);
  const [tooltip, setTooltip] = useState<ActiveTooltip | null>(null);
  const [coords, setCoords] = useState<{ left: number; top: number; ready: boolean }>({
    left: 0,
    top: 0,
    ready: false,
  });
  const activeElRef = useRef<HTMLElement | null>(null);
  const bubbleRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const showForTarget = (target: EventTarget | null) => {
      if (!(target instanceof Element)) return;
      const el = target.closest('[data-tooltip]') as HTMLElement | null;
      if (!el) {
        activeElRef.current = null;
        setTooltip(null);
        return;
      }
      const text = el.getAttribute('data-tooltip')?.trim();
      if (!text) {
        activeElRef.current = null;
        setTooltip(null);
        return;
      }
      const posAttr = el.getAttribute('data-tooltip-pos');
      const preferredPos: 'top' | 'bottom' = posAttr === 'bottom' ? 'bottom' : 'top';
      activeElRef.current = el;
      setTooltip({
        text,
        anchorRect: el.getBoundingClientRect(),
        preferredPos,
      });
    };

    const handleMouseOver = (e: MouseEvent) => {
      showForTarget(e.target);
    };

    const handleMouseOut = (e: MouseEvent) => {
      const activeEl = activeElRef.current;
      if (!activeEl) return;
      const next = e.relatedTarget;
      if (next instanceof Element && activeEl.contains(next)) {
        return;
      }
      if (next instanceof Element && next.closest('[data-tooltip]') === activeEl) {
        return;
      }
      activeElRef.current = null;
      setTooltip(null);
    };

    const handleFocusIn = (e: FocusEvent) => {
      showForTarget(e.target);
    };

    const handleFocusOut = () => {
      activeElRef.current = null;
      setTooltip(null);
    };

    const handleDismiss = () => {
      activeElRef.current = null;
      setTooltip(null);
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleDismiss();
      }
    };

    document.addEventListener('mouseover', handleMouseOver);
    document.addEventListener('mouseout', handleMouseOut);
    document.addEventListener('focusin', handleFocusIn);
    document.addEventListener('focusout', handleFocusOut);
    window.addEventListener('scroll', handleDismiss, true);
    window.addEventListener('resize', handleDismiss);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mouseover', handleMouseOver);
      document.removeEventListener('mouseout', handleMouseOut);
      document.removeEventListener('focusin', handleFocusIn);
      document.removeEventListener('focusout', handleFocusOut);
      window.removeEventListener('scroll', handleDismiss, true);
      window.removeEventListener('resize', handleDismiss);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  useLayoutEffect(() => {
    if (!tooltip || !bubbleRef.current) {
      setCoords((prev) => (prev.ready ? { ...prev, ready: false } : prev));
      return;
    }

    const bubbleRect = bubbleRef.current.getBoundingClientRect();
    const { anchorRect, preferredPos } = tooltip;
    const gap = 8;
    const margin = 8;

    let top =
      preferredPos === 'bottom'
        ? anchorRect.bottom + gap
        : anchorRect.top - bubbleRect.height - gap;

    if (preferredPos === 'top' && top < margin) {
      top = anchorRect.bottom + gap;
    } else if (
      preferredPos === 'bottom' &&
      top + bubbleRect.height > window.innerHeight - margin
    ) {
      top = anchorRect.top - bubbleRect.height - gap;
    }
    top = Math.max(
      margin,
      Math.min(window.innerHeight - bubbleRect.height - margin, top),
    );

    const centeredLeft = anchorRect.left + anchorRect.width / 2 - bubbleRect.width / 2;
    const clampedLeft = Math.max(
      margin,
      Math.min(window.innerWidth - bubbleRect.width - margin, centeredLeft),
    );

    setCoords({ left: clampedLeft, top, ready: true });
  }, [tooltip]);

  if (!mounted || !tooltip || typeof document === 'undefined') {
    return null;
  }

  return createPortal(
    <div
      ref={bubbleRef}
      role="tooltip"
      className="portal-tooltip"
      style={{
        left: `${coords.left}px`,
        top: `${coords.top}px`,
        opacity: coords.ready ? 1 : 0,
      }}
    >
      {tooltip.text}
    </div>,
    document.body,
  );
}
