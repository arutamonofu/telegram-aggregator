'use client';

import type { Element as HastElement, ElementContent, Root as HastRoot } from 'hast';
import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import {
  hitTestHighlightAtPoint,
  registerCssCustomHighlights,
} from '@/lib/ast/css_highlights';
import { decodeDateFlags, formatTelegramDynamicDate } from '@/lib/ast/formatted_date';
import { parseHastInlineStyle } from '@/lib/ast/shiki_highlighter';
import type {
  CodeBlockAstNode,
  CustomEmojiAstNode,
  DiffAstNode,
  ElementAstNode,
  FormattedDateAstNode,
  RagHighlightInput,
  RichMessageContainer,
  TelegramAstNode,
  TextLeafNode,
  WidgetAstNode,
} from '@/lib/ast/types';
import type { DensityMode } from '@/lib/types/feed';
import {
  DEFAULT_ALLOWED_LINK_DOMAINS,
  type InspectedLink,
  inspectLinkSafety,
} from '@/lib/ast/url_safety';
import { ExternalLinkModal } from './ExternalLinkModal';

export interface CitationSyncContextValue {
  activeCiteId: string | null;
  lockedCiteId: string | null;
  setHoverCiteId: (citeId: string | null) => void;
  toggleLockCiteId: (citeId: string) => void;
  focusCiteInDom: (citeId: string) => void;
}

const CitationSyncContext = createContext<CitationSyncContextValue | null>(null);

export function CitationSyncProvider({ children }: { children: React.ReactNode }) {
  const [hoverCiteId, setHoverCiteId] = useState<string | null>(null);
  const [lockedCiteId, setLockedCiteId] = useState<string | null>(null);

  const activeCiteId = hoverCiteId ?? lockedCiteId;

  const toggleLockCiteId = useCallback((citeId: string) => {
    setLockedCiteId((prev) => (prev === citeId ? null : citeId));
  }, []);

  const focusCiteInDom = useCallback((citeId: string) => {
    setLockedCiteId(citeId);
    if (typeof document !== 'undefined') {
      const target = document.querySelector(`[data-cite="${CSS.escape(citeId)}"]`);
      if (target && typeof (target as HTMLElement).scrollIntoView === 'function') {
        (target as HTMLElement).scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, []);

  return (
    <CitationSyncContext.Provider
      value={{
        activeCiteId,
        lockedCiteId,
        setHoverCiteId,
        toggleLockCiteId,
        focusCiteInDom,
      }}
    >
      {children}
    </CitationSyncContext.Provider>
  );
}

export function useCitationSync(): CitationSyncContextValue | null {
  return useContext(CitationSyncContext);
}

function renderHastContentToReact(
  node: HastRoot | ElementContent,
  keyPrefix: string,
): React.ReactNode {
  if (node.type === 'text') {
    return node.value;
  }
  if (node.type === 'root') {
    return node.children.map((child, idx) =>
      renderHastContentToReact(child as ElementContent, `${keyPrefix}-r${idx}`),
    );
  }
  if (node.type !== 'element') {
    return null;
  }

  const el = node as HastElement;
  const props: Record<string, unknown> = { key: keyPrefix };

  if (el.properties) {
    for (const [rawKey, rawVal] of Object.entries(el.properties)) {
      if (rawVal === undefined || rawVal === null || rawVal === false) continue;
      if (rawKey === 'className' || rawKey === 'class') {
        props.className = Array.isArray(rawVal) ? rawVal.join(' ') : String(rawVal);
      } else if (rawKey === 'style') {
        const parsedStyle = parseHastInlineStyle(rawVal);
        if (parsedStyle) props.style = parsedStyle;
      } else if (rawKey === 'tabindex') {
        props.tabIndex = Number(rawVal);
      } else if (rawKey === 'tabIndex') {
        props.tabIndex = rawVal;
      } else {
        props[rawKey] = Array.isArray(rawVal) ? rawVal.join(' ') : rawVal;
      }
    }
  }

  const renderedChildren = el.children.map((child, idx) =>
    renderHastContentToReact(child, `${keyPrefix}-c${idx}`),
  );

  const Tag = el.tagName as keyof React.JSX.IntrinsicElements;
  return React.createElement(Tag, props, ...renderedChildren);
}

function CodeBlockView({ node }: { node: CodeBlockAstNode }) {
  const [copied, setCopied] = useState(false);
  const lineCount = node.code.split('\n').length;
  const isCollapsible = lineCount > 15;
  const [expanded, setExpanded] = useState(!isCollapsible);

  const handleCopy = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      void navigator.clipboard.writeText(node.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <div
      className="my-3 overflow-hidden rounded-xl border text-xs"
      style={{
        borderColor: node.citeIds.length > 0 ? '#f59e0b' : 'var(--border-subtle)',
        backgroundColor: 'var(--bg-muted)',
      }}
      data-testid="ast-code-block"
      data-highlighter={node.highlighterEngine}
      data-language={node.language}
      data-cite={node.citeIds[0]}
    >
      <div
        className="flex items-center justify-between border-b px-3 py-1.5 text-[11px]"
        style={{
          borderColor: 'var(--border-subtle)',
          color: 'var(--text-secondary)',
        }}
      >
        <div className="flex items-center gap-2">
          <span className="typo-code font-semibold" style={{ color: 'var(--text-primary)' }}>
            {node.language}
          </span>
          <span
            className="rounded px-1.5 py-0.5 text-[10px]"
            style={{
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            {node.highlighterEngine}
          </span>
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="rounded-md px-2 py-0.5 text-[11px] font-medium hover:opacity-80"
          style={{
            backgroundColor: 'var(--bg-surface)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          {copied ? 'Скопировано' : 'Копировать'}
        </button>
      </div>
      <div
        className="ast-code-hast typo-code relative overflow-x-auto p-3"
        style={!expanded ? { maxHeight: '22em', overflowY: 'hidden' } : undefined}
      >
        {renderHastContentToReact(node.hast, `code-${node.startUtf16}`)}
        {!expanded ? (
          <div
            className="pointer-events-none absolute inset-x-0 bottom-0 h-14"
            style={{
              background: 'linear-gradient(to bottom, transparent, var(--bg-muted))',
            }}
          />
        ) : null}
      </div>
      {isCollapsible ? (
        <div
          className="border-t px-3 py-1.5 text-center"
          style={{ borderColor: 'var(--border-subtle)' }}
        >
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            data-testid="code-collapse-toggle"
            className="text-xs font-medium"
            style={{ color: 'var(--accent-primary)' }}
          >
            {expanded ? 'Свернуть листинг' : `Развернуть листинг (${lineCount} строк)`}
          </button>
        </div>
      ) : null}
    </div>
  );
}

function ExpandableBlockquoteView({
  node,
  children,
}: {
  node: ElementAstNode;
  children: React.ReactNode;
}) {
  const [expanded, setExpanded] = useState(!node.collapsed);

  return (
    <blockquote
      className="my-2.5 rounded-r-xl border-l-3 border-slate-300 bg-slate-50/90 px-3.5 py-2 text-slate-700"
      data-collapsed={!expanded ? 'true' : 'false'}
      data-expandable={node.collapsed ? 'true' : 'false'}
    >
      <div className={node.collapsed && !expanded ? 'line-clamp-2 overflow-hidden' : ''}>
        {children}
      </div>
      {node.collapsed ? (
        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          className="mt-1.5 text-xs font-medium text-sky-700 hover:text-sky-900"
        >
          {expanded ? 'Свернуть цитату' : 'Развернуть цитату'}
        </button>
      ) : null}
    </blockquote>
  );
}

function SpoilerView({ children }: { children: React.ReactNode }) {
  const [revealed, setRevealed] = useState(false);

  return (
    <span
      role="button"
      tabIndex={0}
      onClick={() => setRevealed((prev) => !prev)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          setRevealed((prev) => !prev);
        }
      }}
      data-revealed={revealed ? 'true' : 'false'}
      aria-label={revealed ? 'Скрыть спойлер' : 'Показать скрытый текст'}
      className={`inline cursor-pointer rounded px-1 transition-all select-none ${
        revealed
          ? 'bg-slate-100 text-slate-900'
          : 'bg-slate-200/90 text-transparent blur-[3px] hover:bg-slate-300/80'
      }`}
    >
      {children}
    </span>
  );
}

function FormattedDateView({ node }: { node: FormattedDateAstNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [inViewport, setInViewport] = useState(true);
  const timeRef = useRef<HTMLTimeElement | null>(null);

  const flags = decodeDateFlags(node.dateFlags, node.dateTimeFormat);

  useEffect(() => {
    if (!flags.relative || typeof IntersectionObserver === 'undefined' || !timeRef.current) {
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        setInViewport(Boolean(entries[0]?.isIntersecting));
      },
      { threshold: 0.1 },
    );
    observer.observe(timeRef.current);
    return () => observer.disconnect();
  }, [flags.relative]);

  useEffect(() => {
    if (!flags.relative || !inViewport) return;
    const timer = setInterval(() => setNowMs(Date.now()), 30000);
    return () => clearInterval(timer);
  }, [flags.relative, inViewport]);

  const formatted = formatTelegramDynamicDate({
    unixTime: node.unixTime,
    dateFlags: node.dateFlags,
    dateTimeFormat: node.dateTimeFormat,
    fallbackText: node.fallbackText,
    nowMs,
  });

  const handleCopyTimestamp = () => {
    if (node.unixTime === undefined) return;
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      void navigator.clipboard.writeText(String(node.unixTime));
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    }
  };

  const handleAddToCalendar = () => {
    if (node.unixTime === undefined || typeof document === 'undefined') return;
    const startDate = new Date(node.unixTime * 1000);
    const pad = (n: number) => String(n).padStart(2, '0');
    const toIcs = (d: Date) =>
      `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}T${pad(
        d.getUTCHours(),
      )}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}Z`;
    const endDate = new Date((node.unixTime + 3600) * 1000);
    const ics = [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'BEGIN:VEVENT',
      `DTSTART:${toIcs(startDate)}`,
      `DTEND:${toIcs(endDate)}`,
      `SUMMARY:${node.fallbackText || 'Событие из Telegram'}`,
      'END:VEVENT',
      'END:VCALENDAR',
    ].join('\r\n');
    const blob = new Blob([ics], { type: 'text/calendar;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `event-${node.unixTime}.ics`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <span className="relative inline-block">
      <time
        ref={timeRef}
        dateTime={formatted.isoString || undefined}
        data-unix-time={node.unixTime}
        data-date-flags={node.dateFlags}
        data-exceeds-limit={node.exceedsLimit ? 'true' : 'false'}
        title={`${formatted.fullLocalString} (нажмите для копирования timestamp)`}
        onClick={() => setMenuOpen((prev) => !prev)}
        onContextMenu={(e) => {
          e.preventDefault();
          setMenuOpen((prev) => !prev);
        }}
        className="cursor-pointer rounded border-b border-dotted px-1 py-0.5 text-[0.95em] font-medium"
        style={{
          borderColor: 'var(--accent-primary)',
          backgroundColor: 'var(--accent-primary-bg)',
          color: 'var(--text-primary)',
        }}
      >
        {formatted.displayText}
      </time>
      {menuOpen ? (
        <span
          role="menu"
          className="surface-card absolute left-0 top-full z-20 mt-1 flex w-64 flex-col gap-1.5 rounded-xl p-2.5 text-xs shadow-lg"
        >
          <span className="font-medium">{formatted.fullLocalString}</span>
          {formatted.isoString ? (
            <span className="typo-code text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              {formatted.isoString}
            </span>
          ) : null}
          {node.unixTime !== undefined ? (
            <>
              <button
                type="button"
                onClick={handleCopyTimestamp}
                className="surface-muted mt-1 rounded-lg px-2.5 py-1 text-left font-medium"
              >
                {copied ? 'Timestamp скопирован' : `Копировать Unix: ${node.unixTime}`}
              </button>
              <button
                type="button"
                onClick={handleAddToCalendar}
                data-testid="add-to-calendar-btn"
                className="surface-muted rounded-lg px-2.5 py-1 text-left font-medium"
              >
                Добавить событие в календарь (.ics)
              </button>
            </>
          ) : null}
        </span>
      ) : null}
    </span>
  );
}

function CustomEmojiView({
  node,
  children,
  density = 'comfortable',
}: {
  node: CustomEmojiAstNode;
  children: React.ReactNode;
  density?: DensityMode;
}) {
  const [imgFailed, setImgFailed] = useState(false);

  return (
    <span
      className="inline-flex items-center align-text-bottom"
      data-custom-emoji-id={node.documentId ?? ''}
      data-lottie-url={node.lottieUrl ?? ''}
      data-twemoji-url={node.twemojiUrl}
      data-static-fallback={density === 'compact' ? 'true' : 'false'}
      title={node.alt}
    >
      {!imgFailed && density !== 'compact' ? (
        <img
          src={node.twemojiUrl}
          alt={node.alt}
          loading="lazy"
          onError={() => setImgFailed(true)}
          className="inline-block h-[1.15em] w-[1.15em] align-middle"
        />
      ) : (
        <span className="inline-block">{node.alt || children}</span>
      )}
    </span>
  );
}

function CopyableChipView({
  kind,
  children,
}: {
  kind: 'ton_address' | 'bank_card_number';
  children: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const extractText = (n: React.ReactNode): string => {
    if (typeof n === 'string' || typeof n === 'number') return String(n);
    if (Array.isArray(n)) return n.map(extractText).join('');
    if (React.isValidElement<{ children?: React.ReactNode }>(n)) {
      return extractText(n.props.children);
    }
    return '';
  };

  const handleCopy = () => {
    const val = extractText(children);
    if (typeof navigator !== 'undefined' && navigator.clipboard && val) {
      void navigator.clipboard.writeText(val);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      data-testid={`copy-chip-${kind}`}
      title={
        kind === 'ton_address'
          ? 'TON-адрес (копирование без автозапуска платежа)'
          : 'Номер карты (копирование без автозапуска платежа)'
      }
      className="surface-muted typo-code inline-flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-[0.88em]"
    >
      <span>{kind === 'ton_address' ? 'TON' : 'CARD'}</span>
      <span>{children}</span>
      <span className="text-[10px]" style={{ color: 'var(--accent-primary)' }}>
        {copied ? '✓' : 'Копировать'}
      </span>
    </button>
  );
}

function DiffEntityView({
  node,
  children,
}: {
  node: DiffAstNode;
  children: React.ReactNode;
}) {
  if (node.variant === 'diff_delete') {
    return (
      <del className="rounded bg-rose-50 px-1 text-rose-800 line-through" data-diff="delete">
        {children}
      </del>
    );
  }
  if (node.variant === 'diff_insert') {
    return (
      <ins className="rounded bg-emerald-50 px-1 text-emerald-900 no-underline" data-diff="insert">
        {children}
      </ins>
    );
  }
  return (
    <span className="inline-flex flex-wrap items-baseline gap-1" data-diff="replace">
      {node.oldText ? (
        <del className="rounded bg-rose-50 px-1 text-rose-700 line-through">{node.oldText}</del>
      ) : null}
      <ins className="rounded bg-emerald-50 px-1 text-emerald-900 no-underline">{children}</ins>
    </span>
  );
}

function WidgetView({ node }: { node: WidgetAstNode }) {
  return (
    <span
      className="my-1 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-800"
      data-widget-kind={node.payload.kind}
    >
      <span className="font-semibold">{node.payload.title}</span>
      {node.payload.subtitle ? (
        <span className="text-slate-500">{node.payload.subtitle}</span>
      ) : null}
    </span>
  );
}

function RagTextLeafView({ node }: { node: TextLeafNode }) {
  const sync = useCitationSync();
  if (node.citeIds.length === 0) {
    return <>{node.text}</>;
  }

  const primaryCiteId = node.citeIds[0];
  const isActive = Boolean(sync?.activeCiteId && node.citeIds.includes(sync.activeCiteId));

  return (
    <mark
      data-cite={primaryCiteId}
      data-cites={node.citeIds.length > 1 ? node.citeIds.join(' ') : undefined}
      data-cite-active={isActive ? 'true' : 'false'}
      tabIndex={0}
      onMouseEnter={() => sync?.setHoverCiteId(primaryCiteId)}
      onMouseLeave={() => sync?.setHoverCiteId(null)}
      onFocus={() => sync?.setHoverCiteId(primaryCiteId)}
      onBlur={() => sync?.setHoverCiteId(null)}
      onClick={() => sync?.toggleLockCiteId(primaryCiteId)}
      className={`cursor-pointer rounded-xs px-0.5 transition-colors focus:outline-2 focus:outline-amber-500 ${
        isActive
          ? 'rag-cite-mark-active bg-amber-300/90 text-slate-950 ring-1 ring-amber-500'
          : 'bg-amber-100/90 text-slate-900 hover:bg-amber-200/90'
      }`}
    >
      {node.text}
    </mark>
  );
}

function AstNodeList({
  nodes,
  onLinkClick,
  density,
}: {
  nodes: TelegramAstNode[];
  onLinkClick: (rawHref: string, e: React.MouseEvent<HTMLAnchorElement>) => void;
  density?: DensityMode;
}) {
  return (
    <>
      {nodes.map((node, index) => {
        const key = `${node.kind}-${node.startUtf16}-${node.endUtf16}-${index}`;
        if (node.kind === 'text') {
          return <RagTextLeafView key={key} node={node} />;
        }
        if (node.kind === 'code_block') {
          return <CodeBlockView key={key} node={node} />;
        }
        if (node.kind === 'formatted_date') {
          return <FormattedDateView key={key} node={node} />;
        }
        if (node.kind === 'widget') {
          return <WidgetView key={key} node={node} />;
        }

        const renderedChildren = (
          <AstNodeList nodes={node.children} onLinkClick={onLinkClick} density={density} />
        );

        if (node.kind === 'custom_emoji') {
          return (
            <CustomEmojiView key={key} node={node} density={density}>
              {renderedChildren}
            </CustomEmojiView>
          );
        }
        if (node.kind === 'diff') {
          return (
            <DiffEntityView key={key} node={node}>
              {renderedChildren}
            </DiffEntityView>
          );
        }

        switch (node.tag) {
          case 'bold':
            return (
              <strong key={key} className="font-semibold text-slate-950">
                {renderedChildren}
              </strong>
            );
          case 'italic':
            return (
              <em key={key} className="italic">
                {renderedChildren}
              </em>
            );
          case 'underline':
            return (
              <u key={key} className="underline decoration-slate-400 underline-offset-2">
                {renderedChildren}
              </u>
            );
          case 'strikethrough':
            return (
              <s key={key} className="line-through decoration-slate-400">
                {renderedChildren}
              </s>
            );
          case 'code':
            return (
              <code
                key={key}
                className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[0.9em] text-slate-800"
              >
                {renderedChildren}
              </code>
            );
          case 'blockquote':
            return (
              <ExpandableBlockquoteView key={key} node={node}>
                {renderedChildren}
              </ExpandableBlockquoteView>
            );
          case 'spoiler':
            return <SpoilerView key={key}>{renderedChildren}</SpoilerView>;
          case 'ton_address':
          case 'bank_card_number':
            return (
              <CopyableChipView key={key} kind={node.tag}>
                {renderedChildren}
              </CopyableChipView>
            );
          case 'text_link':
          case 'url':
          case 'mention':
          case 'text_mention':
          case 'email':
          case 'phone_number': {
            const href = node.href ?? '#';
            return (
              <a
                key={key}
                href={href}
                onClick={(e) => onLinkClick(href, e)}
                rel="noopener noreferrer"
                className="text-sky-700 underline decoration-sky-300 underline-offset-2 hover:text-sky-900"
              >
                {renderedChildren}
              </a>
            );
          }
          case 'hashtag':
          case 'cashtag':
          case 'bot_command':
            return (
              <span key={key} className="font-medium text-sky-700">
                {renderedChildren}
              </span>
            );
          default:
            return <span key={key}>{renderedChildren}</span>;
        }
      })}
    </>
  );
}

export interface AstRendererProps {
  nodes: TelegramAstNode[];
  richContainers?: RichMessageContainer[];
  allowedDomains?: readonly string[];
  className?: string;
  density?: DensityMode;
  cssHighlightSpans?: RagHighlightInput[];
}

function AstRendererInner({
  nodes,
  richContainers,
  allowedDomains = DEFAULT_ALLOWED_LINK_DOMAINS,
  className = '',
  density = 'comfortable',
  cssHighlightSpans = [],
}: AstRendererProps) {
  const [interceptedLink, setInterceptedLink] = useState<InspectedLink | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const registeredRangesRef = useRef<Array<{ claimId: string; range: Range }>>([]);
  const sync = useCitationSync();

  useEffect(() => {
    if (!bodyRef.current || cssHighlightSpans.length === 0) {
      registeredRangesRef.current = [];
      return;
    }
    const { ranges } = registerCssCustomHighlights(bodyRef.current, cssHighlightSpans);
    registeredRangesRef.current = ranges;
  }, [nodes, cssHighlightSpans]);

  const handleLinkClick = useCallback(
    (rawHref: string, event: React.MouseEvent<HTMLAnchorElement>) => {
      const inspected = inspectLinkSafety(rawHref, allowedDomains);
      if (inspected.requiresInterception) {
        event.preventDefault();
        setInterceptedLink(inspected);
      } else if (typeof window !== 'undefined' && !rawHref.startsWith('tg:')) {
        event.preventDefault();
        window.open(inspected.normalizedUrl, '_blank', 'noopener,noreferrer');
      }
    },
    [allowedDomains],
  );

  const handleBodyClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (registeredRangesRef.current.length === 0) return;
      const hitClaimId = hitTestHighlightAtPoint(
        e.clientX,
        e.clientY,
        registeredRangesRef.current,
      );
      if (hitClaimId) {
        sync?.toggleLockCiteId(hitClaimId);
      }
    },
    [sync],
  );

  return (
    <div
      ref={bodyRef}
      onClick={handleBodyClick}
      className={`ast-post-body typo-body whitespace-pre-wrap break-words ${className}`}
    >
      <AstNodeList nodes={nodes} onLinkClick={handleLinkClick} density={density} />
      {richContainers && richContainers.length > 0 ? (
        <div className="mt-3 space-y-2">
          {richContainers.map((container, idx) => (
            <div
              key={`${container.containerType}-${idx}`}
              className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700"
              data-testid="rich-message-container"
            >
              {container.title ? (
                <div className="mb-1.5 font-semibold text-slate-900">{container.title}</div>
              ) : null}
              <div className="flex flex-wrap gap-2">
                {container.widgets.map((widget, wIdx) => (
                  <div
                    key={`${widget.kind}-${wIdx}`}
                    className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5"
                  >
                    <div className="font-medium text-slate-900">{widget.title}</div>
                    {widget.items?.map((item, iIdx) => (
                      <div key={iIdx} className="text-[11px] text-slate-600">
                        {item.label}: <span className="font-medium text-slate-900">{item.value}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : null}
      <ExternalLinkModal link={interceptedLink} onClose={() => setInterceptedLink(null)} />
    </div>
  );
}

/**
 * Deterministic AST renderer component.
 * Automatically binds to an ancestor `CitationSyncProvider` when present,
 * or mounts an isolated provider locally so standalone embeds still synchronize hover/focus.
 */
export function AstRenderer(props: AstRendererProps) {
  const existingSync = useCitationSync();
  if (existingSync) {
    return <AstRendererInner {...props} />;
  }
  return (
    <CitationSyncProvider>
      <AstRendererInner {...props} />
    </CitationSyncProvider>
  );
}
