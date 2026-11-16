'use client';

import React, { useMemo, useState } from 'react';
import { parseTelegramPostAstSync } from '../../lib/ast/parser';
import type { RagHighlightInput } from '../../lib/ast/types';
import { canSharePostExternally } from '../../lib/entitlements';
import { formatRuDateTime } from '../../lib/format_date';
import type {
  CleanPostItem,
  DensityMode,
  ZeroTrustUserSession,
} from '../../lib/types/feed';
import { AstRenderer } from '../ast/AstRenderer';
import { CommentSummaryDrawer } from '../comments/CommentSummaryDrawer';
import {
  IconCheck,
  IconChevronUp,
  IconExpand,
  IconImage,
  IconLayers,
  IconLock,
  IconRss,
  IconSend,
  IconShare,
  IconSparkles,
} from '../ui/Icons';
import { AlbumGrid } from './AlbumGrid';

export interface PostCardProps {
  post: CleanPostItem;
  session: ZeroTrustUserSession;
  density?: DensityMode;
  isActive?: boolean;
  cssHighlightSpans?: RagHighlightInput[];
}

export function PostCard({
  post,
  session,
  density = 'comfortable',
  isActive = false,
  cssHighlightSpans,
}: PostCardProps) {
  const [showOriginalHeadline, setShowOriginalHeadline] = useState(false);
  const [expandedInCompact, setExpandedInCompact] = useState(false);
  const [shareStatus, setShareStatus] = useState<string | null>(null);

  const parsedAst = useMemo(
    () => parseTelegramPostAstSync(post.text, post.entities),
    [post.text, post.entities]
  );

  const shareCheck = canSharePostExternally(post, session);
  const hasOriginalHeadline =
    Boolean(post.originalHeadline) && post.originalHeadline !== post.neutralHeadline;

  const isCompactCollapsed = density === 'compact' && !expandedInCompact;

  const headlineDuplicatesBody = useMemo(() => {
    const cleanHead = post.neutralHeadline
      .replace(/^[🔥⚡🚨❗️❗🔴🟢📢🚀✨💼🎯]+\s*/u, '')
      .replace(/…$/, '')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
    const cleanBody = post.text
      .replace(/^[🔥⚡🚨❗️❗🔴🟢📢🚀✨💼🎯]+\s*/u, '')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
    if (!cleanHead || !cleanBody) return false;
    return cleanBody.startsWith(cleanHead.slice(0, Math.min(cleanHead.length, 48)));
  }, [post.neutralHeadline, post.text]);

  const showHeadlineRow = hasOriginalHeadline || (!headlineDuplicatesBody && isCompactCollapsed);

  const displayedTitle =
    showOriginalHeadline && post.originalHeadline
      ? post.originalHeadline
      : post.neutralHeadline;

  const handleShareClick = () => {
    if (!shareCheck.allowed) {
      setShareStatus(shareCheck.reason ?? 'Доступно только внутри команды');
      return;
    }
    const handle = post.channelHandle.replace(/^@/, '');
    const shareUrl =
      (post.sourcePlatform ?? 'TG') === 'TG'
        ? `https://t.me/${handle}/${post.postId}`
        : `${post.channelHandle} — ${post.neutralHeadline}`;
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      void navigator.clipboard.writeText(shareUrl);
    }
    setShareStatus('Ссылка скопирована');
    setTimeout(() => setShareStatus(null), 2500);
  };

  return (
    <article
      data-post-id={post.id}
      className={`post-card ${isActive ? 'post-card-active' : ''}`}
    >
      <div className="row-between" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
        <div className="row" style={{ gap: '0.4rem', flexWrap: 'wrap' }}>
          <span
            className="icon-pill"
            data-tooltip={`Источник: ${post.sourcePlatform ?? 'TG'}`}
            aria-label={`Источник: ${post.sourcePlatform ?? 'TG'}`}
          >
            {(post.sourcePlatform ?? 'TG') === 'RSS' ? (
              <IconRss size={12} />
            ) : (post.sourcePlatform ?? 'TG') === 'MAX' ? (
              <span className="font-mono text-[10px]">MAX</span>
            ) : (
              <IconSend size={12} />
            )}
          </span>
          <strong className="typo-meta" style={{ color: 'var(--text-primary)' }}>
            {post.channelTitle}
          </strong>
          <span className="typo-meta">{post.channelHandle}</span>
          {post.isPrivateChannel ? (
            <span
              className="icon-pill"
              data-tooltip="Закрытый корпоративный канал · внешний экспорт ограничен"
            >
              <IconLock size={12} />
              <span className="sr-only">Закрытый канал</span>
            </span>
          ) : null}
          {post.duplicateCount && post.duplicateCount > 0 ? (
            <span
              className="icon-pill tabular-nums"
              data-tooltip={`Скрыто повторов в других каналах: ${post.duplicateCount}`}
            >
              <IconLayers size={12} />
              <span>{post.duplicateCount}</span>
            </span>
          ) : null}
        </div>

        <span className="typo-meta tabular-nums">{formatRuDateTime(post.publishedAt)}</span>
      </div>

      {showHeadlineRow ? (
        <div style={{ marginTop: '0.45rem' }}>
          <div className="row" style={{ gap: '0.45rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <h2 className="typo-h3" style={{ flex: 1, minWidth: '220px' }}>
              {displayedTitle}
            </h2>

            {hasOriginalHeadline ? (
              <button
                type="button"
                className={showOriginalHeadline ? 'icon-btn icon-btn-active' : 'icon-btn'}
                onClick={() => setShowOriginalHeadline((prev) => !prev)}
                aria-pressed={showOriginalHeadline}
                aria-label={showOriginalHeadline ? 'Оригинал автора · вернуть' : 'Заголовок очищен'}
                data-tooltip={
                  showOriginalHeadline
                    ? 'Показан оригинал автора · вернуть очищенный'
                    : 'Заголовок очищен от кликбейта · сравнить с оригиналом'
                }
              >
                <IconSparkles size={14} />
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      <div style={{ marginTop: '0.45rem' }}>
        {isCompactCollapsed ? (
          <p className="typo-body line-clamp-2" style={{ color: 'var(--text-secondary)' }}>
            {post.text}
          </p>
        ) : (
          <AstRenderer
            nodes={parsedAst.nodes}
            richContainers={parsedAst.richContainers}
            density={density}
            cssHighlightSpans={cssHighlightSpans}
          />
        )}
      </div>

      {post.media.length > 0 ? (
        isCompactCollapsed ? (
          <div style={{ marginTop: '0.35rem' }}>
            <button
              type="button"
              className="icon-pill"
              onClick={() => setExpandedInCompact(true)}
              aria-label={`Показать вложения (${post.media.length})`}
              data-tooltip={`Показать вложения (${post.media.length})`}
            >
              <IconImage size={12} />
              <span>{post.media.length}</span>
            </button>
          </div>
        ) : (
          <AlbumGrid attachments={post.media} density={density} />
        )
      ) : null}

      <div className="row-between" style={{ marginTop: '0.6rem', flexWrap: 'wrap', gap: '0.4rem' }}>
        <div className="row" style={{ gap: '0.25rem', flexWrap: 'wrap' }}>
          {density === 'compact' ? (
            <button
              type="button"
              className="icon-btn"
              onClick={() => setExpandedInCompact((prev) => !prev)}
              aria-label={expandedInCompact ? 'Свернуть' : 'Развернуть пост'}
              data-tooltip={expandedInCompact ? 'Свернуть пост' : 'Развернуть пост полностью'}
            >
              {expandedInCompact ? <IconChevronUp size={15} /> : <IconExpand size={14} />}
            </button>
          ) : null}

          <button
            type="button"
            className="icon-btn"
            onClick={handleShareClick}
            disabled={!shareCheck.allowed}
            aria-label={shareCheck.allowed ? 'Поделиться' : 'Экспорт ограничен'}
            data-tooltip={
              shareStatus ??
              (shareCheck.allowed
                ? 'Скопировать ссылку на первоисточник'
                : shareCheck.reason ?? 'Запрещено политикой компании')
            }
          >
            {!shareCheck.allowed ? (
              <IconLock size={14} />
            ) : shareStatus === 'Ссылка скопирована' ? (
              <IconCheck size={15} />
            ) : (
              <IconShare size={15} />
            )}
          </button>

          {shareStatus ? <span className="typo-meta">{shareStatus}</span> : null}
        </div>
      </div>

      <CommentSummaryDrawer
        post={post}
        density={density}
      />
    </article>
  );
}
