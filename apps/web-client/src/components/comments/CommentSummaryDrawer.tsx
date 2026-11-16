'use client';

import React, { useState } from 'react';
import { fetchCommentSummary } from '../../lib/api/client';
import type {
  CleanPostItem,
  CommentSummaryResponse,
  DensityMode,
  SelectedCommentItem,
} from '../../lib/types/feed';
import { IconMessage } from '../ui/Icons';

export interface CommentSummaryDrawerProps {
  post: CleanPostItem;
  density?: DensityMode;
  initialSummary?: CommentSummaryResponse | null;
}

type RubricKey = 'bug_report' | 'counter_argument' | 'user_case';

const RUBRIC_META: readonly {
  category: RubricKey;
  title: string;
  badgeClass: string;
}[] = [
  {
    category: 'bug_report',
    title: 'Баг-репорты и опровержения',
    badgeClass: 'badge badge-unsupported',
  },
  {
    category: 'counter_argument',
    title: 'Контраргументы',
    badgeClass: 'badge badge-pending',
  },
  {
    category: 'user_case',
    title: 'Пользовательские кейсы',
    badgeClass: 'badge badge-entailed',
  },
];

export function CommentSummaryDrawer({
  post,
  density = 'comfortable',
  initialSummary = null,
}: CommentSummaryDrawerProps) {
  const [open, setOpen] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [summary, setSummary] = useState<CommentSummaryResponse | null>(initialSummary);
  const [error, setError] = useState<string | null>(null);

  const loadSummary = async (): Promise<CommentSummaryResponse | null> => {
    if (summary) {
      return summary;
    }
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchCommentSummary(post.id);
      setSummary(payload);
      return payload;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка загрузки обсуждения');
      return null;
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = async () => {
    const nextOpen = !open;
    setOpen(nextOpen);
    if (nextOpen && !summary) {
      await loadSummary();
    }
  };

  const groupComments = (items: SelectedCommentItem[], cat: RubricKey) =>
    items.filter((item) => {
      if (cat === 'user_case') {
        return item.category === 'user_case' || item.category === 'expertise';
      }
      return item.category === cat;
    });

  const commentCount = post.commentCount ?? 0;
  if (commentCount === 0 && !initialSummary) {
    return null;
  }

  return (
    <div className="comment-summary-block">
      <div className="row" style={{ flexWrap: 'wrap', gap: '0.35rem' }}>
        <button
          type="button"
          className={open ? 'icon-btn icon-btn-active tabular-nums' : 'icon-btn tabular-nums'}
          onClick={() => void handleToggle()}
          aria-expanded={open}
          aria-label={open ? 'Скрыть выжимку' : `Выжимка обсуждения (${commentCount})`}
          data-tooltip={
            open
              ? 'Скрыть выжимку комментариев'
              : `Выжимка обсуждения (${commentCount} комментариев)`
          }
        >
          <IconMessage size={14} />
          <span>{commentCount}</span>
        </button>
      </div>

      {open ? (
        <div className="comment-summary-panel">
          {loading ? (
            <p className="typo-meta row" style={{ gap: '0.4rem' }} role="status">
              <span className="loading-spinner" aria-hidden="true" />
              <span>Собираем ключевые мысли из комментариев…</span>
            </p>
          ) : null}
          {error ? (
            <p className="typo-meta" style={{ color: 'var(--verify-unsupported-accent)' }}>
              {error}
            </p>
          ) : null}

          {summary ? (
            <div className="stack" style={{ gap: '0.55rem' }}>
              <p className="typo-body">{summary.digest_text}</p>

              {density === 'compact' ? (
                <div className="row" style={{ gap: '0.35rem', flexWrap: 'wrap' }}>
                  {summary.selected.map((item) => (
                    <span
                      key={item.comment_id}
                      className="badge badge-neutral"
                      title={item.text}
                    >
                      {item.author_alias} • ↑{item.likes ?? item.confirmed_count ?? 0}
                    </span>
                  ))}
                </div>
              ) : (
                RUBRIC_META.map((rubric) => {
                  const items = groupComments(summary.selected, rubric.category);
                  if (items.length === 0) {
                    return null;
                  }
                  return (
                    <div key={rubric.category} className="comment-rubric-group">
                      <span className={rubric.badgeClass}>{rubric.title}</span>
                      <div className="stack" style={{ gap: '0.35rem', marginTop: '0.3rem' }}>
                        {items.map((item) => (
                          <div key={item.comment_id} className="comment-quote-row">
                            <div className="row-between" style={{ gap: '0.4rem' }}>
                              <span className="typo-meta" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                {item.author_alias} • ↑{item.likes ?? item.confirmed_count ?? 0}
                              </span>
                              <span className="typo-meta tabular-nums">
                                #{item.comment_id}
                              </span>
                            </div>
                            <p className="typo-meta" style={{ color: 'var(--text-primary)' }}>
                              {item.text}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
