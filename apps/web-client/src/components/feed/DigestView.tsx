'use client';

import React, { useMemo, useState } from 'react';
import { formatRuDateTime } from '../../lib/format_date';
import type {
  DensityMode,
  DigestCard,
  DigestFeedResponse,
  DuplicateChannelRef,
  RagSourcePostItem,
} from '../../lib/types/feed';
import { CitationHoverCard } from '../rag/CitationHoverCard';
import {
  IconArrowRight,
  IconCheck,
  IconCheckDouble,
  IconChevronDown,
  IconChevronUp,
  IconLayers,
} from '../ui/Icons';

export interface DigestViewProps {
  feed: DigestFeedResponse;
  density: DensityMode;
  sourcesByPostId: Record<string, RagSourcePostItem>;
  reviewedCardIds: ReadonlySet<string>;
  activeCardIndex?: number;
  isRefreshing?: boolean;
  lastRefreshedAt?: string | null;
  onMarkCardReviewed: (cardId: string) => void;
  onResetReviewed?: () => void;
  onFilterCleanByCluster: (clusterId: string, title: string) => void;
  onNavigateToCleanFeed: () => void;
  onOpenSources?: () => void;
}

function buildDedupeLabel(card: DigestCard, otherChannels: DuplicateChannelRef[]): string {
  const count = Math.max(card.otherChannelsCount, otherChannels.length);
  if (count === 0) {
    return `Первоисточник: ${card.primarySource.channelHandle}`;
  }
  return `Первоисточник: ${card.primarySource.channelHandle} • ещё ${count}`;
}

export function DigestView({
  feed,
  density,
  sourcesByPostId,
  reviewedCardIds,
  activeCardIndex = -1,
  isRefreshing = false,
  lastRefreshedAt = null,
  onMarkCardReviewed,
  onResetReviewed,
  onFilterCleanByCluster,
  onNavigateToCleanFeed,
  onOpenSources,
}: DigestViewProps) {
  const [expandedCardIds, setExpandedCardIds] = useState<Record<string, boolean>>({});

  const totalCards = feed.cards.length;
  const reviewedCount = useMemo(
    () => feed.cards.filter((c) => reviewedCardIds.has(c.clusterId)).length,
    [feed.cards, reviewedCardIds]
  );

  const isInboxZero = totalCards > 0 && reviewedCount >= totalCards;

  const toggleExpanded = (clusterId: string) => {
    setExpandedCardIds((prev) => ({ ...prev, [clusterId]: !prev[clusterId] }));
  };

  return (
    <section aria-label="Сводка новостей" className="stack">
      {isRefreshing ? (
        <div className="loading-banner" role="status">
          <span className="loading-spinner" aria-hidden="true" />
          <span>Обновляем посты и пересобираем сводку из подключённых каналов…</span>
        </div>
      ) : null}

      {totalCards === 0 && !isRefreshing ? (
        <div className="digest-card" role="status">
          <h2 className="typo-h3">По выбранным источникам пока нет сводки</h2>
          <p className="typo-meta" style={{ marginTop: '0.25rem' }}>
            Сбросьте фильтр в боковой панели или подключите новые каналы.
          </p>
          <div className="row" style={{ gap: '0.5rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
            {onOpenSources ? (
              <button type="button" className="btn-primary" onClick={onOpenSources}>
                Подключить каналы
              </button>
            ) : null}
            <button type="button" className="btn-secondary" onClick={onNavigateToCleanFeed}>
              Перейти в полную ленту
            </button>
          </div>
        </div>
      ) : null}

      {isInboxZero ? (
        <div className="inbox-zero-card" role="status">
          <h2 className="typo-h2">Вы всё прочитали</h2>
          <p className="typo-body" style={{ marginTop: '0.25rem' }}>
            Все {totalCards} тем текущего выпуска просмотрены. Новые публикации появятся после
            следующего обновления без отвлекающих счётчиков.
          </p>
          <div className="row" style={{ gap: '0.5rem', marginTop: '0.7rem', flexWrap: 'wrap' }}>
            <button type="button" className="btn-secondary" onClick={onNavigateToCleanFeed}>
              Открыть полную ленту постов
            </button>
            {onResetReviewed ? (
              <button type="button" className="btn-secondary" onClick={onResetReviewed}>
                Снять отметки
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="stack">
        {feed.cards.map((card, index) => {
          const isReviewed = reviewedCardIds.has(card.clusterId);
          const isActive = index === activeCardIndex;
          const isExpanded = expandedCardIds[card.clusterId] ?? density === 'comfortable';
          const dedupeLabel = buildDedupeLabel(card, card.otherChannels);

          return (
            <article
              key={card.clusterId}
              data-card-id={card.clusterId}
              className={`digest-card ${isReviewed ? 'digest-card-reviewed' : ''} ${
                isActive ? 'digest-card-active' : ''
              }`}
            >
              <div className="row-between" style={{ gap: '0.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: '220px' }}>
                  <div className="row" style={{ gap: '0.45rem', marginBottom: '0.3rem', flexWrap: 'wrap' }}>
                    {card.rubric && card.rubric.trim().toLowerCase() !== 'подписки' ? (
                      <span className="badge badge-neutral">{card.rubric}</span>
                    ) : null}
                    {card.groupingMode === 'channel_latest' && card.otherChannelsCount === 0 ? (
                      <span
                        className="icon-pill"
                        aria-label={`Последние посты · ${card.primarySource.channelHandle}`}
                        data-tooltip="Подборка последних публикаций этого канала"
                      >
                        <IconLayers size={12} />
                        <span>{card.primarySource.channelHandle}</span>
                      </span>
                    ) : (
                      <span className="badge badge-primary-source">
                        {dedupeLabel}
                      </span>
                    )}
                  </div>
                  <h2 className="typo-h3">{card.neutralTitle}</h2>
                </div>

                <div className="row" style={{ gap: '0.25rem' }}>
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={() => toggleExpanded(card.clusterId)}
                    aria-expanded={isExpanded}
                    aria-label={isExpanded ? 'Свернуть' : 'Детали'}
                    data-tooltip={isExpanded ? 'Свернуть детали' : 'Развернуть детали'}
                  >
                    {isExpanded ? <IconChevronUp size={16} /> : <IconChevronDown size={16} />}
                  </button>
                  <button
                    type="button"
                    className={isReviewed ? 'icon-btn icon-btn-active' : 'icon-btn-primary'}
                    onClick={() => onMarkCardReviewed(card.clusterId)}
                    aria-label={isReviewed ? 'Прочитано' : 'Отметить прочитанным'}
                    data-tooltip={isReviewed ? 'Снять отметку прочитанного' : 'Отметить тему прочитанной'}
                  >
                    {isReviewed ? <IconCheckDouble size={16} /> : <IconCheck size={16} />}
                  </button>
                </div>
              </div>

              {isExpanded ? (
                <div className="stack" style={{ marginTop: '0.65rem', gap: '0.5rem' }}>
                  {(() => {
                    let citationSeq = 0;
                    return card.summaryBullets.map((bullet, bIdx) => (
                      <div key={`${card.clusterId}-b-${bIdx}`} className="digest-bullet">
                        <p className="typo-body">
                          {bullet.text}{' '}
                          {bullet.citations.map((cit, cIdx) => {
                            citationSeq += 1;
                            const seq = citationSeq;
                            const source = sourcesByPostId[cit.postId];
                            return (
                              <span key={`${cit.postId}-${cIdx}`} style={{ marginLeft: '4px' }}>
                                <CitationHoverCard
                                  index={seq}
                                  citation={{
                                    postId: cit.postId,
                                    channelHandle: cit.channelHandle,
                                    quotePreview: cit.quoteText,
                                    utf16Start: cit.startUtf16,
                                    utf16End: cit.endUtf16,
                                  }}
                                  source={source}
                                  onOpenSource={() =>
                                    onFilterCleanByCluster(card.clusterId, card.neutralTitle)
                                  }
                                />
                              </span>
                            );
                          })}
                        </p>
                      </div>
                    ));
                  })()}

                  <div className="row-between" style={{ marginTop: '0.25rem', flexWrap: 'wrap', gap: '0.4rem' }}>
                    <span className="typo-meta tabular-nums">
                      Опубликовано: {formatRuDateTime(card.primarySource.publishedAt)}
                    </span>
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={() => onFilterCleanByCluster(card.clusterId, card.neutralTitle)}
                      aria-label="Все посты по теме →"
                      data-tooltip="Открыть все посты по теме в ленте"
                      style={{ color: 'var(--accent-primary)' }}
                    >
                      <IconArrowRight size={15} />
                    </button>
                  </div>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
