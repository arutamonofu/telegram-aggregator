'use client';

import React, { useEffect, useState } from 'react';
import type { DensityMode, MediaAttachmentItem } from '../../lib/types/feed';
import { probeMediaRangeAvailability } from '../../lib/api/client';
import { IconChevronLeft, IconChevronRight, IconClose } from '../ui/Icons';

export interface AlbumGridProps {
  attachments: MediaAttachmentItem[];
  density?: DensityMode;
}

export function AlbumGrid({ attachments, density = 'comfortable' }: AlbumGridProps) {
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const [rangeWarning, setRangeWarning] = useState<string | null>(null);
  const [isProbingMedia, setIsProbingMedia] = useState<boolean>(false);

  useEffect(() => {
    if (lightboxIdx === null) {
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setLightboxIdx(null);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        setLightboxIdx((prev) =>
          prev === null ? null : (prev + 1) % attachments.length
        );
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        setLightboxIdx((prev) =>
          prev === null ? null : (prev - 1 + attachments.length) % attachments.length
        );
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [lightboxIdx, attachments.length]);

  if (attachments.length === 0) {
    return null;
  }

  const count = attachments.length;
  const gridClass =
    density === 'compact'
      ? 'album-grid album-grid-compact'
      : count === 1
      ? 'album-grid album-grid-1'
      : count <= 4
      ? 'album-grid album-grid-2x2'
      : 'album-grid album-grid-3xN';

  const openLightbox = async (index: number) => {
    setLightboxIdx(index);
    const item = attachments[index];
    if (item && (item.mediaType === 'video' || item.mediaType === 'document')) {
      setIsProbingMedia(true);
      setRangeWarning(null);
      try {
        const probe = await probeMediaRangeAvailability(item);
        if (probe.status === 'gone') {
          setRangeWarning(
            'Оригинал медиа удалён в источнике — показана сохранённая копия.'
          );
        } else if (probe.status === 'offline') {
          setRangeWarning(
            'Нет связи с медиа-сервером — показано локальное превью.'
          );
        } else {
          setRangeWarning(null);
        }
      } finally {
        setIsProbingMedia(false);
      }
    } else {
      setIsProbingMedia(false);
      setRangeWarning(null);
    }
  };

  const activeItem = lightboxIdx !== null ? attachments[lightboxIdx] : null;

  return (
    <div style={{ marginTop: '0.5rem' }}>
      <div className={gridClass} role="group" aria-label="Медиа-вложения">
        {attachments.map((item, idx) => {
          const aspectRatio =
            item.width && item.height && item.width > 0 && item.height > 0
              ? `${item.width} / ${item.height}`
              : '16 / 10';

          return (
            <button
              key={item.mediaId}
              type="button"
              className="album-tile"
              style={{ aspectRatio }}
              onClick={() => void openLightbox(idx)}
              aria-label={item.caption ?? `Открыть вложение ${idx + 1}`}
            >
              <div className="album-tile-placeholder" style={{ padding: 0 }}>
                {item.thumbnailDataUrl ? (
                  <img
                    src={item.thumbnailDataUrl}
                    alt={item.caption ?? `Вложение #${idx + 1}`}
                  />
                ) : (
                  <>
                    <span className="badge badge-neutral">
                      {item.mediaType === 'photo'
                        ? 'Фото'
                        : item.mediaType === 'video'
                          ? 'Видео'
                          : 'Файл'}
                    </span>
                    <span className="typo-meta" style={{ textAlign: 'center', padding: '0 8px' }}>
                      {item.caption ?? `Вложение #${idx + 1}`}
                    </span>
                  </>
                )}
                {item.fileGone ? (
                  <span
                    className="badge badge-unsupported"
                    title="Оригинал удалён автором, показана сохранённая копия"
                    style={{ position: 'absolute', bottom: '6px', right: '6px' }}
                  >
                    Из кэша
                  </span>
                ) : item.durationSec ? (
                  <span
                    className="badge badge-neutral tabular-nums"
                    style={{ position: 'absolute', bottom: '6px', right: '6px' }}
                  >
                    {item.durationSec}с
                  </span>
                ) : null}
              </div>
            </button>
          );
        })}
      </div>

      {activeItem ? (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label="Просмотр вложения"
          onClick={() => setLightboxIdx(null)}
        >
          <div
            className="modal-card"
            style={{ maxWidth: '640px' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="row-between" style={{ marginBottom: '0.5rem' }}>
              <strong className="typo-h3">
                {activeItem.caption ?? `Вложение ${(lightboxIdx ?? 0) + 1} из ${count}`}
              </strong>
              <button
                type="button"
                className="icon-btn"
                onClick={() => setLightboxIdx(null)}
                aria-label="Закрыть ✕"
                data-tooltip="Закрыть (Esc)"
                data-tooltip-pos="bottom"
              >
                <IconClose size={17} />
              </button>
            </div>

            <div className="album-lightbox-frame">
              {activeItem.thumbnailDataUrl ? (
                <img
                  src={activeItem.thumbnailDataUrl}
                  alt={activeItem.caption ?? `Вложение ${(lightboxIdx ?? 0) + 1}`}
                />
              ) : (
                <>
                  <span className="badge badge-neutral">
                    {activeItem.mediaType === 'photo'
                      ? 'Фото'
                      : activeItem.mediaType === 'video'
                        ? 'Видео'
                        : 'Файл'}
                  </span>
                  <p className="typo-meta">
                    {activeItem.caption ?? 'Предпросмотр недоступен в автономном режиме'}
                  </p>
                </>
              )}
              {activeItem.width && activeItem.height ? (
                <p className="typo-meta tabular-nums">
                  Размер: {activeItem.width}×{activeItem.height}
                  {activeItem.durationSec ? ` • ${activeItem.durationSec} с` : ''}
                </p>
              ) : null}
            </div>

            {isProbingMedia ? (
              <p className="typo-meta row" style={{ gap: '0.4rem', marginTop: '0.45rem' }} role="status">
                <span className="loading-spinner" aria-hidden="true" />
                <span>Проверяем доступность оригинала…</span>
              </p>
            ) : null}

            {rangeWarning ? (
              <p
                className="typo-meta"
                style={{ color: 'var(--verify-unsupported-accent)', marginTop: '0.45rem' }}
              >
                {rangeWarning}
              </p>
            ) : null}

            {count > 1 ? (
              <div className="row-between" style={{ marginTop: '0.75rem' }}>
                <button
                  type="button"
                  className="icon-btn"
                  onClick={() =>
                    setLightboxIdx(((lightboxIdx ?? 0) - 1 + count) % count)
                  }
                  aria-label="Предыдущее вложение"
                  data-tooltip="Предыдущее (←)"
                >
                  <IconChevronLeft size={18} />
                </button>
                <span className="typo-meta tabular-nums">
                  {(lightboxIdx ?? 0) + 1} / {count}
                </span>
                <button
                  type="button"
                  className="icon-btn"
                  onClick={() => setLightboxIdx(((lightboxIdx ?? 0) + 1) % count)}
                  aria-label="Следующее вложение"
                  data-tooltip="Следующее (→)"
                >
                  <IconChevronRight size={18} />
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
