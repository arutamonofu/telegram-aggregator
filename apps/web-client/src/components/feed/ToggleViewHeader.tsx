'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { DensityMode, ThemeMode, ViewMode } from '../../lib/types/feed';
import {
  IconFilter,
  IconRefresh,
  IconSliders,
} from '../ui/Icons';

export interface ToggleViewHeaderProps {
  mode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  density: DensityMode;
  onDensityChange: (density: DensityMode) => void;
  theme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
  focusMode: boolean;
  onToggleFocusMode: () => void;
  splitMode?: boolean;
  sidebarCollapsed?: boolean;
  onExpandSidebar?: () => void;
  onOpenMobileSidebar?: () => void;
  onOpenCommandPalette: () => void;
  onOpenSettings?: () => void;
  digestCount: number;
  cleanCount: number;
  activeClusterLabel?: string | null;
  onClearClusterFilter?: () => void;
  activeChannelLabel?: string | null;
  onClearChannelFilter?: () => void;
  isRefreshing?: boolean;
  lastRefreshedAt?: string | null;
  onRefreshFeed?: () => void;
}

const THEME_OPTIONS: readonly { id: ThemeMode; label: string }[] = [
  { id: 'light', label: 'Светлая' },
  { id: 'dark', label: 'Тёмная' },
  { id: 'sepia', label: 'Сепия' },
];

export function ToggleViewHeader({
  mode,
  onModeChange,
  density,
  onDensityChange,
  theme,
  onThemeChange,
  focusMode,
  onToggleFocusMode,
  splitMode = false,
  onOpenMobileSidebar,
  digestCount,
  cleanCount,
  activeClusterLabel,
  onClearClusterFilter,
  activeChannelLabel,
  onClearChannelFilter,
  isRefreshing = false,
  lastRefreshedAt = null,
  onRefreshFeed,
}: ToggleViewHeaderProps) {
  const [viewMenuOpen, setViewMenuOpen] = useState(false);
  const viewMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!viewMenuOpen) return;
    const handlePointerDown = (e: MouseEvent) => {
      if (viewMenuRef.current && !viewMenuRef.current.contains(e.target as Node)) {
        setViewMenuOpen(false);
      }
    };
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setViewMenuOpen(false);
      }
    };
    window.addEventListener('mousedown', handlePointerDown);
    window.addEventListener('keydown', handleEsc);
    return () => {
      window.removeEventListener('mousedown', handlePointerDown);
      window.removeEventListener('keydown', handleEsc);
    };
  }, [viewMenuOpen]);

  const showClusterBanner = Boolean(activeClusterLabel && mode === 'clean_originals');
  const showChannelBanner = Boolean(activeChannelLabel && mode !== 'search');

  return (
    <header className="top-control-bar" aria-label="Панель управления чтением">
      <div className="row-between" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
        <div className="row" style={{ gap: '0.45rem', flexWrap: 'wrap' }}>
          {onOpenMobileSidebar && !focusMode ? (
            <button
              type="button"
              className="btn-secondary mobile-sidebar-trigger"
              onClick={onOpenMobileSidebar}
              data-tooltip="Открыть фильтры и каналы"
              data-tooltip-pos="bottom"
              aria-label="Открыть фильтры и каналы"
            >
              <IconFilter size={15} />
            </button>
          ) : null}

          <nav className="segmented-tabs desktop-mode-tabs" aria-label="Переключение режима">
            <button
              type="button"
              className={mode === 'digest' ? 'active' : ''}
              onClick={() => onModeChange('digest')}
            >
              Сводка ({digestCount})
            </button>
            <button
              type="button"
              className={mode === 'clean_originals' ? 'active' : ''}
              onClick={() => onModeChange('clean_originals')}
              data-tooltip="Полная хронологическая лента (V)"
              data-tooltip-pos="bottom"
            >
              Лента ({cleanCount})
            </button>
            {!splitMode ? (
              <button
                type="button"
                className={mode === 'search' ? 'active' : ''}
                onClick={() => onModeChange('search')}
                data-tooltip="Поиск по постам и ответы по источникам (/)"
                data-tooltip-pos="bottom"
              >
                Поиск
              </button>
            ) : null}
          </nav>
        </div>

        <div className="row" style={{ gap: '0.35rem', flexWrap: 'wrap' }}>

          <button
            type="button"
            className="icon-btn-primary"
            onClick={onRefreshFeed}
            disabled={isRefreshing || !onRefreshFeed}
            data-testid="refresh-feed-btn"
            aria-label="Обновить ленту"
            data-tooltip={
              isRefreshing
                ? 'Обновление ленты…'
                : lastRefreshedAt
                  ? `Обновить (последнее: ${lastRefreshedAt})`
                  : 'Обновить ленту'
            }
            data-tooltip-pos="bottom"
          >
            <IconRefresh size={16} className={isRefreshing ? 'spin-icon' : ''} />
          </button>

          <div className="relative" ref={viewMenuRef}>
            <button
              type="button"
              className={viewMenuOpen || focusMode ? 'icon-btn icon-btn-active' : 'icon-btn'}
              onClick={() => setViewMenuOpen((prev) => !prev)}
              aria-expanded={viewMenuOpen}
              aria-haspopup="menu"
              aria-label="Вид"
              data-testid="view-options-btn"
              data-tooltip="Вид: тема, плотность и режим чтения"
              data-tooltip-pos="bottom"
            >
              <IconSliders size={16} />
            </button>

            {viewMenuOpen ? (
              <div className="view-popover stack" role="menu" style={{ gap: '0.65rem' }}>
                <div>
                  <div className="typo-meta" style={{ fontSize: '11px', marginBottom: '0.25rem' }}>
                    Тема оформления
                  </div>
                  <div className="segmented-tabs w-full" role="group" aria-label="Тема оформления">
                    {THEME_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        className={theme === opt.id ? 'active flex-1' : 'flex-1'}
                        onClick={() => onThemeChange(opt.id)}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="typo-meta" style={{ fontSize: '11px', marginBottom: '0.25rem' }}>
                    Вид карточек
                  </div>
                  <div className="segmented-tabs w-full" role="group" aria-label="Плотность ленты">
                    <button
                      type="button"
                      className={density === 'comfortable' ? 'active flex-1' : 'flex-1'}
                      onClick={() => onDensityChange('comfortable')}
                    >
                      Обычная
                    </button>
                    <button
                      type="button"
                      className={density === 'compact' ? 'active flex-1' : 'flex-1'}
                      onClick={() => onDensityChange('compact')}
                    >
                      Компактная
                    </button>
                  </div>
                  <div className="typo-meta" style={{ fontSize: '10.5px', marginTop: '0.25rem' }}>
                    {density === 'comfortable'
                      ? 'Полный текст и вложения без лишних заголовков'
                      : 'Краткие превью по 2 строки для быстрого просмотра'}
                  </div>
                </div>

                <button
                  type="button"
                  className={focusMode ? 'btn-primary w-full' : 'btn-secondary w-full'}
                  onClick={() => {
                    onToggleFocusMode();
                    setViewMenuOpen(false);
                  }}
                  aria-pressed={focusMode}
                >
                  {focusMode ? 'Показать боковые панели' : 'Режим чтения без панелей'}
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {showClusterBanner ? (
        <div
          className="row-between"
          style={{
            width: '100%',
            padding: '0.35rem 0.65rem',
            borderRadius: '8px',
            background: 'var(--bg-muted)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <span className="typo-meta">
            Фильтр по теме: <strong>{activeClusterLabel}</strong>
          </span>
          {onClearClusterFilter ? (
            <button type="button" className="btn-link" onClick={onClearClusterFilter}>
              Сбросить фильтр
            </button>
          ) : null}
        </div>
      ) : null}

      {showChannelBanner ? (
        <div
          className="row-between"
          style={{
            width: '100%',
            padding: '0.35rem 0.65rem',
            borderRadius: '8px',
            background: 'var(--bg-muted)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <span className="typo-meta">
            Канал: <strong>{activeChannelLabel}</strong>
          </span>
          {onClearChannelFilter ? (
            <button type="button" className="btn-link" onClick={onClearChannelFilter}>
              Все каналы
            </button>
          ) : null}
        </div>
      ) : null}
    </header>
  );
}
