'use client';

import React from 'react';
import type { ViewMode } from '@/lib/types/feed';

export interface MobileNavBarProps {
  activeMode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  onOpenSources: () => void;
}

export function MobileNavBar({
  activeMode,
  onModeChange,
  onOpenSources,
}: MobileNavBarProps) {
  return (
    <nav
      aria-label="Мобильная навигация"
      data-testid="mobile-tab-bar"
      className="surface-card fixed inset-x-0 bottom-0 z-40 flex items-center justify-around px-2 py-2 text-[11px] md:hidden"
      style={{ borderTop: '1px solid var(--border-subtle)' }}
    >
      <button
        type="button"
        onClick={() => onModeChange('digest')}
        className="rounded-lg px-2.5 py-1.5 font-medium"
        style={{
          color: activeMode === 'digest' ? 'var(--accent-primary)' : 'var(--text-secondary)',
          backgroundColor: activeMode === 'digest' ? 'var(--accent-primary-bg)' : 'transparent',
        }}
      >
        Сводка
      </button>
      <button
        type="button"
        onClick={() => onModeChange('clean_originals')}
        className="rounded-lg px-2.5 py-1.5 font-medium"
        style={{
          color:
            activeMode === 'clean_originals' ? 'var(--accent-primary)' : 'var(--text-secondary)',
          backgroundColor:
            activeMode === 'clean_originals' ? 'var(--accent-primary-bg)' : 'transparent',
        }}
      >
        Лента
      </button>
      <button
        type="button"
        onClick={() => onModeChange('search')}
        className="rounded-lg px-2.5 py-1.5 font-medium"
        style={{
          color: activeMode === 'search' ? 'var(--accent-primary)' : 'var(--text-secondary)',
          backgroundColor: activeMode === 'search' ? 'var(--accent-primary-bg)' : 'transparent',
        }}
      >
        Поиск
      </button>
      <button
        type="button"
        onClick={onOpenSources}
        className="rounded-lg px-2.5 py-1.5 font-medium"
        style={{ color: 'var(--text-secondary)' }}
      >
        Фильтры
      </button>
    </nav>
  );
}
