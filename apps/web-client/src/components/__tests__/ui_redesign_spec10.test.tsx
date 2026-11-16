import { afterEach, describe, expect, it } from 'bun:test';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import React from 'react';
import HomePage from '@/app/page';
import OfflineFallbackPage from '@/app/~offline/page';
import { CitationSyncProvider } from '@/components/ast/AstRenderer';
import { PostCard } from '@/components/feed/PostCard';
import { SEED_CLEAN_POSTS } from '@/lib/api/seed_data';
import {
  buildDomUtf16TextMap,
  fuzzyReanchorUtf16Span,
  resolveDomRangeForUtf16Span,
} from '@/lib/ast/css_highlights';
import {
  buildB2BExportBundle,
  canSharePostExternally,
  evaluateRagEntitlements,
} from '@/lib/entitlements';
import {
  decryptLocalVaultChunk,
  runLocalFts5Search,
  tokenizeDualFts5,
} from '@/lib/local-vault/fts5_and_crypto';
import type { ZeroTrustUserSession } from '@/lib/types/feed';
import {
  computeCompensatedScrollTop,
  createDirectionAwareMeasurer,
} from '@/lib/virtualization/scroll_stabilizer';

afterEach(() => {
  cleanup();
});

function makeSession(overrides: Partial<ZeroTrustUserSession> = {}): ZeroTrustUserSession {
  return {
    userId: 'u-spec10',
    email: 'spec10@workspace.local',
    authMethod: 'passkey',
    planTier: 'PROSUMER',
    finopsZone: 'GREEN',
    workspaceScope: 'personal',
    rbacRole: 'ANALYST',
    ragQuotaTotal: 30,
    ragQuotaUsed: 6,
    frontierQuotaRemaining: 3,
    localInferenceOnlyRf: false,
    crossBorderConsentAccepted: true,
    subscribedChannels: [],
    appliedPresetIds: [],
    desktopAgentConnected: false,
    agentStatus: 'offline_cache',
    desktopAgentEndpoint: 'http://127.0.0.1:48120',
    readClusterIds: [],
    ...overrides,
  };
}

describe('Section 10 UI/UX & Architecture Specification', () => {
  it('evaluates unified FinOps RAG quota, 80% warning, 30/30 FTS5 fallback, Yellow/Red/Guardrail zones', () => {
    const ok = evaluateRagEntitlements(makeSession({ planTier: 'BASIC', ragQuotaUsed: 10 }));
    expect(ok.quotaRemaining).toBe(20);
    expect(ok.isQuotaWarning).toBe(false);
    expect(ok.canRunCloudRag).toBe(true);

    const warn = evaluateRagEntitlements(makeSession({ planTier: 'BASIC', ragQuotaUsed: 25 }));
    expect(warn.isQuotaWarning).toBe(true);
    expect(warn.canRunCloudRag).toBe(true);

    const exhausted = evaluateRagEntitlements(makeSession({ planTier: 'BASIC', ragQuotaUsed: 30 }));
    expect(exhausted.isQuotaExhausted).toBe(true);
    expect(exhausted.canRunCloudRag).toBe(false);

    const yellow = evaluateRagEntitlements(makeSession({ finopsZone: 'YELLOW', frontierQuotaRemaining: 2 }));
    expect(yellow.showFrontierEscalation).toBe(true);
    expect(yellow.frontierQuotaRemaining).toBe(2);

    const red = evaluateRagEntitlements(makeSession({ finopsZone: 'RED' }));
    expect(red.isRedZoneFastOverview).toBe(true);
    expect(red.statusChipLabel).toContain('14 дн');
  });

  it('enforces B2B private channel share blocking and builds B2B export bundle', () => {
    const session = makeSession({ workspaceScope: 'team_b2b', rbacRole: 'ADMIN' });
    const privatePost = { ...SEED_CLEAN_POSTS[0], isPrivateChannel: true };
    const publicPost = { ...SEED_CLEAN_POSTS[0], isPrivateChannel: false };

    expect(canSharePostExternally(privatePost, session).allowed).toBe(false);
    expect(canSharePostExternally(publicPost, session).allowed).toBe(true);

    const bundle = buildB2BExportBundle({
      target: 'notion',
      workspaceName: 'Security Ops',
      generationSessionId: 'gen-1',
      rbacRole: 'ADMIN',
      verifiedSources: [
        {
          postId: 'post-bgp-42',
          channelHandle: '@infra_postmortem',
          publishedAt: '2026-11-15T08:15:00Z',
          quote: 'потеряно 14% трафика',
        },
      ],
    });
    expect(bundle.target).toBe('notion');
    expect(bundle.verifiedSources.length).toBe(1);
  });

  it('performs dual-tokenization local FTS5 search with UTF-16 spans and validates 24-byte nonce vault chunk', () => {
    const tokens = tokenizeDualFts5('PostgreSQL GIN_index bgp-route');
    expect(tokens).toContain('postgresql');
    expect(tokens).toContain('gin');

    const res = runLocalFts5Search('BGP трафика', SEED_CLEAN_POSTS);
    expect(res.matches.length).toBeGreaterThan(0);
    expect(res.matches[0].rag_highlights.length).toBeGreaterThan(0);

    const newsPosts = [
      {
        ...SEED_CLEAN_POSTS[0],
        id: 'news-front-1',
        postId: 2001,
        neutralHeadline: 'Сводка по ударам по тылам',
        text:
          'Война на Украине (на 13.11.26): Удары по тылам – мысли по поводу...\n' +
          'Сегодня ночью противник атаковал НПЗ. На фронте это скоро скажется.',
      },
      {
        ...SEED_CLEAN_POSTS[1],
        id: 'news-teaser-3',
        postId: 2003,
        neutralHeadline: 'Война на Украине: обстановка на фронтах...',
        text:
          'Война на Украине (на 13.11.26): Обстановка на фронтах...\n' +
          'Свежий обзор ситуации на всех участках фронта. Что происходит, изменения, перспективы...\n' +
          'Ссылка на видео в моей группе в ВК: https://vkvideo.ru/video-213276384_456243671',
      },
      {
        ...SEED_CLEAN_POSTS[1],
        id: 'news-unrelated-2',
        postId: 2002,
        neutralHeadline: 'Новости транспорта и кино',
        text: 'Никакие изменения тарифов на проезд по городу сегодня не планируются, есть новые маршруты, последние билеты в продаже.',
      },
    ];
    const newsRes = runLocalFts5Search('какие новости последние по фронту на украине?', newsPosts);
    expect(newsRes.matches.length).toBe(2);
    expect(newsRes.matches[0].post.id).toBe('news-front-1');
    expect(newsRes.matches[1].post.id).toBe('news-teaser-3');
    const highlightedWords = newsRes.matches[0].rag_highlights.map((h) =>
      'startUtf16' in h ? newsPosts[0].text.slice(h.startUtf16, h.endUtf16) : '',
    );
    expect(highlightedWords).toContain('Украине');
    expect(highlightedWords).toContain('фронте');
    expect(highlightedWords).not.toContain('на');
    expect(highlightedWords).not.toContain('по');

    expect(() =>
      decryptLocalVaultChunk(
        {
          opaqueAlias: 'ch_7f9a2b1c',
          ciphertextB64: 'AAAA',
          nonceHex: 'abcd',
        },
        new Map()
      )
    ).toThrow(/expected 24 bytes/i);
  });

  it('resolves DOM UTF-16 ranges and fuzzy-reanchors shifted spans', () => {
    const div = document.createElement('div');
    div.textContent = 'Сбой BGP: потеряно 14% трафика ЦОД.';
    const textMap = buildDomUtf16TextMap(div);
    const range = resolveDomRangeForUtf16Span(textMap, 10, 30, document);
    expect(range).not.toBeNull();
    expect(range?.toString()).toBe('потеряно 14% трафика');

    const reanchored = fuzzyReanchorUtf16Span(
      'Префикс! Сбой BGP: потеряно 14% трафика ЦОД.',
      'потеряно 14% трафика',
      10,
      30
    );
    expect(reanchored).not.toBeNull();
    expect('Префикс! Сбой BGP: потеряно 14% трафика ЦОД.'.slice(reanchored![0], reanchored![1])).toBe(
      'потеряно 14% трафика'
    );
  });

  it('stabilizes scroll measurements with direction-aware measurer and scrollTop compensation', () => {
    const measurer = createDirectionAwareMeasurer(220);
    expect(measurer.measureElement('row-1', { scrollHeight: 180 }, 'forward')).toBe(180);
    expect(measurer.measureElement('row-1', { scrollHeight: 240 }, 'backward')).toBe(180);

    const nextScroll = computeCompensatedScrollTop(1000, 380, 500);
    expect(nextScroll).toBe(120);
  });

  it('supports reversible AI headline toggle and private B2B lock in PostCard', () => {
    const privatePost = {
      ...SEED_CLEAN_POSTS[0],
      originalHeadline: 'ШОК! BGP упал во всей Европе!',
      isPrivateChannel: true,
    };

    render(
      React.createElement(
        CitationSyncProvider,
        null,
        React.createElement(PostCard, {
          post: privatePost,
          session: makeSession({ workspaceScope: 'team_b2b' }),
        })
      )
    );

    expect(screen.getByText(privatePost.neutralHeadline)).toBeTruthy();
    const toggleBtn = screen.getByRole('button', { name: 'Заголовок очищен' });
    fireEvent.click(toggleBtn);
    expect(screen.getByText('ШОК! BGP упал во всей Европе!')).toBeTruthy();
    expect(screen.getByText('Закрытый канал')).toBeTruthy();
    expect((screen.getByRole('button', { name: 'Экспорт ограничен' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('supports global keyboard shortcuts ([ sidebar, V mode toggle, F focus mode) and View popover in HomePage', () => {
    render(React.createElement(CitationSyncProvider, null, React.createElement(HomePage)));

    // Initially in digest mode with sidebar visible
    expect(screen.getByLabelText('Навигация и источники')).toBeTruthy();
    expect(screen.getByLabelText('Сводка новостей')).toBeTruthy();

    // Open View popover and toggle theme / density
    fireEvent.click(screen.getByTestId('view-options-btn'));
    expect(screen.getByRole('button', { name: 'Тёмная' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Компактная' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Тёмная' }));
    expect(document.documentElement.dataset.theme).toBe('dark');

    // Press V -> switches to Clean Originals
    fireEvent.keyDown(window, { key: 'v' });
    expect(screen.getByLabelText('Полная лента постов')).toBeTruthy();

    // Press F -> Focus Mode hides sidebar
    fireEvent.keyDown(window, { key: 'f' });
    expect(screen.queryByLabelText('Навигация и источники')).toBeNull();

    // Press Esc -> exits Focus Mode
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.getByLabelText('Навигация и источники')).toBeTruthy();

    // Press [ -> smoothly collapses sidebar in-place and shows floating expand button outside center header
    fireEvent.keyDown(window, { key: '[' });
    const sidebar = screen.getByTestId('left-sidebar');
    expect(sidebar.className).toContain('sidebar-left-collapsed');
    const expandBtn = screen.getByTestId('expand-sidebar-btn');
    expect(expandBtn.className).toContain('is-visible');
    expect(expandBtn.closest('.top-control-bar')).toBeNull();

    // Click floating expand button -> restores sidebar
    fireEvent.click(expandBtn);
    expect(sidebar.className).not.toContain('sidebar-left-collapsed');
  });

  it('renders desktop two-column split workspace on >=1100px with always-visible right search pane and no inspector', () => {
    const prevWidth = window.innerWidth;
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      writable: true,
      value: 1440,
    });

    try {
      const { container } = render(
        React.createElement(CitationSyncProvider, null, React.createElement(HomePage)),
      );

      // Left feed + right search pane are both visible simultaneously
      expect(screen.getByLabelText('Сводка новостей')).toBeTruthy();
      expect(screen.getByTestId('search-pane')).toBeTruthy();
      expect(container.querySelector('.workspace-split')).toBeTruthy();
      expect(screen.queryByTestId('ai-chat-dock')).toBeNull();
      expect(screen.queryByTestId('right-inspector')).toBeNull();

      // Search pane cannot be hidden via button or ] shortcut
      expect(screen.queryByTestId('toggle-search-pane-btn')).toBeNull();
      expect(screen.queryByLabelText('Скрыть панель поиска')).toBeNull();

      // ToggleViewHeader shows Сводка & Лента (no redundant Поиск tab while split pane is open)
      expect(screen.getByRole('button', { name: /Сводка \(/ })).toBeTruthy();
      expect(screen.getByRole('button', { name: /Лента \(/ })).toBeTruthy();
      const desktopTabs = container.querySelector('.desktop-mode-tabs');
      expect(desktopTabs?.textContent).not.toContain('Поиск');

      // Quota counter lives inside the AI mode toggle button, time window lives in footer near input
      const aiTab = screen.getByRole('button', { name: /ИИ-ответ/i });
      expect(aiTab.textContent).toContain('30/30');
      const footer = container.querySelector('.search-pane-footer');
      expect(footer?.querySelector('[aria-label="Период поиска"]')).toBeTruthy();

      // Switch right pane to keyword search ("По словам") and run FTS5 search without leaving Digest
      fireEvent.click(screen.getByRole('button', { name: 'По словам' }));
      const kwInput = screen.getByLabelText('Поисковый запрос') as HTMLInputElement;
      fireEvent.change(kwInput, { target: { value: 'BGP' } });
      fireEvent.submit(kwInput.closest('form')!);
      expect(screen.getByText(/Найдено совпадений:/)).toBeTruthy();
      expect(screen.getByLabelText('Сводка новостей')).toBeTruthy();
      expect(screen.queryByTestId('rag-query-bubble')).toBeNull();
      expect(kwInput.value).toBe('BGP');

      // Pressing ] does not collapse right search pane
      fireEvent.keyDown(window, { key: ']' });
      expect(screen.getByTestId('search-pane')).toBeTruthy();
      expect(screen.queryByTestId('ai-chat-dock')).toBeNull();
    } finally {
      Object.defineProperty(window, 'innerWidth', {
        configurable: true,
        writable: true,
        value: prevWidth,
      });
    }
  });

  it('renders PWA offline fallback page with local FTS5 search', () => {
    render(React.createElement(OfflineFallbackPage));
    expect(screen.getByText('Офлайн-режим PWA')).toBeTruthy();

    const input = screen.getByLabelText('Локальный поиск FTS5') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'BGP' } });
    fireEvent.submit(input.closest('form')!);
    expect(screen.getByText(/@infra_postmortem/i)).toBeTruthy();
  });
});
