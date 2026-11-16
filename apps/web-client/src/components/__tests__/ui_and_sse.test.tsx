import { afterEach, describe, expect, it } from 'bun:test';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import HomePage from '@/app/page';
import { CitationSyncProvider } from '@/components/ast/AstRenderer';
import { DigestView } from '@/components/feed/DigestView';
import { PostCard } from '@/components/feed/PostCard';
import { TwoTierResults } from '@/components/rag/TwoTierResults';
import { parseTelegramChannelImportLines } from '@/lib/api/client';
import {
  INDUSTRY_PRESETS,
  SEED_CLEAN_POSTS,
  SEED_DEMO_RAG_CLAIMS,
  SEED_DEMO_STANCE_ROWS,
} from '@/lib/api/seed_data';
import {
  dispatchParsedSseFrame,
  IncrementalSseParser,
  streamRagQuery,
} from '@/lib/sse/rag_stream';
import type { RagStreamCallbacks } from '@/lib/sse/rag_stream';

afterEach(() => {
  cleanup();
});

describe('SSE Protocol Consumer (SPEC 7.5.3 / 5.4)', () => {
  it('parses chunked SSE frames, : keep-alive heartbeats, stance_row, and source_posts', async () => {
    const parser = new IncrementalSseParser();
    const frames: Array<{ event: string; data: string }> = [];
    const comments: string[] = [];

    parser.pushChunk(
      ': keep-alive\n\nevent: pipeline_stage\ndata: {"stage":"hybrid_qdrant"',
      (f) => frames.push(f),
      (c) => comments.push(c),
    );
    expect(comments).toEqual(['keep-alive']);
    expect(frames.length).toBe(0);

    parser.pushChunk(
      ',"message":"Qdrant Top-30"}\n\nevent: source_posts\ndata: {"source_posts":[]}\n\n',
      (f) => frames.push(f),
      (c) => comments.push(c),
    );
    expect(frames.length).toBe(2);
    expect(frames[0].event).toBe('pipeline_stage');
    expect(frames[1].event).toBe('source_posts');

    // Full ReadableStream end-to-end test via streamRagQuery
    const encoder = new TextEncoder();
    const ssePayload = [
      ': keep-alive\n\n',
      'event: pipeline_stage\ndata: {"stage":"nli","message":"NLI Verified"}\n\n',
      `event: verified_claim\ndata: ${JSON.stringify(SEED_DEMO_RAG_CLAIMS[0])}\n\n`,
      `event: stance_row\ndata: ${JSON.stringify(SEED_DEMO_STANCE_ROWS[0])}\n\n`,
      `event: source_posts\ndata: ${JSON.stringify({ source_posts: [{ post: SEED_CLEAN_POSTS[0], rag_highlights: [{ claimId: 'c1', startUtf16: 13, endUtf16: 33 }] }] })}\n\n`,
      'event: done\ndata: {"status":"completed","mode":"qa","verified_claims_count":1,"total_latency_ms":120}\n\n',
    ];

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const part of ssePayload) {
          controller.enqueue(encoder.encode(part));
        }
        controller.close();
      },
    });

    const mockFetch = async () =>
      new Response(stream, {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream; charset=utf-8',
          'Cache-Control': 'no-cache, no-transform',
          'X-Accel-Buffering': 'no',
        },
      });

    let keepAliveCount = 0;
    const receivedClaims: string[] = [];
    const receivedStances: string[] = [];
    let receivedPosts = 0;
    let doneStatus = '';

    await streamRagQuery(
      {
        user_id: 'u1',
        query: 'BGP',
        mode: 'qa',
        channel_ids: ['-1001001'],
      },
      {
        onKeepAlive: () => {
          keepAliveCount += 1;
        },
        onClaim: (c) => {
          receivedClaims.push(c.claim_id);
        },
        onStanceRow: (r) => {
          receivedStances.push(r.aspect);
        },
        onSourcePosts: (posts) => {
          receivedPosts = posts.length;
        },
        onDone: (meta) => {
          doneStatus = meta.status;
        },
      },
      { fetchImpl: mockFetch },
    );

    expect(keepAliveCount).toBe(1);
    expect(receivedClaims).toEqual(['c1']);
    expect(receivedStances.length).toBe(1);
    expect(receivedPosts).toBe(1);
    expect(doneStatus).toBe('completed');

    // Verify compatibility with SPEC `posts` key in addition to backend `source_posts`
    let specKeyCount = 0;
    dispatchParsedSseFrame(
      {
        event: 'source_posts',
        data: JSON.stringify({ posts: [{ post: SEED_CLEAN_POSTS[0], rag_highlights: [] }] }),
      },
      {
        onSourcePosts: (posts) => {
          specKeyCount = posts.length;
        },
      } satisfies RagStreamCallbacks,
    );
    expect(specKeyCount).toBe(1);

    // Verify backend snake_case payload normalization (utf16_range, verbatim_text, text_raw)
    let normalizedClaimQuote = '';
    let normalizedStart = -1;
    let normalizedPostText = '';
    dispatchParsedSseFrame(
      {
        event: 'verified_claim',
        data: JSON.stringify({
          claim_id: 'c9',
          topic_group: 'Вакансии',
          statement: 'Открыта позиция ML-инженера.',
          citations: [
            {
              post_id: 412,
              channel_id: '-100555',
              channel_title: 'Job Offer Inside',
              utf16_range: [5, 19],
              verbatim_text: 'ML-инженер удалённо',
            },
          ],
        }),
      },
      {
        onClaim: (c) => {
          normalizedClaimQuote = c.citations[0]?.quote ?? '';
          normalizedStart = c.citations[0]?.start_utf16 ?? -1;
        },
      },
    );
    dispatchParsedSseFrame(
      {
        event: 'source_posts',
        data: JSON.stringify({
          source_posts: [
            {
              post: {
                post_id: 412,
                channel_id: '-100555',
                channel_username: 'JobOfferInside',
                channel_title: 'Job Offer Inside',
                published_at: '2026-11-15T18:30:00Z',
                text_raw: 'Ищем ML-инженер удалённо',
              },
              rag_highlights: [{ claimId: 'c9', startUtf16: 5, endUtf16: 19 }],
            },
          ],
        }),
      },
      {
        onSourcePosts: (posts) => {
          normalizedPostText = posts[0]?.post.text ?? '';
        },
      },
    );
    expect(normalizedClaimQuote).toBe('ML-инженер удалённо');
    expect(normalizedStart).toBe(5);
    expect(normalizedPostText).toBe('Ищем ML-инженер удалённо');
  });
});

describe('Zero-Trust Onboarding, Channel/Addlist Parser & Toggle View UI (SPEC 5.1, 5.2)', () => {
  it('parses t.me links, @usernames, and t.me/addlist folders while rejecting private invite links', () => {
    const results = parseTelegramChannelImportLines(
      'https://t.me/s/infra_postmortem/42\n@llm_systems_ru\nhttps://t.me/addlist/eng_2026\nhttps://t.me/+PrivateInvite123',
    );
    expect(results.length).toBe(4);
    expect(results[0]).toMatchObject({ valid: true, kind: 'channel', normalizedHandle: '@infra_postmortem' });
    expect(results[1]).toMatchObject({ valid: true, kind: 'channel', normalizedHandle: '@llm_systems_ru' });
    expect(results[2].valid).toBe(true);
    expect(results[2].kind).toBe('addlist');
    expect(results[2].resolvedChannels.length).toBeGreaterThan(0);
    expect(results[3].valid).toBe(false);
    expect(results[3].errorReason).toContain('Desktop Agent');
  });

  it('switches between AI Digest and Clean Originals, opens Repost Tree & Lightbox, and opens OnboardingModal without tel inputs', async () => {
    const { container } = render(
      React.createElement(CitationSyncProvider, null, React.createElement(HomePage)),
    );

    // Zero unread badges or messenger chat counters
    expect(container.querySelector('[data-unread-count]')).toBeNull();
    expect(screen.queryByText(/непрочитанных сообщений/i)).toBeNull();

    // Starts in Digest mode without the legacy "Сводка новостей" header banner
    expect(screen.getByLabelText('Сводка новостей')).toBeTruthy();
    expect(screen.queryByText('Сводка новостей')).toBeNull();

    // Primary source badge is rendered and right inspector is not present
    expect(screen.getAllByText(/Первоисточник:/i).length).toBeGreaterThan(0);
    expect(screen.queryByLabelText('Правая панель-инспектор')).toBeNull();

    // Switch to Clean Originals via cluster deeplink
    const jumpBtns = screen.getAllByRole('button', { name: 'Все посты по теме →' });
    fireEvent.click(jumpBtns[0]);
    expect(screen.getByLabelText('Полная лента постов')).toBeTruthy();

    // Grouped album grid + lightbox renders visual image preview
    const clearFilterBtn = screen.getByRole('button', { name: 'Сбросить фильтр' });
    fireEvent.click(clearFilterBtn);
    const albumThumbs = screen.getAllByLabelText(/Открыть вложение|Демонстрация|grouped_id/i);
    expect(albumThumbs.length).toBeGreaterThan(0);
    fireEvent.click(albumThumbs[0]);
    const lightboxDialog = screen.getByRole('dialog', { name: 'Просмотр вложения' });
    expect(lightboxDialog).toBeTruthy();
    expect(lightboxDialog.querySelector('img')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Закрыть ✕' }));

    // Central ToggleViewHeader no longer duplicates LeftSidebar settings button
    expect(screen.queryByRole('button', { name: 'Настройки' })).toBeNull();

    // Open "Каналы и папки" modal via Папки button in LeftSidebar (5 presets, no account tab)
    fireEvent.click(screen.getByRole('button', { name: 'Папки' }));
    const channelsModal = screen.getByRole('dialog', { name: 'Каналы и папки' });
    expect(channelsModal).toBeTruthy();
    expect(screen.queryByRole('tab', { name: 'Аккаунт и доступ' })).toBeNull();
    expect(container.querySelector('input[type="tel"]')).toBeNull();
    expect(INDUSTRY_PRESETS.length).toBe(5);
    for (const preset of INDUSTRY_PRESETS) {
      expect(channelsModal.textContent).toContain(preset.title);
    }
    fireEvent.click(screen.getByRole('button', { name: 'Закрыть ✕' }));

    // Open "Настройки аккаунта" modal via bottom-left Account button in LeftSidebar
    fireEvent.click(screen.getByTestId('account-settings-btn'));
    const accountModal = screen.getByRole('dialog', { name: 'Настройки аккаунта' });
    expect(accountModal).toBeTruthy();
    expect(screen.queryByRole('tab', { name: 'Папки и подборки' })).toBeNull();
    expect(accountModal.textContent).toContain('Вход в аккаунт (без номера телефона)');
    expect(container.querySelector('input[type="tel"]')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Закрыть ✕' }));

    // Clicking + Добавить in LeftSidebar opens "Каналы и папки" directly on "Добавить каналы" tab
    fireEvent.click(screen.getByRole('button', { name: '+ Добавить' }));
    const addTab = screen.getByRole('tab', { name: 'Добавить каналы' });
    expect(addTab.getAttribute('aria-selected')).toBe('true');
  });

  it('renders Two-Tier RAG results, switches to local FTS5 search, and shows stance matrix', async () => {
    const { container } = render(
      React.createElement(
        CitationSyncProvider,
        null,
        React.createElement(TwoTierResults, {
          session: {
            userId: 'demo_user',
            email: 'demo@workspace.local',
            authMethod: 'passkey',
            planTier: 'PROSUMER',
            finopsZone: 'GREEN',
            workspaceScope: 'personal',
            rbacRole: 'ANALYST',
            ragQuotaTotal: 30,
            ragQuotaUsed: 4,
            frontierQuotaRemaining: 3,
            localInferenceOnlyRf: false,
            crossBorderConsentAccepted: true,
            subscribedChannels: [],
            appliedPresetIds: [],
            desktopAgentConnected: false,
            agentStatus: 'offline_cache',
            desktopAgentEndpoint: 'http://127.0.0.1:48120',
            readClusterIds: [],
          },
          density: 'comfortable',
        }),
      ),
    );

    expect(screen.getByText('Поиск по вашим каналам')).toBeTruthy();
    expect(screen.queryByText('Аргументы «ЗА» и «ПРОТИВ»')).toBeNull();
    expect(screen.queryByText('Ответ с опорой на посты')).toBeNull();

    // Switch to local FTS5 search
    fireEvent.click(screen.getByRole('button', { name: 'Поиск по тексту (бесплатно)' }));
    const searchInput = screen.getByLabelText('Поисковый запрос') as HTMLInputElement;
    fireEvent.change(searchInput, { target: { value: 'BGP' } });
    fireEvent.submit(searchInput.closest('form')!);

    await waitFor(() => {
      expect(screen.getByText('Поиск по сохранённым постам')).toBeTruthy();
      expect(container.textContent).toContain('Найдено совпадений: 1');
    });
  });

  it('hides duplicate headline on live Telegram posts and collapses body in compact mode', () => {
    const livePost = {
      ...SEED_CLEAN_POSTS[0],
      id: 'tg-job-1',
      originalHeadline: undefined,
      neutralHeadline: 'Ищем ML-инженера на удалёнку',
      text: 'Ищем ML-инженера на удалёнку\n\nЗарплата от 300к, стек PyTorch и vLLM.',
    };

    const { rerender } = render(
      React.createElement(
        CitationSyncProvider,
        null,
        React.createElement(PostCard, {
          post: livePost,
          session: {
            userId: 'u1',
            email: 'u@local',
            authMethod: 'passkey',
            planTier: 'PROSUMER',
            finopsZone: 'GREEN',
            workspaceScope: 'personal',
            rbacRole: 'ANALYST',
            ragQuotaTotal: 30,
            ragQuotaUsed: 0,
            frontierQuotaRemaining: 3,
            localInferenceOnlyRf: false,
            crossBorderConsentAccepted: true,
            subscribedChannels: [],
            appliedPresetIds: [],
            desktopAgentConnected: false,
            agentStatus: 'offline_cache',
            desktopAgentEndpoint: 'http://127.0.0.1:48120',
            readClusterIds: [],
          },
          density: 'comfortable',
        }),
      ),
    );

    expect(screen.queryByRole('heading', { level: 2 })).toBeNull();

    rerender(
      React.createElement(
        CitationSyncProvider,
        null,
        React.createElement(PostCard, {
          post: livePost,
          session: {
            userId: 'u1',
            email: 'u@local',
            authMethod: 'passkey',
            planTier: 'PROSUMER',
            finopsZone: 'GREEN',
            workspaceScope: 'personal',
            rbacRole: 'ANALYST',
            ragQuotaTotal: 30,
            ragQuotaUsed: 0,
            frontierQuotaRemaining: 3,
            localInferenceOnlyRf: false,
            crossBorderConsentAccepted: true,
            subscribedChannels: [],
            appliedPresetIds: [],
            desktopAgentConnected: false,
            agentStatus: 'offline_cache',
            desktopAgentEndpoint: 'http://127.0.0.1:48120',
            readClusterIds: [],
          },
          density: 'compact',
        }),
      ),
    );

    expect(screen.getByRole('button', { name: 'Развернуть пост' })).toBeTruthy();
  });

  it('shows channel_latest badge and loading banner in DigestView when refreshing', () => {
    render(
      React.createElement(
        CitationSyncProvider,
        null,
        React.createElement(DigestView, {
          feed: {
            mode: 'digest',
            user_id: 'u1',
            finops_zone: 'GREEN',
            reduce_model: 'local',
            total_candidate_clusters: 1,
            selected_cards_count: 1,
            rubrics: ['Подписки'],
            cards: [
              {
                clusterId: 'c-live-1',
                rubric: 'Подписки',
                neutralTitle: 'Job Offer Inside: Вакансия ML',
                summaryBullets: [],
                primarySource: {
                  channelId: 'ch_job',
                  channelTitle: 'Job Offer Inside',
                  channelHandle: '@JobOfferInside',
                  originScore: 0.9,
                  postId: '10',
                  publishedAt: '2026-11-15T10:00:00Z',
                },
                otherChannelsCount: 0,
                otherChannels: [],
                importanceScore: 0.85,
                groupingMode: 'channel_latest',
              },
            ],
          },
          density: 'comfortable',
          sourcesByPostId: {},
          reviewedCardIds: new Set<string>(),
          isRefreshing: true,
          onMarkCardReviewed: () => {},
          onFilterCleanByCluster: () => {},
          onNavigateToCleanFeed: () => {},
        }),
      ),
    );

    expect(screen.getByLabelText('Последние посты · @JobOfferInside')).toBeTruthy();
    expect(screen.queryByText('Подписки')).toBeNull();
    expect(screen.getByText(/Обновляем посты и пересобираем сводку/)).toBeTruthy();
  });

  it('shows refusal card (not empty boxes) when backend SSE returns no_verified_data and no local match exists', async () => {
    const origFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(
        'event: pipeline_stage\ndata: {"stage":"routing"}\n\nevent: source_posts\ndata: {"source_posts":[]}\n\nevent: done\ndata: {"status":"no_verified_data","rows_count":0}\n\n',
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
      )) as unknown as typeof fetch;

    try {
      render(
        React.createElement(
          CitationSyncProvider,
          null,
          React.createElement(TwoTierResults, {
            session: {
              userId: 'demo_user',
              email: 'demo@workspace.local',
              authMethod: 'passkey',
              planTier: 'PROSUMER',
              finopsZone: 'GREEN',
              workspaceScope: 'personal',
              rbacRole: 'ANALYST',
              ragQuotaTotal: 30,
              ragQuotaUsed: 4,
              frontierQuotaRemaining: 3,
              localInferenceOnlyRf: false,
              crossBorderConsentAccepted: true,
              subscribedChannels: [],
              appliedPresetIds: [],
              desktopAgentConnected: false,
              agentStatus: 'offline_cache',
              desktopAgentEndpoint: 'http://127.0.0.1:48120',
              readClusterIds: [],
            },
            density: 'comfortable',
            availablePosts: [],
          }),
        ),
      );

      const searchInput = screen.getByLabelText('Поисковый запрос') as HTMLInputElement;
      fireEvent.change(searchInput, { target: { value: 'несуществующий_запрос_xyz' } });
      fireEvent.submit(searchInput.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText('В ваших каналах нет точных данных по запросу')).toBeTruthy();
      });
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('allows unsubscribing from channels in LeftSidebar', async () => {
    render(
      React.createElement(CitationSyncProvider, null, React.createElement(HomePage)),
    );

    const unsubBtn = screen.getByRole('button', { name: 'Отписаться от @infra_postmortem' });
    fireEvent.click(unsubBtn);
    expect(screen.queryByRole('button', { name: 'Отписаться от @infra_postmortem' })).toBeNull();
  });

  it('routes queries from the bottom AI chat island into Search with consistent reading-container width', async () => {
    const origFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(
        'event: source_posts\ndata: {"source_posts":[]}\n\nevent: done\ndata: {"status":"completed"}\n\n',
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
      )) as unknown as typeof fetch;

    try {
      const { container } = render(
        React.createElement(CitationSyncProvider, null, React.createElement(HomePage)),
      );

      expect(screen.getByTestId('ai-chat-dock')).toBeTruthy();
      const dockInput = screen.getByLabelText('Запрос для ИИ-ответа по источникам') as HTMLInputElement;
      fireEvent.change(dockInput, { target: { value: 'BGP' } });
      fireEvent.submit(dockInput.closest('form')!);

      await waitFor(() => {
        expect(screen.getByLabelText('Поиск и анализ по источникам')).toBeTruthy();
      });

      expect(container.querySelector('.reading-container')).toBeTruthy();
      expect(container.querySelector('.reading-container-wide')).toBeNull();
      expect(screen.queryByTestId('ai-chat-dock')).toBeNull();
      const searchInput = screen.getByLabelText('Поисковый запрос') as HTMLInputElement;
      expect(searchInput.value).toBe('BGP');
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('displays portal tooltip without native title duplication and renders without right inspector', async () => {
    const { container } = render(
      React.createElement(CitationSyncProvider, null, React.createElement(HomePage)),
    );
    expect(screen.queryByTestId('right-inspector')).toBeNull();
    const dualTooltipElements = container.querySelectorAll('[data-tooltip][title]');
    expect(dualTooltipElements.length).toBe(0);

    const refreshBtn = screen.getByTestId('refresh-feed-btn');
    fireEvent.mouseOver(refreshBtn);
    await waitFor(() => {
      const portalTooltip = document.body.querySelector('[role="tooltip"]');
      expect(portalTooltip).toBeTruthy();
      expect(portalTooltip?.textContent).toMatch(/Обнов(ить|ление)/);
    });
  });

  it('numbers citation badges by actual source card position and resolves live post ids on click/hover', async () => {
    const livePosts = [
      {
        ...SEED_CLEAN_POSTS[0],
        id: 'tg-chan1-101',
        postId: '101',
        channelId: 'ch_1',
        channelTitle: 'Канал 1',
        channelHandle: '@chan1',
        text: 'Первый источник про логистику и мосты.',
      },
      {
        ...SEED_CLEAN_POSTS[1],
        id: 'tg-chan2-777',
        postId: '777',
        channelId: 'ch_2',
        channelTitle: 'Канал 2',
        channelHandle: '@chan2',
        text: 'Второй источник: открыт новый дата-центр в Новосибирске.',
      },
    ];

    const ssePayload = [
      'event: pipeline_stage\ndata: {"stage":"nli_verification"}\n\n',
      `event: verified_claim\ndata: ${JSON.stringify({
        claim_id: 'c1',
        topic_group: 'Инфра',
        statement: 'Первый факт про мосты.',
        citations: [
          {
            post_id: 101,
            channel_id: 'ch_1',
            channel_title: 'Канал 1',
            utf16_range: [0, 38],
            verbatim_text: 'Первый источник про логистику и мосты.',
          },
        ],
      })}\n\n`,
      `event: verified_claim\ndata: ${JSON.stringify({
        claim_id: 'c2',
        topic_group: 'ЦОД',
        statement: 'Открыт новый дата-центр в Новосибирске.',
        citations: [
          {
            post_id: 777,
            channel_id: 'ch_2',
            channel_title: 'Канал 2',
            utf16_range: [17, 57],
            verbatim_text: 'открыт новый дата-центр в Новосибирске.',
          },
        ],
      })}\n\n`,
      `event: source_posts\ndata: ${JSON.stringify({
        source_posts: [
          {
            post: {
              post_id: 101,
              channel_id: 'ch_1',
              channel_username: 'chan1',
              channel_title: 'Канал 1',
              published_at: '2026-11-15T10:00:00Z',
              text_raw: 'Первый источник про логистику и мосты.',
            },
            rag_highlights: [{ claimId: 'c1', startUtf16: 0, endUtf16: 38 }],
          },
          {
            post: {
              post_id: 777,
              channel_id: 'ch_2',
              channel_username: 'chan2',
              channel_title: 'Канал 2',
              published_at: '2026-11-15T11:00:00Z',
              text_raw: 'Второй источник: открыт новый дата-центр в Новосибирске.',
            },
            rag_highlights: [{ claimId: 'c2', startUtf16: 17, endUtf16: 57 }],
          },
        ],
      })}\n\n`,
      'event: done\ndata: {"status":"ok","mode":"qa","verified_claims_count":2}\n\n',
    ].join('');

    const origFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(ssePayload, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })) as unknown as typeof fetch;

    try {
      const { container } = render(
        React.createElement(
          CitationSyncProvider,
          null,
          React.createElement(TwoTierResults, {
            session: {
              userId: 'demo_user',
              email: 'demo@workspace.local',
              authMethod: 'passkey',
              planTier: 'PROSUMER',
              finopsZone: 'GREEN',
              workspaceScope: 'personal',
              rbacRole: 'ANALYST',
              ragQuotaTotal: 30,
              ragQuotaUsed: 4,
              frontierQuotaRemaining: 3,
              localInferenceOnlyRf: false,
              crossBorderConsentAccepted: true,
              subscribedChannels: [],
              appliedPresetIds: [],
              desktopAgentConnected: false,
              agentStatus: 'offline_cache',
              desktopAgentEndpoint: 'http://127.0.0.1:48120',
              readClusterIds: [],
            },
            density: 'comfortable',
            availablePosts: livePosts,
          }),
        ),
      );

      const searchInput = screen.getByLabelText('Поисковый запрос') as HTMLInputElement;
      fireEvent.change(searchInput, { target: { value: 'новости дата-центр' } });
      fireEvent.submit(searchInput.closest('form')!);

      const secondCiteBtn = await waitFor(() =>
        screen.getByRole('button', { name: /Сноска \[2\]/ }),
      );
      expect(secondCiteBtn.textContent?.trim()).toBe('[2]');

      fireEvent.mouseEnter(secondCiteBtn);
      expect(screen.getByRole('tooltip').textContent).toContain('открыт новый дата-центр в Новосибирске');

      fireEvent.click(secondCiteBtn);
      const sourceCards = container.querySelectorAll('article.rag-source-card');
      expect(sourceCards.length).toBe(2);
      expect(sourceCards[1].className).toContain('rag-source-active');
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('parses event: direct_answer, renders indirect direct-answer-card, and deduplicates duplicate post badges', async () => {
    const livePosts = [
      {
        ...SEED_CLEAN_POSTS[0],
        id: 'tg-mig-1',
        postId: '1',
        channelId: 'ch_mig',
        channelTitle: 'МИГ России',
        channelHandle: '@mig41',
        text: 'В Одесской области был атакован порт Черноморск. Уничтожены два локомотива.',
      },
    ];

    const ssePayload = [
      'event: pipeline_stage\ndata: {"stage":"nli_verification"}\n\n',
      `event: direct_answer\ndata: ${JSON.stringify({
        status: 'indirect',
        text: 'Прямого подтверждения этому в каналах нет. Сообщается об атаке на порт Черноморск [1].',
        citations: [
          {
            post_id: 1,
            channel_id: 'ch_mig',
            channel_title: 'МИГ России',
            utf16_range: [0, 48],
            verbatim_text: 'В Одесской области был атакован порт Черноморск.',
          },
          {
            post_id: 1,
            channel_id: 'ch_mig',
            channel_title: 'МИГ России',
            utf16_range: [49, 75],
            verbatim_text: 'Уничтожены два локомотива.',
          },
        ],
      })}\n\n`,
      `event: verified_claim\ndata: ${JSON.stringify({
        claim_id: 'c1',
        topic_group: 'События',
        statement: 'В Одесской области был атакован порт Черноморск.',
        citations: [
          {
            post_id: 1,
            channel_id: 'ch_mig',
            channel_title: 'МИГ России',
            utf16_range: [0, 48],
            verbatim_text: 'В Одесской области был атакован порт Черноморск.',
          },
          {
            post_id: 1,
            channel_id: 'ch_mig',
            channel_title: 'МИГ России',
            utf16_range: [49, 75],
            verbatim_text: 'Уничтожены два локомотива.',
          },
        ],
      })}\n\n`,
      `event: source_posts\ndata: ${JSON.stringify({
        source_posts: [
          {
            post: {
              post_id: 1,
              channel_id: 'ch_mig',
              channel_username: 'mig41',
              channel_title: 'МИГ России',
              published_at: '2026-11-15T12:28:00Z',
              text_raw: 'В Одесской области был атакован порт Черноморск. Уничтожены два локомотива.',
            },
            rag_highlights: [{ claimId: 'c1', startUtf16: 0, endUtf16: 48 }],
          },
        ],
      })}\n\n`,
      'event: done\ndata: {"status":"ok","mode":"qa","answer_status":"indirect","verified_claims_count":1}\n\n',
    ].join('');

    const origFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(ssePayload, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })) as unknown as typeof fetch;

    try {
      render(
        React.createElement(
          CitationSyncProvider,
          null,
          React.createElement(TwoTierResults, {
            session: {
              userId: 'demo_user',
              email: 'demo@workspace.local',
              authMethod: 'passkey',
              planTier: 'PROSUMER',
              finopsZone: 'GREEN',
              workspaceScope: 'personal',
              rbacRole: 'ANALYST',
              ragQuotaTotal: 30,
              ragQuotaUsed: 4,
              frontierQuotaRemaining: 3,
              localInferenceOnlyRf: false,
              crossBorderConsentAccepted: true,
              subscribedChannels: [],
              appliedPresetIds: [],
              desktopAgentConnected: false,
              agentStatus: 'offline_cache',
              desktopAgentEndpoint: 'http://127.0.0.1:48120',
              readClusterIds: [],
            },
            density: 'comfortable',
            availablePosts: livePosts,
          }),
        ),
      );

      const searchInput = screen.getByLabelText('Поисковый запрос') as HTMLInputElement;
      fireEvent.change(searchInput, { target: { value: 'украина сдалась в войне?' } });
      fireEvent.submit(searchInput.closest('form')!);

      const card = await waitFor(() => screen.getByTestId('direct-answer-card'));
      expect(card.textContent).toContain('Прямого подтверждения в каналах нет');
      expect(card.textContent).toContain('Прямого подтверждения этому в каналах нет.');

      // Дублирующиеся цитаты из одного поста схлопнуты в один бейдж [1] внутри карточки
      const cardBadges = card.querySelectorAll('button.citation-badge');
      expect(cardBadges.length).toBe(1);
    } finally {
      globalThis.fetch = origFetch;
    }
  });
});
