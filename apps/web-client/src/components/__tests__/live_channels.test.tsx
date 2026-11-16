import { afterEach, describe, expect, it } from 'bun:test';
import http from 'node:http';
import net from 'node:net';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import React from 'react';
import { ToggleViewHeader } from '@/components/feed/ToggleViewHeader';
import { LeftSidebar } from '@/components/layout/LeftSidebar';
import {
  clearClientLiveChannelCache,
  parseTelegramChannelImportLines,
  syncAllChannelFeeds,
} from '@/lib/api/client';
import {
  clearLiveFetcherCache,
  fetchLiveChannelsBatch,
  formatLiveFetchError,
} from '@/lib/live/live_fetcher';
import {
  buildDigestCardsFromCleanPosts,
  extractTelegramChannelTitle,
  parseRssOrAtomXml,
  parseTelegramPublicHtml,
} from '@/lib/live/tme_public_parser';

afterEach(() => {
  cleanup();
  clearLiveFetcherCache();
  clearClientLiveChannelCache();
});

const SAMPLE_TME_HTML = `
<div class="tgme_channel_info_header_title"><span>Job Offer Inside</span></div>
<div class="tgme_widget_message_wrap" data-post="JobOfferInside/412">
  <div class="tgme_widget_message_text js-message_text">
    <b>Senior Go Developer (Remote)</b><br>Вакансия в финтех: <a href="https://example.com/jobs/412">откликнуться</a> #golang @hr_team
  </div>
  <a class="tgme_widget_message_photo_wrap" style="background-image:url('https://cdn.example.org/photo412.jpg')"></a>
  <time datetime="2026-11-15T18:30:00+00:00">18:30</time>
</div>
`;

const SAMPLE_RSS_XML = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Engineering Feed</title>
    <item>
      <title>PostgreSQL 18 JSONB Benchmarks</title>
      <description>Новые тесты производительности GIN-индексов.</description>
      <link>https://eng.example.com/pg18</link>
      <pubDate>Sun, 15 Nov 2026 15:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>`;

describe('Live Public Channel & RSS Sync + Manual Refresh', () => {
  it('parses public t.me/s/ HTML into CleanPostItem[] with UTF-16 entities and media', () => {
    const items = parseTelegramPublicHtml(SAMPLE_TME_HTML, {
      channelId: 'ch_jobofferinside',
      handle: '@JobOfferInside',
    });

    expect(items.length).toBe(1);
    const post = items[0];
    expect(post.id).toBe('tg-JobOfferInside-412');
    expect(post.channelTitle).toBe('Job Offer Inside');
    expect(post.channelHandle).toBe('@JobOfferInside');
    expect(post.neutralHeadline).toContain('Senior Go Developer');
    expect(post.media.length).toBe(1);
    expect(post.media[0].thumbnailDataUrl).toBe('https://cdn.example.org/photo412.jpg');

    const linkEnt = post.entities.find((e) => e.type === 'text_link');
    expect(linkEnt).toBeTruthy();
    expect(post.text.slice(linkEnt!.offset, linkEnt!.offset + linkEnt!.length)).toBe('откликнуться');
  });

  it('parses RSS XML feeds and builds DigestCard[] summaries', () => {
    const rssItems = parseRssOrAtomXml(SAMPLE_RSS_XML, {
      channelId: 'rss_eng',
      handle: 'https://eng.example.com/rss.xml',
    });

    expect(rssItems.length).toBe(1);
    expect(rssItems[0].sourcePlatform).toBe('RSS');
    expect(rssItems[0].neutralHeadline).toBe('PostgreSQL 18 JSONB Benchmarks');

    const cards = buildDigestCardsFromCleanPosts(rssItems);
    expect(cards.length).toBe(1);
    expect(cards[0].primarySource.channelId).toBe('rss_eng');
    expect(cards[0].summaryBullets[0].citations[0].postId).toBe(rssItems[0].id);
  });

  it('accepts RSS URLs in parseTelegramChannelImportLines alongside @channel', () => {
    const parsed = parseTelegramChannelImportLines(
      '@JobOfferInside\nhttps://habr.com/ru/rss/all/all/'
    );
    expect(parsed.length).toBe(2);
    expect(parsed[0].valid).toBe(true);
    expect(parsed[0].normalizedHandle).toBe('@JobOfferInside');
    expect(parsed[1].valid).toBe(true);
    expect(parsed[1].resolvedChannels[0].platform).toBe('RSS');
  });

  it('renders «Обновить ленту» button in ToggleViewHeader and triggers onRefreshFeed', () => {
    let refreshed = false;
    render(
      React.createElement(ToggleViewHeader, {
        mode: 'digest',
        onModeChange: () => {},
        density: 'comfortable',
        onDensityChange: () => {},
        theme: 'light',
        onThemeChange: () => {},
        focusMode: false,
        onToggleFocusMode: () => {},
        sidebarCollapsed: false,
        onExpandSidebar: () => {},
        onOpenCommandPalette: () => {},
        onOpenSettings: () => {},
        digestCount: 2,
        cleanCount: 4,
        lastRefreshedAt: '01:30:00',
        onRefreshFeed: () => {
          refreshed = true;
        },
      })
    );

    const btn = screen.getByTestId('refresh-feed-btn');
    expect(btn.getAttribute('aria-label')).toBe('Обновить ленту');
    expect(btn.getAttribute('data-tooltip')).toContain('01:30:00');
    fireEvent.click(btn);
    expect(refreshed).toBe(true);
  });

  it('extracts real Telegram channel title and avoids duplicating handle/title in LeftSidebar', () => {
    const resolvedTitle = extractTelegramChannelTitle(SAMPLE_TME_HTML);
    expect(resolvedTitle).toBe('Job Offer Inside');

    const { rerender } = render(
      React.createElement(LeftSidebar, {
        collapsed: false,
        onToggleCollapse: () => {},
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
          subscribedChannels: [
            {
              channelId: 'ch_jobofferinside',
              handle: '@JobOfferInside',
              title: 'JobOfferInside',
              sourceType: 'imported_link',
              platform: 'TG',
            },
          ],
          appliedPresetIds: [],
          desktopAgentConnected: false,
          agentStatus: 'offline_cache',
          desktopAgentEndpoint: 'http://127.0.0.1:48120',
          readClusterIds: [],
        },
        onWorkspaceScopeChange: () => {},
        platformFilter: 'ALL',
        onPlatformFilterChange: () => {},
        selectedPresetFilter: 'ALL',
        onSelectPresetFilter: () => {},
        onOpenOnboarding: () => {},
        onUnsubscribeChannel: () => {},
      }),
    );

    // Before real title is fetched, username is not duplicated twice
    expect(screen.getByText('@JobOfferInside')).toBeTruthy();
    expect(screen.queryByText('JobOfferInside')).toBeNull();

    // Once real title is resolved, real name is shown above and @handle below
    rerender(
      React.createElement(LeftSidebar, {
        collapsed: false,
        onToggleCollapse: () => {},
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
          subscribedChannels: [
            {
              channelId: 'ch_jobofferinside',
              handle: '@JobOfferInside',
              title: resolvedTitle!,
              sourceType: 'imported_link',
              platform: 'TG',
            },
          ],
          appliedPresetIds: [],
          desktopAgentConnected: false,
          agentStatus: 'offline_cache',
          desktopAgentEndpoint: 'http://127.0.0.1:48120',
          readClusterIds: [],
        },
        onWorkspaceScopeChange: () => {},
        platformFilter: 'ALL',
        onPlatformFilterChange: () => {},
        selectedPresetFilter: 'ALL',
        onSelectPresetFilter: () => {},
        onOpenOnboarding: () => {},
        onUnsubscribeChannel: () => {},
      }),
    );

    expect(screen.getByText('Job Offer Inside')).toBeTruthy();
    expect(screen.getByText('@JobOfferInside')).toBeTruthy();
  });

  it('bounds live fetch concurrency to 2 and deduplicates singleflight and cached channels', async () => {
    let activeRequests = 0;
    let maxActiveRequests = 0;
    let totalFetchCalls = 0;

    const mockFetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      const username = decodeURIComponent(url.split('/s/')[1] ?? 'chan');
      totalFetchCalls += 1;
      activeRequests += 1;
      maxActiveRequests = Math.max(maxActiveRequests, activeRequests);
      await new Promise((resolve) => setTimeout(resolve, 15));
      activeRequests -= 1;

      const html = `
        <div class="tgme_channel_info_header_title"><span>Title ${username}</span></div>
        <div class="tgme_widget_message_wrap" data-post="${username}/101">
          <div class="tgme_widget_message_text js-message_text">Post from ${username}</div>
          <time datetime="2026-11-16T01:00:00Z">01:00</time>
        </div>
      `;
      return new Response(html, { status: 200 });
    }) as typeof fetch;

    const channels = Array.from({ length: 6 }, (_, idx) => ({
      channelId: `ch_batch_${idx + 1}`,
      handle: `@batch_chan_${idx + 1}`,
    }));

    const [firstBatch, overlapBatch] = await Promise.all([
      fetchLiveChannelsBatch(channels, {
        maxConcurrency: 2,
        fetchImpl: mockFetch,
      }),
      fetchLiveChannelsBatch(channels.slice(0, 2), {
        maxConcurrency: 2,
        fetchImpl: mockFetch,
      }),
    ]);

    expect(maxActiveRequests).toBeLessThanOrEqual(2);
    expect(totalFetchCalls).toBe(6);
    expect(firstBatch.items.length).toBe(6);
    expect(overlapBatch.items.length).toBe(2);
    expect(Object.keys(firstBatch.errors).length).toBe(0);

    const cachedBatch = await fetchLiveChannelsBatch(channels, {
      maxConcurrency: 2,
      fetchImpl: mockFetch,
    });
    expect(totalFetchCalls).toBe(6);
    expect(cachedBatch.items.length).toBe(6);
  });

  it('retries transient fetch failed errors, formats diagnostics, and preserves stale cache on refresh failure', async () => {
    const attemptsByChannel = new Map<string, number>();
    let shouldFailCachedChannel = false;

    const mockFetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      const username = decodeURIComponent(url.split('/s/')[1] ?? '');
      const count = (attemptsByChannel.get(username) ?? 0) + 1;
      attemptsByChannel.set(username, count);

      if (username === 'flaky_chan') {
        if (count === 1) {
          throw Object.assign(new TypeError('fetch failed'), {
            cause: Object.assign(new Error('read ECONNRESET'), { code: 'ECONNRESET' }),
          });
        }
        return new Response(
          `
            <div class="tgme_channel_info_header_title"><span>Flaky Recovered</span></div>
            <div class="tgme_widget_message_wrap" data-post="flaky_chan/12">
              <div class="tgme_widget_message_text js-message_text">Recovered post</div>
              <time datetime="2026-11-16T02:00:00Z">02:00</time>
            </div>
          `,
          { status: 200 },
        );
      }

      if (username === 'timeout_chan') {
        const err = new Error('The operation was aborted due to timeout');
        err.name = 'TimeoutError';
        throw err;
      }

      if (username === 'broken_chan') {
        throw Object.assign(new TypeError('fetch failed'), {
          cause: Object.assign(new Error('read ECONNRESET'), { code: 'ECONNRESET' }),
        });
      }

      if (username === 'cached_chan') {
        if (shouldFailCachedChannel) {
          throw Object.assign(new TypeError('fetch failed'), {
            cause: Object.assign(new Error('read ECONNRESET'), { code: 'ECONNRESET' }),
          });
        }
        return new Response(
          `
            <div class="tgme_channel_info_header_title"><span>Cached Channel</span></div>
            <div class="tgme_widget_message_wrap" data-post="cached_chan/77">
              <div class="tgme_widget_message_text js-message_text">Cached live post</div>
              <time datetime="2026-11-16T02:30:00Z">02:30</time>
            </div>
          `,
          { status: 200 },
        );
      }

      return new Response('', { status: 404 });
    }) as typeof fetch;

    const res = await fetchLiveChannelsBatch(
      [
        { channelId: 'ch_flaky', handle: '@flaky_chan' },
        { channelId: 'ch_timeout', handle: '@timeout_chan' },
        { channelId: 'ch_broken', handle: '@broken_chan' },
        { channelId: 'ch_cached', handle: '@cached_chan' },
      ],
      {
        maxConcurrency: 2,
        maxRetries: 1,
        retryDelayMs: 5,
        fetchImpl: mockFetch,
      },
    );

    expect(res.errors['@flaky_chan']).toBeUndefined();
    expect(res.items.some((item) => item.channelId === 'ch_flaky')).toBe(true);
    expect(res.errors['@timeout_chan']).toContain('таймаут сети/VPN');
    expect(res.errors['@broken_chan']).toContain('Соединение с t.me прервано');
    expect(res.errors['@broken_chan']).toContain('ECONNRESET');
    expect(formatLiveFetchError(Object.assign(new Error('getaddrinfo ENOTFOUND t.me'), { code: 'ENOTFOUND' }))).toContain('Ошибка DNS');

    // Simulate a forced refresh > 5s later where @cached_chan fails; stale items must survive
    shouldFailCachedChannel = true;
    const origNow = Date.now;
    const baseNow = origNow();
    Date.now = () => baseNow + 6_000;
    try {
      const staleFallback = await fetchLiveChannelsBatch(
        [{ channelId: 'ch_cached', handle: '@cached_chan' }],
        {
          forceRefresh: true,
          maxRetries: 1,
          retryDelayMs: 5,
          fetchImpl: mockFetch,
        },
      );
      expect(Object.keys(staleFallback.errors).length).toBe(0);
      expect(staleFallback.items.length).toBe(1);
      expect(staleFallback.items[0].channelTitle).toBe('Cached Channel');
    } finally {
      Date.now = origNow;
    }
  });

  it('skips synthetic preset channels in syncAllChannelFeeds and fetches only newly added live channels incrementally', async () => {
    const requestedLiveBatches: string[][] = [];
    const origFetch = globalThis.fetch;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith('/api/v1/live/channels')) {
        const body = JSON.parse(String(init?.body ?? '{}')) as {
          channels?: Array<{ channelId: string; handle: string }>;
        };
        const list = body.channels ?? [];
        requestedLiveBatches.push(list.map((c) => c.handle));

        const items = list.map((ch, idx) => ({
          id: `tg-${ch.channelId}-${idx + 1}`,
          channelId: ch.channelId,
          channelTitle: `Resolved ${ch.handle}`,
          channelHandle: ch.handle,
          postId: idx + 1,
          publishedAt: `2026-11-16T03:0${idx}:00Z`,
          neutralHeadline: `Headline ${ch.handle}`,
          sourcePlatform: 'TG' as const,
          clusterId: `cluster-${ch.channelId}`,
          text: `Live text from ${ch.handle}`,
          entities: [],
          media: [],
          originScore: 0.9,
        }));

        const channelTitles = Object.fromEntries(
          list.map((ch) => [ch.channelId, `Resolved ${ch.handle}`]),
        );

        return new Response(
          JSON.stringify({
            items,
            errors: {},
            channelTitles,
            fetchedAt: new Date().toISOString(),
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }

      return new Response(JSON.stringify({ error: 'offline' }), { status: 502 });
    }) as unknown as typeof fetch;

    try {
      const presetChannels = [
        {
          channelId: '-1001001',
          handle: '@infra_postmortem',
          title: 'Infra Postmortem',
          sourceType: 'public_preset' as const,
          platform: 'TG' as const,
        },
        {
          channelId: '-1003001',
          handle: '@product_economics',
          title: 'Product & FinOps',
          sourceType: 'public_preset' as const,
          platform: 'TG' as const,
        },
      ];

      const firstLive = {
        channelId: 'ch_first_live',
        handle: '@first_live',
        title: '',
        sourceType: 'imported_link' as const,
        platform: 'TG' as const,
      };

      const firstSync = await syncAllChannelFeeds({
        userId: 'u1',
        channels: [...presetChannels, firstLive],
      });

      expect(requestedLiveBatches).toEqual([['@first_live']]);
      expect(firstSync.cleanItems.some((p) => p.channelId === 'ch_first_live')).toBe(true);

      const secondLive = {
        channelId: 'ch_second_live',
        handle: '@second_live',
        title: '',
        sourceType: 'imported_link' as const,
        platform: 'TG' as const,
      };

      const secondSync = await syncAllChannelFeeds({
        userId: 'u1',
        channels: [
          ...presetChannels,
          { ...firstLive, title: firstSync.channelTitles.ch_first_live ?? '' },
          secondLive,
        ],
      });

      expect(requestedLiveBatches).toEqual([['@first_live'], ['@second_live']]);
      expect(secondSync.cleanItems.some((p) => p.channelId === 'ch_first_live')).toBe(true);
      expect(secondSync.cleanItems.some((p) => p.channelId === 'ch_second_live')).toBe(true);
      expect(
        secondSync.digest.cards.some((c) => c.primarySource.channelId === 'ch_first_live'),
      ).toBe(true);
      expect(
        secondSync.digest.cards.some((c) => c.primarySource.channelId === 'ch_second_live'),
      ).toBe(true);
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('tunnels live RSS/HTTP requests through SOCKS5 and HTTP CONNECT proxies without local DNS', async () => {
    const targetServer = http.createServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'application/rss+xml' });
      res.end(SAMPLE_RSS_XML);
    });
    await new Promise<void>((resolve) => targetServer.listen(0, '127.0.0.1', resolve));
    const targetPort = (targetServer.address() as net.AddressInfo).port;

    let socksRequestedHost = '';
    const socksProxy = net.createServer((clientSocket) => {
      let stage = 0;
      let buf = Buffer.alloc(0);
      clientSocket.on('data', (chunk: Buffer) => {
        buf = Buffer.concat([buf, chunk]);
        if (stage === 0 && buf.length >= 2) {
          const nMethods = buf[1];
          if (buf.length < 2 + nMethods) return;
          buf = buf.subarray(2 + nMethods);
          stage = 1;
          clientSocket.write(Buffer.from([0x05, 0x00]));
        }
        if (stage === 1 && buf.length >= 5) {
          const atyp = buf[3];
          if (atyp === 0x03) {
            const hostLen = buf[4];
            if (buf.length < 5 + hostLen + 2) return;
            socksRequestedHost = buf.subarray(5, 5 + hostLen).toString('utf8');
            const rest = buf.subarray(5 + hostLen + 2);
            stage = 2;
            const upstream = net.connect(targetPort, '127.0.0.1', () => {
              clientSocket.write(Buffer.from([0x05, 0x00, 0x00, 0x01, 127, 0, 0, 1, 0, 0]));
              if (rest.length > 0) upstream.write(rest);
              clientSocket.pipe(upstream);
              upstream.pipe(clientSocket);
            });
            upstream.on('error', () => clientSocket.destroy());
          }
        }
      });
    });
    await new Promise<void>((resolve) => socksProxy.listen(0, '127.0.0.1', resolve));
    const socksPort = (socksProxy.address() as net.AddressInfo).port;

    let httpConnectTarget = '';
    const httpConnectProxy = http.createServer();
    httpConnectProxy.on('connect', (req, clientSocket, head) => {
      httpConnectTarget = req.url ?? '';
      const upstream = net.connect(targetPort, '127.0.0.1', () => {
        clientSocket.write('HTTP/1.1 200 Connection Established\r\n\r\n');
        if (head.length > 0) upstream.write(head);
        upstream.pipe(clientSocket);
        clientSocket.pipe(upstream);
      });
      upstream.on('error', () => clientSocket.destroy());
    });
    await new Promise<void>((resolve) => httpConnectProxy.listen(0, '127.0.0.1', resolve));
    const httpProxyPort = (httpConnectProxy.address() as net.AddressInfo).port;

    try {
      const socksRes = await fetchLiveChannelsBatch(
        [
          {
            channelId: 'rss_socks',
            handle: 'http://remote-only-domain.invalid/feed.xml',
            platform: 'RSS',
          },
        ],
        {
          bypassCache: true,
          proxyUrl: `socks5h://127.0.0.1:${socksPort}`,
          timeoutMs: 4000,
          maxRetries: 0,
        },
      );
      expect(socksRequestedHost).toBe('remote-only-domain.invalid');
      expect(Object.keys(socksRes.errors)).toEqual([]);
      expect(socksRes.items.length).toBe(1);
      expect(socksRes.items[0].channelTitle).toBe('Engineering Feed');

      const httpRes = await fetchLiveChannelsBatch(
        [
          {
            channelId: 'rss_http_connect',
            handle: 'http://remote-connect-domain.invalid/feed.xml',
            platform: 'RSS',
          },
        ],
        {
          bypassCache: true,
          proxyUrl: `http://127.0.0.1:${httpProxyPort}`,
          timeoutMs: 4000,
          maxRetries: 0,
        },
      );
      expect(httpConnectTarget).toBe('remote-connect-domain.invalid:80');
      expect(Object.keys(httpRes.errors)).toEqual([]);
      expect(httpRes.items.length).toBe(1);
      expect(httpRes.items[0].channelTitle).toBe('Engineering Feed');
    } finally {
      socksProxy.close();
      httpConnectProxy.close();
      targetServer.close();
    }
  });

  it('skips direct t.me requests when skipDirectTme is enabled and loads via mirror/RSS bridge', async () => {
    const requestedUrls: string[] = [];
    const mockFetch = (async (input: RequestInfo | URL) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.includes('r.jina.ai')) {
        return new Response('plain markdown without widget', { status: 200 });
      }
      if (url.includes('tg.i-c-a.su/rss/')) {
        return new Response(SAMPLE_RSS_XML, { status: 200 });
      }
      return new Response('', { status: 404 });
    }) as typeof fetch;

    const res = await fetchLiveChannelsBatch(
      [{ channelId: 'ch_skip_direct', handle: '@skip_direct_chan' }],
      {
        bypassCache: true,
        skipDirectTme: true,
        fetchImpl: mockFetch,
      },
    );

    expect(requestedUrls.some((u) => u.startsWith('https://t.me/'))).toBe(false);
    expect(Object.keys(res.errors)).toEqual([]);
    expect(res.items.length).toBe(1);
    expect(res.items[0].sourcePlatform).toBe('TG');
    expect(res.items[0].channelHandle).toBe('@skip_direct_chan');
  });
});
