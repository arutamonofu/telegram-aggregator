'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DigestView } from '../components/feed/DigestView';
import { PostCard } from '../components/feed/PostCard';
import { ToggleViewHeader } from '../components/feed/ToggleViewHeader';
import { LeftSidebar } from '../components/layout/LeftSidebar';
import { MobileNavBar } from '../components/layout/MobileNavBar';
import {
  AccountSettingsModal,
  type AccountSettingsTab,
} from '../components/onboarding/AccountSettingsModal';
import {
  OnboardingModal,
  type OnboardingTab,
} from '../components/onboarding/OnboardingModal';
import { AiChatDock } from '../components/rag/AiChatDock';
import {
  TwoTierResults,
  type ExternalSearchRequest,
} from '../components/rag/TwoTierResults';
import { TooltipLayer } from '../components/ui/TooltipLayer';
import { evaluateRagEntitlements } from '../lib/entitlements';
import {
  probeDesktopAgentHealth,
  syncAllChannelFeeds,
} from '../lib/api/client';
import {
  INDUSTRY_PRESETS,
  SEED_CLEAN_POSTS,
  SEED_DEMO_RAG_SOURCE_POSTS,
  SEED_DIGEST_RESPONSE,
} from '../lib/api/seed_data';
import type {
  CleanPostItem,
  DensityMode,
  DigestFeedResponse,
  RagSourcePostItem,
  SourcePlatform,
  SubscribedChannel,
  ThemeMode,
  UserFolder,
  ViewMode,
  WorkspaceScope,
  ZeroTrustUserSession,
} from '../lib/types/feed';

const STARTER_PRESETS = INDUSTRY_PRESETS.slice(0, 2);

const INITIAL_FOLDERS: UserFolder[] = STARTER_PRESETS.map((p) => ({
  id: p.id,
  name: p.title,
  channelIds: p.channels.map((c) => c.channelId),
}));

const INITIAL_SESSION: ZeroTrustUserSession = {
  userId: 'demo-user-01',
  email: 'analyst@workspace.local',
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
  subscribedChannels: STARTER_PRESETS.flatMap((preset) =>
    preset.channels.map((ch) => ({
      channelId: ch.channelId,
      handle: ch.handle,
      title: ch.title,
      sourceType: 'public_preset' as const,
      platform: 'TG' as const,
      topic: ch.topic,
    }))
  ),
  appliedPresetIds: STARTER_PRESETS.map((p) => p.id),
  desktopAgentConnected: false,
  agentStatus: 'offline_cache',
  desktopAgentEndpoint: 'http://127.0.0.1:48120',
  readClusterIds: [],
};

export default function HomePage() {
  const [mode, setMode] = useState<ViewMode>('digest');
  const [density, setDensity] = useState<DensityMode>('comfortable');
  const [theme, setTheme] = useState<ThemeMode>('light');
  const [focusMode, setFocusMode] = useState<boolean>(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState<boolean>(false);
  const [isDesktopSplit, setIsDesktopSplit] = useState<boolean>(false);
  const [searchFocusSignal, setSearchFocusSignal] = useState<number>(0);

  useEffect(() => {
    const syncSplit = () => {
      setIsDesktopSplit(window.innerWidth >= 1100);
    };
    syncSplit();
    window.addEventListener('resize', syncSplit);
    return () => window.removeEventListener('resize', syncSplit);
  }, []);

  const [session, setSession] = useState<ZeroTrustUserSession>(INITIAL_SESSION);
  const [folders, setFolders] = useState<UserFolder[]>(INITIAL_FOLDERS);
  const [digestFeed, setDigestFeed] = useState<DigestFeedResponse>(SEED_DIGEST_RESPONSE);
  const [cleanPosts, setCleanPosts] = useState<CleanPostItem[]>(SEED_CLEAN_POSTS);

  const [platformFilter, setPlatformFilter] = useState<SourcePlatform | 'ALL'>('ALL');
  const [selectedPresetFilter, setSelectedPresetFilter] = useState<string | 'ALL'>('ALL');
  const [selectedChannelId, setSelectedChannelId] = useState<string | null>(null);
  const [activeClusterId, setActiveClusterId] = useState<string | null>(null);
  const [activeClusterLabel, setActiveClusterLabel] = useState<string | null>(null);

  const [reviewedCardIds, setReviewedCardIds] = useState<Set<string>>(() => new Set());
  const [activeItemIndex, setActiveItemIndex] = useState<number>(0);
  const [onboardingOpen, setOnboardingOpen] = useState<boolean>(false);
  const [onboardingTab, setOnboardingTab] = useState<OnboardingTab>('presets');
  const [accountModalOpen, setAccountModalOpen] = useState<boolean>(false);
  const [accountModalTab, setAccountModalTab] = useState<AccountSettingsTab>('account');
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<string | null>(null);
  const [liveWarning, setLiveWarning] = useState<string | null>(null);
  const [hydratedStorage, setHydratedStorage] = useState<boolean>(false);
  const [pendingSearchRequest, setPendingSearchRequest] =
    useState<ExternalSearchRequest | null>(null);
  const searchReqSeqRef = useRef<number>(0);
  const refreshSeqRef = useRef<number>(0);
  const pendingChannelAddSyncRef = useRef<boolean>(false);
  const subscribedChannelsRef = useRef<SubscribedChannel[]>(session.subscribedChannels);
  subscribedChannelsRef.current = session.subscribedChannels;

  const channelIdsKey = useMemo(
    () => session.subscribedChannels.map((c) => c.channelId).join(','),
    [session.subscribedChannels],
  );
  const ragEntitlements = useMemo(() => evaluateRagEntitlements(session), [session]);

  const openOnboardingModal = useCallback((tab: OnboardingTab) => {
    setOnboardingTab(tab);
    setOnboardingOpen(true);
  }, []);

  const openAccountModal = useCallback((tab: AccountSettingsTab = 'account') => {
    setAccountModalTab(tab);
    setAccountModalOpen(true);
  }, []);

  // Restore saved subscriptions from localStorage on client mount
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = window.localStorage.getItem('calmfeed.session.v1');
      if (raw) {
        const parsed = JSON.parse(raw) as {
          subscribedChannels?: SubscribedChannel[];
          appliedPresetIds?: string[];
          folders?: UserFolder[];
          ragQuotaUsed?: number;
        };
        if (Array.isArray(parsed.subscribedChannels)) {
          setSession((prev) => ({
            ...prev,
            subscribedChannels: parsed.subscribedChannels ?? prev.subscribedChannels,
            appliedPresetIds: Array.isArray(parsed.appliedPresetIds)
              ? parsed.appliedPresetIds
              : prev.appliedPresetIds,
            ragQuotaUsed:
              typeof parsed.ragQuotaUsed === 'number'
                ? parsed.ragQuotaUsed
                : prev.ragQuotaUsed,
          }));
        }
        if (Array.isArray(parsed.folders)) {
          setFolders(parsed.folders);
        }
      }
    } catch {
      // Ignore storage read errors
    } finally {
      setHydratedStorage(true);
    }
  }, []);

  // Persist subscriptions to localStorage
  useEffect(() => {
    if (!hydratedStorage || typeof window === 'undefined') return;
    try {
      window.localStorage.setItem(
        'calmfeed.session.v1',
        JSON.stringify({
          subscribedChannels: session.subscribedChannels,
          appliedPresetIds: session.appliedPresetIds,
          folders,
          ragQuotaUsed: session.ragQuotaUsed,
        }),
      );
    } catch {
      // Ignore storage write errors
    }
  }, [
    hydratedStorage,
    session.subscribedChannels,
    session.appliedPresetIds,
    session.ragQuotaUsed,
    folders,
  ]);

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.dataset.theme = theme;
    }
  }, [theme]);

  const refreshFeeds = useCallback(
    async (forceLive = false) => {
      const seq = ++refreshSeqRef.current;
      setIsRefreshing(true);
      setLiveWarning(null);
      try {
        const [synced, agent] = await Promise.all([
          syncAllChannelFeeds({
            userId: session.userId,
            channels: subscribedChannelsRef.current,
            forceLive,
          }),
          probeDesktopAgentHealth(),
        ]);
        if (seq !== refreshSeqRef.current) return;
        setDigestFeed(synced.digest);
        setCleanPosts(synced.cleanItems);
        const timeStr = new Intl.DateTimeFormat('ru-RU', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }).format(new Date(synced.fetchedAt));
        setLastRefreshedAt(timeStr);

        const errEntries = Object.entries(synced.liveErrors);
        if (errEntries.length > 0) {
          setLiveWarning(
            errEntries.map(([ch, msg]) => `${ch}: ${msg}`).join(' • '),
          );
        }

        setSession((prev) => {
          const nextChannels = prev.subscribedChannels.map((ch) => {
            const resolved = synced.channelTitles[ch.channelId]?.trim();
            if (!resolved) return ch;
            const normResolved = resolved.replace(/^@/, '').toLowerCase();
            const normHandle = ch.handle.trim().replace(/^@/, '').toLowerCase();
            if (normResolved === normHandle || resolved === ch.title) {
              return ch;
            }
            return { ...ch, title: resolved };
          });
          return {
            ...prev,
            subscribedChannels: nextChannels,
            desktopAgentConnected: agent.connected,
            agentStatus: agent.connected ? 'online' : 'offline_cache',
          };
        });
      } finally {
        if (seq === refreshSeqRef.current) {
          setIsRefreshing(false);
        }
      }
    },
    [session.userId, channelIdsKey],
  );

  useEffect(() => {
    if (!hydratedStorage) return;
    pendingChannelAddSyncRef.current = false;
    void refreshFeeds(false);
  }, [hydratedStorage, refreshFeeds]);

  useEffect(() => {
    const handlePopState = () => {
      if (accountModalOpen) {
        setAccountModalOpen(false);
        return;
      }
      if (onboardingOpen) {
        setOnboardingOpen(false);
        return;
      }
      if (mobileSidebarOpen) {
        setMobileSidebarOpen(false);
        return;
      }
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [accountModalOpen, onboardingOpen, mobileSidebarOpen]);

  const sourcesByPostId = useMemo(() => {
    const map: Record<string, RagSourcePostItem> = {};
    for (const src of SEED_DEMO_RAG_SOURCE_POSTS) {
      map[src.post.id] = src;
    }
    for (const post of cleanPosts) {
      if (!map[post.id]) {
        map[post.id] = { post, rag_highlights: [] };
      }
    }
    return map;
  }, [cleanPosts]);

  const channelPlatformMap = useMemo(() => {
    const map = new Map<string, SourcePlatform>();
    for (const ch of session.subscribedChannels) {
      map.set(ch.channelId, ch.platform ?? 'TG');
    }
    for (const post of cleanPosts) {
      if (!map.has(post.channelId)) {
        map.set(post.channelId, post.sourcePlatform ?? 'TG');
      }
    }
    return map;
  }, [session.subscribedChannels, cleanPosts]);

  const activePresetChannelIds = useMemo(() => {
    if (selectedPresetFilter === 'ALL') return null;
    const customFolder = folders.find((f) => f.id === selectedPresetFilter);
    if (customFolder) {
      return new Set(customFolder.channelIds);
    }
    const preset = INDUSTRY_PRESETS.find((p) => p.id === selectedPresetFilter);
    return preset ? new Set(preset.channels.map((c) => c.channelId)) : null;
  }, [selectedPresetFilter, folders]);

  const activeChannelLabel = useMemo(() => {
    if (!selectedChannelId) return null;
    const ch = session.subscribedChannels.find((c) => c.channelId === selectedChannelId);
    return ch ? (ch.title || ch.handle) : selectedChannelId;
  }, [selectedChannelId, session.subscribedChannels]);

  const filteredCleanPosts = useMemo(() => {
    return cleanPosts.filter((post) => {
      if (platformFilter !== 'ALL' && (post.sourcePlatform ?? 'TG') !== platformFilter) {
        return false;
      }
      if (activePresetChannelIds && !activePresetChannelIds.has(post.channelId)) {
        return false;
      }
      if (selectedChannelId && post.channelId !== selectedChannelId) {
        return false;
      }
      if (activeClusterId && post.clusterId !== activeClusterId) {
        return false;
      }
      if (session.workspaceScope === 'personal' && post.isPrivateChannel) {
        return false;
      }
      return true;
    });
  }, [
    cleanPosts,
    platformFilter,
    activePresetChannelIds,
    selectedChannelId,
    activeClusterId,
    session.workspaceScope,
  ]);

  const filteredDigestFeed = useMemo<DigestFeedResponse>(() => {
    const cards = digestFeed.cards.filter((card) => {
      if (platformFilter !== 'ALL') {
        const plat = channelPlatformMap.get(card.primarySource.channelId) ?? 'TG';
        if (plat !== platformFilter) return false;
      }
      if (activePresetChannelIds && !activePresetChannelIds.has(card.primarySource.channelId)) {
        return false;
      }
      if (selectedChannelId && card.primarySource.channelId !== selectedChannelId) {
        return false;
      }
      return true;
    });
    return {
      ...digestFeed,
      cards,
      selected_cards_count: cards.length,
    };
  }, [
    digestFeed,
    platformFilter,
    channelPlatformMap,
    activePresetChannelIds,
    selectedChannelId,
  ]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTyping =
        Boolean(target) &&
        (target?.tagName === 'INPUT' ||
          target?.tagName === 'TEXTAREA' ||
          target?.tagName === 'SELECT' ||
          target?.isContentEditable);

      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        if (isDesktopSplit && !focusMode) {
          setSearchFocusSignal((n) => n + 1);
        } else {
          setMode('search');
        }
        return;
      }

      if (event.key === 'Escape') {
        if (accountModalOpen) {
          setAccountModalOpen(false);
        } else if (onboardingOpen) {
          setOnboardingOpen(false);
        } else if (mobileSidebarOpen) {
          setMobileSidebarOpen(false);
        } else if (focusMode) {
          setFocusMode(false);
        }
        return;
      }

      if (isTyping) {
        return;
      }

      const lower = event.key.toLowerCase();
      if (event.key === '/') {
        event.preventDefault();
        if (isDesktopSplit && !focusMode) {
          setSearchFocusSignal((n) => n + 1);
        } else {
          setMode('search');
        }
      } else if (event.key === '[') {
        event.preventDefault();
        setSidebarCollapsed((prev) => !prev);
      } else if (lower === 'v') {
        event.preventDefault();
        setMode((prev) => (prev === 'digest' ? 'clean_originals' : 'digest'));
      } else if (lower === 'f') {
        event.preventDefault();
        setFocusMode((prev) => !prev);
      } else if (lower === 'j') {
        event.preventDefault();
        const maxLen =
          mode === 'digest'
            ? filteredDigestFeed.cards.length
            : filteredCleanPosts.length;
        if (maxLen > 0) {
          setActiveItemIndex((prev) => Math.min(maxLen - 1, prev + 1));
        }
      } else if (lower === 'k') {
        event.preventDefault();
        setActiveItemIndex((prev) => Math.max(0, prev - 1));
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    accountModalOpen,
    onboardingOpen,
    mobileSidebarOpen,
    focusMode,
    isDesktopSplit,
    mode,
    filteredDigestFeed.cards.length,
    filteredCleanPosts.length,
  ]);

  const handleFilterCleanByCluster = (clusterId: string, title: string) => {
    setActiveClusterId(clusterId);
    setActiveClusterLabel(title);
    setMode('clean_originals');
  };

  const handleMarkCardReviewed = (clusterId: string) => {
    setReviewedCardIds((prev) => {
      const next = new Set(prev);
      if (next.has(clusterId)) {
        next.delete(clusterId);
      } else {
        next.add(clusterId);
      }
      return next;
    });
  };

  const handleConsumeRagQuota = () => {
    setSession((prev) => ({
      ...prev,
      ragQuotaUsed: Math.min(prev.ragQuotaTotal, prev.ragQuotaUsed + 1),
    }));
  };

  const handleUnsubscribeChannel = (channelId: string) => {
    if (selectedChannelId === channelId) {
      setSelectedChannelId(null);
    }
    setSession((prev) => ({
      ...prev,
      subscribedChannels: prev.subscribedChannels.filter(
        (c) => c.channelId !== channelId
      ),
    }));
  };

  const handleWorkspaceScopeChange = (workspaceScope: WorkspaceScope) => {
    setSession((prev) => ({ ...prev, workspaceScope }));
  };

  const handleApplyPreset = (presetId: string) => {
    const preset = INDUSTRY_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setSession((prev) => {
      const existing = new Map(prev.subscribedChannels.map((c) => [c.channelId, c]));
      for (const ch of preset.channels) {
        existing.set(ch.channelId, {
          channelId: ch.channelId,
          handle: ch.handle,
          title: ch.title,
          sourceType: 'public_preset',
          platform: 'TG',
          topic: ch.topic,
        });
      }
      const applied = Array.from(new Set([...prev.appliedPresetIds, presetId]));
      return {
        ...prev,
        appliedPresetIds: applied,
        subscribedChannels: Array.from(existing.values()),
      };
    });
    setFolders((prev) => {
      const chIds = preset.channels.map((c) => c.channelId);
      const exists = prev.some((f) => f.id === preset.id);
      if (exists) {
        return prev.map((f) =>
          f.id === preset.id
            ? { ...f, channelIds: Array.from(new Set([...f.channelIds, ...chIds])) }
            : f,
        );
      }
      return [...prev, { id: preset.id, name: preset.title, channelIds: chIds }];
    });
    setSelectedChannelId(null);
    setSelectedPresetFilter(presetId);
  };

  const handleCreateFolder = (name: string, channelIds: string[]) => {
    const id = `folder_${Date.now()}`;
    setFolders((prev) => [...prev, { id, name, channelIds }]);
  };

  const handleUpdateFolder = (
    folderId: string,
    patch: Partial<Pick<UserFolder, 'name' | 'channelIds'>>,
  ) => {
    setFolders((prev) =>
      prev.map((f) => (f.id === folderId ? { ...f, ...patch } : f)),
    );
  };

  const handleDeleteFolder = (folderId: string) => {
    setFolders((prev) => prev.filter((f) => f.id !== folderId));
    if (selectedPresetFilter === folderId) {
      setSelectedPresetFilter('ALL');
    }
  };

  const handleAddChannels = (channels: SubscribedChannel[]) => {
    const existing = new Map(subscribedChannelsRef.current.map((c) => [c.channelId, c]));
    for (const ch of channels) {
      existing.set(ch.channelId, ch);
    }
    const nextList = Array.from(existing.values());
    const nextKey = nextList.map((c) => c.channelId).join(',');
    subscribedChannelsRef.current = nextList;
    if (nextKey !== channelIdsKey) {
      pendingChannelAddSyncRef.current = true;
    }
    setSession((prev) => {
      const prevMap = new Map(prev.subscribedChannels.map((c) => [c.channelId, c]));
      for (const ch of channels) {
        prevMap.set(ch.channelId, ch);
      }
      return {
        ...prev,
        subscribedChannels: Array.from(prevMap.values()),
      };
    });
  };

  const splitActive = isDesktopSplit && !focusMode;
  const showLeftSidebar = !focusMode;
  const leftFeedMode = splitActive && mode === 'search' ? 'digest' : mode;

  const twoTierProps = {
    session,
    density,
    availablePosts: filteredCleanPosts,
    externalQuery: pendingSearchRequest,
    onExternalQueryConsumed: () => setPendingSearchRequest(null),
    onConsumeRagQuota: handleConsumeRagQuota,
    onAcceptPrivacyConsent: () =>
      setSession((prev) => ({
        ...prev,
        crossBorderConsentAccepted: true,
      })),
  };

  return (
    <div
      className={`app-body density-${density} ${
        focusMode ? 'app-shell-focus' : ''
      } ${splitActive ? 'app-split-desktop' : ''} ${
        showLeftSidebar && !sidebarCollapsed ? 'app-has-left-sidebar' : ''
      }`}
    >
      {showLeftSidebar ? (
        <LeftSidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
          mobileOpen={mobileSidebarOpen}
          onCloseMobile={() => setMobileSidebarOpen(false)}
          session={session}
          folders={folders}
          onWorkspaceScopeChange={handleWorkspaceScopeChange}
          platformFilter={platformFilter}
          onPlatformFilterChange={setPlatformFilter}
          selectedPresetFilter={selectedPresetFilter}
          onSelectPresetFilter={setSelectedPresetFilter}
          selectedChannelId={selectedChannelId}
          onSelectChannel={setSelectedChannelId}
          onOpenOnboarding={() => openOnboardingModal('presets')}
          onOpenFolders={() => openOnboardingModal('presets')}
          onOpenAddChannels={() => openOnboardingModal('import')}
          onOpenAccountSettings={() => openAccountModal('account')}
          onOpenAgentSettings={() => openAccountModal('agent')}
          onUnsubscribeChannel={handleUnsubscribeChannel}
        />
      ) : null}

      <main className="center-workspace">
        <div
          className={splitActive ? 'workspace-split' : 'reading-container'}
        >
          <div className={splitActive ? 'feed-column' : undefined}>
            <ToggleViewHeader
              mode={leftFeedMode}
              onModeChange={(nextMode) => {
                if (splitActive && nextMode === 'search') {
                  setSearchFocusSignal((n) => n + 1);
                  return;
                }
                setMode(nextMode);
                setActiveItemIndex(0);
              }}
              density={density}
              onDensityChange={setDensity}
              theme={theme}
              onThemeChange={setTheme}
              focusMode={focusMode}
              onToggleFocusMode={() => setFocusMode((prev) => !prev)}
              splitMode={splitActive}
              sidebarCollapsed={sidebarCollapsed}
              onExpandSidebar={() => setSidebarCollapsed(false)}
              onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
              onOpenCommandPalette={() => {
                if (splitActive) {
                  setSearchFocusSignal((n) => n + 1);
                } else {
                  setMode('search');
                }
              }}
              digestCount={filteredDigestFeed.cards.length}
              cleanCount={filteredCleanPosts.length}
              activeClusterLabel={activeClusterLabel}
              onClearClusterFilter={() => {
                setActiveClusterId(null);
                setActiveClusterLabel(null);
              }}
              activeChannelLabel={activeChannelLabel}
              onClearChannelFilter={() => setSelectedChannelId(null)}
              isRefreshing={isRefreshing}
              lastRefreshedAt={lastRefreshedAt}
              onRefreshFeed={() => void refreshFeeds(true)}
            />

            {liveWarning ? (
              <div className="quota-banner quota-banner-warn" role="status">
                <span>{liveWarning}</span>
              </div>
            ) : null}

            <div style={{ marginTop: '1rem' }}>
              {leftFeedMode === 'digest' ? (
                <DigestView
                  feed={filteredDigestFeed}
                  density={density}
                  sourcesByPostId={sourcesByPostId}
                  reviewedCardIds={reviewedCardIds}
                  activeCardIndex={activeItemIndex}
                  isRefreshing={isRefreshing}
                  lastRefreshedAt={lastRefreshedAt}
                  onMarkCardReviewed={handleMarkCardReviewed}
                  onResetReviewed={() => setReviewedCardIds(new Set())}
                  onFilterCleanByCluster={handleFilterCleanByCluster}
                  onNavigateToCleanFeed={() => setMode('clean_originals')}
                  onOpenSources={() => openOnboardingModal('import')}
                />
              ) : null}

              {leftFeedMode === 'clean_originals' ? (
                <section aria-label="Полная лента постов" className="stack">
                  {isRefreshing ? (
                    <div className="loading-banner" role="status">
                      <span className="loading-spinner" aria-hidden="true" />
                      <span>Загружаем свежие посты из подключённых каналов…</span>
                    </div>
                  ) : null}

                  {filteredCleanPosts.length === 0 && !isRefreshing ? (
                    <div className="digest-card">
                      <p className="typo-body">
                        По выбранным фильтрам постов нет. Сбросьте фильтр источников или
                        подключите новые каналы.
                      </p>
                      <div className="row" style={{ gap: '0.5rem', marginTop: '0.7rem', flexWrap: 'wrap' }}>
                        {selectedChannelId || selectedPresetFilter !== 'ALL' || platformFilter !== 'ALL' || activeClusterId ? (
                          <button
                            type="button"
                            className="btn-secondary"
                            onClick={() => {
                              setSelectedChannelId(null);
                              setSelectedPresetFilter('ALL');
                              setPlatformFilter('ALL');
                              setActiveClusterId(null);
                              setActiveClusterLabel(null);
                            }}
                          >
                            Сбросить все фильтры
                          </button>
                        ) : null}
                        <button
                          type="button"
                          className="btn-primary"
                          onClick={() => openOnboardingModal('import')}
                        >
                          Добавить каналы
                        </button>
                      </div>
                    </div>
                  ) : (
                    filteredCleanPosts.map((post, idx) => (
                      <PostCard
                        key={post.id}
                        post={post}
                        session={session}
                        density={density}
                        isActive={idx === activeItemIndex}
                      />
                    ))
                  )}
                </section>
              ) : null}

              {!splitActive && leftFeedMode === 'search' ? (
                <TwoTierResults {...twoTierProps} layoutVariant="full" />
              ) : null}
            </div>

            {!splitActive && leftFeedMode !== 'search' && !focusMode ? (
              <AiChatDock
                quotaRemaining={ragEntitlements.quotaRemaining}
                quotaTotal={ragEntitlements.quotaTotal}
                canRunCloudRag={ragEntitlements.canRunCloudRag}
                onSubmitQuery={(queryText) => {
                  searchReqSeqRef.current += 1;
                  setPendingSearchRequest({
                    id: searchReqSeqRef.current,
                    text: queryText,
                    mode: ragEntitlements.canRunCloudRag ? 'rag_synthesis' : 'local_fts5',
                  });
                  setMode('search');
                  setActiveItemIndex(0);
                }}
              />
            ) : null}
          </div>

          {splitActive ? (
            <div className="search-column">
              <TwoTierResults
                {...twoTierProps}
                layoutVariant="pane"
                focusSignal={searchFocusSignal}
              />
            </div>
          ) : null}
        </div>
      </main>

      <MobileNavBar
        activeMode={mode}
        onModeChange={setMode}
        onOpenSources={() => setMobileSidebarOpen(true)}
      />

      <OnboardingModal
        open={onboardingOpen}
        initialTab={onboardingTab}
        session={session}
        folders={folders}
        onClose={() => setOnboardingOpen(false)}
        onApplyPreset={handleApplyPreset}
        onCreateFolder={handleCreateFolder}
        onUpdateFolder={handleUpdateFolder}
        onDeleteFolder={handleDeleteFolder}
        onAddChannels={handleAddChannels}
        onChannelsImported={() => {
          if (!pendingChannelAddSyncRef.current) {
            void refreshFeeds(true);
          }
        }}
      />

      <AccountSettingsModal
        open={accountModalOpen}
        initialTab={accountModalTab}
        session={session}
        onClose={() => setAccountModalOpen(false)}
        onUpdateSession={(patch) => setSession((prev) => ({ ...prev, ...patch }))}
      />

      <TooltipLayer />
    </div>
  );
}
