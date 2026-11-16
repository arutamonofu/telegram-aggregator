'use client';

import React from 'react';
import { INDUSTRY_PRESETS } from '@/lib/api/seed_data';
import type {
  SourcePlatform,
  UserFolder,
  WorkspaceScope,
  ZeroTrustUserSession,
} from '@/lib/types/feed';
import {
  IconClose,
  IconFolderCog,
  IconPanelLeftClose,
  IconPanelLeftOpen,
  IconPlus,
  IconSettings,
} from '@/components/ui/Icons';

export interface LeftSidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen?: boolean;
  onCloseMobile?: () => void;
  session: ZeroTrustUserSession;
  folders?: UserFolder[];
  onWorkspaceScopeChange: (scope: WorkspaceScope) => void;
  platformFilter: SourcePlatform | 'ALL';
  onPlatformFilterChange: (platform: SourcePlatform | 'ALL') => void;
  selectedPresetFilter: string | 'ALL';
  onSelectPresetFilter: (presetId: string | 'ALL') => void;
  selectedChannelId?: string | null;
  onSelectChannel?: (channelId: string | null) => void;
  onOpenOnboarding: () => void;
  onOpenFolders?: () => void;
  onOpenAddChannels?: () => void;
  onOpenAccountSettings?: () => void;
  onOpenAgentSettings?: () => void;
  onUnsubscribeChannel: (channelId: string) => void;
}

const WORKSPACE_LABELS: Array<{ id: WorkspaceScope; label: string }> = [
  { id: 'personal', label: 'Личное' },
  { id: 'team_b2b', label: 'Команда' },
  { id: 'hybrid', label: 'Гибрид' },
];

const PLATFORM_LABELS: Array<{ id: SourcePlatform | 'ALL'; label: string }> = [
  { id: 'ALL', label: 'Все источники' },
  { id: 'TG', label: 'Telegram' },
  { id: 'MAX', label: 'MAX' },
  { id: 'RSS', label: 'RSS-ленты' },
];

export function LeftSidebar({
  collapsed,
  onToggleCollapse,
  mobileOpen = false,
  onCloseMobile,
  session,
  folders,
  onWorkspaceScopeChange,
  platformFilter,
  onPlatformFilterChange,
  selectedPresetFilter,
  onSelectPresetFilter,
  selectedChannelId = null,
  onSelectChannel,
  onOpenOnboarding,
  onOpenFolders,
  onOpenAddChannels,
  onOpenAccountSettings,
  onOpenAgentSettings,
  onUnsubscribeChannel,
}: LeftSidebarProps) {
  const isDesktopCollapsed = collapsed && !mobileOpen;

  const privateCount = session.subscribedChannels.filter(
    (c) => c.sourceType === 'private_agent',
  ).length;

  const showWorkspaceSwitcher =
    privateCount > 0 ||
    session.workspaceScope !== 'personal' ||
    session.planTier === 'B2B_WORKSPACE';

  const platformCounts: Record<SourcePlatform | 'ALL', number> = {
    ALL: session.subscribedChannels.length,
    TG: session.subscribedChannels.filter((c) => (c.platform ?? 'TG') === 'TG').length,
    MAX: session.subscribedChannels.filter((c) => c.platform === 'MAX').length,
    RSS: session.subscribedChannels.filter((c) => c.platform === 'RSS').length,
  };

  const visiblePlatforms = PLATFORM_LABELS.filter(
    (pf) => pf.id === 'ALL' || platformCounts[pf.id] > 0 || platformFilter === pf.id,
  );

  const hasPrivateOrAgent = session.desktopAgentConnected || privateCount > 0;
  const agentColor =
    session.agentStatus === 'online'
      ? 'var(--agent-online)'
      : session.agentStatus === 'offline_cache'
        ? 'var(--agent-offline)'
        : 'var(--agent-error)';

  const agentLabel =
    session.agentStatus === 'online'
      ? `Агент подключён • ${privateCount} закрытых`
      : session.agentStatus === 'session_expired'
        ? 'Требуется вход в агент'
        : hasPrivateOrAgent
          ? 'Агент не в сети · тексты из кэша'
          : 'Закрытые каналы · подключить';

  return (
    <>
      <button
        type="button"
        onClick={onToggleCollapse}
        data-testid="expand-sidebar-btn"
        aria-label="Показать панель навигации"
        aria-hidden={!isDesktopCollapsed}
        tabIndex={isDesktopCollapsed ? 0 : -1}
        data-tooltip="Показать панель каналов ([)"
        data-tooltip-pos="right"
        className={`sidebar-floating-expand-btn ${isDesktopCollapsed ? 'is-visible' : ''}`}
      >
        <IconPanelLeftOpen size={18} />
      </button>

      {mobileOpen ? (
        <div
          className="sidebar-backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      ) : null}

      <aside
        aria-label="Навигация и источники"
        aria-hidden={isDesktopCollapsed}
        data-testid="left-sidebar"
        className={`sidebar-left p-3.5 text-xs ${
          isDesktopCollapsed ? 'sidebar-left-collapsed' : ''
        } ${mobileOpen ? 'sidebar-left-mobile-open' : ''}`}
      >
        <div
          className="flex items-center justify-between gap-2 pb-3"
          style={{ borderBottom: '1px solid var(--border-subtle)' }}
        >
          <div className="flex items-center gap-2">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-lg font-bold text-white"
              style={{ backgroundColor: 'var(--accent-primary)' }}
            >
              CF
            </div>
            <div>
              <div className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                CalmFeed
              </div>
              <div className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                {session.subscribedChannels.length} источников
              </div>
            </div>
          </div>

          {mobileOpen && onCloseMobile ? (
            <button
              type="button"
              onClick={onCloseMobile}
              aria-label="Закрыть меню"
              data-tooltip="Закрыть меню"
              data-tooltip-pos="bottom"
              className="icon-btn"
            >
              <IconClose size={16} />
            </button>
          ) : (
            <button
              type="button"
              onClick={onToggleCollapse}
              data-testid="collapse-sidebar-btn"
              data-tooltip="Свернуть панель ([)"
              data-tooltip-pos="bottom"
              aria-label="Свернуть левую панель"
              className="icon-btn"
            >
              <IconPanelLeftClose size={16} />
            </button>
          )}
        </div>

        {showWorkspaceSwitcher ? (
          <div className="mt-3.5">
            <div
              className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide"
              style={{ color: 'var(--text-secondary)' }}
            >
              Пространство
            </div>
            <div
              className="grid grid-cols-1 gap-1"
              role="group"
              aria-label="Выбор рабочего пространства"
            >
              {WORKSPACE_LABELS.map((ws) => {
                const active = session.workspaceScope === ws.id;
                return (
                  <button
                    key={ws.id}
                    type="button"
                    onClick={() => onWorkspaceScopeChange(ws.id)}
                    data-testid={`workspace-scope-${ws.id}`}
                    className="flex items-center justify-between rounded-lg px-2.5 py-1.5 text-left font-medium transition-colors"
                    style={{
                      backgroundColor: active ? 'var(--accent-primary-bg)' : 'transparent',
                      color: active ? 'var(--accent-primary)' : 'var(--text-primary)',
                      border: active
                        ? '1px solid var(--accent-primary)'
                        : '1px solid transparent',
                    }}
                  >
                    <span>{ws.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}

        {visiblePlatforms.length > 2 ? (
          <div className="mt-4">
            <div
              className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide"
              style={{ color: 'var(--text-secondary)' }}
            >
              Платформы
            </div>
            <div className="space-y-1">
              {visiblePlatforms.map((pf) => {
                const active = platformFilter === pf.id;
                return (
                  <button
                    key={pf.id}
                    type="button"
                    onClick={() => {
                      onPlatformFilterChange(pf.id);
                      onCloseMobile?.();
                    }}
                    data-testid={`platform-filter-${pf.id}`}
                    className="flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left transition-colors"
                    style={{
                      backgroundColor: active ? 'var(--bg-muted)' : 'transparent',
                      fontWeight: active ? 600 : 400,
                    }}
                  >
                    <span>{pf.label}</span>
                    <span className="tabular-nums text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                      {platformCounts[pf.id]}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}

        {/* Folders & Channels Tree */}
        <div className="mt-4 flex-1 overflow-y-auto overflow-x-hidden">
          <div className="mb-1.5 flex items-center justify-between gap-1">
            <span
              className="text-[11px] font-semibold uppercase tracking-wide"
              style={{ color: 'var(--text-secondary)' }}
            >
              Папки и каналы
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => {
                  onCloseMobile?.();
                  (onOpenFolders ?? onOpenOnboarding)();
                }}
                className="icon-btn"
                aria-label="Папки"
                data-tooltip="Настроить папки и подборки"
              >
                <IconFolderCog size={15} />
              </button>
              <button
                type="button"
                onClick={() => {
                  onCloseMobile?.();
                  (onOpenAddChannels ?? onOpenOnboarding)();
                }}
                className="icon-btn"
                style={{ color: 'var(--accent-primary)' }}
                aria-label="+ Добавить"
                data-tooltip="Добавить канал, папку или RSS"
              >
                <IconPlus size={15} />
              </button>
            </div>
          </div>

          <div className="space-y-1">
            <button
              type="button"
              onClick={() => {
                onSelectPresetFilter('ALL');
                onSelectChannel?.(null);
                onCloseMobile?.();
              }}
              className="flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left"
              style={{
                backgroundColor:
                  selectedPresetFilter === 'ALL' && !selectedChannelId
                    ? 'var(--bg-muted)'
                    : 'transparent',
                fontWeight: selectedPresetFilter === 'ALL' && !selectedChannelId ? 600 : 400,
              }}
            >
              <span>Все подписки</span>
              <span className="tabular-nums text-[11px]">
                {session.subscribedChannels.length}
              </span>
            </button>

            {folders
              ? folders.map((folder) => {
                  const subscribedInFolder = folder.channelIds.filter((id) =>
                    session.subscribedChannels.some((s) => s.channelId === id),
                  ).length;
                  const active = selectedPresetFilter === folder.id && !selectedChannelId;
                  return (
                    <button
                      key={folder.id}
                      type="button"
                      onClick={() => {
                        onSelectPresetFilter(folder.id);
                        onSelectChannel?.(null);
                        onCloseMobile?.();
                      }}
                      className="flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left"
                      style={{
                        backgroundColor: active ? 'var(--bg-muted)' : 'transparent',
                        fontWeight: active ? 600 : 400,
                      }}
                    >
                      <span className="truncate">{folder.name}</span>
                      <span className="tabular-nums text-[11px]">{subscribedInFolder}</span>
                    </button>
                  );
                })
              : INDUSTRY_PRESETS.filter((p) => session.appliedPresetIds.includes(p.id)).map(
                  (preset) => {
                    const subscribedInPreset = preset.channels.filter((ch) =>
                      session.subscribedChannels.some((s) => s.channelId === ch.channelId),
                    ).length;
                    if (subscribedInPreset === 0) return null;
                    const active = selectedPresetFilter === preset.id && !selectedChannelId;
                    return (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => {
                          onSelectPresetFilter(preset.id);
                          onSelectChannel?.(null);
                          onCloseMobile?.();
                        }}
                        className="flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left"
                        style={{
                          backgroundColor: active ? 'var(--bg-muted)' : 'transparent',
                          fontWeight: active ? 600 : 400,
                        }}
                      >
                        <span className="truncate">{preset.title}</span>
                        <span className="tabular-nums text-[11px]">{subscribedInPreset}</span>
                      </button>
                    );
                  },
                )}
          </div>

          {session.subscribedChannels.length > 0 ? (
            <div className="mt-3 space-y-1">
              <div
                className="px-1.5 text-[10px] font-semibold uppercase tracking-wide"
                style={{ color: 'var(--text-muted)' }}
              >
                Каналы
              </div>
              {session.subscribedChannels.map((ch) => {
                const isChannelSelected = selectedChannelId === ch.channelId;
                const displayTitle = (ch.title ?? '').trim();
                const normalizedTitle = displayTitle.replace(/^@/, '').toLowerCase();
                const normalizedHandle = ch.handle.trim().replace(/^@/, '').toLowerCase();
                const hasRealTitle =
                  Boolean(displayTitle) && normalizedTitle !== normalizedHandle;
                const primaryLabel = hasRealTitle ? displayTitle : ch.handle;
                return (
                  <div
                    key={ch.channelId}
                    className={`channel-row ${isChannelSelected ? 'channel-row-active' : ''}`}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        onSelectChannel?.(isChannelSelected ? null : ch.channelId);
                        onCloseMobile?.();
                      }}
                      className="min-w-0 flex-1 text-left"
                      data-tooltip={`Фильтр по каналу: ${primaryLabel}`}
                    >
                      <div
                        className="truncate text-[11.5px]"
                        style={{
                          color: isChannelSelected
                            ? 'var(--accent-primary)'
                            : 'var(--text-primary)',
                          fontWeight: isChannelSelected ? 600 : 500,
                        }}
                      >
                        {primaryLabel}
                      </div>
                      {hasRealTitle ? (
                        <div
                          className="truncate text-[10.5px]"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          {ch.handle}
                        </div>
                      ) : null}
                    </button>
                    <button
                      type="button"
                      onClick={() => onUnsubscribeChannel(ch.channelId)}
                      aria-label={`Отписаться от ${ch.handle}`}
                      data-testid={`unsubscribe-${ch.channelId}`}
                      data-tooltip={`Отписаться от ${ch.handle}`}
                      className="icon-btn shrink-0"
                      style={{ minWidth: '24px', height: '24px', padding: '0 4px' }}
                    >
                      <IconClose size={13} />
                    </button>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>

        {/* Bottom-left Account & Settings Bar (AI-chat style) */}
        <div
          className="mt-3 pt-3"
          style={{ borderTop: '1px solid var(--border-subtle)' }}
        >
          <button
            type="button"
            onClick={() => {
              onCloseMobile?.();
              (onOpenAccountSettings ?? onOpenAgentSettings ?? onOpenOnboarding)();
            }}
            data-testid="account-settings-btn"
            aria-label="Настройки аккаунта"
            data-tooltip="Настройки аккаунта и закрытых каналов"
            className="surface-muted flex w-full min-w-0 items-center gap-2.5 rounded-xl px-2.5 py-2 text-left transition-colors"
          >
            <div
              className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold"
              style={{
                backgroundColor: 'var(--accent-primary-bg)',
                color: 'var(--accent-primary)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              {(session.email.slice(0, 2) || 'CF').toUpperCase()}
              {hasPrivateOrAgent ? (
                <span
                  className="absolute h-2 w-2 rounded-full"
                  style={{
                    backgroundColor: agentColor,
                    right: '-1px',
                    bottom: '-1px',
                    border: '1.5px solid var(--bg-surface)',
                  }}
                />
              ) : null}
            </div>
            <div className="min-w-0 flex-1">
              <div
                className="truncate text-[11.5px] font-semibold"
                style={{ color: 'var(--text-primary)' }}
              >
                {session.email || 'Аккаунт'}
              </div>
              <div
                className="truncate text-[10.5px]"
                style={{ color: 'var(--text-secondary)' }}
              >
                {hasPrivateOrAgent ? agentLabel : 'Настройки аккаунта и доступа'}
              </div>
            </div>
            <IconSettings
              size={15}
              className="shrink-0"
            />
          </button>
        </div>
      </aside>
    </>
  );
}
