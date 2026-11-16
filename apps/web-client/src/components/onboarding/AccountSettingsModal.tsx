'use client';

import React, { useEffect, useState } from 'react';
import { probeDesktopAgentHealth } from '../../lib/api/client';
import {
  buildB2BExportBundle,
  evaluateRagEntitlements,
  formatRbacRoleLabel,
} from '../../lib/entitlements';
import type {
  AgentConnectionStatus,
  RbacRole,
  WorkspaceScope,
  ZeroTrustUserSession,
} from '../../lib/types/feed';
import { IconClose } from '../ui/Icons';

export type AccountSettingsTab = 'account' | 'agent';

export interface AccountSettingsModalProps {
  open: boolean;
  initialTab?: AccountSettingsTab;
  session: ZeroTrustUserSession;
  onClose: () => void;
  onUpdateSession: (next: Partial<ZeroTrustUserSession>) => void;
}

type ExportTarget = 'slack' | 'notion' | 'telegram' | 'webhook';

const WORKSPACE_SCOPES: readonly { id: WorkspaceScope; label: string }[] = [
  { id: 'personal', label: 'Личное' },
  { id: 'team_b2b', label: 'Команда' },
  { id: 'hybrid', label: 'Гибрид' },
];

/**
 * Dedicated modal for account access, AI quota/privacy (152-FZ), B2B workspace settings,
 * and private-channel Desktop Agent connection.
 */
export function AccountSettingsModal({
  open,
  initialTab = 'account',
  session,
  onClose,
  onUpdateSession,
}: AccountSettingsModalProps) {
  const [activeTab, setActiveTab] = useState<AccountSettingsTab>(initialTab);
  const [isProbingAgent, setIsProbingAgent] = useState<boolean>(false);
  const [agentCheckMsg, setAgentCheckMsg] = useState<string | null>(null);
  const [exportTarget, setExportTarget] = useState<ExportTarget>('notion');
  const [exportPreview, setExportPreview] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setActiveTab(initialTab);
    }
  }, [open, initialTab]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  const entitlements = evaluateRagEntitlements(session);

  const handleProbeAgent = async () => {
    setIsProbingAgent(true);
    setAgentCheckMsg('Проверяем связь с локальным помощником…');
    try {
      const res = await probeDesktopAgentHealth();
      const nextStatus: AgentConnectionStatus = res.connected
        ? 'online'
        : 'offline_cache';
      onUpdateSession({
        desktopAgentConnected: res.connected,
        agentStatus: nextStatus,
      });
      setAgentCheckMsg(
        res.connected
          ? `Локальный помощник подключён (версия ${res.version ?? '1.0'})`
          : 'Программа на компьютере не найдена — используется обычный облачный режим для открытых каналов.',
      );
    } finally {
      setIsProbingAgent(false);
    }
  };

  const handleGenerateExport = () => {
    const bundle = buildB2BExportBundle({
      target: exportTarget,
      workspaceName: 'Команда',
      generationSessionId: 'gen-session-01',
      rbacRole: session.rbacRole,
      verifiedSources: [
        {
          postId: 'post-bgp-42',
          channelHandle: '@infra_postmortem',
          publishedAt: '2026-11-15T08:15:00Z',
          quote: 'потеряно 14% трафика',
        },
      ],
    });
    setExportPreview(
      `Пакет для ${bundle.target.toUpperCase()} готов • источников: ${bundle.verifiedSources.length}`,
    );
  };

  return (
    <div
      className="modal-backdrop density-comfortable"
      role="dialog"
      aria-modal="true"
      aria-label="Настройки аккаунта"
      onClick={onClose}
    >
      <div
        className="modal-card"
        style={{ maxWidth: '640px' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="row-between" style={{ marginBottom: '0.75rem', gap: '0.5rem' }}>
          <div>
            <h2 className="typo-h2">Настройки аккаунта</h2>
            <p className="typo-meta">
              Профиль, лимиты ИИ-поиска, приватность и доступ к закрытым каналам
            </p>
          </div>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Закрыть ✕"
            data-tooltip="Закрыть (Esc)"
            data-tooltip-pos="bottom"
          >
            <IconClose size={17} />
          </button>
        </div>

        <div
          className="segmented-tabs"
          role="tablist"
          style={{ marginBottom: '0.9rem', flexWrap: 'wrap' }}
        >
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'account'}
            className={activeTab === 'account' ? 'active' : ''}
            onClick={() => setActiveTab('account')}
          >
            Аккаунт и доступ
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'agent'}
            className={activeTab === 'agent' ? 'active' : ''}
            onClick={() => setActiveTab('agent')}
          >
            Закрытые каналы
          </button>
        </div>

        {activeTab === 'account' ? (
          <div className="stack" style={{ gap: '0.85rem' }}>
            <div className="rag-panel">
              <div className="row-between" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
                <strong className="typo-meta">Лимит ИИ-поиска на месяц</strong>
                <span className="badge badge-neutral tabular-nums">
                  Осталось {entitlements.quotaRemaining} из {entitlements.quotaTotal}
                </span>
              </div>
              <p className="typo-meta" style={{ marginTop: '0.25rem' }}>
                Обычный поиск по тексту постов работает без ограничений и не расходует лимит ИИ.
              </p>
            </div>

            <div>
              <strong className="typo-meta">Вход в аккаунт (без номера телефона):</strong>
              <div className="row" style={{ gap: '0.45rem', marginTop: '0.35rem', flexWrap: 'wrap' }}>
                {(['passkey', 'email_otp', 'oauth_google'] as const).map((method) => (
                  <button
                    key={method}
                    type="button"
                    className={
                      session.authMethod === method ? 'btn-primary' : 'btn-secondary'
                    }
                    onClick={() => onUpdateSession({ authMethod: method })}
                  >
                    {method === 'passkey'
                      ? 'Passkey (отпечаток / ключ)'
                      : method === 'email_otp'
                        ? 'Ссылка на Email'
                        : 'Корпоративный вход'}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <strong className="typo-meta">Рабочее пространство:</strong>
              <div className="row" style={{ gap: '0.45rem', marginTop: '0.35rem', flexWrap: 'wrap' }}>
                {WORKSPACE_SCOPES.map((ws) => (
                  <button
                    key={ws.id}
                    type="button"
                    className={session.workspaceScope === ws.id ? 'btn-primary' : 'btn-secondary'}
                    onClick={() => onUpdateSession({ workspaceScope: ws.id })}
                  >
                    {ws.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="rag-panel stack" style={{ gap: '0.45rem' }}>
              <strong className="typo-meta">Приватность ИИ-поиска (152-ФЗ):</strong>
              <label className="checkbox-inline">
                <input
                  type="checkbox"
                  checked={session.crossBorderConsentAccepted}
                  onChange={(e) =>
                    onUpdateSession({ crossBorderConsentAccepted: e.target.checked })
                  }
                />
                <span>Разрешить отправку поисковых запросов в облачную ИИ-модель</span>
              </label>
              <label className="checkbox-inline">
                <input
                  type="checkbox"
                  checked={session.localInferenceOnlyRf}
                  onChange={(e) => onUpdateSession({ localInferenceOnlyRf: e.target.checked })}
                />
                <span>Обрабатывать запросы только на серверах в РФ</span>
              </label>
            </div>

            <details className="rag-panel">
              <summary className="typo-meta cursor-pointer font-medium">
                Командный доступ и экспорт (B2B)
              </summary>
              <div className="stack" style={{ gap: '0.7rem', marginTop: '0.6rem' }}>
                <div>
                  <strong className="typo-meta">
                    Роль в команде ({formatRbacRoleLabel(session.rbacRole)}):
                  </strong>
                  <div className="row" style={{ gap: '0.45rem', marginTop: '0.35rem', flexWrap: 'wrap' }}>
                    {(['ADMIN', 'ANALYST', 'VIEWER'] as const).map((role: RbacRole) => (
                      <button
                        key={role}
                        type="button"
                        className={session.rbacRole === role ? 'btn-primary' : 'btn-secondary'}
                        onClick={() => onUpdateSession({ rbacRole: role })}
                      >
                        {formatRbacRoleLabel(role)}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <strong className="typo-meta">Экспорт дайджестов:</strong>
                  <div className="row" style={{ gap: '0.45rem', marginTop: '0.35rem', flexWrap: 'wrap' }}>
                    <select
                      className="select-input"
                      value={exportTarget}
                      onChange={(e) => setExportTarget(e.target.value as ExportTarget)}
                      aria-label="Цель экспорта B2B"
                    >
                      <option value="notion">Notion</option>
                      <option value="slack">Slack</option>
                      <option value="telegram">Telegram-канал</option>
                      <option value="webhook">JSON Webhook</option>
                    </select>
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={handleGenerateExport}
                    >
                      Подготовить пакет
                    </button>
                  </div>
                  {exportPreview ? (
                    <p className="typo-meta" style={{ marginTop: '0.35rem' }}>
                      {exportPreview}
                    </p>
                  ) : null}
                </div>
              </div>
            </details>
          </div>
        ) : (
          <div className="stack" style={{ gap: '0.65rem' }}>
            <p className="typo-body">
              Открытые каналы и RSS работают прямо в браузере. Если вы хотите читать{' '}
              <strong>закрытые чаты и приватные каналы</strong>, установите локальный помощник на
              свой компьютер — данные останутся только у вас:
            </p>
            <ol className="stack" style={{ gap: '0.35rem', paddingLeft: '1.2rem' }}>
              <li className="typo-body">
                Установите приложение-помощник CalmFeed на ваш компьютер.
              </li>
              <li className="typo-body">
                Войдите в Telegram внутри локального приложения — пароли и ключи не покидают ваше
                устройство.
              </li>
              <li className="typo-body">
                Нажмите кнопку ниже, чтобы связать браузер с приложением.
              </li>
            </ol>

            <div className="row" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn-primary"
                disabled={isProbingAgent}
                onClick={() => void handleProbeAgent()}
              >
                {isProbingAgent ? (
                  <>
                    <span className="loading-spinner" aria-hidden="true" />
                    <span>Проверяем…</span>
                  </>
                ) : (
                  'Проверить подключение'
                )}
              </button>
              <span className="typo-meta">
                Статус: {session.desktopAgentConnected ? 'Подключён' : 'Не подключён (работают открытые каналы)'}
              </span>
            </div>
            {agentCheckMsg ? <p className="typo-meta">{agentCheckMsg}</p> : null}

            <details className="rag-panel" style={{ padding: '10px 14px' }}>
              <summary className="typo-meta cursor-pointer font-medium">
                Технические подробности (для специалистов)
              </summary>
              <p className="typo-meta" style={{ marginTop: '0.4rem' }}>
                Локальный агент (Electron/Rust) работает по адресу <code>127.0.0.1:48120</code>,
                хранит индекс FTS5 и ключи XChaCha20 на диске и отдаёт медиа закрытых каналов
                напрямую без загрузки в облако.
              </p>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
