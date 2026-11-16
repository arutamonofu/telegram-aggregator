'use client';

import React, { useEffect, useState } from 'react';
import {
  parseTelegramChannelImportLines,
  registerTelegramChannel,
  type ParsedImportEntry,
} from '../../lib/api/client';
import { INDUSTRY_PRESETS } from '../../lib/api/seed_data';
import type {
  SubscribedChannel,
  UserFolder,
  ZeroTrustUserSession,
} from '../../lib/types/feed';
import { IconClose } from '../ui/Icons';

export type OnboardingTab = 'presets' | 'import';

export interface OnboardingModalProps {
  open: boolean;
  initialTab?: OnboardingTab;
  session: ZeroTrustUserSession;
  folders?: UserFolder[];
  onClose: () => void;
  onApplyPreset: (presetId: string) => void;
  onCreateFolder?: (name: string, channelIds: string[]) => void;
  onUpdateFolder?: (
    folderId: string,
    patch: Partial<Pick<UserFolder, 'name' | 'channelIds'>>,
  ) => void;
  onDeleteFolder?: (folderId: string) => void;
  onAddChannels?: (channels: SubscribedChannel[]) => void;
  onChannelsImported: () => void;
}

const SAMPLE_IMPORT_LINES =
  'https://t.me/addlist/tech-architecture-ru\n@ru_devops\n@ai_papers_ru';

export function OnboardingModal({
  open,
  initialTab = 'presets',
  session,
  folders = [],
  onClose,
  onApplyPreset,
  onCreateFolder,
  onUpdateFolder,
  onDeleteFolder,
  onAddChannels,
  onChannelsImported,
}: OnboardingModalProps) {
  const [activeTab, setActiveTab] = useState<OnboardingTab>(initialTab);
  const [selectedPresetId, setSelectedPresetId] = useState<string>(
    session.appliedPresetIds[0] ?? INDUSTRY_PRESETS[0]?.id ?? ''
  );

  const [newFolderName, setNewFolderName] = useState<string>('');
  const [newFolderChannelIds, setNewFolderChannelIds] = useState<string[]>([]);

  const [importText, setImportText] = useState<string>('');
  const [parsedImports, setParsedImports] = useState<ParsedImportEntry[]>([]);
  const [importStatus, setImportStatus] = useState<string | null>(null);
  const [isImporting, setIsImporting] = useState<boolean>(false);

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

  const selectedPreset =
    INDUSTRY_PRESETS.find((p) => p.id === selectedPresetId) ?? INDUSTRY_PRESETS[0];

  const handleAnalyzeImport = () => {
    const parsed = parseTelegramChannelImportLines(importText);
    setParsedImports(parsed);
    setImportStatus(
      parsed.length > 0
        ? `Распознано строк: ${parsed.length}`
        : 'Вставьте хотя бы одну ссылку или @канал',
    );
  };

  const handleConfirmImport = async () => {
    const list = parseTelegramChannelImportLines(importText);
    setParsedImports(list);
    if (list.length === 0) {
      setImportStatus('Вставьте ссылку на канал, папку или RSS-ленту.');
      return;
    }

    const newlyAdded: SubscribedChannel[] = [];
    setIsImporting(true);
    setImportStatus('Проверяем и подключаем источники…');
    try {
      for (const item of list) {
        if (item.valid && item.resolvedChannels.length > 0) {
          for (const ch of item.resolvedChannels) {
            const isRss = ch.platform === 'RSS' || /^https?:\/\//i.test(ch.handle);
            const res = isRss ? { registered: false } : await registerTelegramChannel(ch.handle);
            newlyAdded.push({
              ...ch,
              title: res.title || ch.title || '',
            });
          }
        } else if (item.valid && item.normalizedHandle) {
          const isRss = /^https?:\/\//i.test(item.normalizedHandle);
          const res = isRss
            ? { registered: false }
            : await registerTelegramChannel(item.normalizedHandle);
          newlyAdded.push({
            channelId:
              res.channelId ?? `ch_${item.normalizedHandle.replace(/^@/, '').toLowerCase()}`,
            handle: item.normalizedHandle,
            title: res.title ?? item.title ?? '',
            sourceType: item.kind === 'addlist' ? 'imported_addlist' : 'imported_link',
            platform: isRss ? 'RSS' : 'TG',
          });
        }
      }
      if (newlyAdded.length > 0) {
        onAddChannels?.(newlyAdded);
      }
      setImportStatus(
        newlyAdded.length > 0
          ? `Подключено источников: ${newlyAdded.length}. Лента обновляется…`
          : 'Не найдено доступных открытых каналов',
      );
      if (newlyAdded.length > 0) {
        onChannelsImported();
      }
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <div
      className="modal-backdrop density-comfortable"
      role="dialog"
      aria-modal="true"
      aria-label="Каналы и папки"
      onClick={onClose}
    >
      <div
        className="modal-card"
        style={{ maxWidth: '700px' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="row-between" style={{ marginBottom: '0.75rem', gap: '0.5rem' }}>
          <div>
            <h2 className="typo-h2">Каналы и папки</h2>
            <p className="typo-meta">
              Подписки работают без ввода номера телефона, SMS-кодов и доступа к вашей переписке
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
            aria-selected={activeTab === 'presets'}
            className={activeTab === 'presets' ? 'active' : ''}
            onClick={() => setActiveTab('presets')}
          >
            Папки и подборки
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'import'}
            className={activeTab === 'import' ? 'active' : ''}
            onClick={() => setActiveTab('import')}
          >
            Добавить каналы
          </button>
        </div>

        {activeTab === 'presets' ? (
          <div className="stack" style={{ gap: '0.85rem' }}>
            {onCreateFolder ? (
              <div className="rag-panel stack" style={{ gap: '0.55rem' }}>
                <strong className="typo-h3">Ваши папки</strong>
                <div className="row" style={{ gap: '0.45rem', flexWrap: 'wrap' }}>
                  <input
                    type="text"
                    className="text-input"
                    style={{ flex: 1, minWidth: '180px' }}
                    value={newFolderName}
                    onChange={(e) => setNewFolderName(e.target.value)}
                    placeholder="Название новой папки (например: Вакансии)"
                    aria-label="Название новой папки"
                  />
                  <button
                    type="button"
                    className="btn-primary"
                    disabled={!newFolderName.trim()}
                    onClick={() => {
                      if (!newFolderName.trim()) return;
                      onCreateFolder(newFolderName.trim(), newFolderChannelIds);
                      setNewFolderName('');
                      setNewFolderChannelIds([]);
                    }}
                  >
                    Создать папку
                  </button>
                </div>

                {session.subscribedChannels.length > 0 ? (
                  <div className="row" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
                    {session.subscribedChannels.map((ch) => {
                      const checked = newFolderChannelIds.includes(ch.channelId);
                      return (
                        <label key={ch.channelId} className="checkbox-inline">
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={(e) => {
                              setNewFolderChannelIds((prev) =>
                                e.target.checked
                                  ? [...prev, ch.channelId]
                                  : prev.filter((id) => id !== ch.channelId),
                              );
                            }}
                          />
                          <span>{ch.title || ch.handle}</span>
                        </label>
                      );
                    })}
                  </div>
                ) : null}

                {folders.length > 0 ? (
                  <div className="stack" style={{ gap: '0.45rem', marginTop: '0.25rem' }}>
                    {folders.map((folder) => (
                      <div
                        key={folder.id}
                        className="surface-card rounded-xl p-2.5 stack"
                        style={{ gap: '0.4rem' }}
                      >
                        <div className="row-between" style={{ gap: '0.4rem', flexWrap: 'wrap' }}>
                          <strong className="typo-meta" style={{ color: 'var(--text-primary)' }}>
                            {folder.name} ({folder.channelIds.filter((id) =>
                              session.subscribedChannels.some((s) => s.channelId === id),
                            ).length})
                          </strong>
                          {onDeleteFolder ? (
                            <button
                              type="button"
                              className="btn-secondary"
                              onClick={() => onDeleteFolder(folder.id)}
                            >
                              Удалить папку
                            </button>
                          ) : null}
                        </div>
                        {onUpdateFolder && session.subscribedChannels.length > 0 ? (
                          <div className="row" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
                            {session.subscribedChannels.map((ch) => {
                              const inFolder = folder.channelIds.includes(ch.channelId);
                              return (
                                <label key={`${folder.id}-${ch.channelId}`} className="checkbox-inline">
                                  <input
                                    type="checkbox"
                                    checked={inFolder}
                                    onChange={(e) => {
                                      const nextIds = e.target.checked
                                        ? Array.from(new Set([...folder.channelIds, ch.channelId]))
                                        : folder.channelIds.filter((id) => id !== ch.channelId);
                                      onUpdateFolder(folder.id, { channelIds: nextIds });
                                    }}
                                  />
                                  <span>{ch.title || ch.handle}</span>
                                </label>
                              );
                            })}
                          </div>
                        ) : null}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="typo-meta">Папок пока нет — создайте папку и выберите нужные каналы.</p>
                )}
              </div>
            ) : null}

            <div className="stack" style={{ gap: '0.45rem' }}>
              <strong className="typo-meta">Готовые тематические подборки:</strong>
              {INDUSTRY_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  className={`preset-option ${
                    selectedPreset?.id === preset.id ? 'preset-option-active' : ''
                  }`}
                  onClick={() => setSelectedPresetId(preset.id)}
                >
                  <div className="row-between">
                    <strong className="typo-h3">{preset.title}</strong>
                    <span className="badge badge-neutral">
                      {preset.channels.length} каналов
                    </span>
                  </div>
                  <p className="typo-meta">{preset.description}</p>
                </button>
              ))}
            </div>

            {selectedPreset ? (
              <div className="rag-panel">
                <span className="typo-meta">Входят каналы:</span>
                <p className="typo-body" style={{ marginTop: '0.25rem' }}>
                  {selectedPreset.channels.map((c) => `${c.title} (${c.handle})`).join(', ')}
                </p>
                <div style={{ marginTop: '0.6rem' }}>
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => {
                      onApplyPreset(selectedPreset.id);
                      onClose();
                    }}
                  >
                    Подключить подборку «{selectedPreset.title}»
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {activeTab === 'import' ? (
          <div className="stack" style={{ gap: '0.65rem' }}>
            <div className="row-between" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
              <p className="typo-body">
                Укажите ссылки на открытые каналы Telegram, папки или RSS-ленты (по одной на строке):
              </p>
              <button
                type="button"
                className="btn-link"
                onClick={() => setImportText(SAMPLE_IMPORT_LINES)}
              >
                Вставить пример
              </button>
            </div>
            <textarea
              className="text-input"
              rows={4}
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
              placeholder={'https://t.me/channel_name\n@channel_name\nhttps://t.me/addlist/...\nhttps://site.com/rss.xml'}
              aria-label="Ссылки на каналы или папки"
            />
            <div className="row" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn-primary"
                disabled={isImporting}
                onClick={() => void handleConfirmImport()}
              >
                {isImporting ? (
                  <>
                    <span className="loading-spinner" aria-hidden="true" />
                    <span>Подключаем…</span>
                  </>
                ) : (
                  'Проверить и добавить'
                )}
              </button>
              <button
                type="button"
                className="btn-secondary"
                disabled={isImporting}
                onClick={handleAnalyzeImport}
              >
                Только проверить список
              </button>
            </div>
            {importStatus ? <p className="typo-meta">{importStatus}</p> : null}
            {parsedImports.length > 0 ? (
              <div className="stack" style={{ gap: '0.3rem' }}>
                {parsedImports.map((item, idx) => (
                  <div key={idx} className="row-between" style={{ gap: '0.5rem' }}>
                    <span className="typo-meta truncate">{item.rawInput}</span>
                    <span
                      className={
                        !item.valid
                          ? 'badge badge-unsupported'
                          : 'badge badge-entailed'
                      }
                    >
                      {item.valid
                        ? item.kind === 'addlist'
                          ? `Папка (${item.resolvedChannels.length} каналов)`
                          : `Готово: ${item.normalizedHandle ?? item.title ?? ''}`
                        : item.errorReason ?? 'Некорректная ссылка'}
                    </span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
