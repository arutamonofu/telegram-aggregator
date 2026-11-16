'use client';

import React, { useState } from 'react';
import { IconArrowUp, IconSparkles } from '../ui/Icons';

export interface AiChatDockProps {
  onSubmitQuery: (query: string) => void;
  quotaRemaining?: number;
  quotaTotal?: number;
  canRunCloudRag?: boolean;
}

/**
 * Floating bottom AI chat island ("ИИ-ответ по источникам") that routes
 * submitted questions straight into the Search/RAG view.
 */
export function AiChatDock({
  onSubmitQuery,
  canRunCloudRag = true,
}: AiChatDockProps) {
  const [draft, setDraft] = useState('');

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed) return;
    setDraft('');
    onSubmitQuery(trimmed);
  };

  return (
    <div className="ai-chat-dock" data-testid="ai-chat-dock">
      <form
        className="ai-chat-island"
        onSubmit={handleSubmit}
        aria-label="ИИ-ответ по источникам"
      >
        <span className="ai-chat-sparkle" aria-hidden="true">
          <IconSparkles size={15} />
        </span>

        <input
          type="text"
          className="ai-chat-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={
            canRunCloudRag
              ? 'Спросить ИИ по источникам…'
              : 'Найти по источникам…'
          }
          aria-label="Запрос для ИИ-ответа по источникам"
        />

        <button
          type="submit"
          className="ai-chat-send-btn"
          disabled={!draft.trim()}
          aria-label="Отправить запрос в ИИ-поиск"
          data-tooltip="Спросить"
        >
          <IconArrowUp size={16} />
        </button>
      </form>
    </div>
  );
}
