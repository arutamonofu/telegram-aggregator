'use client';

import React from 'react';
import type { InspectedLink } from '@/lib/ast/url_safety';

export interface ExternalLinkModalProps {
  link: InspectedLink | null;
  onClose: () => void;
  onConfirmOpen?: (url: string) => void;
}

export function ExternalLinkModal({ link, onClose, onConfirmOpen }: ExternalLinkModalProps) {
  if (!link) {
    return null;
  }

  const handleProceed = () => {
    if (!link.isAllowedProtocol) {
      return;
    }
    if (onConfirmOpen) {
      onConfirmOpen(link.normalizedUrl);
    } else if (typeof window !== 'undefined') {
      window.open(link.normalizedUrl, '_blank', 'noopener,noreferrer');
    }
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/35 p-4 backdrop-blur-xs"
      role="dialog"
      aria-modal="true"
      aria-labelledby="external-link-modal-title"
      data-testid="external-link-modal"
    >
      <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-6 shadow-xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <span className="inline-flex items-center rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-800 ring-1 ring-amber-200">
              Проверка внешней ссылки
            </span>
            <h2
              id="external-link-modal-title"
              className="mt-2 text-lg font-semibold text-slate-900"
            >
              Переход на внешний ресурс
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
            aria-label="Закрыть предупреждение"
          >
            ✕
          </button>
        </div>

        <p className="mt-2 text-sm text-slate-600">
          Домен не входит в доверенный белый список или использует интернационализированные
          символы. Проверьте декодированный адрес перед открытием:
        </p>

        {link.warningReason ? (
          <div
            className={`mt-3 rounded-xl border px-3.5 py-2.5 text-xs font-medium ${
              link.hasMixedScripts || !link.isAllowedProtocol
                ? 'border-rose-200 bg-rose-50 text-rose-800'
                : 'border-amber-200 bg-amber-50 text-amber-800'
            }`}
            data-testid="link-warning-banner"
          >
            {link.warningReason}
          </div>
        ) : null}

        <div className="mt-4 space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-3.5">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
              Декодированный адрес (Unicode)
            </div>
            <div
              className="mt-1 break-all font-mono text-sm font-medium text-slate-900"
              data-testid="decoded-unicode-url"
            >
              {link.decodedFullUrl}
            </div>
          </div>

          {link.decodedHost && (
            <div className="grid grid-cols-2 gap-3 border-t border-slate-200 pt-2.5 text-xs">
              <div>
                <div className="text-slate-500">Домен (Unicode)</div>
                <div
                  className="mt-0.5 break-all font-mono font-semibold text-slate-800"
                  data-testid="decoded-unicode-host"
                >
                  {link.decodedHost}
                </div>
              </div>
              <div>
                <div className="text-slate-500">ASCII / Punycode</div>
                <div
                  className="mt-0.5 break-all font-mono text-slate-600"
                  data-testid="ascii-punycode-host"
                >
                  {link.asciiHost}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="mt-5 flex items-center justify-end gap-2.5">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Отмена
          </button>
          <button
            type="button"
            disabled={!link.isAllowedProtocol}
            onClick={handleProceed}
            className="btn-primary rounded-xl px-4 py-2 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40"
          >
            Перейти на сайт
          </button>
        </div>
      </div>
    </div>
  );
}
