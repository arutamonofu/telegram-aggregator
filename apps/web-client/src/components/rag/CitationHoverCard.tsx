'use client';

import React, { useState } from 'react';
import type { RagSourcePostItem, VerificationStatus } from '../../lib/types/feed';

export interface CitationPreviewData {
  postId: string;
  channelHandle: string;
  quotePreview: string;
  utf16Start: number;
  utf16End: number;
}

export interface CitationHoverCardProps {
  index: number;
  citation: CitationPreviewData;
  source?: RagSourcePostItem;
  verificationStatus?: VerificationStatus;
  onOpenSource: (postId: string) => void;
}

const STATUS_BADGE_MAP: Record<VerificationStatus, { label: string; className: string }> = {
  entailed: { label: 'Подтверждено цитатой', className: 'badge badge-entailed' },
  pending: { label: 'Проверяется…', className: 'badge badge-pending' },
  unsupported: { label: 'Нет точной цитаты', className: 'badge badge-unsupported' },
};

export function CitationHoverCard({
  index,
  citation,
  source,
  verificationStatus = 'entailed',
  onOpenSource,
}: CitationHoverCardProps) {
  const [hovered, setHovered] = useState(false);

  const badge = STATUS_BADGE_MAP[verificationStatus];
  const snippet = source
    ? source.post.text.slice(citation.utf16Start, citation.utf16End) || citation.quotePreview
    : citation.quotePreview;

  return (
    <span
      className="citation-wrapper"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        className={`citation-badge citation-${verificationStatus}`}
        onClick={() => onOpenSource(citation.postId)}
        onFocus={() => setHovered(true)}
        onBlur={() => setHovered(false)}
        aria-label={`Сноска [${index}]: ${source?.post.channelHandle ?? citation.channelHandle}`}
      >
        [{index}]
      </button>

      {hovered ? (
        <div className="citation-popover" role="tooltip">
          <div className="row-between" style={{ gap: '0.4rem', marginBottom: '0.3rem' }}>
            <strong className="typo-meta" style={{ color: 'var(--text-primary)' }}>
              {source?.post.channelHandle ?? citation.channelHandle}
            </strong>
            <span className={badge.className}>{badge.label}</span>
          </div>
          <p className="typo-meta" style={{ color: 'var(--text-primary)' }}>
            «{snippet}»
          </p>
          <p className="typo-meta" style={{ marginTop: '0.3rem', fontSize: '11.5px' }}>
            Нажмите, чтобы перейти к посту
          </p>
        </div>
      ) : null}
    </span>
  );
}
