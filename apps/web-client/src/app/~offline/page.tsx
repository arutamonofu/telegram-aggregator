'use client';

import React, { useState } from 'react';
import { SEED_CLEAN_POSTS, SEED_DIGEST_RESPONSE } from '../../lib/api/seed_data';
import { runLocalFts5Search } from '../../lib/local-vault/fts5_and_crypto';
import type { RagSourcePostItem } from '../../lib/types/feed';

export default function OfflineFallbackPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RagSourcePostItem[]>([]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const res = runLocalFts5Search(query, SEED_CLEAN_POSTS);
    setResults(res.matches);
  };

  return (
    <main className="reading-container" style={{ paddingTop: '2rem', paddingBottom: '3rem' }}>
      <div className="edition-header-card">
        <span className="badge badge-pending">Офлайн-режим PWA</span>
        <h1 className="typo-h2" style={{ marginTop: '0.4rem' }}>
          Нет подключения к сети — доступен локальный кэш и FTS5
        </h1>
        <p className="typo-meta" style={{ marginTop: '0.25rem' }}>
          Последний сохранённый выпуск: {SEED_DIGEST_RESPONSE.cards.length} тем (слот{' '}
          {SEED_DIGEST_RESPONSE.scheduleWindow ?? '09:00'})
        </p>
      </div>

      <form className="row" style={{ gap: '0.5rem', marginTop: '1rem' }} onSubmit={handleSearch}>
        <input
          type="search"
          className="text-input"
          style={{ flex: 1 }}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Локальный поиск FTS5 по сохранённым постам..."
          aria-label="Локальный поиск FTS5"
        />
        <button type="submit" className="btn-primary">
          Искать локально
        </button>
      </form>

      <div className="stack" style={{ marginTop: '1rem' }}>
        {results.length > 0
          ? results.map((src) => (
              <article key={src.post.id} className="rag-source-card">
                <strong className="typo-meta">{src.post.channelHandle}</strong>
                <p className="typo-body" style={{ marginTop: '0.25rem' }}>
                  {src.post.text}
                </p>
              </article>
            ))
          : SEED_DIGEST_RESPONSE.cards.map((card) => (
              <article key={card.clusterId} className="digest-card">
                <span className="badge badge-neutral">{card.rubric}</span>
                <h2 className="typo-h3" style={{ marginTop: '0.35rem' }}>
                  {card.neutralTitle}
                </h2>
                <p className="typo-meta" style={{ marginTop: '0.25rem' }}>
                  {card.primarySource.channelHandle} • {card.primarySource.publishedAt}
                </p>
              </article>
            ))}
      </div>
    </main>
  );
}
