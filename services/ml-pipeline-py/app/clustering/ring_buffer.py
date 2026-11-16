"""Кольцевой буфер в RAM (15 мин) для устранения race-condition до индексации HNSW/Qdrant."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass(slots=True)
class RingBufferEntry:
    """Элемент горячего 15-минутного буфера в оперативной памяти."""

    post_id: int
    cluster_id: str
    channel_id: int | str
    published_at: datetime
    embedding: np.ndarray
    text: str
    entities: set[str] = field(default_factory=set)


class HotRingBuffer:
    """15-минутный кольцевой буфер свежих постов с векторным brute-force поиском Top-K."""

    def __init__(
        self,
        *,
        window_minutes: int = 15,
        max_size: int = 15000,
    ) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.max_size = max_size
        self._entries: deque[RingBufferEntry] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self._entries)

    def evict_expired(self, now: datetime) -> int:
        """Удаляет записи старше `window_minutes` относительно `now`."""
        now_utc = _ensure_utc(now)
        cutoff = now_utc - self.window
        removed = 0
        # Считаем по порядку (если записи приходят не строго монотонно, фильтруем аккуратно)
        if not self._entries:
            return 0
        if all(_ensure_utc(item.published_at) >= cutoff for item in self._entries):
            return 0
        kept = deque(maxlen=self.max_size)
        for item in self._entries:
            if _ensure_utc(item.published_at) >= cutoff:
                kept.append(item)
            else:
                removed += 1
        self._entries = kept
        return removed

    def add(self, entry: RingBufferEntry) -> None:
        entry.published_at = _ensure_utc(entry.published_at)
        vec = np.asarray(entry.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        entry.embedding = (vec / norm).astype(np.float32) if norm > 1e-12 else vec
        self.evict_expired(entry.published_at)
        self._entries.append(entry)

    def search_top_k(
        self,
        query_embedding: np.ndarray,
        now: datetime,
        *,
        k: int = 20,
    ) -> list[tuple[RingBufferEntry, float]]:
        """Возвращает Top-K ближайших записей по косинусному сходству за последние 15 мин."""
        self.evict_expired(now)
        if not self._entries or k <= 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 1e-12:
            return []
        q = q / q_norm

        entries = list(self._entries)
        mat = np.stack([item.embedding for item in entries], axis=0)
        sims = np.clip(mat @ q, -1.0, 1.0)
        order = np.argsort(-sims)[:k]
        return [(entries[int(idx)], float(sims[int(idx)])) for idx in order]
