"""Двухуровневый семантический кэш (разделы 7.3.2 и 7.8.5 SPEC.md).

Уровень 1:
    PublicCacheKey  = SHA256(Sorted(channel_ids) || TimeBucket || ExactEntities(numbers, tickers, law_ids, cli_flags))
    PrivateCacheKey = SHA256(tenant_id || K_tenant_salt || Sorted(channel_aliases) || TimeBucket || ExactEntities)
    Векторное сравнение (tau_cache in [0.88..0.95]) выполняется ИСКЛЮЧИТЕЛЬНО внутри
    совпавшего хэш-бакета SHA-256, что исключает коллизии вида 8080 vs 8081.

Уровень 2:
    Кэш предрасчетных карточек событийных кластеров (CanonicalEventCard) и оценок
    реранкинга для пользователей с уникальной корзиной каналов S_u.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import cosine_similarity
from app.preprocessing.entities_ner import (
    ExtractedEntities,
    extract_entities,
    format_exact_entities_signature,
)


def _canonical_channel_list(channel_ids: Sequence[int | str]) -> list[str]:
    unique = {str(cid).strip() for cid in channel_ids if str(cid).strip()}

    def _sort_key(val: str) -> tuple[int, int, str]:
        if val.lstrip("-").isdigit():
            return (0, int(val), val)
        return (1, 0, val)

    return sorted(unique, key=_sort_key)


def normalize_time_bucket(
    time_bucket: str | int,
    *,
    reference_time: datetime | None = None,
) -> str:
    """Нормализует временной бакет (например, '24h:2026-11-15' или число дней окна)."""
    if isinstance(time_bucket, str):
        cleaned = time_bucket.strip()
        if cleaned:
            return cleaned
    now = reference_time or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    days = int(time_bucket) if not isinstance(time_bucket, str) else 90
    if days <= 1:
        return f"1d:{now.strftime('%Y-%m-%dT%H')}"
    return f"{days}d:{now.strftime('%Y-%m-%d')}"


def _resolve_entities_signature(
    exact_entities: ExtractedEntities | str | Mapping[str, Sequence[str]],
) -> str:
    if isinstance(exact_entities, ExtractedEntities):
        return format_exact_entities_signature(exact_entities)
    if isinstance(exact_entities, str):
        return format_exact_entities_signature(extract_entities(exact_entities))
    ent = ExtractedEntities(
        ner=sorted({str(x) for x in exact_entities.get("ner", [])}),
        numbers=sorted({str(x) for x in exact_entities.get("numbers", [])}),
        tickers=sorted({str(x) for x in exact_entities.get("tickers", [])}),
        law_ids=sorted({str(x) for x in exact_entities.get("law_ids", [])}),
        cli_flags=sorted({str(x) for x in exact_entities.get("cli_flags", [])}),
    )
    return format_exact_entities_signature(ent)


def build_public_cache_key(
    channel_ids: Sequence[int | str],
    time_bucket: str | int,
    exact_entities: ExtractedEntities | str | Mapping[str, Sequence[str]],
    *,
    reference_time: datetime | None = None,
) -> str:
    """Вычисляет PublicCacheKey = SHA256(Sorted(channel_ids) || TimeBucket || ExactEntities)."""
    sorted_channels = ",".join(_canonical_channel_list(channel_ids))
    bucket = normalize_time_bucket(time_bucket, reference_time=reference_time)
    entities_sig = _resolve_entities_signature(exact_entities)
    raw = f"public||{sorted_channels}||{bucket}||{entities_sig}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_private_cache_key(
    tenant_id: str,
    tenant_salt: str,
    channel_aliases: Sequence[str],
    time_bucket: str | int,
    exact_entities: ExtractedEntities | str | Mapping[str, Sequence[str]],
    *,
    reference_time: datetime | None = None,
) -> str:
    """Вычисляет PrivateCacheKey = SHA256(tenant_id || K_tenant_salt || Sorted(channel_aliases) || TimeBucket || ExactEntities)."""
    if not tenant_id or not tenant_salt:
        raise ValueError("tenant_id and tenant_salt must be non-empty for PrivateCacheKey")
    sorted_aliases = ",".join(sorted({a.strip() for a in channel_aliases if a.strip()}))
    bucket = normalize_time_bucket(time_bucket, reference_time=reference_time)
    entities_sig = _resolve_entities_signature(exact_entities)
    raw = f"private||{tenant_id}||{tenant_salt}||{sorted_aliases}||{bucket}||{entities_sig}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_query_for_rerank(query: str) -> str:
    """Детерминированный хэш нормализованного текста запроса для Level 2 кэша реранкера."""
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class SemanticCacheEntry:
    """Запись Level 1 семантического кэша внутри конкретного хэш-бакета SHA-256."""

    cache_key: str
    query_text: str
    query_embedding: list[float]
    response_payload: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    is_private: bool = False
    tenant_id: str | None = None

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "query_text": self.query_text,
            "query_embedding": self.query_embedding,
            "response_payload": self.response_payload,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_private": self.is_private,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SemanticCacheEntry:
        return cls(
            cache_key=str(data["cache_key"]),
            query_text=str(data["query_text"]),
            query_embedding=[float(x) for x in data["query_embedding"]],
            response_payload=dict(data["response_payload"]),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            expires_at=datetime.fromisoformat(str(data["expires_at"])),
            is_private=bool(data.get("is_private", False)),
            tenant_id=data.get("tenant_id"),
        )


@dataclass(slots=True)
class Level1CacheHit:
    """Результат попадания в Level 1 семантического кэша."""

    cache_key: str
    similarity: float
    entry: SemanticCacheEntry


class TwoLevelSemanticCache:
    """Двухуровневый семантический кэш (SHA-256 бакеты + кэш кластерных карточек и реранка)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        redis_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.redis_client = redis_client
        self._l1_buckets: dict[str, list[SemanticCacheEntry]] = {}
        self._l2_event_cards: dict[str, dict[str, Any]] = {}
        self._l2_rerank_scores: dict[tuple[str, int], float] = {}

    def _load_bucket(self, cache_key: str) -> list[SemanticCacheEntry]:
        if cache_key in self._l1_buckets:
            return self._l1_buckets[cache_key]
        if self.redis_client is not None:
            raw = self.redis_client.get(f"semcache:l1:{cache_key}")
            if raw:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                items = [SemanticCacheEntry.from_dict(item) for item in json.loads(raw)]
                self._l1_buckets[cache_key] = items
                return items
        return []

    def _save_bucket(self, cache_key: str, entries: list[SemanticCacheEntry], ttl_sec: int) -> None:
        self._l1_buckets[cache_key] = entries
        if self.redis_client is not None:
            payload = json.dumps([e.to_dict() for e in entries], ensure_ascii=False)
            self.redis_client.set(f"semcache:l1:{cache_key}", payload, ex=max(1, ttl_sec))

    def get_level1(
        self,
        cache_key: str,
        query_embedding: Sequence[float] | np.ndarray,
        *,
        tau_cache: float | None = None,
        now: datetime | None = None,
        is_private: bool = False,
        tenant_id: str | None = None,
    ) -> Level1CacheHit | None:
        """Ищет похожий запрос ИСКЛЮЧИТЕЛЬНО внутри совпавшего хэш-бакета `cache_key`."""
        if is_private and not tenant_id:
            raise ValueError("Private contour lookup forbidden without tenant_id")

        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        threshold = self.settings.cache_tau_green if tau_cache is None else float(tau_cache)
        bucket_entries = self._load_bucket(cache_key)
        if not bucket_entries:
            return None

        active_entries = [e for e in bucket_entries if not e.is_expired(current_time)]
        if len(active_entries) != len(bucket_entries):
            self._l1_buckets[cache_key] = active_entries

        best_entry: SemanticCacheEntry | None = None
        best_sim = -1.0
        for entry in active_entries:
            if entry.is_private != is_private:
                continue
            if is_private and entry.tenant_id != tenant_id:
                continue
            sim = cosine_similarity(query_embedding, entry.query_embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= threshold:
            return Level1CacheHit(
                cache_key=cache_key,
                similarity=round(best_sim, 6),
                entry=best_entry,
            )
        return None

    def put_level1(
        self,
        cache_key: str,
        query_text: str,
        query_embedding: Sequence[float] | np.ndarray,
        response_payload: dict[str, Any],
        *,
        is_news_query: bool = True,
        ttl_sec: int | None = None,
        now: datetime | None = None,
        is_private: bool = False,
        tenant_id: str | None = None,
    ) -> SemanticCacheEntry:
        """Сохраняет ответ RAG в хэш-бакет `cache_key` с TTL (1-3ч новости, 24-72ч ретро)."""
        if is_private and not tenant_id:
            raise ValueError("Private contour store forbidden without tenant_id")

        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        effective_ttl = (
            int(ttl_sec)
            if ttl_sec is not None
            else (self.settings.cache_news_ttl_sec if is_news_query else self.settings.cache_retro_ttl_sec)
        )
        entry = SemanticCacheEntry(
            cache_key=cache_key,
            query_text=query_text,
            query_embedding=[float(x) for x in np.asarray(query_embedding, dtype=np.float32).tolist()],
            response_payload=dict(response_payload),
            created_at=current_time,
            expires_at=current_time + timedelta(seconds=effective_ttl),
            is_private=is_private,
            tenant_id=tenant_id,
        )
        existing = [e for e in self._load_bucket(cache_key) if not e.is_expired(current_time)]
        existing.append(entry)
        self._save_bucket(cache_key, existing, effective_ttl)
        return entry

    def put_event_card(self, cluster_id: str, card: dict[str, Any]) -> None:
        """Сохраняет предрасчетную карточку событийного кластера (Уровень 2 / Shared Map)."""
        self._l2_event_cards[str(cluster_id)] = dict(card)
        if self.redis_client is not None:
            self.redis_client.set(
                f"semcache:l2:cluster:{cluster_id}",
                json.dumps(card, ensure_ascii=False),
                ex=self.settings.cache_retro_ttl_sec,
            )

    def get_event_card(self, cluster_id: str) -> dict[str, Any] | None:
        cid = str(cluster_id)
        if cid in self._l2_event_cards:
            return dict(self._l2_event_cards[cid])
        if self.redis_client is not None:
            raw = self.redis_client.get(f"semcache:l2:cluster:{cid}")
            if raw:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                card = json.loads(raw)
                self._l2_event_cards[cid] = card
                return dict(card)
        return None

    def get_event_cards(
        self,
        cluster_ids: Sequence[str],
        subscribed_channel_ids: Sequence[int | str] | None = None,
    ) -> list[dict[str, Any]]:
        """Извлекает предрасчетные карточки кластеров с фильтрацией по подписке S_u."""
        sub_set = {str(cid) for cid in subscribed_channel_ids} if subscribed_channel_ids is not None else None
        cards: list[dict[str, Any]] = []
        for cid in cluster_ids:
            card = self.get_event_card(cid)
            if card is None:
                continue
            if sub_set is not None:
                card_channels = {str(c) for c in card.get("channel_ids", [])}
                if card_channels and card_channels.isdisjoint(sub_set):
                    continue
            cards.append(card)
        return cards

    def put_rerank_score(self, query_hash: str, post_id: int, score: float) -> None:
        self._l2_rerank_scores[(query_hash, int(post_id))] = float(score)

    def get_rerank_score(self, query_hash: str, post_id: int) -> float | None:
        return self._l2_rerank_scores.get((query_hash, int(post_id)))
