"""Гибридный поиск (Weighted RRF, 5 веток Prefetch) и кросс-энкодерный реранкинг Qwen3-Reranker-0.6B."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.cache.semantic_cache import TwoLevelSemanticCache, hash_query_for_rerank
from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import Qwen3EmbeddingClient
from app.embeddings.qwen3_reranker import Qwen3RerankerClient
from app.preprocessing.entities_ner import has_exact_lexical_triggers
from app.retrieval.qdrant_repo import QdrantPublicChannelsRepo

DEFAULT_SEARCH_RERANK_INSTRUCTION = (
    "Given a user search query, judge whether the Telegram post contains direct factual "
    "information, code, or data that answers the query."
)


@dataclass(slots=True)
class RetrievedPostCandidate:
    """Уникальный пост-кандидат после группировки по `post_id` и реранкинга."""

    post_id: int
    channel_id: str
    channel_username: str
    channel_title: str
    text: str
    published_at: datetime
    rrf_score: float
    rerank_score: float | None = None
    cluster_id: str | None = None
    origin_post_id: int | None = None
    has_code: bool = False
    has_ocr: bool = False
    has_comments: bool = False
    source_url: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def build_reranker_document(self) -> str:
        """Формирует полный контекст поста + сработавших модальностей для кросс-энкодера."""
        parts = [self.text]
        code_txt = str(self.payload.get("code_text") or "").strip()
        ocr_txt = str(self.payload.get("ocr_text") or "").strip()
        comments_txt = str(self.payload.get("comments_digest_text") or "").strip()
        if code_txt:
            parts.append(f"[Code]\n{code_txt}")
        if ocr_txt:
            parts.append(f"[OCR]\n{ocr_txt}")
        if comments_txt:
            parts.append(f"[Comments]\n{comments_txt}")
        return "\n\n".join(p for p in parts if p)


@dataclass(slots=True)
class HybridSearchResult:
    """Результат работы конвейера гибридного поиска и кросс-энкодерного реранкинга."""

    query: str
    has_exact_entities: bool
    bm25_weight: float
    rrf_candidates_count: int
    reranked_posts: list[RetrievedPostCandidate]
    latency_ms: float
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0


def _parse_published_at(payload: Mapping[str, Any]) -> datetime:
    iso_val = payload.get("published_at_iso")
    if isinstance(iso_val, str) and iso_val:
        dt = datetime.fromisoformat(iso_val)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    raw = payload.get("published_at")
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if isinstance(raw, str) and raw:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


class HybridSearchService:
    """Конвейер: Weighted RRF (`query_points_groups`) -> Top-30 -> Qwen3-Reranker-0.6B -> Top-8..12."""

    def __init__(
        self,
        qdrant_repo: QdrantPublicChannelsRepo,
        *,
        settings: Settings | None = None,
        embedder: Qwen3EmbeddingClient | None = None,
        reranker: Qwen3RerankerClient | None = None,
        semantic_cache: TwoLevelSemanticCache | None = None,
    ) -> None:
        self.settings = settings or qdrant_repo.settings or get_settings()
        self.qdrant_repo = qdrant_repo
        self.embedder = embedder or qdrant_repo.embedder
        self.reranker = reranker or Qwen3RerankerClient(self.settings)
        self.semantic_cache = semantic_cache

    def search_and_rerank(
        self,
        *,
        query: str,
        subscribed_channel_ids: Sequence[int | str],
        published_after: datetime,
        top_k: int | None = None,
        group_limit: int | None = None,
        enable_reranker: bool = True,
        cluster_id: str | None = None,
        full_post_texts: Mapping[int, str] | None = None,
    ) -> HybridSearchResult:
        """Выполняет гибридный поиск Top-30 групп и реранкинг в Top-K документов."""
        t0 = time.perf_counter()
        target_top_k = top_k or self.settings.rerank_top_k_green
        target_group_limit = group_limit or self.settings.hybrid_group_limit

        has_exact = has_exact_lexical_triggers(query)
        bm25_weight = (
            self.settings.rrf_w_bm25_exact if has_exact else self.settings.rrf_w_bm25_base
        )

        # Запрос кодируется строго с Instruct-префиксом (раздел 7.2.1 SPEC)
        query_vec = self.embedder.encode_query(query)

        t_ret0 = time.perf_counter()
        groups_res = self.qdrant_repo.query_hybrid_groups(
            query_text=query,
            query_dense_512d=query_vec,
            subscribed_channel_ids=subscribed_channel_ids,
            published_after=published_after,
            has_exact_entities=has_exact,
            group_limit=target_group_limit,
            group_size=1,
            cluster_id=cluster_id,
        )
        retrieval_ms = round((time.perf_counter() - t_ret0) * 1000.0, 2)

        candidates: list[RetrievedPostCandidate] = []
        for group in groups_res.groups:
            if not group.hits:
                continue
            hit = group.hits[0]
            payload = dict(hit.payload or {})
            pid = int(payload.get("post_id", group.id))
            parent_text = (
                full_post_texts[pid]
                if full_post_texts is not None and pid in full_post_texts
                else str(payload.get("full_text") or payload.get("text") or "")
            )
            ch_id = str(payload.get("channel_id", ""))
            ch_user = str(payload.get("channel_username", "")).lstrip("@")
            ch_title = str(payload.get("channel_title", ch_user or ch_id))
            src_url = str(payload.get("source_url") or "")
            if not src_url and ch_user:
                src_url = f"https://t.me/{ch_user}/{pid}"

            candidates.append(
                RetrievedPostCandidate(
                    post_id=pid,
                    channel_id=ch_id,
                    channel_username=ch_user,
                    channel_title=ch_title,
                    text=parent_text,
                    published_at=_parse_published_at(payload),
                    rrf_score=float(hit.score),
                    cluster_id=str(payload["cluster_id"]) if payload.get("cluster_id") else None,
                    origin_post_id=(
                        int(payload["origin_post_id"])
                        if payload.get("origin_post_id") is not None
                        else pid
                    ),
                    has_code=bool(payload.get("has_code", False)),
                    has_ocr=bool(payload.get("has_ocr", False)),
                    has_comments=bool(payload.get("has_comments", False)),
                    source_url=src_url,
                    payload=payload,
                )
            )

        rerank_ms = 0.0
        if candidates and enable_reranker:
            t_rr0 = time.perf_counter()
            q_hash = hash_query_for_rerank(query)
            uncached_indices: list[int] = []
            uncached_docs: list[str] = []

            for idx, cand in enumerate(candidates):
                cached_score = (
                    self.semantic_cache.get_rerank_score(q_hash, cand.post_id)
                    if self.semantic_cache is not None
                    else None
                )
                if cached_score is not None:
                    cand.rerank_score = cached_score
                else:
                    uncached_indices.append(idx)
                    uncached_docs.append(cand.build_reranker_document())

            if uncached_docs:
                scores = self.reranker.score_batch(
                    query,
                    uncached_docs,
                    instruction=DEFAULT_SEARCH_RERANK_INSTRUCTION,
                )
                for idx, sc in zip(uncached_indices, scores):
                    candidates[idx].rerank_score = float(sc)
                    if self.semantic_cache is not None:
                        self.semantic_cache.put_rerank_score(
                            q_hash, candidates[idx].post_id, float(sc)
                        )

            candidates.sort(
                key=lambda c: (
                    c.rerank_score if c.rerank_score is not None else -1.0,
                    c.rrf_score,
                ),
                reverse=True,
            )
            rerank_ms = round((time.perf_counter() - t_rr0) * 1000.0, 2)
        else:
            candidates.sort(key=lambda c: c.rrf_score, reverse=True)

        selected = candidates[: max(1, target_top_k)] if candidates else []
        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return HybridSearchResult(
            query=query,
            has_exact_entities=has_exact,
            bm25_weight=bm25_weight,
            rrf_candidates_count=len(candidates),
            reranked_posts=selected,
            latency_ms=total_ms,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
        )
