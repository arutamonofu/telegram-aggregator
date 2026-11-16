"""Single-Pass Incremental Clustering в скользящем окне 72 часа (раздел 7.2.3 SPEC.md)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

import numpy as np

from app.clustering.origin_score import (
    ClusterOriginResult,
    OriginCandidate,
    recompute_cluster_origin,
    resolve_effective_published_at,
)
from app.clustering.ring_buffer import HotRingBuffer, RingBufferEntry
from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import Qwen3EmbeddingClient, cosine_similarity
from app.embeddings.qwen3_reranker import Qwen3RerankerClient
from app.preprocessing.entities_ner import extract_entities, jaccard_similarity

_TOKEN_RE = re.compile(r"[a-zа-яё0-9%]+", re.IGNORECASE)


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def compute_bm25_norm(
    query_text: str,
    medoid_text: str,
    *,
    k1: float = 1.2,
    b: float = 0.75,
    avg_doc_len: float = 35.0,
) -> float:
    """Нормализованная к [0, 1] оценка BM25(P_new, M_j) / BM25(M_j, M_j)."""
    q_tokens = [t[:5] for t in _TOKEN_RE.findall(query_text.lower())]
    d_tokens = [t[:5] for t in _TOKEN_RE.findall(medoid_text.lower())]
    if not q_tokens or not d_tokens:
        return 0.0

    d_counts = Counter(d_tokens)
    q_unique = set(q_tokens)
    d_unique = set(d_tokens)
    d_len = len(d_tokens)
    length_norm = 1.0 - b + b * (d_len / max(avg_doc_len, 1.0))

    def _bm25_raw(term_set: set[str]) -> float:
        total = 0.0
        for term in term_set:
            tf = d_counts.get(term, 0)
            if tf == 0:
                continue
            num = tf * (k1 + 1.0)
            den = tf + k1 * length_norm
            total += num / den
        return total

    max_score = _bm25_raw(d_unique)
    if max_score <= 1e-12:
        return 0.0
    raw = _bm25_raw(q_unique)
    return round(max(0.0, min(1.0, raw / max_score)), 6)


def compute_composite_similarity(
    *,
    cosine_centroid: float,
    bm25_norm: float,
    j_ent: float,
    delta_hours: float,
    w_cos: float = 0.55,
    w_bm25: float = 0.20,
    w_ent: float = 0.25,
    lambda_time_decay: float = 0.003,
) -> float:
    """Точная формула $S(P_{\\text{new}}, C_j)$ из раздела 7.2.3 SPEC.md:

    $$S(P_{\\text{new}}, C_j) = w_1 \\cos(e(P_{\\text{new}}), \\mu_j)
      + w_2 \\text{BM25}_{\\text{norm}}(P_{\\text{new}}, M_j)
      + w_3 J_{\\text{ent}}(E(P_{\\text{new}}), E(M_j))
      - \\lambda \\Delta t$$
    """
    val = (
        w_cos * cosine_centroid
        + w_bm25 * bm25_norm
        + w_ent * j_ent
        - lambda_time_decay * max(0.0, delta_hours)
    )
    return round(val, 6)


@dataclass(slots=True)
class ClusterMember:
    """Пост в составе кластера событий $C_j$."""

    post_id: int
    channel_id: int | str
    first_version_text: str
    current_text: str
    embedding: np.ndarray
    entities: set[str]
    published_at: datetime
    edit_date: datetime | None = None
    effective_published_at: datetime | None = None
    explicit_cite_score: float = 0.0
    channel_authority: float = 0.0
    originality_score: float = 1.0
    has_primary_evidence: bool = False
    origin_score: float = 0.0

    def __post_init__(self) -> None:
        self.published_at = _ensure_utc(self.published_at)
        if self.effective_published_at is None:
            self.effective_published_at = self.published_at
        else:
            self.effective_published_at = _ensure_utc(self.effective_published_at)
        if self.edit_date is not None:
            self.edit_date = _ensure_utc(self.edit_date)


@dataclass(slots=True)
class EventClusterState:
    """Состояние кластера $C_j$: центроид $\\mu_j$, медоид $M_j$ и первоисточник."""

    cluster_id: str
    centroid: np.ndarray
    embedding_sum: np.ndarray
    medoid_post_id: int
    medoid_text: str
    medoid_entities: set[str]
    medoid_embedding: np.ndarray
    last_updated_at: datetime
    origin_post_id: int
    origin_channel_id: int | str
    members: list[ClusterMember] = field(default_factory=list)

    def refresh_centroid_and_medoid(self) -> None:
        if not self.members:
            return
        emb_sum = np.sum([m.embedding for m in self.members], axis=0).astype(np.float32)
        self.embedding_sum = emb_sum
        norm = float(np.linalg.norm(emb_sum))
        self.centroid = (emb_sum / norm).astype(np.float32) if norm > 1e-12 else emb_sum

        # Медоид M_j = член кластера с максимальным косинусным сходством к центроиду
        best_member = self.members[0]
        best_sim = -2.0
        for member in self.members:
            sim = cosine_similarity(member.embedding, self.centroid)
            if sim > best_sim:
                best_sim = sim
                best_member = member
            elif math.isclose(sim, best_sim, rel_tol=0.0, abs_tol=1e-7):
                if (member.effective_published_at, member.post_id) < (
                    best_member.effective_published_at,
                    best_member.post_id,
                ):
                    best_member = member
        self.medoid_post_id = best_member.post_id
        self.medoid_text = best_member.current_text
        self.medoid_entities = set(best_member.entities)
        self.medoid_embedding = best_member.embedding
        self.last_updated_at = max(m.published_at for m in self.members)


@dataclass(slots=True)
class SimilarityBreakdown:
    """Подробный разбор оценки кандидата $C_j$."""

    cluster_id: str
    medoid_post_id: int
    cosine_centroid: float
    bm25_norm: float
    j_ent: float
    delta_hours: float
    composite_score: float
    reranker_score: float | None = None
    decision: str = ""


@dataclass(slots=True)
class ClusterAssignmentResult:
    """Результат потоковой кластеризации нового поста или его правки."""

    cluster_id: str
    is_new_cluster: bool
    decision: str
    origin_post_id: int
    origin_channel_id: int | str
    best_candidate: SimilarityBreakdown | None = None
    evaluated_candidates: list[SimilarityBreakdown] = field(default_factory=list)
    origin_result: ClusterOriginResult | None = None


QdrantCandidateFetcher = Callable[[np.ndarray, datetime, int], Sequence[str]]


class SinglePassClusterer:
    """Потоковый Single-Pass кластеризатор (окно 72 ч, RingBuffer 15 мин + Top-20 Qdrant)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Qwen3EmbeddingClient | None = None,
        reranker: Qwen3RerankerClient | None = None,
        ring_buffer: HotRingBuffer | None = None,
        qdrant_candidate_fetcher: QdrantCandidateFetcher | None = None,
        bm25_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Qwen3EmbeddingClient(self.settings)
        self.reranker = reranker or Qwen3RerankerClient(self.settings)
        self.ring_buffer = ring_buffer or HotRingBuffer(
            window_minutes=self.settings.ring_buffer_minutes,
            max_size=self.settings.ring_buffer_max_size,
        )
        self.qdrant_candidate_fetcher = qdrant_candidate_fetcher
        self._bm25_fn = bm25_fn or compute_bm25_norm
        self.clusters: dict[str, EventClusterState] = {}
        self.post_to_cluster: dict[int, str] = {}
        self._next_cluster_seq = 1

    def _new_cluster_id(self) -> str:
        cid = f"clu_{self._next_cluster_seq}"
        self._next_cluster_seq += 1
        return cid

    def register_existing_cluster(self, cluster: EventClusterState) -> None:
        """Регистрирует готовый кластер (например, для unit-тестов или гидратации из БД)."""
        cluster.refresh_centroid_and_medoid()
        self._recompute_origin_in_place(cluster)
        self.clusters[cluster.cluster_id] = cluster
        for m in cluster.members:
            self.post_to_cluster[m.post_id] = cluster.cluster_id
            self.ring_buffer.add(
                RingBufferEntry(
                    post_id=m.post_id,
                    cluster_id=cluster.cluster_id,
                    channel_id=m.channel_id,
                    published_at=m.published_at,
                    embedding=m.embedding,
                    text=m.current_text,
                    entities=set(m.entities),
                )
            )

    def _recompute_origin_in_place(self, cluster: EventClusterState) -> ClusterOriginResult:
        candidates = [
            OriginCandidate(
                post_id=m.post_id,
                channel_id=m.channel_id,
                effective_published_at=m.effective_published_at or m.published_at,
                explicit_cite_score=m.explicit_cite_score,
                channel_authority=m.channel_authority,
                originality_score=m.originality_score,
                has_primary_evidence=m.has_primary_evidence,
            )
            for m in cluster.members
        ]
        res = recompute_cluster_origin(
            candidates,
            alpha=self.settings.origin_alpha,
            beta=self.settings.origin_beta,
            delta=self.settings.origin_delta,
            eta=self.settings.origin_eta,
            gamma_per_min=self.settings.origin_gamma_per_min,
        )
        for member in cluster.members:
            member.origin_score = res.scores_by_post[member.post_id]
        cluster.origin_post_id = res.origin_post_id
        cluster.origin_channel_id = res.origin_channel_id
        return res

    def _active_candidate_clusters(
        self,
        query_emb: np.ndarray,
        now: datetime,
    ) -> list[EventClusterState]:
        now_utc = _ensure_utc(now)
        window = timedelta(hours=self.settings.sliding_window_hours)

        candidate_ids: list[str] = []
        seen: set[str] = set()

        # 1. Кандидаты из RAM Ring Buffer (последние 15 минут)
        for entry, _ in self.ring_buffer.search_top_k(
            query_emb,
            now_utc,
            k=self.settings.qdrant_top_k_candidates,
        ):
            if entry.cluster_id not in seen and entry.cluster_id in self.clusters:
                seen.add(entry.cluster_id)
                candidate_ids.append(entry.cluster_id)

        # 2. Кандидаты из Qdrant Top-20 (если подключен fetcher) либо из активного 72h кэша
        if self.qdrant_candidate_fetcher is not None:
            for cid in self.qdrant_candidate_fetcher(
                query_emb,
                now_utc,
                self.settings.qdrant_top_k_candidates,
            ):
                if cid not in seen and cid in self.clusters:
                    seen.add(cid)
                    candidate_ids.append(cid)
        else:
            # Локальный поиск по центроидам активных кластеров в окне 72 часов
            scored_local: list[tuple[float, str]] = []
            for cid, clu in self.clusters.items():
                if cid in seen:
                    continue
                if abs((now_utc - _ensure_utc(clu.last_updated_at)).total_seconds()) <= window.total_seconds():
                    sim = cosine_similarity(query_emb, clu.centroid)
                    scored_local.append((sim, cid))
            scored_local.sort(key=lambda item: item[0], reverse=True)
            for _, cid in scored_local[: self.settings.qdrant_top_k_candidates]:
                seen.add(cid)
                candidate_ids.append(cid)

        # Строго фильтруем 72-часовое скользящее окно
        active: list[EventClusterState] = []
        for cid in candidate_ids:
            clu = self.clusters[cid]
            age_sec = abs((now_utc - _ensure_utc(clu.last_updated_at)).total_seconds())
            if age_sec <= window.total_seconds():
                active.append(clu)
        return active

    def score_cluster_candidate(
        self,
        *,
        post_text: str,
        post_embedding: np.ndarray,
        post_entities: set[str],
        published_at: datetime,
        cluster: EventClusterState,
        cos_override: float | None = None,
        bm25_override: float | None = None,
        j_ent_override: float | None = None,
    ) -> SimilarityBreakdown:
        """Вычисляет формулу $S(P_{\\text{new}}, C_j)$ для конкретного кластера."""
        pub_utc = _ensure_utc(published_at)
        medoid_member = next(
            (m for m in cluster.members if m.post_id == cluster.medoid_post_id),
            None,
        )
        medoid_ts = (
            _ensure_utc(medoid_member.published_at)
            if medoid_member is not None
            else _ensure_utc(cluster.last_updated_at)
        )
        delta_hours = max(0.0, (pub_utc - medoid_ts).total_seconds() / 3600.0)

        cos_val = (
            float(cos_override)
            if cos_override is not None
            else cosine_similarity(post_embedding, cluster.centroid)
        )
        bm25_val = (
            float(bm25_override)
            if bm25_override is not None
            else self._bm25_fn(post_text, cluster.medoid_text)
        )
        j_ent_val = (
            float(j_ent_override)
            if j_ent_override is not None
            else jaccard_similarity(post_entities, cluster.medoid_entities)
        )

        s_val = compute_composite_similarity(
            cosine_centroid=cos_val,
            bm25_norm=bm25_val,
            j_ent=j_ent_val,
            delta_hours=delta_hours,
            w_cos=self.settings.w_cos,
            w_bm25=self.settings.w_bm25,
            w_ent=self.settings.w_ent,
            lambda_time_decay=self.settings.lambda_time_decay,
        )
        return SimilarityBreakdown(
            cluster_id=cluster.cluster_id,
            medoid_post_id=cluster.medoid_post_id,
            cosine_centroid=round(cos_val, 6),
            bm25_norm=round(bm25_val, 6),
            j_ent=round(j_ent_val, 6),
            delta_hours=round(delta_hours, 6),
            composite_score=s_val,
        )

    def ingest_post(
        self,
        *,
        post_id: int,
        channel_id: int | str,
        text: str,
        published_at: datetime,
        embedding: np.ndarray | None = None,
        entities: set[str] | None = None,
        edit_date: datetime | None = None,
        first_version_text: str | None = None,
        explicit_cite_score: float = 0.0,
        channel_authority: float = 0.0,
        originality_score: float = 1.0,
        has_primary_evidence: bool = False,
    ) -> ClusterAssignmentResult:
        """Выполняет Single-Pass кластеризацию нового поста $P_{\\text{new}}$."""
        pub_utc = _ensure_utc(published_at)
        emb = (
            np.asarray(embedding, dtype=np.float32)
            if embedding is not None
            else self.embedder.encode_document(text, reject_instruct_prefix=False)
        )
        emb_norm = float(np.linalg.norm(emb))
        if emb_norm > 1e-12:
            emb = (emb / emb_norm).astype(np.float32)

        ents = entities if entities is not None else extract_entities(text).all_canonical
        initial_text = first_version_text if first_version_text is not None else text
        effective_ts, _, _ = resolve_effective_published_at(
            pub_utc,
            edit_date,
            initial_text,
            text,
            change_threshold=self.settings.stub_edit_change_threshold,
        )

        new_member = ClusterMember(
            post_id=post_id,
            channel_id=channel_id,
            first_version_text=initial_text,
            current_text=text,
            embedding=emb,
            entities=set(ents),
            published_at=pub_utc,
            edit_date=edit_date,
            effective_published_at=effective_ts,
            explicit_cite_score=explicit_cite_score,
            channel_authority=channel_authority,
            originality_score=originality_score,
            has_primary_evidence=has_primary_evidence,
        )

        candidates = self._active_candidate_clusters(emb, pub_utc)
        breakdowns = [
            self.score_cluster_candidate(
                post_text=text,
                post_embedding=emb,
                post_entities=new_member.entities,
                published_at=pub_utc,
                cluster=clu,
            )
            for clu in candidates
        ]
        breakdowns.sort(key=lambda b: b.composite_score, reverse=True)

        tau_low = self.settings.clustering_tau_low
        tau_high = self.settings.clustering_tau_high
        rerank_thresh = self.settings.rerank_merge_threshold

        chosen_cluster: EventClusterState | None = None
        chosen_breakdown: SimilarityBreakdown | None = None
        final_decision = "new_cluster"

        if breakdowns:
            top = breakdowns[0]
            if top.composite_score >= tau_high:
                top.decision = "merge_auto"
                chosen_cluster = self.clusters[top.cluster_id]
                chosen_breakdown = top
                final_decision = "merge_auto"
            elif top.composite_score >= tau_low:
                # Серая зона [0.72, 0.88): проверяем строго по медоиду M_j через Qwen3-Reranker-0.6B
                grey_candidates = [
                    b for b in breakdowns if tau_low <= b.composite_score < tau_high
                ][:3]
                for item in grey_candidates:
                    clu = self.clusters[item.cluster_id]
                    rerank_prob = self.reranker.score_pair(text, clu.medoid_text)
                    item.reranker_score = rerank_prob
                    if rerank_prob >= rerank_thresh:
                        item.decision = "merge_reranked"
                        chosen_cluster = clu
                        chosen_breakdown = item
                        final_decision = "merge_reranked"
                        break
                    item.decision = "reject_rerank"
                if chosen_cluster is None:
                    chosen_breakdown = breakdowns[0]
                    final_decision = "reject_rerank"
            else:
                top.decision = "reject_below_tau_low"
                chosen_breakdown = top
                final_decision = "reject_below_tau_low"

        if chosen_cluster is not None:
            chosen_cluster.members.append(new_member)
            chosen_cluster.refresh_centroid_and_medoid()
            origin_res = self._recompute_origin_in_place(chosen_cluster)
            self.post_to_cluster[post_id] = chosen_cluster.cluster_id
            self.ring_buffer.add(
                RingBufferEntry(
                    post_id=post_id,
                    cluster_id=chosen_cluster.cluster_id,
                    channel_id=channel_id,
                    published_at=pub_utc,
                    embedding=emb,
                    text=text,
                    entities=set(new_member.entities),
                )
            )
            return ClusterAssignmentResult(
                cluster_id=chosen_cluster.cluster_id,
                is_new_cluster=False,
                decision=final_decision,
                origin_post_id=chosen_cluster.origin_post_id,
                origin_channel_id=chosen_cluster.origin_channel_id,
                best_candidate=chosen_breakdown,
                evaluated_candidates=breakdowns,
                origin_result=origin_res,
            )

        # Создаем новый кластер
        new_cid = self._new_cluster_id()
        new_cluster = EventClusterState(
            cluster_id=new_cid,
            centroid=emb.copy(),
            embedding_sum=emb.copy(),
            medoid_post_id=post_id,
            medoid_text=text,
            medoid_entities=set(new_member.entities),
            medoid_embedding=emb.copy(),
            last_updated_at=pub_utc,
            origin_post_id=post_id,
            origin_channel_id=channel_id,
            members=[new_member],
        )
        origin_res = self._recompute_origin_in_place(new_cluster)
        self.clusters[new_cid] = new_cluster
        self.post_to_cluster[post_id] = new_cid
        self.ring_buffer.add(
            RingBufferEntry(
                post_id=post_id,
                cluster_id=new_cid,
                channel_id=channel_id,
                published_at=pub_utc,
                embedding=emb,
                text=text,
                entities=set(new_member.entities),
            )
        )
        return ClusterAssignmentResult(
            cluster_id=new_cid,
            is_new_cluster=True,
            decision=final_decision,
            origin_post_id=new_cluster.origin_post_id,
            origin_channel_id=new_cluster.origin_channel_id,
            best_candidate=chosen_breakdown,
            evaluated_candidates=breakdowns,
            origin_result=origin_res,
        )

    def handle_post_edited(
        self,
        *,
        post_id: int,
        new_text: str,
        edit_date: datetime,
        new_embedding: np.ndarray | None = None,
    ) -> ClusterAssignmentResult | None:
        """Обрабатывает редактирование поста: пересчет effective_ts (>30% -> edit_date) и OriginScore."""
        cid = self.post_to_cluster.get(post_id)
        if cid is None or cid not in self.clusters:
            return None
        cluster = self.clusters[cid]
        member = next((m for m in cluster.members if m.post_id == post_id), None)
        if member is None:
            return None

        edit_utc = _ensure_utc(edit_date)
        member.current_text = new_text
        member.edit_date = edit_utc
        member.entities = extract_entities(new_text).all_canonical
        if new_embedding is not None:
            member.embedding = np.asarray(new_embedding, dtype=np.float32)
        else:
            member.embedding = self.embedder.encode_document(new_text, reject_instruct_prefix=False)

        eff_ts, _, switched = resolve_effective_published_at(
            member.published_at,
            edit_utc,
            member.first_version_text,
            new_text,
            change_threshold=self.settings.stub_edit_change_threshold,
        )
        member.effective_published_at = eff_ts

        cluster.refresh_centroid_and_medoid()
        origin_res = self._recompute_origin_in_place(cluster)
        return ClusterAssignmentResult(
            cluster_id=cluster.cluster_id,
            is_new_cluster=False,
            decision="edit_switched_to_edit_date" if switched else "edit_minor_preserved",
            origin_post_id=cluster.origin_post_id,
            origin_channel_id=cluster.origin_channel_id,
            origin_result=origin_res,
        )
