"""Ступень 3 воронки комментариев: MMR-диверсификация, Context-Ancestor Closure и кэш бакетов."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from app.comments.bug_rescue import BugReportRescuer
from app.comments.graph_pruner import (
    RawComment,
    ScoredComment,
    filter_and_score_comment_tree,
)
from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import Qwen3EmbeddingClient, cosine_similarity


def resolve_comment_bucket(
    comment_count: int,
    buckets: Sequence[int] = (20, 50, 150, 500),
) -> int:
    """Отображает число комментариев поста в один из бакетов кэширования 20/50/150/500."""
    sorted_buckets = sorted(buckets)
    for b in sorted_buckets:
        if comment_count <= b:
            return b
    return sorted_buckets[-1]


@dataclass(slots=True)
class SelectedCommentItem:
    """Финально отобранный комментарий для индексации `dense_comments` и LLM-суммаризации."""

    comment_id: int
    reply_to_comment_id: int | None
    graph_score: float
    is_rescued_bug_report: bool
    confirmations_count: int
    raw_text: str
    formatted_text: str
    ancestor_preview: str | None = None


@dataclass(slots=True)
class CommentFunnelResult:
    """Результат работы on-demand воронки комментариев для поста."""

    post_id: int
    total_raw_comments: int
    bucket: int
    cache_hit: bool
    kept_after_pruning: int
    selected_comments: list[SelectedCommentItem] = field(default_factory=list)
    digest_text: str = ""
    dense_comments_vector: list[float] | None = None


class CommentBucketCache:
    """In-memory (или Redis-адаптер) кэш сжатых комментариев по `(post_id, bucket)`."""

    def __init__(self) -> None:
        self._store: dict[tuple[int, int], CommentFunnelResult] = {}

    def get(self, post_id: int, bucket: int) -> CommentFunnelResult | None:
        return self._store.get((post_id, bucket))

    def set(self, post_id: int, bucket: int, value: CommentFunnelResult) -> None:
        self._store[(post_id, bucket)] = value

    def invalidate_post(self, post_id: int) -> None:
        keys = [k for k in self._store if k[0] == post_id]
        for k in keys:
            del self._store[k]


def _truncate_clean(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


class CommentFunnelSummarizer:
    """Трехступенчатая воронка сжатия комментариев с кэшированием по бакетам 20/50/150/500."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Qwen3EmbeddingClient | None = None,
        rescuer: BugReportRescuer | None = None,
        cache: CommentBucketCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Qwen3EmbeddingClient(self.settings)
        self.rescuer = rescuer or BugReportRescuer(self.settings)
        self.cache = cache or CommentBucketCache()

    def _base_utility(self, item: ScoredComment) -> float:
        base = item.graph_score
        if item.is_rescued_bug_report:
            base += 0.85 * item.rescue_score
        base += 0.08 * math_log1p(item.reactions_count)
        return base

    def summarize(
        self,
        *,
        post_id: int,
        comments: Sequence[RawComment],
        max_selected: int = 8,
        force_refresh: bool = False,
        compute_embedding: bool = True,
    ) -> CommentFunnelResult:
        total_count = len(comments)
        bucket = resolve_comment_bucket(total_count, self.settings.comment_buckets)

        if not force_refresh:
            cached = self.cache.get(post_id, bucket)
            if cached is not None:
                return CommentFunnelResult(
                    post_id=cached.post_id,
                    total_raw_comments=total_count,
                    bucket=bucket,
                    cache_hit=True,
                    kept_after_pruning=cached.kept_after_pruning,
                    selected_comments=list(cached.selected_comments),
                    digest_text=cached.digest_text,
                    dense_comments_vector=(
                        list(cached.dense_comments_vector)
                        if cached.dense_comments_vector is not None
                        else None
                    ),
                )

        # Ступень 1 и 2: очистка и топологический GraphScore
        scored, all_cleaned = filter_and_score_comment_tree(
            comments,
            wd=self.settings.comment_graph_wd,
            wu=self.settings.comment_graph_wu,
            wp=self.settings.comment_graph_wp,
        )
        # Ступень 2.5: спасение одиночных баг-репортов и поправок (GraphScore == 0)
        candidates = self.rescuer.rescue_zero_graph_comments(scored)
        if not candidates:
            empty_res = CommentFunnelResult(
                post_id=post_id,
                total_raw_comments=total_count,
                bucket=bucket,
                cache_hit=False,
                kept_after_pruning=0,
            )
            self.cache.set(post_id, bucket, empty_res)
            return empty_res

        # Сортируем кандидатов по базовой ценности и готовим эмбеддинги для MMR-дедупликации
        candidates.sort(key=lambda c: (self._base_utility(c), c.reactions_count, -c.comment_id), reverse=True)
        cand_embs = self.embedder.encode_documents(
            [c.text for c in candidates],
            reject_instruct_prefix=False,
        )
        utilities = [self._base_utility(c) for c in candidates]
        max_util = max(utilities) or 1.0
        norm_rels = [u / max_util for u in utilities]

        # Ступень 3: MMR-диверсификация + схлопывание дублей в счетчик подтверждений
        selected_indices: list[int] = []
        confirmations_by_selected_idx: dict[int, int] = {}
        consumed: set[int] = set()
        dedup_thresh = self.settings.comment_mmr_dedup_threshold
        mmr_lambda = self.settings.comment_mmr_lambda

        # 1. Сначала поглощаем очевидные дубли, группируя к первому/лучшему представителю
        # и выбираем до max_selected разнородных комментариев через MMR.
        for _ in range(min(max_selected, len(candidates))):
            best_idx: int | None = None
            best_mmr = float("-inf")

            for idx in range(len(candidates)):
                if idx in consumed:
                    continue
                # Проверяем сходство с уже выбранными: если >= dedup_thresh,
                # засчитываем как подтверждение ближайшего выбранного и исключаем из выбора.
                if selected_indices:
                    max_sim = -1.0
                    nearest_sel = selected_indices[0]
                    for sel_idx in selected_indices:
                        sim = cosine_similarity(cand_embs[idx], cand_embs[sel_idx])
                        if sim > max_sim:
                            max_sim = sim
                            nearest_sel = sel_idx
                    if max_sim >= dedup_thresh:
                        confirmations_by_selected_idx[nearest_sel] = (
                            confirmations_by_selected_idx.get(nearest_sel, 0) + 1
                        )
                        consumed.add(idx)
                        continue
                    mmr_val = mmr_lambda * norm_rels[idx] - (1.0 - mmr_lambda) * max_sim
                else:
                    mmr_val = norm_rels[idx]

                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = idx

            if best_idx is None:
                break
            selected_indices.append(best_idx)
            consumed.add(best_idx)
            confirmations_by_selected_idx.setdefault(best_idx, 0)

        # Проверим оставшиеся еще не просмотренные кандидаты на предмет подтверждений для выбранных
        for idx in range(len(candidates)):
            if idx in consumed:
                continue
            max_sim = -1.0
            nearest_sel = -1
            for sel_idx in selected_indices:
                sim = cosine_similarity(cand_embs[idx], cand_embs[sel_idx])
                if sim > max_sim:
                    max_sim = sim
                    nearest_sel = sel_idx
            if nearest_sel >= 0 and max_sim >= dedup_thresh:
                confirmations_by_selected_idx[nearest_sel] += 1
                consumed.add(idx)

        # Замыкание по предкам (Context-Ancestor Closure до 200 символов)
        selected_comment_ids = {candidates[i].comment_id for i in selected_indices}
        selected_items: list[SelectedCommentItem] = []

        for sel_idx in selected_indices:
            c = candidates[sel_idx]
            conf = confirmations_by_selected_idx.get(sel_idx, 0)
            pid = c.reply_to_comment_id
            ancestor_preview: str | None = None

            if pid is not None and pid in all_cleaned and pid not in selected_comment_ids:
                ancestor_preview = _truncate_clean(
                    all_cleaned[pid],
                    max_chars=self.settings.comment_ancestor_max_chars,
                )
                prefix = f'[#c{c.comment_id} -> ответ на #c{pid} "{ancestor_preview}"] '
            elif pid is not None and pid in all_cleaned:
                prefix = f"[#c{c.comment_id} -> ответ на #c{pid}] "
            else:
                prefix = f"[#c{c.comment_id}] "

            conf_suffix = (
                f" [Подтверждено в {conf} аналогичных комментариях]"
                if conf > 0
                else ""
            )
            formatted = f"{prefix}{c.text}{conf_suffix}"
            selected_items.append(
                SelectedCommentItem(
                    comment_id=c.comment_id,
                    reply_to_comment_id=c.reply_to_comment_id,
                    graph_score=c.graph_score,
                    is_rescued_bug_report=c.is_rescued_bug_report,
                    confirmations_count=conf,
                    raw_text=c.text,
                    formatted_text=formatted,
                    ancestor_preview=ancestor_preview,
                )
            )

        digest_text = "\n".join(item.formatted_text for item in selected_items)
        dense_vec: list[float] | None = None
        if compute_embedding and digest_text.strip():
            # Кодируем комментарии строго без Instruct-префикса (документное пространство Qwen3)
            vec = self.embedder.encode_document(digest_text, reject_instruct_prefix=False)
            dense_vec = np.asarray(vec, dtype=np.float32).tolist()

        result = CommentFunnelResult(
            post_id=post_id,
            total_raw_comments=total_count,
            bucket=bucket,
            cache_hit=False,
            kept_after_pruning=len(candidates),
            selected_comments=selected_items,
            digest_text=digest_text,
            dense_comments_vector=dense_vec,
        )
        self.cache.set(post_id, bucket, result)
        return result


def math_log1p(val: int) -> float:
    import math

    return math.log1p(max(0, val))
