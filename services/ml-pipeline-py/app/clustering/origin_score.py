"""Расчет OriginScore и защита от грабберов (переключение effective date при правке > 30%)."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from app.preprocessing.normalizer import normalize_text_nfkc


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def compute_edit_change_ratio(first_version_text: str, current_text: str) -> float:
    """Доля изменившихся символов (от 0.0 до 1.0) относительно первой версии поста."""
    first_norm = normalize_text_nfkc(first_version_text).strip()
    curr_norm = normalize_text_nfkc(current_text).strip()
    if not first_norm and not curr_norm:
        return 0.0
    if not first_norm or not curr_norm:
        return 1.0
    similarity = difflib.SequenceMatcher(None, first_norm, curr_norm).ratio()
    return round(1.0 - similarity, 6)


def resolve_effective_published_at(
    published_at: datetime,
    edit_date: datetime | None,
    first_version_text: str,
    current_text: str,
    *,
    change_threshold: float = 0.30,
) -> tuple[datetime, float, bool]:
    """Определяет $t_i^{\\text{effective}}$ с защитой от «заглушек-грабберов».

    Раздел 7.2.3 SPEC.md:
    Если у поста зафиксировано редактирование (`edit_date`) и при этом изменилось
    более 30% символов относительно первой версии (`change_ratio > 0.30`),
    эффективной меткой времени становится `edit_date`, а не исходный `published_at`.
    Возвращает `(effective_ts, change_ratio, switched_to_edit_date)`.
    """
    pub_utc = _ensure_utc(published_at)
    change_ratio = compute_edit_change_ratio(first_version_text, current_text)
    if edit_date is not None and change_ratio > change_threshold:
        edit_utc = _ensure_utc(edit_date)
        if edit_utc >= pub_utc:
            return edit_utc, change_ratio, True
    return pub_utc, change_ratio, False


@dataclass(slots=True)
class OriginCandidate:
    """Кандидат на роль первоисточника внутри событийного кластера $C_j$."""

    post_id: int
    channel_id: int | str
    effective_published_at: datetime
    explicit_cite_score: float = 0.0
    channel_authority: float = 0.0
    originality_score: float = 1.0
    has_primary_evidence: bool = False


@dataclass(slots=True)
class ClusterOriginResult:
    """Результат пересчета первоисточника кластера."""

    origin_post_id: int
    origin_channel_id: int | str
    scores_by_post: dict[int, float]
    t_min: datetime


def compute_origin_score(
    candidate: OriginCandidate,
    *,
    t_min: datetime,
    alpha: float = 0.50,
    beta: float = 0.25,
    delta: float = 0.20,
    eta: float = 0.25,
    gamma_per_min: float = 0.50,
) -> float:
    """Вычисляет OriginScore по формуле раздела 7.2.3 SPEC.md.

    $$\\text{OriginScore}(P_i) = \\alpha S_{\\text{explicit\\_cite}}
      + \\beta \\log(1 + \\text{Authority}_{c_i})
      + \\delta \\text{Originality}
      + \\eta \\mathbb{I}[\\text{PrimaryEvidence}]
      - \\gamma (t_i^{\\text{effective}} - t_{\\min})$$
    где $(t_i^{\\text{effective}} - t_{\\min})$ измеряется в минутах.
    """
    t_eff = _ensure_utc(candidate.effective_published_at)
    t_min_utc = _ensure_utc(t_min)
    delay_minutes = max(0.0, (t_eff - t_min_utc).total_seconds() / 60.0)

    authority_term = math.log1p(max(0.0, candidate.channel_authority))
    evidence_flag = 1.0 if candidate.has_primary_evidence else 0.0

    score = (
        alpha * candidate.explicit_cite_score
        + beta * authority_term
        + delta * candidate.originality_score
        + eta * evidence_flag
        - gamma_per_min * delay_minutes
    )
    return round(score, 6)


def recompute_cluster_origin(
    candidates: Sequence[OriginCandidate],
    *,
    alpha: float = 0.50,
    beta: float = 0.25,
    delta: float = 0.20,
    eta: float = 0.25,
    gamma_per_min: float = 0.50,
) -> ClusterOriginResult:
    """Пересчитывает `OriginScore` и выбирает `origin_post_id` среди всех постов кластера."""
    if not candidates:
        raise ValueError("Cannot compute cluster origin for empty candidate list")

    t_min = min(_ensure_utc(c.effective_published_at) for c in candidates)
    scores_by_post: dict[int, float] = {}
    best_candidate = candidates[0]
    best_score = float("-inf")

    for cand in candidates:
        score = compute_origin_score(
            cand,
            t_min=t_min,
            alpha=alpha,
            beta=beta,
            delta=delta,
            eta=eta,
            gamma_per_min=gamma_per_min,
        )
        scores_by_post[cand.post_id] = score
        if score > best_score:
            best_score = score
            best_candidate = cand
        elif math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-9):
            # Детерминированный тай-брейк: кто опубликовал раньше, затем меньший post_id
            cand_ts = _ensure_utc(cand.effective_published_at)
            best_ts = _ensure_utc(best_candidate.effective_published_at)
            if (cand_ts, cand.post_id) < (best_ts, best_candidate.post_id):
                best_candidate = cand

    return ClusterOriginResult(
        origin_post_id=best_candidate.post_id,
        origin_channel_id=best_candidate.channel_id,
        scores_by_post=scores_by_post,
        t_min=t_min,
    )
