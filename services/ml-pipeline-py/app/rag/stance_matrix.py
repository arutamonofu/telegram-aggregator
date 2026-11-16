"""4-шаговый конвейер Stance-Stratified Synthesis (матрицы «ЗА vs ПРОТИВ», раздел 7.4.3 SPEC.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import Qwen3EmbeddingClient, cosine_similarity
from app.preprocessing.normalizer import normalize_text_nfkc
from app.preprocessing.offsets import PostUtf16OffsetMap, build_utf16_offset_map
from app.rag.nli_gate import (
    _CON_POLARITY_CUES,
    _PRO_POLARITY_CUES,
    NliVerificationGate,
)
from app.rag.smith_waterman import smith_waterman_align
from app.rag.synthesizer import IndexedSourcePost, VerifiedCitation

StanceLabel = Literal["PRO", "CON", "MIXED", "NEUTRAL"]
StanceSide = Literal["PRO", "CON"]

_ASPECT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Стоимость и бюджет",
        ("стоим", "цен", "дорог", "дешев", "бюджет", "затрат", "расход", "эконом", "тариф", "выгод"),
    ),
    (
        "Производительность и скорость",
        ("скорост", "быстр", "ускор", "замедл", "латент", "rps", "нагруз", "память", "cpu", "gpu", "масштаб"),
    ),
    (
        "Надежность и безопасность",
        ("сбой", "риск", "уязвим", "ошибк", "утечк", "безопас", "стабиль", "отказ", "штраф", "блокир"),
    ),
    (
        "Архитектура и эксплуатация",
        ("внедр", "миграц", "поддерж", "интеграц", "документ", "сложн", "удоб", "прост", "совмест", "конфиг"),
    ),
)

DEFAULT_ASPECT = "Общая оценка"

_STRONG_PRO_RE = re.compile(
    r"\b(?:рекомендуем|выгодно|ускоряет|снижает\s+затраты|экономит|преимущество|плюс|удобно|поддерживаем)\b",
    re.IGNORECASE,
)
_STRONG_CON_RE = re.compile(
    r"\b(?:риск|уязвимость|дороже|замедляет|проблема|недостаток|минус|критикуем|не\s+рекомендуем|сбой|деградац)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class StancePostInput:
    """Входной пост для конвейера Stance-Stratified Synthesis."""

    post_id: int
    channel_id: str
    channel_username: str
    channel_title: str
    msg_id: int
    published_at: datetime
    text: str
    deeplink: str = ""
    is_ad: bool = False
    stance_hint: StanceLabel | None = None
    offset_map: PostUtf16OffsetMap | None = None

    def to_indexed_post(self) -> IndexedSourcePost:
        omap = self.offset_map or build_utf16_offset_map(self.text, post_id=self.post_id)
        pub = (
            self.published_at.replace(tzinfo=timezone.utc)
            if self.published_at.tzinfo is None
            else self.published_at.astimezone(timezone.utc)
        )
        link = self.deeplink
        if not link:
            user = self.channel_username.lstrip("@").strip()
            link = f"https://t.me/{user}/{self.msg_id}" if user else f"https://t.me/c/{self.channel_id}/{self.msg_id}"
        return IndexedSourcePost(
            post_id=self.post_id,
            channel_id=str(self.channel_id),
            channel_username=self.channel_username.lstrip("@"),
            channel_title=self.channel_title,
            msg_id=self.msg_id,
            published_at=pub,
            text=omap.text_nfkc,
            deeplink=link,
            offset_map=omap,
        )


@dataclass(slots=True)
class StanceCandidateArg:
    """Атомарный аргумент ЗА/ПРОТИВ до MMR-квотирования и двойной NLI-проверки."""

    arg_id: str
    side: StanceSide
    aspect: str
    statement: str
    post_id: int
    sentence_id: int | None
    channel_id: str
    exact_quote: str
    relevance_score: float = 1.0


@dataclass(slots=True)
class VerifiedStancePoint:
    """Верифицированный аргумент колонки PRO или CON со строгой ссылкой на источник."""

    claim_id: str
    side: StanceSide
    aspect: str
    statement: str
    citations: list[VerifiedCitation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "side": self.side,
            "aspect": self.aspect,
            "statement": self.statement,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass(slots=True)
class StanceMatrixRow:
    """Строка матрицы «ЗА vs ПРОТИВ» по конкретному аспекту обсуждения."""

    aspect: str
    pro_points: list[VerifiedStancePoint] = field(default_factory=list)
    con_points: list[VerifiedStancePoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "pro_points": [p.to_dict() for p in self.pro_points],
            "con_points": [c.to_dict() for c in self.con_points],
        }


@dataclass(slots=True)
class StanceMatrixResult:
    """Итог 4-шагового конвейера Stance-Stratified Synthesis."""

    query: str
    stance_counts: dict[str, int]
    selected_pro_count: int
    selected_con_count: int
    rejected_nli_count: int
    rejected_polarity_count: int
    rows: list[StanceMatrixRow]
    source_posts_payload: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "stance_counts": dict(self.stance_counts),
            "selected_pro_count": self.selected_pro_count,
            "selected_con_count": self.selected_con_count,
            "rejected_nli_count": self.rejected_nli_count,
            "rejected_polarity_count": self.rejected_polarity_count,
            "rows": [r.to_dict() for r in self.rows],
            "source_posts_payload": self.source_posts_payload,
        }


def _score_sentence_polarity(sentence: str) -> tuple[int, int]:
    norm = normalize_text_nfkc(sentence).lower().replace("ё", "е")
    pro_hits = sum(1 for cue in _PRO_POLARITY_CUES if cue in norm) + len(_STRONG_PRO_RE.findall(norm))
    con_hits = sum(1 for cue in _CON_POLARITY_CUES if cue in norm) + len(_STRONG_CON_RE.findall(norm))
    return pro_hits, con_hits


def infer_argument_aspect(text: str, explicit_aspect: str | None = None) -> str:
    """Определяет аспект сравнения (Стоимость, Производительность, Безопасность и др.)."""
    if explicit_aspect and explicit_aspect.strip():
        return explicit_aspect.strip()
    norm = normalize_text_nfkc(text).lower().replace("ё", "е")
    best_aspect = DEFAULT_ASPECT
    best_hits = 0
    for aspect, keywords in _ASPECT_RULES:
        hits = sum(1 for kw in keywords if kw in norm)
        if hits > best_hits:
            best_hits = hits
            best_aspect = aspect
    return best_aspect


def classify_post_stance(
    post: StancePostInput,
    *,
    query: str = "",
) -> StanceLabel:
    """Шаг 1 Stance-Stratified Synthesis: классификация поста в PRO | CON | MIXED | NEUTRAL."""
    if post.stance_hint in ("PRO", "CON", "MIXED", "NEUTRAL"):
        return post.stance_hint
    pro_hits, con_hits = _score_sentence_polarity(post.text)
    if pro_hits > 0 and con_hits > 0:
        return "MIXED"
    if pro_hits > 0:
        return "PRO"
    if con_hits > 0:
        return "CON"
    # Простая проверка на наличие явной нейтральной информативности vs отсутствия позиции
    _ = query
    return "NEUTRAL"


def extract_atomic_stance_arguments(
    posts: Sequence[StancePostInput],
    *,
    query: str = "",
) -> tuple[dict[str, int], list[StanceCandidateArg], list[StanceCandidateArg]]:
    """Шаг 1-2: отбраковка NEUTRAL/is_ad и разбор аргументов по сторонам PRO и CON."""
    stance_counts: dict[str, int] = {"PRO": 0, "CON": 0, "MIXED": 0, "NEUTRAL": 0}
    pro_args: list[StanceCandidateArg] = []
    con_args: list[StanceCandidateArg] = []
    arg_seq = 1

    for post in posts:
        if post.is_ad:
            continue
        label = classify_post_stance(post, query=query)
        stance_counts[label] += 1
        if label == "NEUTRAL":
            continue

        omap = post.offset_map or build_utf16_offset_map(post.text, post_id=post.post_id)
        sentences = omap.sentences
        for sent in sentences:
            s_text = sent.text.strip()
            if len(s_text) < 8:
                continue
            p_hits, c_hits = _score_sentence_polarity(s_text)
            if p_hits == 0 and c_hits == 0:
                if label in ("PRO", "CON") and sent.sentence_id == 1:
                    side: StanceSide = label
                else:
                    continue
            elif p_hits > c_hits:
                side = "PRO"
            elif c_hits > p_hits:
                side = "CON"
            else:
                if label in ("PRO", "CON"):
                    side = label
                else:
                    continue

            aspect = infer_argument_aspect(s_text)
            rel = 1.0 + 0.25 * max(p_hits, c_hits)
            cand = StanceCandidateArg(
                arg_id=f"arg_{arg_seq}",
                side=side,
                aspect=aspect,
                statement=s_text,
                post_id=post.post_id,
                sentence_id=sent.sentence_id,
                channel_id=str(post.channel_id),
                exact_quote=s_text,
                relevance_score=rel,
            )
            arg_seq += 1
            if side == "PRO":
                pro_args.append(cand)
            else:
                con_args.append(cand)

    return stance_counts, pro_args, con_args


def mmr_quota_sample_arguments(
    candidates: Sequence[StanceCandidateArg],
    *,
    embedder: Qwen3EmbeddingClient,
    max_k: int = 8,
    max_per_channel: int = 2,
    mmr_lambda: float = 0.70,
) -> list[StanceCandidateArg]:
    """Шаг 3: MMR-квотирование (<= K_max=8 на сторону, <= 2 аргументов от одного канала)."""
    if not candidates or max_k <= 0 or max_per_channel <= 0:
        return []

    max_rel = max(max(c.relevance_score, 1e-6) for c in candidates)
    norm_rels = [c.relevance_score / max_rel for c in candidates]
    emb_matrix = embedder.encode_documents([c.statement for c in candidates])

    selected_indices: list[int] = []
    channel_counts: dict[str, int] = {}
    remaining = set(range(len(candidates)))

    while remaining and len(selected_indices) < max_k:
        best_idx = -1
        best_score = -1e9
        for idx in remaining:
            ch_id = str(candidates[idx].channel_id)
            if channel_counts.get(ch_id, 0) >= max_per_channel:
                continue
            rel = norm_rels[idx]
            if not selected_indices:
                score = rel
            else:
                max_sim = max(
                    cosine_similarity(emb_matrix[idx], emb_matrix[sel])
                    for sel in selected_indices
                )
                score = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        selected_indices.append(best_idx)
        chosen_ch = str(candidates[best_idx].channel_id)
        channel_counts[chosen_ch] = channel_counts.get(chosen_ch, 0) + 1
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


class StanceMatrixSynthesizer:
    """Конвейер синтеза верифицированной матрицы «ЗА vs ПРОТИВ» (раздел 7.4.3)."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        nli_gate: NliVerificationGate | None = None,
        embedder: Qwen3EmbeddingClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.nli_gate = nli_gate or NliVerificationGate(self.settings)
        self.embedder = embedder or Qwen3EmbeddingClient(self.settings)

    def build_verified_matrix(
        self,
        *,
        query: str,
        posts: Sequence[StancePostInput],
        candidate_pro: Sequence[StanceCandidateArg] | None = None,
        candidate_con: Sequence[StanceCandidateArg] | None = None,
        mmr_lambda: float = 0.70,
    ) -> StanceMatrixResult:
        """Выполняет 4 шага Stance-Stratified Synthesis с двойной NLI-проверкой."""
        indexed_posts: dict[int, IndexedSourcePost] = {}
        for post in posts:
            if post.is_ad:
                continue
            indexed_posts[post.post_id] = post.to_indexed_post()

        if candidate_pro is not None or candidate_con is not None:
            stance_counts = {"PRO": 0, "CON": 0, "MIXED": 0, "NEUTRAL": 0}
            for post in posts:
                if not post.is_ad:
                    stance_counts[classify_post_stance(post, query=query)] += 1
            raw_pro = list(candidate_pro or [])
            raw_con = list(candidate_con or [])
        else:
            stance_counts, raw_pro, raw_con = extract_atomic_stance_arguments(posts, query=query)

        k_max = self.settings.stance_max_k_per_side
        per_ch = self.settings.stance_max_per_channel

        sampled_pro = mmr_quota_sample_arguments(
            raw_pro,
            embedder=self.embedder,
            max_k=k_max,
            max_per_channel=per_ch,
            mmr_lambda=mmr_lambda,
        )
        sampled_con = mmr_quota_sample_arguments(
            raw_con,
            embedder=self.embedder,
            max_k=k_max,
            max_per_channel=per_ch,
            mmr_lambda=mmr_lambda,
        )

        highlights_by_post: dict[int, list[dict[str, Any]]] = {}
        rejected_nli = 0
        rejected_polarity = 0
        verified_pro: list[VerifiedStancePoint] = []
        verified_con: list[VerifiedStancePoint] = []

        def _verify_side(items: Sequence[StanceCandidateArg], side: StanceSide) -> list[VerifiedStancePoint]:
            nonlocal rejected_nli, rejected_polarity
            out: list[VerifiedStancePoint] = []
            for idx, item in enumerate(items, start=1):
                doc = indexed_posts.get(item.post_id)
                if doc is None:
                    rejected_nli += 1
                    continue
                alignment = smith_waterman_align(
                    doc.offset_map.text_nfkc,
                    item.exact_quote,
                    offset_map=doc.offset_map,
                    post_id=doc.post_id,
                    sentence_id=item.sentence_id,
                    match_score=self.settings.sw_match_score,
                    mismatch_score=self.settings.sw_mismatch_score,
                    gap_open=self.settings.sw_gap_open_score,
                    gap_extend=self.settings.sw_gap_extend_score,
                    similarity_threshold=self.settings.sw_similarity_threshold,
                    jaccard_threshold=self.settings.sw_jaccard_fallback_threshold,
                )
                if alignment is None:
                    rejected_nli += 1
                    continue

                # Инвариант 1: ENTAILMENT между statement и дословной цитатой verbatim_text
                nli_res = self.nli_gate.check_pair(
                    statement=item.statement,
                    verbatim_text=alignment.verbatim_text,
                )
                if not nli_res.is_entailed:
                    rejected_nli += 1
                    continue

                # Инвариант 2: Отсутствие инверсии полярности относительно колонки PRO/CON
                if not self.nli_gate.check_stance_polarity(alignment.verbatim_text, side):
                    rejected_polarity += 1
                    continue

                claim_id = f"{side.lower()}_{idx}"
                v_cite = VerifiedCitation(
                    post_id=doc.post_id,
                    channel_id=doc.channel_id,
                    channel_title=doc.channel_title,
                    msg_id=doc.msg_id,
                    published_at=doc.published_at.isoformat(),
                    deeplink=doc.deeplink,
                    utf16_range=[alignment.utf16_range[0], alignment.utf16_range[1]],
                    verbatim_text=alignment.verbatim_text,
                    nli_entailment_score=nli_res.entailment_prob,
                    sw_alignment_score=alignment.similarity_score,
                )
                out.append(
                    VerifiedStancePoint(
                        claim_id=claim_id,
                        side=side,
                        aspect=item.aspect,
                        statement=item.statement,
                        citations=[v_cite],
                    )
                )
                highlights_by_post.setdefault(doc.post_id, []).append(
                    {
                        "claimId": claim_id,
                        "startUtf16": alignment.utf16_range[0],
                        "endUtf16": alignment.utf16_range[1],
                    }
                )
            return out

        verified_pro = _verify_side(sampled_pro, "PRO")
        verified_con = _verify_side(sampled_con, "CON")

        rows_map: dict[str, StanceMatrixRow] = {}
        for pt in verified_pro:
            row = rows_map.setdefault(pt.aspect, StanceMatrixRow(aspect=pt.aspect))
            row.pro_points.append(pt)
        for pt in verified_con:
            row = rows_map.setdefault(pt.aspect, StanceMatrixRow(aspect=pt.aspect))
            row.con_points.append(pt)

        rows = sorted(
            rows_map.values(),
            key=lambda r: -(len(r.pro_points) + len(r.con_points)),
        )

        source_posts_payload = [
            {
                "post": indexed_posts[pid].to_clean_post_item(),
                "rag_highlights": highlights,
            }
            for pid, highlights in highlights_by_post.items()
            if pid in indexed_posts
        ]

        return StanceMatrixResult(
            query=query,
            stance_counts=stance_counts,
            selected_pro_count=len(verified_pro),
            selected_con_count=len(verified_con),
            rejected_nli_count=rejected_nli,
            rejected_polarity_count=rejected_polarity,
            rows=rows,
            source_posts_payload=source_posts_payload,
        )
