"""Сборка кластерных дайджестов: Shared Map (предрасчет карточек) + Personal Lightweight Reduce (раздел 7.4.2)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from app.cache.semantic_cache import TwoLevelSemanticCache
from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import Qwen3EmbeddingClient, cosine_similarity
from app.preprocessing.entities_ner import extract_entities
from app.preprocessing.normalizer import normalize_text_nfkc
from app.preprocessing.offsets import PostUtf16OffsetMap, build_utf16_offset_map
from app.rag.nli_gate import NliVerificationGate
from app.rag.query_planner import FinOpsBudgetTracker, plan_task_route
from app.rag.smith_waterman import smith_waterman_align
from app.rag.synthesizer import VerifiedCitation, VerifiedClaim

_CLICKBAIT_RE = re.compile(
    r"(?:^|\b)(?:СРОЧНО|ШОК|МОЛНИЯ|ВНЕЗАПНО|СЕНСАЦИЯ|ОФИЦИАЛЬНО|ВАЖНО|BREAKING|URGENT)\s*[!?:—-]*\s*",
    re.IGNORECASE,
)

_RUBRIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Экономика и финансы",
        ("цб", "ставк", "инфляц", "рубл", "доллар", "нефт", "бирж", "sber", "акци", "ввп", "налог", "бюджет"),
    ),
    (
        "Технологии и ИИ",
        ("llm", "нейросет", "gpt", "claude", "qwen", "модел", "ai", "ии", "gpu", "rag", "код", "python", "релиз"),
    ),
    (
        "Кибербезопасность и инфраструктура",
        ("цод", "bgp", "cve", "уязвим", "ddos", "порт", "8080", "8081", "утечк", "взлом", "nginx", "kubernetes", "сбой"),
    ),
    (
        "Законодательство и регулирование",
        ("фз", "закон", "госдум", "штраф", "ркн", "реестр", "постановлен", "регулир", "правов", "приказ"),
    ),
    (
        "Рынки и бизнес",
        ("рынок", "сделк", "стартап", "инвестиц", "выручк", "прибыл", "бизнес", "компан", "слиян"),
    ),
    (
        "Наука и образование",
        ("исследован", "университет", "наук", "эксперимент", "грант", "статья", "arxiv"),
    ),
)

DEFAULT_RUBRIC = "Общество и события"


@dataclass(slots=True)
class ClusterPostItem:
    """Пост внутри событийного кластера для Shared Map предрасчета."""

    post_id: int
    channel_id: str
    channel_username: str
    channel_title: str
    msg_id: int
    published_at: datetime
    text: str
    deeplink: str = ""
    subscribers_count: int = 0
    is_ad: bool = False
    is_origin: bool = False
    origin_post_id: int | None = None
    cluster_id: str | None = None
    rubric_hint: str | None = None
    engagement_score: float = 0.0
    offset_map: PostUtf16OffsetMap | None = None


@dataclass(slots=True)
class CanonicalEventCard:
    """Предрасчитанная каноническая карточка событийного кластера (Level 2 Shared Map)."""

    cluster_id: str
    neutral_headline: str
    summary_text: str
    compressed_text: str
    rubric: str
    channel_ids: list[str]
    origin_post_id: int
    origin_channel_id: str
    origin_channel_title: str
    origin_deeplink: str
    published_at: datetime
    post_links: list[dict[str, Any]]
    verified_claims: list[VerifiedClaim]
    entities: list[str]
    importance_score: float
    processing_tier: str
    model_id: str
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "neutral_headline": self.neutral_headline,
            "summary_text": self.summary_text,
            "compressed_text": self.compressed_text,
            "rubric": self.rubric,
            "channel_ids": list(self.channel_ids),
            "origin_post_id": self.origin_post_id,
            "origin_channel_id": self.origin_channel_id,
            "origin_channel_title": self.origin_channel_title,
            "origin_deeplink": self.origin_deeplink,
            "published_at": self.published_at.isoformat(),
            "post_links": [dict(p) for p in self.post_links],
            "verified_claims": [c.to_dict() for c in self.verified_claims],
            "entities": list(self.entities),
            "importance_score": self.importance_score,
            "processing_tier": self.processing_tier,
            "model_id": self.model_id,
            "embedding": list(self.embedding) if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CanonicalEventCard:
        claims = [
            VerifiedClaim(
                claim_id=str(c["claim_id"]),
                topic_group=str(c["topic_group"]),
                statement=str(c["statement"]),
                citations=[
                    VerifiedCitation(
                        post_id=int(ct["post_id"]),
                        channel_id=str(ct["channel_id"]),
                        channel_title=str(ct["channel_title"]),
                        msg_id=int(ct["msg_id"]),
                        published_at=str(ct["published_at"]),
                        deeplink=str(ct["deeplink"]),
                        utf16_range=[int(x) for x in ct["utf16_range"]],
                        verbatim_text=str(ct["verbatim_text"]),
                        nli_entailment_score=float(ct["nli_entailment_score"]),
                        sw_alignment_score=float(ct["sw_alignment_score"]),
                    )
                    for ct in c.get("citations", [])
                ],
            )
            for c in data.get("verified_claims", [])
        ]
        pub = datetime.fromisoformat(str(data["published_at"]))
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        emb_raw = data.get("embedding")
        return cls(
            cluster_id=str(data["cluster_id"]),
            neutral_headline=str(data["neutral_headline"]),
            summary_text=str(data["summary_text"]),
            compressed_text=str(data["compressed_text"]),
            rubric=str(data.get("rubric") or DEFAULT_RUBRIC),
            channel_ids=[str(x) for x in data.get("channel_ids", [])],
            origin_post_id=int(data["origin_post_id"]),
            origin_channel_id=str(data["origin_channel_id"]),
            origin_channel_title=str(data.get("origin_channel_title", "")),
            origin_deeplink=str(data.get("origin_deeplink", "")),
            published_at=pub,
            post_links=[dict(p) for p in data.get("post_links", [])],
            verified_claims=claims,
            entities=[str(e) for e in data.get("entities", [])],
            importance_score=float(data.get("importance_score", 0.0)),
            processing_tier=str(data.get("processing_tier", "batch_6h")),
            model_id=str(data.get("model_id", "gemini-2.5-flash-lite-batch")),
            embedding=[float(x) for x in emb_raw] if emb_raw is not None else None,
        )


@dataclass(slots=True)
class DigestClusterCard:
    """Персонализированная карточка события в дайджесте пользователя."""

    cluster_id: str
    rubric: str
    neutral_headline: str
    summary_text: str
    origin_post: dict[str, Any]
    subscribed_post_links: list[dict[str, Any]]
    verified_claims: list[dict[str, Any]]
    relevance_score: float
    channel_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "rubric": self.rubric,
            "neutral_headline": self.neutral_headline,
            "summary_text": self.summary_text,
            "origin_post": dict(self.origin_post),
            "subscribed_post_links": [dict(x) for x in self.subscribed_post_links],
            "verified_claims": [dict(c) for c in self.verified_claims],
            "relevance_score": self.relevance_score,
            "channel_count": self.channel_count,
        }


@dataclass(slots=True)
class PersonalDigestResult:
    """Результат персонального Lightweight Reduce дайджеста (25-35 карточек, 4-7 рубрик)."""

    user_id: str
    finops_zone: str
    reduce_model: str
    total_candidate_clusters: int
    selected_cards_count: int
    rubrics: list[dict[str, Any]]
    cards: list[DigestClusterCard]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "finops_zone": self.finops_zone,
            "reduce_model": self.reduce_model,
            "total_candidate_clusters": self.total_candidate_clusters,
            "selected_cards_count": self.selected_cards_count,
            "rubrics": [dict(r) for r in self.rubrics],
            "cards": [c.to_dict() for c in self.cards],
        }


def neutralize_headline(raw_text: str, *, max_chars: int = 120) -> str:
    """Нейтрализует кликбейтный заголовок/первую фразу и ограничивает длину."""
    norm = normalize_text_nfkc(raw_text)
    first_line = next((ln.strip() for ln in norm.splitlines() if ln.strip()), norm)
    cleaned = _CLICKBAIT_RE.sub("", first_line)
    cleaned = re.sub(r"!{2,}", ".", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" —-:\t\r\n")
    if not cleaned:
        cleaned = first_line.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max_chars - 1].rsplit(" ", 1)[0].strip()
    return (cut or cleaned[: max_chars - 1]) + "…"


def infer_rubric(text: str, explicit_hint: str | None = None) -> str:
    """Определяет тематическую рубрику события из подсказки или ключевых слов."""
    if explicit_hint and explicit_hint.strip():
        return explicit_hint.strip()
    lower = normalize_text_nfkc(text).lower().replace("ё", "е")
    best_rubric = DEFAULT_RUBRIC
    best_hits = 0
    for rubric, keywords in _RUBRIC_KEYWORDS:
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_hits:
            best_hits = hits
            best_rubric = rubric
    return best_rubric


def _build_deeplink(item: ClusterPostItem) -> str:
    if item.deeplink:
        return item.deeplink
    username = item.channel_username.lstrip("@").strip()
    if username:
        return f"https://t.me/{username}/{item.msg_id}"
    return f"https://t.me/c/{item.channel_id}/{item.msg_id}"


def _pick_origin_post(posts: Sequence[ClusterPostItem]) -> ClusterPostItem:
    explicit = [p for p in posts if p.is_origin]
    if explicit:
        return min(explicit, key=lambda p: (p.published_at, p.post_id))
    id_map = {p.post_id: p for p in posts}
    for p in posts:
        if p.origin_post_id is not None and p.origin_post_id in id_map:
            return id_map[p.origin_post_id]
    return min(posts, key=lambda p: (p.published_at, p.post_id))


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _merge_to_rubric_budget(
    rubric_groups: dict[str, list[DigestClusterCard]],
    *,
    max_rubrics: int,
) -> dict[str, list[DigestClusterCard]]:
    if len(rubric_groups) <= max_rubrics:
        return rubric_groups
    ordered = sorted(rubric_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    kept = dict(ordered[: max_rubrics - 1])
    overflow_cards: list[DigestClusterCard] = []
    for _, items in ordered[max_rubrics - 1 :]:
        overflow_cards.extend(items)
    if overflow_cards:
        bucket_name = DEFAULT_RUBRIC if DEFAULT_RUBRIC not in kept else "Прочие события"
        kept.setdefault(bucket_name, []).extend(overflow_cards)
    return kept


class ClusteredDigestService:
    """Сервис сборки дайджестов: Shared Map (предрасчет) + Personal Lightweight Reduce."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        semantic_cache: TwoLevelSemanticCache | None = None,
        nli_gate: NliVerificationGate | None = None,
        embedder: Qwen3EmbeddingClient | None = None,
        finops_tracker: FinOpsBudgetTracker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.semantic_cache = semantic_cache or TwoLevelSemanticCache(self.settings)
        self.nli_gate = nli_gate or NliVerificationGate(self.settings)
        self.embedder = embedder or Qwen3EmbeddingClient(self.settings)
        self.finops_tracker = finops_tracker or FinOpsBudgetTracker(self.settings)

    def precompute_shared_event_card(
        self,
        cluster_id: str,
        posts: Sequence[ClusterPostItem],
        *,
        is_private_contour: bool = False,
        now: datetime | None = None,
    ) -> CanonicalEventCard | None:
        """Выполняет Shared Map: фильтрует рекламу, строит факты с CQE/SW и кэширует карточку."""
        clean_posts = [p for p in posts if not p.is_ad and normalize_text_nfkc(p.text).strip()]
        if not clean_posts:
            return None

        origin = _pick_origin_post(clean_posts)
        max_subs = max(p.subscribers_count for p in clean_posts)
        is_core = max_subs >= self.settings.core_channel_subscribers_threshold

        if is_core:
            route = plan_task_route(
                "STREAMING_MAP_AND_COMMENTS",
                is_private_contour=is_private_contour,
                prefer_batch=False,
                now=now,
            )
            processing_tier = "streaming_core"
        else:
            route = plan_task_route(
                "ASYNC_DIGEST_BATCH",
                is_private_contour=is_private_contour,
                prefer_batch=True,
                now=now,
            )
            processing_tier = "batch_6h"

        offset_maps: dict[int, PostUtf16OffsetMap] = {}
        links_by_channel: dict[str, dict[str, Any]] = {}
        all_entities: list[str] = []
        seen_entities: set[str] = set()

        for post in sorted(clean_posts, key=lambda p: (_ensure_utc(p.published_at), p.post_id)):
            omap = post.offset_map or build_utf16_offset_map(post.text, post_id=post.post_id)
            offset_maps[post.post_id] = omap
            ch_id = str(post.channel_id)
            if ch_id not in links_by_channel:
                links_by_channel[ch_id] = {
                    "post_id": post.post_id,
                    "channel_id": int(ch_id) if ch_id.isdigit() else ch_id,
                    "channel_username": post.channel_username.lstrip("@") or None,
                    "channel_title": post.channel_title,
                    "msg_id": post.msg_id,
                    "published_at": _ensure_utc(post.published_at).isoformat(),
                    "deeplink": _build_deeplink(post),
                    "is_origin": post.post_id == origin.post_id,
                }
            ent = extract_entities(omap.text_nfkc)
            for val in sorted(ent.all_canonical):
                if val not in seen_entities:
                    seen_entities.add(val)
                    all_entities.append(val)

        # Отбираем до 3 репрезентативных фактов по первоисточнику и ключевым клаузам с CQE-проверкой
        origin_map = offset_maps[origin.post_id]
        rubric = infer_rubric(origin_map.text_nfkc, origin.rubric_hint)
        neutral_headline = neutralize_headline(origin_map.text_nfkc)

        candidate_sentences = list(origin_map.sentences[:2])
        for post in clean_posts:
            if post.post_id != origin.post_id and len(candidate_sentences) < 4:
                pmap = offset_maps[post.post_id]
                if pmap.sentences:
                    candidate_sentences.append(pmap.sentences[0])

        verified_claims: list[VerifiedClaim] = []
        seen_statements: set[str] = set()
        claim_idx = 1

        # Сопоставляем клаузы с постами, из которых они взяты
        clause_sources: list[tuple[ClusterPostItem, PostUtf16OffsetMap, Any]] = []
        for s in origin_map.sentences[:2]:
            clause_sources.append((origin, origin_map, s))
        for post in clean_posts:
            if post.post_id == origin.post_id:
                continue
            pmap = offset_maps[post.post_id]
            if pmap.sentences:
                clause_sources.append((post, pmap, pmap.sentences[0]))
            if len(clause_sources) >= 4:
                break

        for src_post, src_map, sent in clause_sources:
            stmt = sent.text.strip()
            if len(stmt) < 8 or stmt in seen_statements:
                continue
            alignment = smith_waterman_align(
                src_map.text_nfkc,
                stmt,
                offset_map=src_map,
                post_id=src_post.post_id,
                sentence_id=sent.sentence_id,
                match_score=self.settings.sw_match_score,
                mismatch_score=self.settings.sw_mismatch_score,
                gap_open=self.settings.sw_gap_open_score,
                gap_extend=self.settings.sw_gap_extend_score,
                similarity_threshold=self.settings.sw_similarity_threshold,
                jaccard_threshold=self.settings.sw_jaccard_fallback_threshold,
            )
            if alignment is None:
                continue
            nli_res = self.nli_gate.check_pair(statement=stmt, verbatim_text=alignment.verbatim_text)
            if not nli_res.is_entailed:
                continue
            seen_statements.add(stmt)
            verified_claims.append(
                VerifiedClaim(
                    claim_id=f"c{claim_idx}",
                    topic_group=rubric,
                    statement=stmt,
                    citations=[
                        VerifiedCitation(
                            post_id=src_post.post_id,
                            channel_id=str(src_post.channel_id),
                            channel_title=src_post.channel_title,
                            msg_id=src_post.msg_id,
                            published_at=_ensure_utc(src_post.published_at).isoformat(),
                            deeplink=_build_deeplink(src_post),
                            utf16_range=[alignment.utf16_range[0], alignment.utf16_range[1]],
                            verbatim_text=alignment.verbatim_text,
                            nli_entailment_score=nli_res.entailment_prob,
                            sw_alignment_score=alignment.similarity_score,
                        )
                    ],
                )
            )
            claim_idx += 1
            if len(verified_claims) >= 3:
                break

        if verified_claims:
            summary_text = " ".join(c.statement for c in verified_claims[:2])
        else:
            summary_text = neutral_headline
        compressed_text = f"{neutral_headline}. {summary_text}" if summary_text != neutral_headline else neutral_headline
        compressed_text = compressed_text[:360]

        unique_channels = sorted({str(p.channel_id) for p in clean_posts})
        total_eng = sum(max(0.0, float(p.engagement_score)) for p in clean_posts)
        importance = round(len(unique_channels) * 1.5 + math.log1p(total_eng) * 0.4 + (1.0 if is_core else 0.0), 4)

        card_emb = self.embedder.encode_documents([f"{neutral_headline} {summary_text}"])[0].tolist()

        card = CanonicalEventCard(
            cluster_id=str(cluster_id),
            neutral_headline=neutral_headline,
            summary_text=summary_text,
            compressed_text=compressed_text,
            rubric=rubric,
            channel_ids=unique_channels,
            origin_post_id=origin.post_id,
            origin_channel_id=str(origin.channel_id),
            origin_channel_title=origin.channel_title,
            origin_deeplink=_build_deeplink(origin),
            published_at=_ensure_utc(origin.published_at),
            post_links=list(links_by_channel.values()),
            verified_claims=verified_claims,
            entities=all_entities[:12],
            importance_score=importance,
            processing_tier=processing_tier,
            model_id=route.model_id,
            embedding=[float(x) for x in card_emb],
        )
        self.semantic_cache.put_event_card(str(cluster_id), card.to_dict())
        return card

    def batch_precompute_shared_map(
        self,
        posts: Sequence[ClusterPostItem],
        *,
        is_private_contour: bool = False,
        now: datetime | None = None,
    ) -> list[CanonicalEventCard]:
        """Группирует поток постов по `cluster_id` и строит Shared Map карточки."""
        by_cluster: dict[str, list[ClusterPostItem]] = {}
        for post in posts:
            if post.is_ad:
                continue
            cid = str(post.cluster_id or f"single_{post.post_id}")
            by_cluster.setdefault(cid, []).append(post)

        cards: list[CanonicalEventCard] = []
        for cid, group in by_cluster.items():
            card = self.precompute_shared_event_card(
                cid,
                group,
                is_private_contour=is_private_contour,
                now=now,
            )
            if card is not None:
                cards.append(card)
        return cards

    def build_personal_digest(
        self,
        *,
        user_id: str,
        subscribed_channel_ids: Sequence[int | str],
        cluster_ids: Sequence[str] | None = None,
        cards: Sequence[CanonicalEventCard] | None = None,
        plan_tier: str | None = None,
        max_cards: int | None = None,
        mmr_lambda: float = 0.70,
        now: datetime | None = None,
    ) -> PersonalDigestResult:
        """Выполняет Lightweight Reduce: фильтр по S_u, подписка + origin_post, MMR (25-35), 4-7 рубрик."""
        state = self.finops_tracker.get_state(user_id, plan_tier=plan_tier or "PROSUMER")
        zone = self.finops_tracker.resolve_zone(state)
        routing = self.finops_tracker.plan_rag_route(
            user_id,
            mode="digest",
            plan_tier=plan_tier or state.plan_tier,
            subscribed_channels_count=len(subscribed_channel_ids),
            now=now,
        )
        sub_set = {str(c).strip() for c in subscribed_channel_ids if str(c).strip()}

        raw_cards: list[CanonicalEventCard] = []
        if cards is not None:
            raw_cards.extend(cards)
        elif cluster_ids is not None:
            for d in self.semantic_cache.get_event_cards(cluster_ids, subscribed_channel_ids=list(sub_set)):
                raw_cards.append(CanonicalEventCard.from_dict(d))

        # Персонализированная фильтрация строго по пересечению с подпиской S_u
        eligible: list[tuple[CanonicalEventCard, list[dict[str, Any]], float]] = []
        for card in raw_cards:
            card_ch_set = {str(c) for c in card.channel_ids}
            if card_ch_set.isdisjoint(sub_set):
                continue
            sub_links = [
                dict(link)
                for link in card.post_links
                if str(link.get("channel_id")) in sub_set
            ]
            # Если ни одна ссылка не сохранилась в post_links, но пересечение по channel_ids есть — не теряем
            if not sub_links:
                continue
            sub_overlap = len({str(l.get("channel_id")) for l in sub_links})
            personal_rel = float(card.importance_score) + sub_overlap * 0.75
            eligible.append((card, sub_links, personal_rel))

        target_limit = max_cards or self.settings.digest_max_cards
        target_limit = max(1, min(35, int(target_limit)))

        # MMR-отбор ключевых карточек (25-35 при достаточном объеме кластеров)
        if eligible:
            max_rel = max(rel for _, _, rel in eligible) or 1.0
            norm_items = [(card, links, rel / max_rel) for card, links, rel in eligible]
            embeddings: list[np.ndarray] = []
            missing_indices: list[int] = []
            for idx, (card, _, _) in enumerate(norm_items):
                if card.embedding:
                    embeddings.append(np.asarray(card.embedding, dtype=np.float32))
                else:
                    embeddings.append(np.zeros(self.settings.embedding_dim_mrl, dtype=np.float32))
                    missing_indices.append(idx)
            if missing_indices:
                texts = [f"{norm_items[i][0].neutral_headline} {norm_items[i][0].summary_text}" for i in missing_indices]
                encoded = self.embedder.encode_documents(texts)
                for pos, idx in enumerate(missing_indices):
                    embeddings[idx] = encoded[pos]

            selected_indices: list[int] = []
            remaining = set(range(len(norm_items)))
            while remaining and len(selected_indices) < target_limit:
                best_idx = -1
                best_score = -1e9
                for idx in remaining:
                    rel = norm_items[idx][2]
                    if not selected_indices:
                        mmr_score = rel
                    else:
                        max_sim = max(
                            cosine_similarity(embeddings[idx], embeddings[sel])
                            for sel in selected_indices
                        )
                        mmr_score = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                if best_idx < 0:
                    break
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

            selected_pairs = [norm_items[i] for i in selected_indices]
        else:
            selected_pairs = []

        selected_cards: list[DigestClusterCard] = []
        for card, sub_links, rel in selected_pairs:
            origin_in_sub = card.origin_channel_id in sub_set
            origin_payload = {
                "post_id": card.origin_post_id,
                "channel_id": int(card.origin_channel_id) if card.origin_channel_id.isdigit() else card.origin_channel_id,
                "channel_title": card.origin_channel_title,
                "deeplink": card.origin_deeplink,
                "published_at": card.published_at.isoformat(),
                "is_subscribed": origin_in_sub,
            }
            selected_cards.append(
                DigestClusterCard(
                    cluster_id=card.cluster_id,
                    rubric=card.rubric,
                    neutral_headline=card.neutral_headline,
                    summary_text=card.compressed_text if routing.use_compressed_event_cards else card.summary_text,
                    origin_post=origin_payload,
                    subscribed_post_links=sub_links,
                    verified_claims=[c.to_dict() for c in card.verified_claims],
                    relevance_score=round(rel, 4),
                    channel_count=len(card.channel_ids),
                )
            )

        grouped: dict[str, list[DigestClusterCard]] = {}
        for item in selected_cards:
            grouped.setdefault(item.rubric, []).append(item)

        grouped = _merge_to_rubric_budget(grouped, max_rubrics=self.settings.digest_max_rubrics)
        rubrics_payload = [
            {
                "rubric": rubric,
                "cards_count": len(items),
                "cards": [c.to_dict() for c in items],
            }
            for rubric, items in grouped.items()
        ]

        return PersonalDigestResult(
            user_id=user_id,
            finops_zone=zone,
            reduce_model=routing.digest_reduce_model,
            total_candidate_clusters=len(eligible),
            selected_cards_count=len(selected_cards),
            rubrics=rubrics_payload,
            cards=selected_cards,
        )
