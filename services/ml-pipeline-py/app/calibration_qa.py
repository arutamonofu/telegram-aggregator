"""Калибровка конвейера (раздел 9.2 SPEC) и проверка Quality Gates (раздел 8.1.4 SPEC)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import log2
from typing import Any

from qdrant_client import QdrantClient

from app.cache.semantic_cache import TwoLevelSemanticCache, build_public_cache_key
from app.config import Settings, get_settings
from app.consumers.post_pipeline import IngestPostInput, PostPipelineService, TelegramEntity
from app.embeddings.qwen3_embedder import cosine_similarity
from app.preprocessing.entities_ner import extract_entities, has_exact_lexical_triggers
from app.preprocessing.offsets import build_utf16_offset_map
from app.rag.nli_gate import NliVerificationGate
from app.rag.synthesizer import GlassBoxRagSynthesizer, IndexedSourcePost
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.qdrant_repo import QdrantPublicChannelsRepo


SPEC_9_2_QUERY = "Какая ошибка возникает в gotd на 225 слое при чтении длинной истории канала?"
SPEC_9_2_GO_CODE = (
    "case *tg.UpdatesChannelDifferenceTooLong:\n"
    "    return client.MessagesGetHistory(ctx, 225)"
)


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _slice_utf16(text: str, start: int, end: int) -> str:
    raw = text.encode("utf-16-le")
    return raw[start * 2 : end * 2].decode("utf-16-le")


def _adjusted_rand_index(labels_true: list[int], labels_pred: list[int]) -> float:
    """Вычисляет Adjusted Rand Index (ARI) без внешних зависимостей."""
    n = len(labels_true)
    if n < 2:
        return 1.0
    true_groups: dict[int, int] = {}
    pred_groups: dict[int, int] = {}
    contingency: dict[tuple[int, int], int] = {}
    for t, p in zip(labels_true, labels_pred):
        true_groups[t] = true_groups.get(t, 0) + 1
        pred_groups[p] = pred_groups.get(p, 0) + 1
        contingency[(t, p)] = contingency.get((t, p), 0) + 1

    def _comb2(k: int) -> float:
        return k * (k - 1) / 2.0

    sum_comb_c = sum(_comb2(v) for v in contingency.values())
    sum_comb_t = sum(_comb2(v) for v in true_groups.values())
    sum_comb_p = sum(_comb2(v) for v in pred_groups.values())
    total_comb = _comb2(n)
    if total_comb == 0:
        return 1.0
    expected = (sum_comb_t * sum_comb_p) / total_comb
    max_index = 0.5 * (sum_comb_t + sum_comb_p)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0 if sum_comb_c == expected else 0.0
    return (sum_comb_c - expected) / denom


def _homogeneity(labels_true: list[int], labels_pred: list[int]) -> float:
    """Вычисляет метрику Homogeneity = 1 - H(C|K) / H(C)."""
    n = len(labels_true)
    if n == 0:
        return 1.0
    true_counts: dict[int, int] = {}
    pred_counts: dict[int, int] = {}
    contingency: dict[tuple[int, int], int] = {}
    for t, p in zip(labels_true, labels_pred):
        true_counts[t] = true_counts.get(t, 0) + 1
        pred_counts[p] = pred_counts.get(p, 0) + 1
        contingency[(t, p)] = contingency.get((t, p), 0) + 1

    h_c = -sum((c / n) * log2(c / n) for c in true_counts.values() if c > 0)
    if h_c <= 1e-12:
        return 1.0
    h_c_given_k = -sum(
        (n_ck / n) * log2(n_ck / pred_counts[k])
        for (_, k), n_ck in contingency.items()
        if n_ck > 0
    )
    return max(0.0, min(1.0, 1.0 - (h_c_given_k / h_c)))


@dataclass(slots=True)
class Spec92Artifacts:
    """Набор входных постов и индексированных представлений для сценария SPEC 9.2."""

    query: str
    subscribed_channel_ids: list[int]
    base_time: datetime
    p_ad: IngestPostInput
    p_target: IngestPostInput
    p_rewrites: list[IngestPostInput]
    p_neg: IngestPostInput


def build_spec_9_2_fixture(now: datetime | None = None) -> Spec92Artifacts:
    """Строит эталонный тестовый набор раздела 9.2 `docs/SPEC.md`."""
    base_time = (now or datetime.now(timezone.utc)).replace(microsecond=0) - timedelta(hours=3)

    # 1) Рекламный пост (P_ad): должен отсеяться на CPU-каскаде L1 (> 0.80):
    # - катафора в 1-м абзаце ("Мало кто знает") без заглавных/версий (+0.22)
    # - скрытый text_url, ведущий на t.me/... (+0.15)
    # - прагматический сдвиг / CTA в последних 30% текста (+0.22)
    # - @username / ссылка в последних 20% текста (+0.28)
    # Итого L1 = 0.87 > 0.80.
    ad_prefix = (
        "Мало кто знает, почему на 225 слое ваш код на gotd падает при чтении длинной истории канала:\n"
        "многие разработчики неделями ищут утечку памяти в цикле обновлений и перебирают таймауты впустую, "
        "хотя проблема решается одним правильным паттерном обработки. "
        "Только до пятницы действует скидка 40% и промокод — забирайте "
    )
    ad_anchor = "курс по программированию"
    ad_suffix = " через @go_promo_hub"
    ad_text = ad_prefix + ad_anchor + ad_suffix
    p_ad = IngestPostInput(
        event_id="spec-9-2-ad",
        post_id=9200,
        channel_id=101,
        channel_username="go_dev_digest",
        channel_title="Go Dev Digest",
        published_at=base_time + timedelta(minutes=5),
        text_raw=ad_text,
        entities=[
            TelegramEntity(
                type="text_url",
                offset=_utf16_len(ad_prefix),
                length=_utf16_len(ad_anchor),
                url="https://t.me/go_promo_hub/42",
            )
        ],
        channel_authority=0.25,
        originality_score=0.2,
    )

    # 2) Целевой пост (P_target): начинается с 🔥 (2 UTF-16 юнита -> сдвиг +1),
    # содержит разбор updates.channelDifferenceTooLong и Go-блок кода (pre language="go").
    target_prose = (
        "🔥 Разбор бага библиотеки gotd на слое 225 протокола MTProto.\n"
        "При чтении длинной истории канала в gotd на 225 слое возникает ошибка "
        "updates.channelDifferenceTooLong, если локальный pts устарел и нужен откат на messages.getHistory.\n"
        "Пример обработчика на языке программирования Go:\n"
    )
    target_text = target_prose + SPEC_9_2_GO_CODE
    p_target = IngestPostInput(
        event_id="spec-9-2-target",
        post_id=9201,
        channel_id=102,
        channel_username="mtproto_internals",
        channel_title="MTProto Internals",
        published_at=base_time,
        text_raw=target_text,
        entities=[
            TelegramEntity(
                type="pre",
                offset=_utf16_len(target_prose),
                length=_utf16_len(SPEC_9_2_GO_CODE),
                language="go",
            )
        ],
        has_informative_media=False,
        explicit_cite_score=0.0,
        channel_authority=0.94,
        originality_score=0.98,
        has_primary_evidence=True,
    )

    # 3) 3 рерайта P_target в других каналах (без блока кода pre и с более поздним published_at)
    rewrite_specs = [
        (
            9202,
            103,
            "go_digest_ru",
            "Go Digest RU",
            base_time + timedelta(minutes=25),
            "Разбор бага библиотеки gotd на слое 225 протокола MTProto.\n"
            "При чтении длинной истории канала в gotd на 225 слое возникает ошибка "
            "updates.channelDifferenceTooLong, если локальный pts устарел (пересказ mtproto_internals).",
        ),
        (
            9203,
            104,
            "backend_weekly",
            "Backend Weekly",
            base_time + timedelta(minutes=45),
            "Разбор бага в gotd на 225 слое MTProto.\n"
            "При чтении длинной истории канала в gotd на 225 слое возникает ошибка "
            "updates.channelDifferenceTooLong при устаревшем pts и требуется вызов messages.getHistory.",
        ),
        (
            9204,
            105,
            "tg_api_notes",
            "Telegram API Notes",
            base_time + timedelta(minutes=70),
            "Разбор проблемы библиотеки gotd на слое 225 MTProto.\n"
            "Если читать длинную историю канала на 225 слое в gotd, возникает ошибка "
            "updates.channelDifferenceTooLong с последующим откатом через messages.getHistory.",
        ),
    ]
    p_rewrites = [
        IngestPostInput(
            event_id=f"spec-9-2-rw-{pid}",
            post_id=pid,
            channel_id=cid,
            channel_username=uname,
            channel_title=title,
            published_at=pub_at,
            text_raw=txt,
            explicit_cite_score=0.75,
            channel_authority=0.45,
            originality_score=0.35,
            has_primary_evidence=False,
        )
        for pid, cid, uname, title, pub_at, txt in rewrite_specs
    ]

    # 4) Нерелевантный технический пост про 224 слой (без dense_code и с другим числом 224)
    p_neg = IngestPostInput(
        event_id="spec-9-2-neg",
        post_id=9205,
        channel_id=106,
        channel_username="mtproto_changelog",
        channel_title="MTProto Changelog",
        published_at=base_time + timedelta(minutes=15),
        text_raw=(
            "На слое 224 схемы Telegram добавлены обновленные флаги приватности подарков, "
            "звездных реакций и параметров оформления профиля без изменений в механизме истории чатов."
        ),
        channel_authority=0.80,
        originality_score=0.90,
        has_primary_evidence=False,
    )

    return Spec92Artifacts(
        query=SPEC_9_2_QUERY,
        subscribed_channel_ids=[101, 102, 103, 104, 105, 106],
        base_time=base_time,
        p_ad=p_ad,
        p_target=p_target,
        p_rewrites=p_rewrites,
        p_neg=p_neg,
    )


def run_spec_9_2_e2e_scenario(
    *,
    now: datetime | None = None,
    total_corpus_channels: int = 10_000,
    matching_corpus_channels: int = 9,
) -> dict[str, Any]:
    """Выполняет сквозной сценарий раздела 9.2 (от инжеста/антиспама до NLI и utf16_range)."""
    settings = Settings()
    fixture = build_spec_9_2_fixture(now=now)
    qdrant_client = QdrantClient(":memory:")
    qdrant_repo = QdrantPublicChannelsRepo(
        client=qdrant_client,
        settings=settings,
        use_remote_bm25_document=False,
    )
    qdrant_repo.ensure_collection(recreate=True)

    pipeline = PostPipelineService(settings=settings, qdrant_repo=qdrant_repo)
    cache = TwoLevelSemanticCache(settings=settings)
    nli_gate = NliVerificationGate(settings=settings)

    all_posts = [
        fixture.p_ad,
        fixture.p_target,
        *fixture.p_rewrites,
        fixture.p_neg,
    ]
    pipeline_results = {post.post_id: pipeline.process_post(post) for post in all_posts}

    ad_res = pipeline_results[fixture.p_ad.post_id]
    # Дополнительно проверяем, что даже при наличии ad-записи в Qdrant pre-filter `is_ad = false`
    # надежно исключает P_ad на этапе поиска.
    qdrant_repo.upsert_post_chunks(
        post_id=fixture.p_ad.post_id,
        channel_id=fixture.p_ad.channel_id,
        channel_username=fixture.p_ad.channel_username,
        channel_title=fixture.p_ad.channel_title,
        text=fixture.p_ad.text_raw,
        published_at=fixture.p_ad.published_at,
        cluster_id="cluster-ad-control",
        origin_post_id=fixture.p_ad.post_id,
        origin_channel_id=fixture.p_ad.channel_id,
        is_ad=True,
        spam_class="ad",
    )

    target_res = pipeline_results[fixture.p_target.post_id]
    rewrite_results = [pipeline_results[rw.post_id] for rw in fixture.p_rewrites]
    neg_res = pipeline_results[fixture.p_neg.post_id]

    target_cluster_id = target_res.cluster_result.cluster_id if target_res.cluster_result else None
    cluster_state = pipeline.clusterer.clusters.get(target_cluster_id or "")
    cluster_member_ids = sorted(m.post_id for m in cluster_state.members) if cluster_state else []

    # Проверка извлечения сущностей и ключа L1-кэша
    q_entities = extract_entities(fixture.query)
    exact_trigger = has_exact_lexical_triggers(fixture.query, entities=q_entities)
    cache_key = build_public_cache_key(
        fixture.subscribed_channel_ids,
        "7d",
        q_entities,
        reference_time=fixture.base_time + timedelta(hours=3),
    )

    # Проверка selectvity & ACORN pre-filter args + dense_code similarity
    q_vec = pipeline.embedder.encode_queries([fixture.query])[0]
    code_vec = pipeline.embedder.encode_documents([SPEC_9_2_GO_CODE])[0]
    dense_code_cos = cosine_similarity(q_vec, code_vec)

    neg_vec = pipeline.embedder.encode_documents([fixture.p_neg.text_raw])[0]
    neg_cos = cosine_similarity(q_vec, neg_vec)

    query_args = qdrant_repo.build_hybrid_query_args(
        query_text=fixture.query,
        query_dense_512d=q_vec,
        subscribed_channel_ids=fixture.subscribed_channel_ids,
        published_after=fixture.base_time - timedelta(hours=1),
        has_exact_entities=exact_trigger,
        group_limit=settings.hybrid_group_limit,
        group_size=1,
    )
    selectivity = matching_corpus_channels / max(1, total_corpus_channels)

    full_texts: dict[int, str] = {
        fixture.p_target.post_id: target_res.preprocessed.offset_map.text_nfkc,
        fixture.p_neg.post_id: neg_res.preprocessed.offset_map.text_nfkc,
    }
    for rw in fixture.p_rewrites:
        full_texts[rw.post_id] = pipeline_results[rw.post_id].preprocessed.offset_map.text_nfkc

    hybrid_search = HybridSearchService(
        qdrant_repo=qdrant_repo,
        settings=settings,
        embedder=pipeline.embedder,
        reranker=pipeline.reranker,
        semantic_cache=cache,
    )
    search_res = hybrid_search.search_and_rerank(
        query=fixture.query,
        subscribed_channel_ids=fixture.subscribed_channel_ids,
        published_after=fixture.base_time - timedelta(hours=1),
        top_k=5,
        group_limit=10,
        enable_reranker=True,
        full_post_texts=full_texts,
    )

    # Подготовка IndexedSourcePost из результатов pipeline для RAG и NLI/SW верификации
    indexed_posts: list[IndexedSourcePost] = []
    for pid, post_obj in [
        (fixture.p_target.post_id, fixture.p_target),
        *[(rw.post_id, rw) for rw in fixture.p_rewrites],
        (fixture.p_neg.post_id, fixture.p_neg),
    ]:
        p_res = pipeline_results[pid]
        indexed_posts.append(
            IndexedSourcePost(
                post_id=pid,
                channel_id=str(post_obj.channel_id),
                channel_username=post_obj.channel_username,
                channel_title=post_obj.channel_title,
                msg_id=pid,
                published_at=post_obj.published_at,
                text=p_res.preprocessed.offset_map.text_nfkc,
                deeplink=f"https://t.me/{post_obj.channel_username}/{pid}",
                offset_map=p_res.preprocessed.offset_map,
                cluster_id=p_res.cluster_result.cluster_id if p_res.cluster_result else None,
            )
        )

    rag_synth = GlassBoxRagSynthesizer(
        settings=settings,
        hybrid_search=hybrid_search,
        semantic_cache=cache,
        nli_gate=nli_gate,
    )
    rag_res = rag_synth.run_rag_query(
        user_id="qa-spec-9-2-user",
        query=fixture.query,
        channel_ids=fixture.subscribed_channel_ids,
        mode="qa",
        time_window_days=7,
        preloaded_posts=indexed_posts,
        now=fixture.base_time + timedelta(hours=3),
    )

    primary_claim = rag_res.verified_claims[0] if rag_res.verified_claims else None
    primary_citation = primary_claim.citations[0] if (primary_claim and primary_claim.citations) else None
    target_text_nfkc = target_res.preprocessed.offset_map.text_nfkc

    utf16_slice = ""
    cp_start = -1
    utf16_shift = 0
    if primary_citation is not None and primary_citation.post_id == fixture.p_target.post_id:
        u_start, u_end = primary_citation.utf16_range
        utf16_slice = _slice_utf16(target_text_nfkc, u_start, u_end)
        cp_start = target_text_nfkc.find(utf16_slice) if utf16_slice else -1
        if cp_start >= 0:
            utf16_shift = u_start - cp_start

    target_vector_names = sorted(
        {v for chunk in target_res.qdrant_chunks for v in chunk.vector_names}
    )
    neg_vector_names = sorted(
        {v for chunk in neg_res.qdrant_chunks for v in chunk.vector_names}
    )

    return {
        "query": fixture.query,
        "ad_verdict": {
            "post_id": ad_res.post_id,
            "is_ad": ad_res.spam_verdict.is_ad,
            "spam_class": ad_res.spam_verdict.spam_class,
            "stage_decided": ad_res.spam_verdict.stage_decided,
            "l1_score": ad_res.spam_verdict.l1_features.l1_score
            if ad_res.spam_verdict.l1_features
            else 0.0,
            "should_persist": ad_res.spam_verdict.should_persist,
        },
        "clustering": {
            "target_cluster_id": target_cluster_id,
            "member_post_ids": cluster_member_ids,
            "rewrite_cluster_ids": [
                rw.cluster_result.cluster_id if rw.cluster_result else None
                for rw in rewrite_results
            ],
            "rewrite_decisions": [
                rw.cluster_result.decision if rw.cluster_result else None
                for rw in rewrite_results
            ],
            "origin_post_id": cluster_state.origin_post_id if cluster_state else None,
            "origin_channel_id": cluster_state.origin_channel_id if cluster_state else None,
            "neg_cluster_id": neg_res.cluster_result.cluster_id if neg_res.cluster_result else None,
        },
        "entities_and_cache": {
            "extracted_all_canonical": sorted(q_entities.all_canonical),
            "exact_trigger": exact_trigger,
            "cache_key": cache_key,
            "bm25_weight": search_res.bm25_weight,
        },
        "retrieval": {
            "selectivity": selectivity,
            "selectivity_under_0_5_pct": selectivity < 0.005,
            "acorn_max_selectivity": settings.acorn_max_selectivity,
            "group_by": query_args.get("group_by"),
            "dense_code_cosine": round(dense_code_cos, 4),
            "threshold_dense_code": settings.threshold_dense_code,
            "neg_cosine": round(neg_cos, 4),
            "target_vector_names": target_vector_names,
            "neg_vector_names": neg_vector_names,
            "reranked_post_ids": [c.post_id for c in search_res.reranked_posts],
            "top1_post_id": search_res.reranked_posts[0].post_id
            if search_res.reranked_posts
            else None,
            "top1_rerank_score": search_res.reranked_posts[0].rerank_score
            if search_res.reranked_posts
            else 0.0,
        },
        "rag_and_nli": {
            "verified_claims_count": len(rag_res.verified_claims),
            "rejected_claims_count": rag_res.rejected_claims_count,
            "fallback_message": rag_res.fallback_message,
            "primary_statement": primary_claim.statement if primary_claim else "",
            "citation_post_id": primary_citation.post_id if primary_citation else None,
            "utf16_range": primary_citation.utf16_range if primary_citation else None,
            "cp_start": cp_start,
            "utf16_emoji_shift": utf16_shift,
            "utf16_sliced_text": utf16_slice,
            "verbatim_text": primary_citation.verbatim_text if primary_citation else "",
            "sw_alignment_score": primary_citation.sw_alignment_score if primary_citation else 0.0,
            "nli_entailment_score": primary_citation.nli_entailment_score
            if primary_citation
            else 0.0,
        },
        "pipeline_artifacts": {
            "pipeline": pipeline,
            "qdrant_repo": qdrant_repo,
            "hybrid_search": hybrid_search,
            "semantic_cache": cache,
            "nli_gate": nli_gate,
            "indexed_posts": indexed_posts,
            "fixture": fixture,
        },
    }


def evaluate_stage1_quality_gates(
    *,
    settings: Settings | None = None,
    spec_9_2_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Выполняет 5-шаговую калибровку раздела 9.2 и проверяет Quality Gates 8.1.4."""
    cfg = settings or Settings()
    e2e = spec_9_2_result or run_spec_9_2_e2e_scenario()
    now = e2e["pipeline_artifacts"]["fixture"].base_time

    # --- Шаг 1: Калибровка фильтра спама (L0 -> L1 -> L2) ---
    spam_pipeline = PostPipelineService(settings=cfg)
    spam_samples: list[tuple[IngestPostInput, bool]] = [
        # Ad / Scam (expected is_ad=True or is_noise=True, should_persist=False)
        (e2e["pipeline_artifacts"]["fixture"].p_ad, True),
        (
            IngestPostInput(
                event_id="cal-scam-1",
                post_id=8101,
                channel_id=201,
                published_at=now,
                text_raw="СРОЧНАЯ РАЗДАЧА 500 TON! Заходи на https://claim-airdrop.example.com/bonus",
            ),
            True,
        ),
        (
            IngestPostInput(
                event_id="cal-ad-2",
                post_id=8102,
                channel_id=205,
                channel_username="backend_feed",
                published_at=now,
                text_raw=(
                    "Мало кто знает, почему ваши сервисы тормозят под высокой нагрузкой:\n"
                    "мы разобрали типовые ошибки профилирования памяти, планировщика горутин и сетевого пула "
                    "в продакшене без лишней теории. В конце модуля открывается скидка 35% по промокоду на "
                    "закрытый курс — переходите к @promo_courses_bot"
                ),
                entities=[
                    TelegramEntity(
                        type="text_url",
                        offset=_utf16_len(
                            "Мало кто знает, почему ваши сервисы тормозят под высокой нагрузкой:\n"
                            "мы разобрали типовые ошибки профилирования памяти, планировщика горутин и сетевого пула "
                            "в продакшене без лишней теории. В конце модуля открывается скидка 35% по промокоду на "
                        ),
                        length=_utf16_len("закрытый курс"),
                        url="https://t.me/promo_courses_bot",
                    )
                ],
            ),
            True,
        ),
        # Legitimate technical / analytical posts with links or mentions (expected is_ad=False)
        (e2e["pipeline_artifacts"]["fixture"].p_target, False),
        (
            IngestPostInput(
                event_id="cal-tech-1",
                post_id=8103,
                channel_id=202,
                published_at=now,
                text_raw=(
                    "Вышел релиз Qdrant v1.17 с поддержкой ACORN для selective pre-filtering "
                    "и улучшенным Weighted RRF; документация доступна на https://qdrant.tech/documentation/."
                ),
            ),
            False,
        ),
        (
            IngestPostInput(
                event_id="cal-news-1",
                post_id=8104,
                channel_id=203,
                published_at=now,
                text_raw=(
                    "ЦБ РФ сохранил ключевую ставку на уровне 21% по итогам заседания, "
                    "подробный релиз опубликован на официальном сайте регулятора https://cbr.ru/press/."
                ),
            ),
            False,
        ),
        (
            IngestPostInput(
                event_id="cal-tech-2",
                post_id=8105,
                channel_id=204,
                published_at=now,
                text_raw=(
                    "В gotd v0.120 исправили обработку Layer 225 и обновили схему messages.getHistory "
                    "для длинных каналов без потери pts."
                ),
            ),
            False,
        ),
    ]
    ad_tp = ad_fn = leg_fp = leg_tn = cpu_handled = 0
    for item, expected_rejected in spam_samples:
        verdict = spam_pipeline.spam_cascade.evaluate(
            item.text_raw,
            entities=[e.model_dump() for e in item.entities],
            source_channel=item.channel_username or str(item.channel_id),
            published_at=item.published_at,
        )
        predicted_rejected = verdict.is_ad or verdict.is_noise or (not verdict.should_persist)
        if verdict.stage_decided in {"L0", "L1"}:
            cpu_handled += 1
        if expected_rejected:
            if predicted_rejected:
                ad_tp += 1
            else:
                ad_fn += 1
        else:
            if predicted_rejected:
                leg_fp += 1
            else:
                leg_tn += 1
    spam_recall = ad_tp / max(1, ad_tp + ad_fn)
    spam_fpr = leg_fp / max(1, leg_fp + leg_tn)
    cpu_share = cpu_handled / max(1, len(spam_samples))

    # --- Шаг 2: Кластеризация и OriginScore ---
    cl_svc = PostPipelineService(settings=cfg)
    cluster_gold: list[tuple[IngestPostInput, int, bool]] = [
        # Cluster 0: gotd layer 225 (target = origin)
        (e2e["pipeline_artifacts"]["fixture"].p_target, 0, True),
        *[(rw, 0, False) for rw in e2e["pipeline_artifacts"]["fixture"].p_rewrites],
        # Cluster 1: CBR 21% (origin = first post)
        (
            IngestPostInput(
                event_id="cal-cbr21-orig",
                post_id=8201,
                channel_id=301,
                channel_username="cbr_official",
                published_at=now,
                text_raw="ЦБ РФ повысил ключевую ставку до 21% годовых. Набиуллина отметила рост инфляционных рисков.",
                has_primary_evidence=True,
                channel_authority=0.95,
                originality_score=0.99,
            ),
            1,
            True,
        ),
        (
            IngestPostInput(
                event_id="cal-cbr21-rw",
                post_id=8202,
                channel_id=302,
                channel_username="fin_news_fast",
                published_at=now + timedelta(minutes=20),
                text_raw="Как сообщает регулятор, ЦБ РФ повысил ключевую ставку до 21% годовых, Набиуллина подтвердила риски.",
                explicit_cite_score=0.80,
                channel_authority=0.40,
                originality_score=0.30,
            ),
            1,
            False,
        ),
        # Cluster 2: CBR 19% (conflicting numerical fact -> MUST separate from Cluster 1)
        (
            IngestPostInput(
                event_id="cal-cbr19-orig",
                post_id=8203,
                channel_id=303,
                channel_username="macro_analytics",
                published_at=now + timedelta(minutes=5),
                text_raw="Аналитик Сидоров прогнозирует снижение ключевой ставки ЦБ до 19% к концу третьего квартала.",
                has_primary_evidence=True,
                channel_authority=0.88,
                originality_score=0.95,
            ),
            2,
            True,
        ),
        # Cluster 3: gotd Layer 224 negative post -> MUST separate from Cluster 0
        (e2e["pipeline_artifacts"]["fixture"].p_neg, 3, True),
    ]
    labels_true: list[int] = []
    labels_pred: list[int] = []
    cid_map: dict[str, int] = {}
    predicted_origins_by_true_group: dict[int, int | None] = {}
    expected_origins_by_true_group: dict[int, int] = {}
    for item, true_grp, is_orig in cluster_gold:
        res = cl_svc.process_post(item)
        if res.cluster_result is None:
            continue
        cid = res.cluster_result.cluster_id
        if cid not in cid_map:
            cid_map[cid] = len(cid_map)
        labels_true.append(true_grp)
        labels_pred.append(cid_map[cid])
        if is_orig:
            expected_origins_by_true_group[true_grp] = item.post_id
        predicted_origins_by_true_group[true_grp] = res.cluster_result.origin_post_id

    ari = _adjusted_rand_index(labels_true, labels_pred)
    homog = _homogeneity(labels_true, labels_pred)
    origin_hits = sum(
        1
        for grp, exp_pid in expected_origins_by_true_group.items()
        if predicted_origins_by_true_group.get(grp) == exp_pid
    )
    origin_acc = origin_hits / max(1, len(expected_origins_by_true_group))

    # --- Шаг 3: Hybrid Search + Reranker (NDCG@10, MRR@5, Recall@20) ---
    ret_client = QdrantClient(":memory:")
    ret_repo = QdrantPublicChannelsRepo(ret_client, settings=cfg, use_remote_bm25_document=False)
    ret_repo.ensure_collection(recreate=True)
    ret_pipe = PostPipelineService(settings=cfg, qdrant_repo=ret_repo)
    corpus_texts: dict[int, str] = {}
    for item, _, _ in cluster_gold:
        res = ret_pipe.process_post(item)
        corpus_texts[item.post_id] = res.preprocessed.offset_map.text_nfkc
    ret_search = HybridSearchService(
        qdrant_repo=ret_repo,
        settings=cfg,
        embedder=ret_pipe.embedder,
        reranker=ret_pipe.reranker,
    )
    eval_queries = [
        (SPEC_9_2_QUERY, 9201, {9201, 9202, 9203, 9204}),
        ("До какого уровня ЦБ РФ повысил ключевую ставку по словам Набиуллиной?", 8201, {8201, 8202}),
        ("Какую ключевую ставку ЦБ прогнозирует Сидоров?", 8203, {8203}),
    ]
    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    sub_all = [102, 103, 104, 105, 106, 301, 302, 303]
    for q_text, top_expected, rel_ids in eval_queries:
        sr = ret_search.search_and_rerank(
            query=q_text,
            subscribed_channel_ids=sub_all,
            published_after=now - timedelta(days=1),
            top_k=10,
            group_limit=20,
            enable_reranker=True,
            full_post_texts=corpus_texts,
        )
        ranked_ids = [c.post_id for c in sr.reranked_posts]
        # Recall@20
        hits_all = len(set(ranked_ids) & rel_ids)
        recalls.append(hits_all / max(1, len(rel_ids)))
        # MRR@5 по наиболее релевантному источнику или кластеру
        rr = 0.0
        for rank_idx, pid in enumerate(ranked_ids[:5], start=1):
            if pid == top_expected:
                rr = 1.0 / rank_idx
                break
        if rr == 0.0:
            for rank_idx, pid in enumerate(ranked_ids[:5], start=1):
                if pid in rel_ids:
                    rr = 1.0 / rank_idx
                    break
        mrrs.append(rr)
        # NDCG@10 (gain=2 для первоисточника, gain=1 для рерайта)
        gains = [
            2.0 if pid == top_expected else (1.0 if pid in rel_ids else 0.0)
            for pid in ranked_ids[:10]
        ]
        dcg = sum(g / log2(i + 2) for i, g in enumerate(gains))
        ideal_gains = sorted(
            [2.0 if pid == top_expected else 1.0 for pid in rel_ids],
            reverse=True,
        )[:10]
        idcg = sum(g / log2(i + 2) for i, g in enumerate(ideal_gains))
        ndcgs.append(dcg / idcg if idcg > 0 else 1.0)

    ndcg_at_10 = sum(ndcgs) / len(ndcgs)
    mrr_at_5 = sum(mrrs) / len(mrrs)
    recall_at_20 = sum(recalls) / len(recalls)

    # --- Шаг 4: Smith-Waterman & NLI Gate + UTF-16 ---
    nli_gate: NliVerificationGate = e2e["pipeline_artifacts"]["nli_gate"]
    rag_check_precision = (
        1.0
        if (
            e2e["rag_and_nli"]["sw_alignment_score"] >= cfg.sw_similarity_threshold
            and e2e["rag_and_nli"]["nli_entailment_score"] >= cfg.nli_entailment_threshold
            and e2e["rag_and_nli"]["citation_post_id"] == 9201
        )
        else 0.0
    )
    # Проверка блокировки галлюцинации (противоречащее число 224 вместо 225, либо выдуманная ошибка)
    target_nfkc = e2e["pipeline_artifacts"]["indexed_posts"][0].text
    hallucination_probes = [
        "В gotd на 224 слое при чтении длинной истории возникает ошибка updates.channelDifferenceTooLong.",
        "В gotd на 225 слое всегда возвращается вымышленная ошибка FloodWait999999 без исключений.",
    ]
    hallucination_leaks = 0
    for claim_text in hallucination_probes:
        nli_res = nli_gate.check_pair(claim_text, target_nfkc)
        if nli_res.is_entailed:
            hallucination_leaks += 1
    hallucination_leakage_rate = hallucination_leaks / len(hallucination_probes)
    utf16_exact_match = (
        1.0
        if (
            e2e["rag_and_nli"]["utf16_emoji_shift"] == 1
            and e2e["rag_and_nli"]["utf16_sliced_text"]
            == e2e["rag_and_nli"]["verbatim_text"]
            and "updates.channelDifferenceTooLong" in e2e["rag_and_nli"]["utf16_sliced_text"]
        )
        else 0.0
    )

    # --- Шаг 5: Semantic Cache Isolation ---
    iso_cache: TwoLevelSemanticCache = e2e["pipeline_artifacts"]["semantic_cache"]
    q_225_ent = extract_entities(SPEC_9_2_QUERY)
    q_224_text = "Какая ошибка возникает в gotd на 224 слое при чтении длинной истории канала?"
    q_224_ent = extract_entities(q_224_text)
    q_vec_225 = ret_pipe.embedder.encode_queries([SPEC_9_2_QUERY])[0]
    q_vec_224 = ret_pipe.embedder.encode_queries([q_224_text])[0]
    key_225 = build_public_cache_key([102, 106], "7d", q_225_ent, reference_time=now)
    key_224 = build_public_cache_key([102, 106], "7d", q_224_ent, reference_time=now)
    iso_cache.put_level1(
        cache_key=key_225,
        query_text=SPEC_9_2_QUERY,
        query_embedding=q_vec_225,
        response_payload={"answer": "updates.channelDifferenceTooLong"},
        now=now,
    )
    miss_on_224 = iso_cache.get_level1(
        cache_key=key_224,
        query_embedding=q_vec_224,
        now=now,
    )
    false_cache_hits = 1 if (miss_on_224 is not None or key_225 == key_224) else 0
    cache_isolation_ok = false_cache_hits == 0

    gates = {
        "spam_cascade": {
            "recall": round(spam_recall, 4),
            "fpr": round(spam_fpr, 4),
            "l0_l1_cpu_share": round(cpu_share, 4),
            "passed": spam_recall >= 0.96 and spam_fpr <= 0.015 and cpu_share >= 0.82,
        },
        "clustering_and_origin": {
            "ari": round(ari, 4),
            "homogeneity": round(homog, 4),
            "origin_top1_accuracy": round(origin_acc, 4),
            "passed": ari >= 0.82 and homog >= 0.88 and origin_acc >= 0.90,
        },
        "hybrid_search_and_rerank": {
            "ndcg_at_10": round(ndcg_at_10, 4),
            "mrr_at_5": round(mrr_at_5, 4),
            "recall_at_20": round(recall_at_20, 4),
            "passed": ndcg_at_10 >= 0.84 and mrr_at_5 >= 0.88 and recall_at_20 >= 0.95,
        },
        "nli_and_smith_waterman": {
            "citation_precision": rag_check_precision,
            "hallucination_leakage_rate": hallucination_leakage_rate,
            "utf16_exact_match": utf16_exact_match,
            "passed": (
                rag_check_precision >= 1.0
                and hallucination_leakage_rate <= 0.005
                and utf16_exact_match >= 1.0
            ),
        },
        "semantic_cache_isolation": {
            "key_225": key_225,
            "key_224": key_224,
            "false_cache_hits": false_cache_hits,
            "passed": cache_isolation_ok,
        },
    }
    gates["all_passed"] = all(g["passed"] for g in gates.values())
    return gates
