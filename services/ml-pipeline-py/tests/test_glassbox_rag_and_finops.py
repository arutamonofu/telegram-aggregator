"""Unit-тесты протокола Glass-box RAG, дайджестов, Stance-синтеза, FinOps и SSE API."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Sequence

import httpx
import pytest
from qdrant_client import QdrantClient

from app.cache.semantic_cache import TwoLevelSemanticCache
from app.config import Settings
from app.main import create_app
from app.preprocessing.offsets import build_utf16_offset_map
from app.rag.digest_builder import ClusteredDigestService, ClusterPostItem
from app.rag.nli_gate import NO_VERIFIED_DATA_STUB, NliVerificationGate
from app.rag.query_planner import (
    FinOpsBudgetTracker,
    RoutingDecision,
    UserUsageState,
    is_deepseek_peak_window,
    plan_task_route,
)
from app.rag.stance_matrix import StanceCandidateArg, StanceMatrixSynthesizer, StancePostInput
from app.rag.synthesizer import (
    DraftCitation,
    DraftClaim,
    GlassBoxRagSynthesizer,
    IndexedSourcePost,
    build_anthropic_citations_payload,
    build_openai_llguidance_payload,
    parse_anthropic_citations_response,
    parse_structured_json_response,
)
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.qdrant_repo import QdrantPublicChannelsRepo


def _make_indexed_post(post_id: int, text: str, *, channel_id: str = "101", title: str = "Infra") -> IndexedSourcePost:
    omap = build_utf16_offset_map(text, post_id=post_id)
    return IndexedSourcePost(
        post_id=post_id,
        channel_id=channel_id,
        channel_username=title.lower(),
        channel_title=title,
        msg_id=post_id,
        published_at=datetime(2026, 11, 15, 11, 0, tzinfo=timezone.utc),
        text=omap.text_nfkc,
        deeplink=f"https://t.me/{title.lower()}/{post_id}",
        offset_map=omap,
    )


def test_anthropic_citations_and_openai_llguidance_formats() -> None:
    doc = _make_indexed_post(
        101,
        "🚀 Сбой в ЦОД: потеряно 14% пакетов из-за BGP. Резервный канал запущен через 12 минут.",
    )

    anthropic_payload = build_anthropic_citations_payload("Что случилось в ЦОД?", [doc])
    assert anthropic_payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert anthropic_payload["messages"][0]["content"][0]["citations"] == {"enabled": True}

    mock_anthropic_resp = {
        "content": [
            {
                "type": "text",
                "text": "В ЦОД зафиксирована потеря 14% пакетов из-за сбоя BGP.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "потеряно 14% пакетов из-за BGP",
                        "document_index": 0,
                        "start_char_index": 14,
                        "end_char_index": 44,
                    }
                ],
            }
        ]
    }
    parsed_anth = parse_anthropic_citations_response(mock_anthropic_resp, [doc])
    assert len(parsed_anth) == 1
    assert parsed_anth[0].citations[0].post_id == 101
    assert parsed_anth[0].citations[0].sentence_id == 1
    assert parsed_anth[0].citations[0].exact_quote == "потеряно 14% пакетов из-за BGP"

    openai_payload = build_openai_llguidance_payload("Что случилось в ЦОД?", [doc], model_id="gpt-6.1-sol")
    assert openai_payload["response_format"]["type"] == "json_schema"
    assert openai_payload["response_format"]["json_schema"]["strict"] is True
    assert openai_payload["extra_body"]["guided_decoding_backend"] == "llguidance"
    assert openai_payload["reasoning_effort"] == "none"
    assert openai_payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    raw_openai_resp = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "mode": "qa",
                            "claims": [
                                {
                                    "claim_id": "c1",
                                    "topic_group": "Сеть",
                                    "statement": "Потеряно 14% пакетов из-за сбоя BGP",
                                    "citations": [
                                        {
                                            "post_id": 101,
                                            "sentence_id": 1,
                                            "exact_quote": "потеряно 14% пакетов из-за BGP",
                                        }
                                    ],
                                }
                            ],
                            "stance_comparison_matrix": [],
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }
    parsed_oai = parse_structured_json_response(raw_openai_resp)
    assert len(parsed_oai) == 1
    assert parsed_oai[0].claim_id == "c1"
    assert parsed_oai[0].citations[0].post_id == 101


def test_nli_gate_verification_and_no_verified_data_stub() -> None:
    doc = _make_indexed_post(501, "🚀 Сбой в ЦОД: потеряно 14% пакетов из-за BGP.")

    class _MixedBackend:
        def synthesize(
            self,
            *,
            query: str,
            documents: Sequence[IndexedSourcePost],
            routing: RoutingDecision,
            mode: str = "qa",
        ) -> list[DraftClaim]:
            _ = query, documents, routing, mode
            return [
                DraftClaim(
                    claim_id="c1",
                    topic_group="Сеть",
                    statement="Из-за BGP потеряно 14% пакетов в ЦОД",
                    citations=[DraftCitation(post_id=501, sentence_id=1, exact_quote="потеряно 14% пакетов из—за BGP")],
                ),
                DraftClaim(
                    claim_id="c2",
                    topic_group="Сеть",
                    statement="Из-за BGP потеряно 95% пакетов в ЦОД",  # галлюцинация числа
                    citations=[DraftCitation(post_id=501, sentence_id=1, exact_quote="потеряно 14% пакетов из-за BGP")],
                ),
            ]

    syn = GlassBoxRagSynthesizer(llm_backend=_MixedBackend(), nli_gate=NliVerificationGate())
    res = syn.run_rag_query(
        user_id="user-nli-1",
        query="Потери пакетов в ЦОД?",
        channel_ids=[101],
        preloaded_posts=[doc],
    )
    assert len(res.verified_claims) == 1
    assert res.rejected_claims_count == 1
    assert res.fallback_message is None
    assert res.verified_claims[0].citations[0].utf16_range == [15, 45]
    assert res.verified_claims[0].citations[0].nli_entailment_score >= 0.75

    class _AllHallucinatedBackend:
        def synthesize(
            self,
            *,
            query: str,
            documents: Sequence[IndexedSourcePost],
            routing: RoutingDecision,
            mode: str = "qa",
        ) -> list[DraftClaim]:
            _ = query, documents, routing, mode
            return [
                DraftClaim(
                    claim_id="c1",
                    topic_group="Фейк",
                    statement="ЦОД закрылся на 100 часов в Токио",
                    citations=[DraftCitation(post_id=501, sentence_id=1, exact_quote="закрылся на 100 часов в Токио")],
                )
            ]

    syn_stub = GlassBoxRagSynthesizer(llm_backend=_AllHallucinatedBackend())
    res_stub = syn_stub.run_rag_query(
        user_id="user-nli-2",
        query="Закрытие ЦОД",
        channel_ids=[101],
        preloaded_posts=[doc],
    )
    assert res_stub.verified_claims == []
    assert res_stub.fallback_message == NO_VERIFIED_DATA_STUB


def test_clustered_digest_shared_map_and_lightweight_reduce() -> None:
    svc = ClusteredDigestService()
    now = datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc)

    rubric_templates = [
        ("Экономика и финансы", "ЦБ обновил прогноз по ставке и инфляции на 2027 год."),
        ("Технологии и ИИ", "Вышла новая открытая модель Qwen для локального RAG и кода."),
        ("Кибербезопасность и инфраструктура", "Инженеры устранили ошибку маршрутизации BGP в крупном ЦОД."),
        ("Законодательство и регулирование", "Вступили в силу поправки к закону 152-ФЗ о защите данных."),
        ("Рынки и бизнес", "Акции промышленного сектора выросли после отчета о выручке."),
        ("Наука и образование", "Университет запустил новый суперкомпьютерный кластер для исследований."),
    ]

    posts: list[ClusterPostItem] = []
    for idx in range(1, 35):
        rub_name, base_text = rubric_templates[(idx - 1) % len(rubric_templates)]
        ch_id = "10" if idx <= 30 else "999"  # кластеры 31..34 не входят в подписку S_u=["10","20"]
        is_core = idx == 1
        posts.append(
            ClusterPostItem(
                post_id=3000 + idx,
                channel_id=ch_id,
                channel_username=f"ch_{ch_id}",
                channel_title=f"Channel {ch_id}",
                msg_id=idx,
                published_at=now - timedelta(minutes=idx),
                text=f"СРОЧНО!! Событие {idx}: {base_text} Подробности подтверждены источниками.",
                subscribers_count=12 if is_core else 2,
                cluster_id=f"cluster_{idx}",
                rubric_hint=rub_name,
                engagement_score=100 + idx * 5,
            )
        )

    # Рекламный пост должен быть отфильтрован на входе Shared Map
    posts.append(
        ClusterPostItem(
            post_id=3999,
            channel_id="10",
            channel_username="ch_10",
            channel_title="Channel 10",
            msg_id=999,
            published_at=now,
            text="Реклама сервиса VPN со скидкой 70%.",
            cluster_id="cluster_ad_only",
            is_ad=True,
        )
    )

    cards = svc.batch_precompute_shared_map(posts, now=now)
    card_ids = {c.cluster_id for c in cards}
    assert "cluster_ad_only" not in card_ids

    by_cid = {c.cluster_id: c for c in cards}
    assert by_cid["cluster_1"].processing_tier == "streaming_core"
    assert not by_cid["cluster_1"].neutral_headline.startswith("СРОЧНО")
    assert by_cid["cluster_2"].processing_tier == "batch_6h"
    assert svc.semantic_cache.get_event_card("cluster_1") is not None

    digest = svc.build_personal_digest(
        user_id="digest-user",
        subscribed_channel_ids=["10", "20"],
        cluster_ids=[f"cluster_{i}" for i in range(1, 35)],
        max_cards=28,
        now=now,
    )
    assert digest.total_candidate_clusters == 30
    assert 25 <= digest.selected_cards_count <= 30
    assert digest.selected_cards_count == 28
    assert 4 <= len(digest.rubrics) <= 7
    for c in digest.cards:
        assert all(str(link["channel_id"]) in {"10", "20"} for link in c.subscribed_post_links)
        assert "post_id" in c.origin_post


def test_stance_matrix_4step_mmr_quota_and_double_nli() -> None:
    syn = StanceMatrixSynthesizer()
    now = datetime(2026, 11, 15, 11, 0, tzinfo=timezone.utc)

    # Создаем 10 каналов с PRO-аргументами (один канал дает 4 аргумента -> квота <=2 на канал)
    # И посты CON/MIXED/NEUTRAL/ad
    posts: list[StancePostInput] = []
    for ch_idx in range(1, 7):
        for k in range(1, 4):
            pid = 4000 + ch_idx * 10 + k
            posts.append(
                StancePostInput(
                    post_id=pid,
                    channel_id=str(ch_idx),
                    channel_username=f"tech_{ch_idx}",
                    channel_title=f"Tech {ch_idx}",
                    msg_id=k,
                    published_at=now,
                    text=f"Вариант {ch_idx}.{k} ускоряет поиск и снижает затраты на GPU при масштабировании.",
                    stance_hint="PRO",
                )
            )
    for ch_idx in range(10, 15):
        pid = 5000 + ch_idx
        posts.append(
            StancePostInput(
                post_id=pid,
                channel_id=str(ch_idx),
                channel_username=f"crit_{ch_idx}",
                channel_title=f"Crit {ch_idx}",
                msg_id=1,
                published_at=now,
                text=f"Внедрение {ch_idx} создает риск утечек, замедляет ответ и увеличивает расходы.",
                stance_hint="CON",
            )
        )
    posts.append(
        StancePostInput(
            post_id=5999,
            channel_id="99",
            channel_username="neutral_ch",
            channel_title="Neutral",
            msg_id=1,
            published_at=now,
            text="Сегодня состоялась рабочая встреча комитета.",
        )
    )
    posts.append(
        StancePostInput(
            post_id=6000,
            channel_id="98",
            channel_username="ad_ch",
            channel_title="Ad",
            msg_id=1,
            published_at=now,
            text="Реклама: купите подписку выгодно!",
            is_ad=True,
        )
    )

    matrix = syn.build_verified_matrix(query="Стоит ли переходить на новый движок?", posts=posts)
    assert matrix.stance_counts["NEUTRAL"] == 1
    assert matrix.selected_pro_count <= 8
    assert matrix.selected_con_count <= 8

    pro_per_ch: dict[str, int] = {}
    for row in matrix.rows:
        for p in row.pro_points:
            ch = str(p.citations[0].channel_id)
            pro_per_ch[ch] = pro_per_ch.get(ch, 0) + 1
    assert all(cnt <= 2 for cnt in pro_per_ch.values())

    # Проверяем второй инвариант NLI: аргумент в колонке PRO с негативной полярностью отклоняется
    inverted_post = StancePostInput(
        post_id=6100,
        channel_id="200",
        channel_username="bad_ch",
        channel_title="Bad Ch",
        msg_id=1,
        published_at=now,
        text="Новый движок критикуют за опасный риск утечек и частые сбои на нагрузке.",
    )
    bad_pro_arg = StanceCandidateArg(
        arg_id="inv_1",
        side="PRO",
        aspect="Надежность и безопасность",
        statement="Новый движок критикуют за опасный риск утечек и частые сбои на нагрузке.",
        post_id=6100,
        sentence_id=1,
        channel_id="200",
        exact_quote="критикуют за опасный риск утечек и частые сбои",
    )
    checked = syn.build_verified_matrix(
        query="Новый движок",
        posts=[inverted_post],
        candidate_pro=[bad_pro_arg],
        candidate_con=[],
    )
    assert checked.selected_pro_count == 0
    assert checked.rejected_polarity_count == 1


def test_finops_cascade_zones_trial_basic_limit_and_deepseek_peak() -> None:
    tracker = FinOpsBudgetTracker()

    # GREEN -> YELLOW -> RED -> GUARDRAIL
    tracker.set_state(UserUsageState(user_id="u_green", plan_tier="PROSUMER", accumulated_tech_cogs_rub=40.0))
    d_green = tracker.plan_rag_route("u_green", deep_analysis=True, now=datetime(2026, 11, 15, 14, 0, tzinfo=timezone.utc))
    assert d_green.finops_zone == "GREEN"
    assert d_green.top_k_docs == 10
    assert d_green.model_tier == "frontier"

    tracker.set_state(
        UserUsageState(
            user_id="u_yellow",
            plan_tier="PROSUMER",
            accumulated_tech_cogs_rub=155.0,
            red_zone_cohort_share=0.08,  # снижает порог желтой зоны до 150 руб. (п. 7.8.8)
        )
    )
    d_yellow = tracker.plan_rag_route("u_yellow")
    assert d_yellow.finops_zone == "YELLOW"
    assert d_yellow.top_k_docs == 7

    tracker.set_state(UserUsageState(user_id="u_red", plan_tier="PROSUMER", accumulated_tech_cogs_rub=270.0))
    d_red = tracker.plan_rag_route("u_red", deep_analysis=True)
    assert d_red.finops_zone == "RED"
    assert d_red.use_compressed_event_cards is True
    assert d_red.top_k_docs == 6
    assert d_red.model_id == "gpt-6-luna"

    tracker.set_state(UserUsageState(user_id="u_guard", plan_tier="PROSUMER", rag_queries_last_minute=6))
    d_guard = tracker.plan_rag_route("u_guard")
    assert d_guard.finops_zone == "GUARDRAIL"
    assert d_guard.challenge_required is True
    assert d_guard.throttle_delay_sec == 3.0
    assert d_guard.enable_gpu_reranker is False
    assert d_guard.top_k_docs == 4

    # Жесткий лимит 30 RAG-запросов для TRIAL и BASIC
    tracker.set_state(UserUsageState(user_id="u_trial", plan_tier="TRIAL", rag_queries_used_period=29))
    assert tracker.plan_rag_route("u_trial").allowed is True
    tracker.record_query("u_trial")
    trial_blocked = tracker.plan_rag_route("u_trial")
    assert trial_blocked.allowed is False
    assert trial_blocked.rejection_reason == "rag_quota_exceeded_30"

    tracker.set_state(UserUsageState(user_id="u_basic", plan_tier="BASIC", rag_queries_used_period=30, rag_queries_today=0))
    basic_blocked = tracker.plan_rag_route("u_basic")
    assert basic_blocked.allowed is False
    assert basic_blocked.rejection_reason == "rag_quota_exceeded_30"

    # Проверка пиковых окон DeepSeek по МСК (будни 04:00-07:00 и 09:00-13:00)
    msk = timezone(timedelta(hours=3))
    weekday_peak = datetime(2026, 11, 16, 10, 30, tzinfo=msk)  # Пн 10:30 МСК
    weekday_off = datetime(2026, 11, 16, 15, 0, tzinfo=msk)    # Пн 15:00 МСК
    weekend_time = datetime(2026, 11, 15, 10, 30, tzinfo=msk)  # Вс 10:30 МСК
    assert is_deepseek_peak_window(weekday_peak) is True
    assert is_deepseek_peak_window(weekday_off) is False
    assert is_deepseek_peak_window(weekend_time) is False

    route_peak = plan_task_route("STREAMING_MAP_AND_COMMENTS", now=weekday_peak)
    assert route_peak.deepseek_blocked_for_peak is True
    assert route_peak.provider != "deepseek"


@pytest.mark.asyncio
async def test_fastapi_sse_streaming_keepalive_and_feed_endpoints() -> None:
    cfg = Settings(sse_keepalive_interval_sec=0.04)
    repo = QdrantPublicChannelsRepo(QdrantClient(":memory:"), settings=cfg)
    repo.ensure_collection()
    cache = TwoLevelSemanticCache(cfg)
    hybrid = HybridSearchService(repo, settings=cfg, semantic_cache=cache)
    tracker = FinOpsBudgetTracker(cfg)

    class _SlowMockLlmBackend:
        def synthesize(
            self,
            *,
            query: str,
            documents: Sequence[IndexedSourcePost],
            routing: RoutingDecision,
            mode: str = "qa",
        ) -> list[DraftClaim]:
            _ = query, routing, mode
            time.sleep(0.11)  # провоцирует минимум один keep-alive тик при interval=0.04s
            doc = documents[0]
            return [
                DraftClaim(
                    claim_id="c1",
                    topic_group="Сеть",
                    statement="Потеряно 14% пакетов в ЦОД из-за BGP",
                    citations=[
                        DraftCitation(
                            post_id=doc.post_id,
                            sentence_id=1,
                            exact_quote="потеряно 14% пакетов из—за BGP",
                        )
                    ],
                )
            ]

    post_doc = _make_indexed_post(901, "🚀 Сбой в ЦОД: потеряно 14% пакетов из-за BGP.")
    digest_svc = ClusteredDigestService(settings=cfg, semantic_cache=cache, finops_tracker=tracker)
    digest_svc.precompute_shared_event_card(
        "cluster_net_1",
        [
            ClusterPostItem(
                post_id=901,
                channel_id="101",
                channel_username="infra",
                channel_title="Infra",
                msg_id=901,
                published_at=datetime(2026, 11, 15, 11, 0, tzinfo=timezone.utc),
                text="🚀 Сбой в ЦОД: потеряно 14% пакетов из-за BGP. Трафик переключен.",
                subscribers_count=10,
                cluster_id="cluster_net_1",
            )
        ],
    )

    app = create_app(
        settings=cfg,
        semantic_cache=cache,
        hybrid_search=hybrid,
        finops_tracker=tracker,
        llm_backend=_SlowMockLlmBackend(),
        digest_service=digest_svc,
        clean_posts_store=[post_doc],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. SSE QA с : keep-alive\n\n
        sse_resp = await client.post(
            "/api/v1/rag/query",
            json={
                "user_id": "sse-user-1",
                "query": "Сколько пакетов потеряно в ЦОД?",
                "mode": "qa",
                "channel_ids": [101],
                "preloaded_posts": [
                    {
                        "post_id": 901,
                        "channel_id": "101",
                        "channel_username": "infra",
                        "channel_title": "Infra",
                        "msg_id": 901,
                        "published_at": "2026-11-15T11:00:00+00:00",
                        "text": "🚀 Сбой в ЦОД: потеряно 14% пакетов из-за BGP.",
                    }
                ],
            },
        )
        assert sse_resp.status_code == 200
        assert sse_resp.headers["content-type"].startswith("text/event-stream")
        body = sse_resp.text
        assert ": keep-alive\n\n" in body
        assert "event: pipeline_stage\n" in body
        assert "event: direct_answer\n" in body
        assert "event: verified_claim\n" in body
        assert "event: source_posts\n" in body
        assert "event: done\n" in body
        assert '"utf16_range": [15, 45]' in body

        # 2. SSE Stance Matrix
        stance_resp = await client.post(
            "/api/v1/rag/query",
            json={
                "user_id": "sse-user-2",
                "query": "Переход на новый балансировщик",
                "mode": "stance_matrix",
                "channel_ids": [101, 102],
                "preloaded_posts": [
                    {
                        "post_id": 911,
                        "channel_id": "101",
                        "channel_title": "Ops",
                        "msg_id": 1,
                        "published_at": "2026-11-15T11:00:00+00:00",
                        "text": "Новый балансировщик ускоряет маршрутизацию и экономит CPU.",
                        "stance_hint": "PRO",
                    },
                    {
                        "post_id": 912,
                        "channel_id": "102",
                        "channel_title": "Sec",
                        "msg_id": 2,
                        "published_at": "2026-11-15T11:05:00+00:00",
                        "text": "Миграция создает риск сбоя и замедляет откат конфигурации.",
                        "stance_hint": "CON",
                    },
                ],
            },
        )
        assert stance_resp.status_code == 200
        assert "event: stance_row\n" in stance_resp.text
        assert "event: done\n" in stance_resp.text

        # 3. Ограничение 30 RAG-запросов для TRIAL -> HTTP 429
        tracker.set_state(UserUsageState(user_id="trial-capped", plan_tier="TRIAL", rag_queries_used_period=30))
        blocked_resp = await client.post(
            "/api/v1/rag/query",
            json={
                "user_id": "trial-capped",
                "query": "Проверка лимита",
                "mode": "qa",
                "channel_ids": [101],
            },
        )
        assert blocked_resp.status_code == 429
        assert blocked_resp.json()["detail"]["error"] == "rag_quota_exceeded_30"

        # 4. GET /api/v1/feed (digest и clean_originals)
        feed_digest = await client.get(
            "/api/v1/feed",
            params={"user_id": "sse-user-1", "mode": "digest", "channel_ids": "101"},
        )
        assert feed_digest.status_code == 200
        digest_json = feed_digest.json()
        assert digest_json["mode"] == "digest"
        assert digest_json["selected_cards_count"] == 1

        feed_orig = await client.get(
            "/api/v1/feed",
            params={"user_id": "sse-user-1", "mode": "clean_originals", "channel_ids": "101"},
        )
        assert feed_orig.status_code == 200
        orig_json = feed_orig.json()
        assert orig_json["mode"] == "clean_originals"
        assert orig_json["total"] == 1
        assert orig_json["items"][0]["post_id"] == 901


def test_openrouter_llm_backend_wiring_and_fenced_json() -> None:
    from app.rag.synthesizer import GlassBoxRagSynthesizer, HttpOpenAILlguidanceBackend

    cfg = Settings(
        llm_api_key="or-test-key",
        llm_base_url="https://openrouter.ai/api/v1",
        llm_model_id="openai/gpt-4o-mini",
        llm_strict_vllm_extra_body=False,
    )
    synth = GlassBoxRagSynthesizer(settings=cfg)
    assert isinstance(synth.llm_backend, HttpOpenAILlguidanceBackend)
    assert synth.llm_backend.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert synth.llm_backend.model_override == "openai/gpt-4o-mini"
    assert synth.llm_backend.include_vllm_extra_body is False

    doc = _make_indexed_post(701, "OpenRouter возвращает ответ в JSON.")
    openrouter_payload = build_openai_llguidance_payload(
        "Что возвращает OpenRouter?",
        [doc],
        model_id="openai/gpt-4o-mini",
        include_vllm_extra_body=False,
    )
    assert "extra_body" not in openrouter_payload
    assert "reasoning_effort" not in openrouter_payload
    assert openrouter_payload["model"] == "openai/gpt-4o-mini"

    fenced = """```json
    {"claims":[{"claim_id":"c1","topic_group":"API","statement":"OpenRouter работает.","citations":[{"post_id":701,"sentence_id":1,"exact_quote":"OpenRouter возвращает ответ в JSON."}]}]}
    ```"""
    parsed = parse_structured_json_response(fenced)
    assert len(parsed) == 1
    assert parsed[0].citations[0].post_id == 701


def test_news_summary_multi_clause_nli_and_extractive_synthesis() -> None:
    news_text = (
        "Война на Украине (на 13.11.26): Удары по тылам – мысли по поводу...\n"
        "Сегодня ночью противник атаковал очередной российский НПЗ. На этот раз в Волгограде.\n"
        "Несмотря на плотную ПВО, если бить по объектам ВПК и мостам через Днепр без ограничений, "
        "это скоро скажется на фронте."
    )
    doc = _make_indexed_post(1001, news_text, channel_id="ch_news", title="News Desk")
    unrelated = _make_indexed_post(
        1002,
        "Вышел новый релиз базы данных PostgreSQL 18 с ускорением вакуума.",
        channel_id="ch_db",
        title="DB Weekly",
    )

    # 1. NLI не должен ложно срабатывать на «несмотря на» и «без ограничений»
    gate = NliVerificationGate()
    idiom_res = gate.check_pair(
        statement="Удары по объектам ВПК и мостам через Днепр скоро скажутся на фронте.",
        verbatim_text=(
            "Несмотря на плотную ПВО, если бить по объектам ВПК и мостам через Днепр без ограничений, "
            "это скоро скажется на фронте."
        ),
    )
    assert idiom_res.label != "CONTRADICTS"
    assert idiom_res.is_entailed is True

    # 2. Контекстная проверка мульти-цитат: отдельные предложения S2 и S3 неполны, а вместе дают ENTAILMENT
    class _MultiClauseNliBackend:
        def predict_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
            _ = hypothesis
            has_npz = "нпз" in premise.lower()
            has_city = "волгограде" in premise.lower()
            if has_npz and has_city:
                return {"ENTAILMENT": 0.86, "NEUTRAL": 0.11, "CONTRADICTS": 0.03}
            return {"ENTAILMENT": 0.08, "NEUTRAL": 0.89, "CONTRADICTS": 0.03}

    multi_gate = NliVerificationGate(backend=_MultiClauseNliBackend())
    syn_multi = GlassBoxRagSynthesizer(nli_gate=multi_gate)
    draft = DraftClaim(
        claim_id="c1",
        topic_group="Атаки на НПЗ",
        statement="Ночью противник атаковал российский НПЗ в Волгограде.",
        citations=[
            DraftCitation(
                post_id=1001,
                sentence_id=2,
                exact_quote="[P1001_S2] Сегодня ночью противник атаковал очередной российский НПЗ.",
            ),
            DraftCitation(
                post_id=1001,
                sentence_id=3,
                exact_quote="На этот раз в Волгограде.",
            ),
        ],
    )
    verified, rejected, sw_fail, _ = syn_multi.verify_draft_claims([draft], {1001: doc})
    assert sw_fail == 0
    assert rejected == 0
    assert len(verified) == 1
    assert len(verified[0].citations) == 2

    # 3. ExtractiveGlassBoxBackend на обзорном новостном запросе формирует подтвержденные тезисы
    syn_ext = GlassBoxRagSynthesizer()
    res = syn_ext.run_rag_query(
        user_id="news-user-1",
        query="какие новости есть по фронту на украине?",
        channel_ids=["ch_news", "ch_db"],
        preloaded_posts=[doc, unrelated],
    )
    assert len(res.verified_claims) >= 1
    assert res.fallback_message is None
    assert all(c.citations[0].post_id == 1001 for c in res.verified_claims)

    # 4. Анти-тизерный реранкинг preloaded_posts, порядок source_posts_payload и даты в промпте LLM
    teaser_text = (
        "Война на Украине (на 13.11.26): Обстановка на фронтах...\n"
        "Свежий обзор ситуации на всех участках фронта. Что происходит, изменения, перспективы...\n"
        "Ссылка на видео в моей группе в ВК: https://vkvideo.ru/video-213276384_456243671"
    )
    teaser_doc = IndexedSourcePost(
        post_id=3001,
        channel_id="ch_news",
        channel_username="yurasumy",
        channel_title="Юрий Подоляка",
        msg_id=3001,
        published_at=datetime(2026, 11, 13, 11, 37, tzinfo=timezone.utc),
        text=teaser_text,
        deeplink="https://t.me/yurasumy/3001",
        offset_map=build_utf16_offset_map(teaser_text, post_id=3001),
    )
    res_ranked = syn_ext.run_rag_query(
        user_id="news-user-2",
        query="какие новости последние по фронту на украине?",
        channel_ids=["ch_news", "ch_db"],
        preloaded_posts=[teaser_doc, unrelated, doc],
    )
    assert len(res_ranked.verified_claims) >= 1
    assert res_ranked.verified_claims[0].citations[0].post_id == 1001
    # Процитированный источник [1] должен идти первым в source_posts_payload
    assert res_ranked.source_posts_payload[0]["post"]["post_id"] == 1001

    from app.rag.synthesizer import build_openai_llguidance_payload

    llm_payload = build_openai_llguidance_payload(
        "какие новости последние по фронту на украине?",
        [doc],
    )
    sys_msg = llm_payload["messages"][0]["content"]
    user_msg = llm_payload["messages"][1]["content"]
    assert "Skip empty video/stream announcements" in sys_msg
    assert f"{doc.published_at:%Y-%m-%d}" in user_msg


def test_direct_answer_qa_indirect_polar_and_hallucination_guard() -> None:
    from app.rag.synthesizer import parse_structured_json_synthesis

    # 1. Бинарный вопрос без прямого подтверждения в постах -> status="indirect" + честный ответ
    post_1 = _make_indexed_post(
        2001,
        (
            "Российские войска продолжают наносить удары по объектам, обеспечивающим снабжение "
            "украинской армии, включая порты и энергетическую инфраструктуру. "
            "В Одесской области был атакован порт Черноморск."
        ),
        channel_id="ch_mig",
        title="МИГ России",
    )
    post_2 = _make_indexed_post(
        2002,
        (
            "Президент Украины Владимир Зеленский заявил о готовности провести референдум о выводе ВСУ "
            "из оставшейся части Донбасса при условии получения всеобъемлющего пакета мер безопасности."
        ),
        channel_id="ch_news2",
        title="Новости",
    )

    syn_ext = GlassBoxRagSynthesizer()
    res_polar = syn_ext.run_rag_query(
        user_id="qa-polar-1",
        query="украина сдалась в войне?",
        channel_ids=["ch_mig", "ch_news2"],
        preloaded_posts=[post_1, post_2],
    )
    assert res_polar.direct_answer is not None
    assert res_polar.direct_answer.status == "indirect"
    assert "Прямого подтверждения этому в подключённых каналах нет" in res_polar.direct_answer.text
    assert len(res_polar.verified_claims) >= 1
    assert len(res_polar.direct_answer.citations) >= 1

    # 2. Прямой фактологический вопрос -> status="direct" с точными фактами
    doc_net = _make_indexed_post(
        2003,
        "🚀 Сбой в ЦОД: потеряно 14% пакетов из-за BGP. Резервный канал запущен через 12 минут.",
    )
    res_direct = syn_ext.run_rag_query(
        user_id="qa-direct-1",
        query="Сколько пакетов потеряно в ЦОД из-за BGP?",
        channel_ids=["101"],
        preloaded_posts=[doc_net],
    )
    assert res_direct.direct_answer is not None
    assert res_direct.direct_answer.status == "direct"
    assert "14%" in res_direct.direct_answer.text

    # 3. Защита от галлюцинации в direct_answer (LLM приписала 99% вместо 14%) -> отклоняется и заменяется
    class _HallucinatedDirectAnswerBackend:
        def __init__(self) -> None:
            self.last_draft_answer: tuple[str | None, str | None] = (None, None)

        def synthesize(
            self,
            *,
            query: str,
            documents: Sequence[IndexedSourcePost],
            routing: RoutingDecision,
            mode: str = "qa",
        ) -> list[DraftClaim]:
            _ = query, documents, routing, mode
            parsed_claims, ans_status, ans_text = parse_structured_json_synthesis(
                {
                    "mode": "qa",
                    "answer_status": "direct",
                    "direct_answer": "В ЦОД потеряно 99% пакетов из-за BGP [c1].",
                    "claims": [
                        {
                            "claim_id": "c1",
                            "topic_group": "Сеть",
                            "statement": "В ЦОД потеряно 14% пакетов из-за BGP.",
                            "citations": [
                                {
                                    "post_id": 2003,
                                    "sentence_id": 1,
                                    "exact_quote": "потеряно 14% пакетов из-за BGP",
                                }
                            ],
                        }
                    ],
                    "stance_comparison_matrix": [],
                }
            )
            self.last_draft_answer = (ans_status, ans_text)
            return parsed_claims

    syn_guard = GlassBoxRagSynthesizer(llm_backend=_HallucinatedDirectAnswerBackend())
    res_guarded = syn_guard.run_rag_query(
        user_id="qa-guard-1",
        query="Сколько пакетов потеряно в ЦОД?",
        channel_ids=["101"],
        preloaded_posts=[doc_net],
    )
    assert res_guarded.direct_answer is not None
    assert "99%" not in res_guarded.direct_answer.text
    assert "14%" in res_guarded.direct_answer.text
    assert res_guarded.direct_answer.status == "direct"

    # 4. Валидный LLM direct_answer со статусом indirect и ссылкой [c1] резолвится в [1] и проходит NLI
    class _ValidIndirectLlmBackend:
        def __init__(self) -> None:
            self.last_draft_answer: tuple[str | None, str | None] = (None, None)

        def synthesize(
            self,
            *,
            query: str,
            documents: Sequence[IndexedSourcePost],
            routing: RoutingDecision,
            mode: str = "qa",
        ) -> list[DraftClaim]:
            _ = query, documents, routing, mode
            parsed_claims, ans_status, ans_text = parse_structured_json_synthesis(
                {
                    "mode": "qa",
                    "answer_status": "indirect",
                    "direct_answer": (
                        "В источниках нет сообщений о капитуляции Украины. "
                        "Напротив, сообщается об ударах по порту Черноморск [c1] и заявлении Зеленского о референдуме [c2]."
                    ),
                    "claims": [
                        {
                            "claim_id": "c1",
                            "topic_group": "События",
                            "statement": "В Одесской области был атакован порт Черноморск.",
                            "citations": [
                                {
                                    "post_id": 2001,
                                    "sentence_id": 2,
                                    "exact_quote": "В Одесской области был атакован порт Черноморск.",
                                }
                            ],
                        },
                        {
                            "claim_id": "c2",
                            "topic_group": "Политика",
                            "statement": "Владимир Зеленский заявил о готовности провести референдум о выводе ВСУ из оставшейся части Донбасса.",
                            "citations": [
                                {
                                    "post_id": 2002,
                                    "sentence_id": 1,
                                    "exact_quote": "Владимир Зеленский заявил о готовности провести референдум о выводе ВСУ из оставшейся части Донбасса",
                                }
                            ],
                        },
                    ],
                    "stance_comparison_matrix": [],
                }
            )
            self.last_draft_answer = (ans_status, ans_text)
            return parsed_claims

    syn_valid_llm = GlassBoxRagSynthesizer(llm_backend=_ValidIndirectLlmBackend())
    res_valid_llm = syn_valid_llm.run_rag_query(
        user_id="qa-indirect-llm-1",
        query="украина сдалась в войне?",
        channel_ids=["ch_mig", "ch_news2"],
        preloaded_posts=[post_1, post_2],
    )
    assert res_valid_llm.direct_answer is not None
    assert res_valid_llm.direct_answer.status == "indirect"
    assert "[1]" in res_valid_llm.direct_answer.text
    assert "[2]" in res_valid_llm.direct_answer.text
    assert "нет сообщений о капитуляции" in res_valid_llm.direct_answer.text
