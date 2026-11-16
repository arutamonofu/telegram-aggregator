"""Сквозные интеграционные тесты разделов 8.1.4 и 9.2 `docs/SPEC.md`."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.calibration_qa import (
    SPEC_9_2_QUERY,
    build_spec_9_2_fixture,
    evaluate_stage1_quality_gates,
    run_spec_9_2_e2e_scenario,
)
from app.cache.semantic_cache import TwoLevelSemanticCache
from app.config import get_settings
from app.consumers.post_pipeline import PostPipelineService
from app.main import create_app
from app.rag.nli_gate import NliVerificationGate
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.qdrant_repo import QdrantPublicChannelsRepo


def _parse_sse_frames(raw_body: str) -> list[tuple[str, Any]]:
    frames: list[tuple[str, Any]] = []
    for block in raw_body.strip().split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if event_name and data_lines:
            frames.append((event_name, json.loads("\n".join(data_lines))))
    return frames


def test_spec_9_2_e2e_gotd_layer_225_pipeline() -> None:
    """Проверяет прохождение всех узлов сценария SPEC 9.2 для gotd на 225 слое."""
    res = run_spec_9_2_e2e_scenario()

    # 1. Антиспам-каскад (L1 > 0.80 -> is_ad = True)
    ad = res["ad_verdict"]
    assert ad["post_id"] == 9200
    assert ad["is_ad"] is True
    assert ad["stage_decided"] == "L1"
    assert ad["l1_score"] > 0.80

    # 2. Кластеризация и OriginScore (P_target + 3 рерайта в одном кластере, P_neg отдельно)
    cl = res["clustering"]
    assert cl["member_post_ids"] == [9201, 9202, 9203, 9204]
    assert all(cid == cl["target_cluster_id"] for cid in cl["rewrite_cluster_ids"])
    assert cl["origin_post_id"] == 9201
    assert cl["origin_channel_id"] == 102
    assert cl["neg_cluster_id"] is not None
    assert cl["neg_cluster_id"] != cl["target_cluster_id"]

    # 3. Сущности и Semantic Cache Key (225 + gotd) + буст w_bm25 = 1.6
    ent = res["entities_and_cache"]
    assert {"225", "gotd"}.issubset(set(ent["extracted_all_canonical"]))
    assert ent["exact_trigger"] is True
    assert len(ent["cache_key"]) == 64
    assert ent["bm25_weight"] == 1.6

    # 4. Гибридный поиск, selectivty < 0.5%, порог dense_code (> 0.55), group_by, reranking (#1 = P_target)
    ret = res["retrieval"]
    assert ret["selectivity_under_0_5_pct"] is True
    assert ret["selectivity"] < 0.005
    assert ret["group_by"] == "post_id"
    assert ret["dense_code_cosine"] > ret["threshold_dense_code"]
    assert ret["threshold_dense_code"] == 0.55
    assert "dense_code" in ret["target_vector_names"]
    assert "dense_code" not in ret["neg_vector_names"]
    assert ret["neg_cosine"] < ret["threshold_dense_code"]
    assert 9200 not in ret["reranked_post_ids"], "Qdrant is_ad=false pre-filter must exclude P_ad"
    assert ret["top1_post_id"] == 9201

    # 5. Glass-box RAG: Smith-Waterman + UTF-16 сдвиг от эмодзи 🔥 (+1) + NLI-гейт >= 0.75
    rag = res["rag_and_nli"]
    assert rag["verified_claims_count"] >= 1
    assert rag["fallback_message"] is None
    assert rag["citation_post_id"] == 9201
    assert rag["utf16_emoji_shift"] == 1
    assert rag["sw_alignment_score"] >= 0.85
    assert rag["nli_entailment_score"] >= 0.75
    assert "updates.channelDifferenceTooLong" in rag["utf16_sliced_text"]
    assert rag["utf16_sliced_text"] == rag["verbatim_text"]


def test_spec_9_2_fastapi_end_to_end_http_and_sse() -> None:
    """Проверяет сквозной HTTP/SSE путь через FastAPI: /v1/posts/process -> /api/v1/rag/query."""
    settings = get_settings()
    fixture = build_spec_9_2_fixture()

    qdrant_repo = QdrantPublicChannelsRepo(
        client=QdrantClient(":memory:"),
        settings=settings,
        use_remote_bm25_document=False,
    )
    qdrant_repo.ensure_collection(recreate=True)
    pipeline = PostPipelineService(settings=settings, qdrant_repo=qdrant_repo)
    cache = TwoLevelSemanticCache(settings=settings)
    nli_gate = NliVerificationGate(settings=settings)
    hybrid_search = HybridSearchService(
        qdrant_repo=qdrant_repo,
        settings=settings,
        embedder=pipeline.embedder,
        reranker=pipeline.reranker,
        semantic_cache=cache,
    )

    app = create_app(
        settings=settings,
        pipeline=pipeline,
        semantic_cache=cache,
        hybrid_search=hybrid_search,
        nli_gate=nli_gate,
    )

    with TestClient(app) as client:
        for post in [
            fixture.p_ad,
            fixture.p_target,
            *fixture.p_rewrites,
            fixture.p_neg,
        ]:
            resp = client.post("/v1/posts/process", json=post.model_dump(mode="json"))
            assert resp.status_code == 200
            body = resp.json()
            if post.post_id == 9200:
                assert body["spam"]["is_ad"] is True
                assert body["spam"]["stage_decided"] == "L1"
            elif post.post_id in {9201, 9202, 9203, 9204}:
                assert body["spam"]["is_ad"] is False
                assert body["cluster"]["origin_post_id"] == 9201
            else:
                assert body["cluster"]["origin_post_id"] == 9205

        sse_resp = client.post(
            "/api/v1/rag/query",
            json={
                "user_id": "http-e2e-user",
                "query": SPEC_9_2_QUERY,
                "mode": "qa",
                "channel_ids": fixture.subscribed_channel_ids,
                "time_window_days": 7,
            },
        )
        assert sse_resp.status_code == 200
        frames = _parse_sse_frames(sse_resp.text)
        events_by_name: dict[str, list[Any]] = {}
        for name, payload in frames:
            events_by_name.setdefault(name, []).append(payload)

        assert "verified_claim" in events_by_name
        assert "done" in events_by_name
        done_payload = events_by_name["done"][-1]
        assert done_payload["status"] == "ok"
        assert done_payload["verified_claims_count"] >= 1

        first_claim = events_by_name["verified_claim"][0]
        first_citation = first_claim["citations"][0]
        assert first_citation["post_id"] == 9201
        assert first_citation["sw_alignment_score"] >= 0.85
        assert first_citation["nli_entailment_score"] >= 0.75
        assert "updates.channelDifferenceTooLong" in first_citation["verbatim_text"]


def test_spec_9_2_five_step_calibration_and_stage1_quality_gates() -> None:
    """Проверяет все 5 шагов калибровки SPEC 9.2 и ML-пороги Quality Gates SPEC 8.1.4."""
    gates = evaluate_stage1_quality_gates()
    assert gates["all_passed"] is True

    assert gates["spam_cascade"]["recall"] >= 0.96
    assert gates["spam_cascade"]["fpr"] <= 0.015
    assert gates["spam_cascade"]["l0_l1_cpu_share"] >= 0.82

    assert gates["clustering_and_origin"]["ari"] >= 0.82
    assert gates["clustering_and_origin"]["homogeneity"] >= 0.88
    assert gates["clustering_and_origin"]["origin_top1_accuracy"] >= 0.90

    assert gates["hybrid_search_and_rerank"]["ndcg_at_10"] >= 0.84
    assert gates["hybrid_search_and_rerank"]["mrr_at_5"] >= 0.88
    assert gates["hybrid_search_and_rerank"]["recall_at_20"] >= 0.95

    assert gates["nli_and_smith_waterman"]["citation_precision"] == 1.0
    assert gates["nli_and_smith_waterman"]["hallucination_leakage_rate"] <= 0.005
    assert gates["nli_and_smith_waterman"]["utf16_exact_match"] == 1.0

    assert gates["semantic_cache_isolation"]["false_cache_hits"] == 0
