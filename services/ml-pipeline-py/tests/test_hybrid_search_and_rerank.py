"""Unit-тесты гибридного поиска Qdrant (`query_points_groups`, Weighted RRF) и реранкинга."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from qdrant_client import QdrantClient, models

from app.cache.semantic_cache import TwoLevelSemanticCache
from app.config import Settings
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.qdrant_repo import QdrantPublicChannelsRepo


def test_build_hybrid_query_args_branches_weights_and_thresholds() -> None:
    cfg = Settings()
    repo = QdrantPublicChannelsRepo(QdrantClient(":memory:"), settings=cfg)
    repo.ensure_collection()
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    q_vec = repo.embedder.encode_query("Как устроено обсуждение новой архитектуры?")

    args_conceptual = repo.build_hybrid_query_args(
        query_text="Как устроено обсуждение новой архитектуры?",
        query_dense_512d=q_vec,
        subscribed_channel_ids=[101, 102],
        published_after=now - timedelta(days=7),
        has_exact_entities=False,
    )
    prefetches = args_conceptual["prefetch"]
    assert len(prefetches) == 5
    assert [p.using for p in prefetches] == [
        "dense_body",
        "bm25",
        "dense_code",
        "dense_ocr",
        "dense_comments",
    ]
    assert [p.score_threshold for p in prefetches] == [0.42, None, 0.55, 0.52, 0.50]
    assert [p.limit for p in prefetches] == [50, 50, 25, 25, 25]

    rrf_query = args_conceptual["query"]
    assert isinstance(rrf_query, models.RrfQuery)
    assert rrf_query.rrf.k == 60
    assert rrf_query.rrf.weights == [1.0, 0.9, 0.6, 0.5, 0.4]

    assert args_conceptual["group_by"] == "post_id"
    assert args_conceptual["group_size"] == 1
    assert args_conceptual["limit"] == 30
    assert args_conceptual["search_params"].quantization.rescore is True
    assert args_conceptual["search_params"].hnsw_ef == 128

    args_exact = repo.build_hybrid_query_args(
        query_text="Инцидент на порту 8080 в модуле SBER по 152-ФЗ",
        query_dense_512d=q_vec,
        subscribed_channel_ids=[101, 102],
        published_after=now - timedelta(days=7),
        has_exact_entities=True,
    )
    assert args_exact["query"].rrf.weights == [1.0, 1.6, 0.6, 0.5, 0.4]


def test_qdrant_prefilter_channel_ad_time_and_grouping() -> None:
    cfg = Settings(longread_char_threshold=400)
    repo = QdrantPublicChannelsRepo(QdrantClient(":memory:"), settings=cfg)
    repo.ensure_collection()
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)

    # Пост 1: длинный органический пост (дробится на 2+ чанка с одним post_id=1001)
    long_text = (
        "Авария в ЦОД: потеряно 14% пакетов из-за ошибки BGP на порту 8080. "
        + ("Подробности инцидента с маршрутизацией трафика на граничных узлах. " * 12)
    )
    chunks_1001 = repo.upsert_post_chunks(
        post_id=1001,
        channel_id=101,
        channel_username="infra_ops",
        channel_title="Infra Ops",
        text=long_text,
        published_at=now - timedelta(hours=2),
        is_ad=False,
    )
    assert len(chunks_1001) >= 2

    # Пост 2: рекламный (должен исключаться пре-фильтром is_ad=False)
    repo.upsert_post_chunks(
        post_id=1002,
        channel_id=101,
        channel_username="infra_ops",
        channel_title="Infra Ops",
        text="Реклама: настройка BGP на порту 8080 со скидкой 50%.",
        published_at=now - timedelta(hours=1),
        is_ad=True,
    )

    # Пост 3: неподписанный канал 999 (должен исключаться по channel_id IN)
    repo.upsert_post_chunks(
        post_id=1003,
        channel_id=999,
        channel_username="foreign_ch",
        channel_title="Foreign Channel",
        text="Пост из чужого канала про BGP и порт 8080.",
        published_at=now - timedelta(hours=1),
        is_ad=False,
    )

    # Пост 4: устаревший пост (раньше published_after)
    repo.upsert_post_chunks(
        post_id=1004,
        channel_id=102,
        channel_username="net_alerts",
        channel_title="Net Alerts",
        text="Старый пост про BGP на порту 8080.",
        published_at=now - timedelta(days=14),
        is_ad=False,
    )

    # Пост 5: мультимодальный пост (code + ocr + comments)
    repo.upsert_post_chunks(
        post_id=1005,
        channel_id=102,
        channel_username="net_alerts",
        channel_title="Net Alerts",
        text="Диагностика балансировщика и логов маршрутизации BGP.",
        code_text="nginx proxy_pass http://127.0.0.1:8080;",
        ocr_text="Слайд разбора инцидента BGP 8080",
        comments_digest_text="Инженеры подтвердили сбой на порту 8080",
        published_at=now - timedelta(hours=3),
        is_ad=False,
    )

    query = "Ошибка BGP на порту 8080 в ЦОД"
    q_vec = repo.embedder.encode_query(query)
    res = repo.query_hybrid_groups(
        query_text=query,
        query_dense_512d=q_vec,
        subscribed_channel_ids=[101, 102],
        published_after=now - timedelta(days=2),
        has_exact_entities=True,
        group_limit=30,
        group_size=1,
    )

    returned_post_ids = {int(g.id) for g in res.groups}
    assert returned_post_ids == {1001, 1005}
    for g in res.groups:
        assert len(g.hits) == 1


def test_hybrid_search_and_rerank_top30_to_top10() -> None:
    cfg = Settings()
    repo = QdrantPublicChannelsRepo(QdrantClient(":memory:"), settings=cfg)
    repo.ensure_collection()
    cache = TwoLevelSemanticCache(cfg)
    svc = HybridSearchService(repo, settings=cfg, semantic_cache=cache)
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)

    for idx in range(1, 18):
        body = (
            f"Инженерный отчет #{idx}: сбой BGP в ЦОД на порту 8080, потеряно {idx}% пакетов."
            if idx <= 14
            else f"Заметка #{idx}: обзор инструментов мониторинга сети без деталей маршрутизации."
        )
        repo.upsert_post_chunks(
            post_id=2000 + idx,
            channel_id=101 if idx % 2 == 0 else 102,
            channel_username=f"chan_{idx}",
            channel_title=f"Channel {idx}",
            text=body,
            published_at=now - timedelta(minutes=idx * 10),
            is_ad=False,
        )

    res = svc.search_and_rerank(
        query="потери пакетов BGP порт 8080",
        subscribed_channel_ids=[101, 102],
        published_after=now - timedelta(days=1),
        top_k=10,
        group_limit=30,
        enable_reranker=True,
    )
    assert res.has_exact_entities is True
    assert res.bm25_weight == 1.6
    assert res.rrf_candidates_count == 14
    assert len(res.reranked_posts) == 10
    assert all(p.rerank_score is not None for p in res.reranked_posts)
    scores = [float(p.rerank_score) for p in res.reranked_posts]
    assert scores == sorted(scores, reverse=True)

    # Повторный запуск использует кэш оценок реранкера (Level 2)
    res_cached = svc.search_and_rerank(
        query="потери пакетов BGP порт 8080",
        subscribed_channel_ids=[101, 102],
        published_after=now - timedelta(days=1),
        top_k=10,
        group_limit=30,
        enable_reranker=True,
    )
    assert [p.post_id for p in res_cached.reranked_posts] == [p.post_id for p in res.reranked_posts]
