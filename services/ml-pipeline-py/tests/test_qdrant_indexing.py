"""Тесты индексации Qdrant `public_channels_v1`, дробления лонгридов > 2000 символов и E2E конвейера."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from app.consumers import IngestPostInput, PostPipelineService, TelegramEntity
from app.main import create_app
from app.retrieval import QdrantPublicChannelsRepo, deterministic_point_id, split_longread_with_header


def test_longread_conditional_split_and_sparse_named_vectors_in_qdrant() -> None:
    """Проверяет условное дробление (>2000 символов) с шапкой и именованные векторы в Qdrant."""
    client = QdrantClient(":memory:")
    repo = QdrantPublicChannelsRepo(client=client)
    repo.ensure_collection(recreate=True)

    service = PostPipelineService(qdrant_repo=repo)
    pub_at = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)

    # 1. Короткий пост (<= 2000 символов): НЕ дробится, dense_code/dense_ocr не заполняются при пустоте
    short_res = service.process_post(
        IngestPostInput(
            event_id="evt_1",
            post_id=101,
            channel_id=1001,
            channel_username="backend_daily",
            channel_title="Backend Daily",
            published_at=pub_at,
            text_raw="PostgreSQL 18 ускорил параллельное чтение индексов и вакуум на 25%.",
        )
    )
    assert len(short_res.qdrant_chunks) == 1
    assert short_res.qdrant_chunks[0].total_chunks == 1
    assert short_res.qdrant_chunks[0].vector_names == ["bm25", "dense_body"]
    assert not short_res.qdrant_chunks[0].text.startswith("[Backend Daily |")

    # 2. Лонгрид (> 2000 символов) + блок кода + OCR + комментарийная шапка
    long_paragraphs = [
        f"Абзац {i}: подробный разбор архитектуры распределительного сервиса PostgreSQL и HNSW индексов Qdrant."
        for i in range(1, 35)
    ]
    long_text = "\n\n".join(long_paragraphs) + "\nSELECT * FROM core_public.posts;"
    assert len(long_text) > 2000

    code_pos = long_text.index("SELECT * FROM core_public.posts;")
    long_res = service.process_post(
        IngestPostInput(
            event_id="evt_2",
            post_id=102,
            channel_id=1001,
            channel_username="backend_daily",
            channel_title="Backend Daily",
            published_at=pub_at,
            text_raw=long_text,
            entities=[
                TelegramEntity(
                    type="pre",
                    offset=code_pos,
                    length=len("SELECT * FROM core_public.posts;"),
                    language="sql",
                )
            ],
            ocr_text="Throughput: 120k QPS on NVMe",
        )
    )
    assert 2 <= len(long_res.qdrant_chunks) <= 3
    for chunk in long_res.qdrant_chunks:
        assert chunk.text.startswith("[Backend Daily | ")
        assert "| 2026-03-20]" in chunk.text

    # У корневого чанка (chunk_id=0) заполнены dense_body, dense_code, dense_ocr, bm25; dense_comments пуст до воронки
    assert long_res.qdrant_chunks[0].vector_names == [
        "bm25",
        "dense_body",
        "dense_code",
        "dense_ocr",
    ]
    assert long_res.qdrant_chunks[1].vector_names == ["bm25", "dense_body"]

    # Проверяем фактическую запись в коллекцию Qdrant :memory:
    points = client.retrieve(
        collection_name="public_channels_v1",
        ids=[deterministic_point_id(102, 0)],
        with_vectors=True,
        with_payload=True,
    )
    assert len(points) == 1
    vectors = points[0].vector
    assert isinstance(vectors, dict)
    assert len(vectors["dense_body"]) == 512
    assert len(vectors["dense_code"]) == 512
    assert len(vectors["dense_ocr"]) == 512
    assert "dense_comments" not in vectors

    # 3. Рекламный пост сохраняется в Qdrant с is_ad = True
    ad_res = service.process_post(
        IngestPostInput(
            event_id="evt_3",
            post_id=103,
            channel_id=1002,
            channel_username="promo_channel",
            channel_title="Promo",
            published_at=pub_at,
            text_raw="Новый курс по архитектуре баз данных. erid: 2Vtzqv9911",
        )
    )
    assert ad_res.spam_verdict.is_ad is True
    assert ad_res.cluster_result is None
    assert len(ad_res.qdrant_chunks) == 1
    assert ad_res.qdrant_chunks[0].payload["is_ad"] is True

    # Проверка поиска через Qdrant query_points с фильтром is_ad=False
    found = client.query_points(
        collection_name="public_channels_v1",
        query=vectors["dense_body"],
        using="dense_body",
        query_filter=models.Filter(
            must=[models.FieldCondition(key="is_ad", match=models.MatchValue(value=False))]
        ),
        limit=5,
        with_payload=True,
    )
    found_post_ids = {p.payload["post_id"] for p in found.points}
    assert 102 in found_post_ids
    assert 103 not in found_post_ids


def test_split_longread_helper_threshold() -> None:
    ts = datetime(2026, 3, 21, tzinfo=timezone.utc)
    short_chunks = split_longread_with_header("Короткий пост.", channel_label="Канал", published_at=ts)
    assert short_chunks == ["Короткий пост."]

    long_text = " ".join(f"Предложение номер {i} о распределенных системах." for i in range(90))
    long_chunks = split_longread_with_header(long_text, channel_label="SysDesign", published_at=ts)
    assert 2 <= len(long_chunks) <= 3
    assert all(ch.startswith("[SysDesign | ") and "2026-03-21]" in ch for ch in long_chunks)


def test_fastapi_endpoints_smoke() -> None:
    app = create_app()
    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    resp = client.post(
        "/v1/posts/process",
        json={
            "event_id": "e1",
            "post_id": 900,
            "channel_id": 10,
            "channel_username": "ru_ai",
            "published_at": "2026-03-20T12:00:00Z",
            "text_raw": "🚀 Вышла модель Qwen3-Embedding-0.6B! Поддержка MRL до 512 координат.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["post_id"] == 900
    assert body["sentences"][0]["tag"] == "P900_S1"
    assert body["spam"]["is_ad"] is False
    assert body["cluster"]["origin_post_id"] == 900


def test_get_settings_handles_shallow_docker_container_path(monkeypatch) -> None:
    """Проверяет отсутствие IndexError в get_settings при пути /app/app/config.py внутри Docker."""
    import sys
    import app.config as app_config

    orig_pytest = sys.modules.pop("pytest", None)
    app_config.get_settings.cache_clear()
    monkeypatch.setattr(app_config, "__file__", "/app/app/config.py")
    try:
        cfg = app_config.get_settings()
        assert cfg.qdrant_collection
    finally:
        app_config.get_settings.cache_clear()
        if orig_pytest is not None:
            sys.modules["pytest"] = orig_pytest
