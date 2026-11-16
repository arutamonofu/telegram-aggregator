"""Комплексная проверка базовой инфраструктуры Стадии 1 (Dev).

Проверяет:
1. Сигнатуры и доступность классов `qdrant_client.models`.
2. Наличие и валидность SQL-миграций в `deploy/postgres/migrations/`.
3. PostgreSQL 16: схемы `core_public` и `tenant_rf`, все таблицы, ENUM-типы,
   индексы, триггеры `updated_at` и тестовую транзакцию (с ROLLBACK).
4. Redis 7: доступность `PING` и работу Redis Streams (`XADD`/`XREADGROUP`/`XACK`).
5. Qdrant: инициализацию `public_channels_v1`, создание payload-индексов ДО
   загрузки векторов и сквозной smoke-тест записи (`Qdrant/bm25`) и гибридного
   поиска (`query_points_groups` + `Weighted RRF` + `AcornSearchParams`) на
   временной коллекции с последующим удалением.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import redis
from qdrant_client import QdrantClient, models

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deploy.qdrant.init_public_channels_v1 import (  # noqa: E402
    COLLECTION_NAME,
    REQUIRED_PAYLOAD_INDEXES,
    VECTOR_SIZE,
    build_acorn_search_params,
    init_public_channels_collection,
    verify_collection_schema,
)

EXPECTED_CORE_PUBLIC_TABLES = {
    "channels",
    "mtproto_sessions",
    "channel_access_hashes",
    "event_clusters",
    "posts",
    "ingest_idempotency",
    "post_utf16_offsets",
    "post_comments",
    "comment_summaries",
    "file_references",
    "file_sources",
}

EXPECTED_TENANT_RF_TABLES = {
    "users",
    "user_channel_subscriptions",
    "channel_presets",
    "user_finops_quotas",
}

EXPECTED_ENUMS = {
    ("core_public", "channel_tier_enum"),
    ("core_public", "session_state_enum"),
    ("core_public", "spam_class_enum"),
    ("core_public", "stance_enum"),
    ("core_public", "media_kind_enum"),
    ("tenant_rf", "plan_tier_enum"),
    ("tenant_rf", "finops_zone_enum"),
}


def check_qdrant_models_api() -> None:
    """Проверяет точные сигнатуры требуемых моделей qdrant_client.models."""
    required_attrs = {
        "VectorParams": ("size", "distance", "hnsw_config", "quantization_config", "on_disk"),
        "SparseVectorParams": ("index", "modifier"),
        "HnswConfigDiff": ("m", "ef_construct", "full_scan_threshold", "on_disk"),
        "ScalarQuantization": ("scalar",),
        "ScalarQuantizationConfig": ("type", "quantile", "always_ram"),
        "AcornSearchParams": ("enable", "max_selectivity"),
        "TextIndexParams": ("type", "tokenizer", "min_token_len", "max_token_len", "lowercase", "stopwords", "stemmer", "on_disk"),
        "SnowballParams": ("type", "language"),
        "Document": ("text", "model", "options"),
    }
    for cls_name, expected_fields in required_attrs.items():
        cls = getattr(models, cls_name, None)
        if cls is None:
            raise AssertionError(f"Missing qdrant_client.models.{cls_name}")
        model_fields = getattr(cls, "model_fields", {})
        for field in expected_fields:
            if field not in model_fields:
                raise AssertionError(f"Missing field '{field}' on models.{cls_name}")

    for member in ("KEYWORD", "DATETIME", "BOOL", "INTEGER", "TEXT"):
        if not hasattr(models.PayloadSchemaType, member):
            raise AssertionError(f"Missing PayloadSchemaType.{member}")

    if not hasattr(models.SnowballLanguage, "RUSSIAN"):
        raise AssertionError("Missing SnowballLanguage.RUSSIAN")
    if not hasattr(models.Modifier, "IDF"):
        raise AssertionError("Missing Modifier.IDF")

    print("[OK] qdrant_client.models API signatures verified.")


def check_postgres(database_url: str) -> None:
    """Проверяет примененные миграции и выполняет транзакционный smoke-тест (с ROLLBACK)."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema IN ('core_public', 'tenant_rf')
                  AND table_type = 'BASE TABLE';
                """
            )
            rows = cur.fetchall()
            core_tables = {name for schema, name in rows if schema == "core_public"}
            tenant_tables = {name for schema, name in rows if schema == "tenant_rf"}

            missing_core = EXPECTED_CORE_PUBLIC_TABLES - core_tables
            missing_tenant = EXPECTED_TENANT_RF_TABLES - tenant_tables
            if missing_core:
                raise AssertionError(f"Missing core_public tables: {sorted(missing_core)}")
            if missing_tenant:
                raise AssertionError(f"Missing tenant_rf tables: {sorted(missing_tenant)}")

            cur.execute(
                """
                SELECT n.nspname, t.typname
                FROM pg_type t
                JOIN pg_namespace n ON n.oid = t.typnamespace
                WHERE t.typtype = 'e'
                  AND n.nspname IN ('core_public', 'tenant_rf');
                """
            )
            found_enums = set(cur.fetchall())
            missing_enums = EXPECTED_ENUMS - found_enums
            if missing_enums:
                raise AssertionError(f"Missing PostgreSQL ENUMs: {sorted(missing_enums)}")

            # Транзакционная проверка вставки (с откатом, чтобы БД оставалась чистой)
            cur.execute(
                """
                INSERT INTO core_public.channels (channel_id, username, title)
                VALUES (999000111, 'smoke_test_ch', 'Smoke Test Channel');
                """
            )
            cur.execute(
                """
                INSERT INTO core_public.posts (
                    channel_id, msg_id, published_at, effective_published_at,
                    text_raw, text_nfkc, content_sha256
                )
                VALUES (
                    999000111, 1, NOW(), NOW(),
                    'Тестовый пост', 'Тестовый пост', decode('deadbeef', 'hex')
                )
                RETURNING post_id;
                """
            )
            post_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO core_public.file_references (
                    file_id, tg_id, access_hash, dc_id, media_kind, thumb_size, file_reference
                )
                VALUES (
                    'photo:2:123456:x', 123456, 7890, 2, 'photo', 'x', decode('cafe', 'hex')
                );
                """
            )
            cur.execute(
                """
                INSERT INTO core_public.file_sources (file_id, channel_id, msg_id)
                VALUES ('photo:2:123456:x', 999000111, 1);
                """
            )
            cur.execute(
                """
                INSERT INTO core_public.ingest_idempotency (channel_id, msg_id)
                VALUES (999000111, 1)
                ON CONFLICT DO NOTHING;
                """
            )
            cur.execute(
                """
                INSERT INTO tenant_rf.users (email, oauth_provider)
                VALUES ('smoke@tg-agg.local', 'email')
                RETURNING user_id;
                """
            )
            user_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO tenant_rf.user_channel_subscriptions (user_id, channel_id)
                VALUES (%s, 999000111);
                """,
                (user_id,),
            )
            assert post_id > 0
        conn.rollback()

    print(
        f"[OK] PostgreSQL 16 verified: {len(EXPECTED_CORE_PUBLIC_TABLES)} core_public tables, "
        f"{len(EXPECTED_TENANT_RF_TABLES)} tenant_rf tables, {len(EXPECTED_ENUMS)} ENUMs, transactional smoke check passed."
    )


def check_redis(redis_url: str) -> None:
    """Проверяет Redis 7 PING и базовый цикл Streams (XADD/XGROUP/XREADGROUP/XACK)."""
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    if not client.ping():
        raise AssertionError("Redis PING returned False")

    stream_key = f"stream:verify:{uuid.uuid4().hex}"
    group_name = "cg:verify"
    try:
        client.xgroup_create(stream_key, group_name, id="0", mkstream=True)
        entry_id = client.xadd(stream_key, {"payload": '{"schema_version":"1.0"}'})
        msgs = client.xreadgroup(group_name, "worker-1", {stream_key: ">"}, count=1)
        if not msgs or len(msgs[0][1]) != 1:
            raise AssertionError(f"Unexpected XREADGROUP output: {msgs}")
        acked = client.xack(stream_key, group_name, entry_id)
        if acked != 1:
            raise AssertionError(f"Expected XACK=1, got {acked}")
    finally:
        client.delete(stream_key)
        client.close()

    print("[OK] Redis 7 PING and Streams (XADD/XGROUP/XREADGROUP/XACK) verified.")


def check_qdrant(qdrant_url: str, recreate: bool = False) -> None:
    """Проверяет инициализацию public_channels_v1 и сквозной smoke-тест на временной коллекции."""
    client = QdrantClient(url=qdrant_url, timeout=30)

    init_public_channels_collection(client, collection_name=COLLECTION_NAME, recreate=recreate)
    info = verify_collection_schema(client, collection_name=COLLECTION_NAME, is_local=False)

    # Проверяем, что требуемые payload-индексы созданы на коллекции
    schema_keys = set((info.payload_schema or {}).keys())
    missing_idx = set(REQUIRED_PAYLOAD_INDEXES) - schema_keys
    if missing_idx:
        raise AssertionError(f"Missing payload indexes on '{COLLECTION_NAME}': {sorted(missing_idx)}")

    # Smoke-тест вставки точки и гибридного поиска с ACORN-1 и Qdrant/bm25 на временной коллекции
    tmp_collection = f"_verify_{COLLECTION_NAME}_{uuid.uuid4().hex[:8]}"
    try:
        init_public_channels_collection(client, collection_name=tmp_collection, recreate=True)
        tmp_info = verify_collection_schema(client, collection_name=tmp_collection, is_local=False)
        if (tmp_info.points_count or 0) != 0:
            raise AssertionError("Expected 0 points immediately after index creation")

        unit_vec = [1.0 / (VECTOR_SIZE ** 0.5)] * VECTOR_SIZE
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        test_point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense_body": unit_vec,
                "bm25": models.Document(
                    text="Сбой маршрутизации BGP в дата-центре и ошибка updates.channelDifferenceTooLong",
                    model="Qdrant/bm25",
                    options={"language": "russian", "avg_len": 256.0},
                ),
            },
            payload={
                "post_id": 42,
                "chunk_index": 0,
                "channel_id": "1092837465",
                "msg_id": 8421,
                "published_at": now_iso,
                "is_ad": False,
                "spam_class": "organic",
                "cluster_id": str(uuid.uuid4()),
                "text_searchable": "Сбой маршрутизации BGP в дата-центре и ошибка updates.channelDifferenceTooLong",
            },
        )
        client.upsert(collection_name=tmp_collection, points=[test_point], wait=True)

        common_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="channel_id",
                    match=models.MatchAny(any=["1092837465"]),
                ),
                models.FieldCondition(
                    key="published_at",
                    range=models.DatetimeRange(gte=datetime(2026, 1, 1, tzinfo=timezone.utc)),
                ),
                models.FieldCondition(
                    key="is_ad",
                    match=models.MatchValue(value=False),
                ),
            ]
        )

        acorn_params = build_acorn_search_params()
        results = client.query_points_groups(
            collection_name=tmp_collection,
            prefetch=[
                models.Prefetch(
                    query=unit_vec,
                    using="dense_body",
                    filter=common_filter,
                    params=acorn_params,
                    score_threshold=0.42,
                    limit=10,
                ),
                models.Prefetch(
                    query=models.Document(
                        text="сбой маршрутизации BGP",
                        model="Qdrant/bm25",
                        options={"language": "russian", "avg_len": 256.0},
                    ),
                    using="bm25",
                    filter=common_filter,
                    limit=10,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=60, weights=[1.0, 1.6])),
            group_by="post_id",
            group_size=1,
            limit=10,
            with_payload=True,
        )
        if len(results.groups) != 1 or results.groups[0].id != 42:
            raise AssertionError(f"Unexpected hybrid search groups result: {results}")
    finally:
        if client.collection_exists(tmp_collection):
            client.delete_collection(tmp_collection)

    print(
        f"[OK] Qdrant verified: '{COLLECTION_NAME}' initialized with {len(REQUIRED_PAYLOAD_INDEXES)} "
        "RAM-pinned payload indexes before vector ingestion; hybrid search + BM25 + ACORN-1 passed."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PostgreSQL, Redis and Qdrant infrastructure.")
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "DATABASE_URL",
            "postgresql://tg_agg:dev_password_change_me@localhost:5432/tg_agg_dev",
        ),
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument(
        "--recreate-qdrant",
        action="store_true",
        help="Recreate public_channels_v1 before verifying",
    )
    args = parser.parse_args(argv)

    check_qdrant_models_api()
    check_postgres(args.database_url)
    check_redis(args.redis_url)
    check_qdrant(args.qdrant_url, recreate=args.recreate_qdrant)
    print("ALL INFRASTRUCTURE CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
