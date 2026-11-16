"""Инициализация коллекции Qdrant `public_channels_v1` (Стадия 1: Dev).

Создает коллекцию по схеме Multi-Representation (4 плотных вектора 512d с SQ8 +
1 разреженный вектор `bm25` с Modifier.IDF) и создает все payload-индексы
с фиксацией в RAM (`on_disk=False`) строго ДО загрузки любых векторов для
корректного построения ребер Filterable HNSW.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Final

from qdrant_client import QdrantClient, models

COLLECTION_NAME: Final[str] = "public_channels_v1"
VECTOR_SIZE: Final[int] = 512
DENSE_VECTOR_NAMES: Final[tuple[str, ...]] = (
    "dense_body",
    "dense_code",
    "dense_ocr",
    "dense_comments",
)
REQUIRED_PAYLOAD_INDEXES: Final[tuple[str, ...]] = (
    "channel_id",
    "published_at",
    "is_ad",
    "post_id",
    "cluster_id",
    "text_searchable",
)


def build_dense_vector_params() -> models.VectorParams:
    """Возвращает конфигурацию 512d плотного вектора (SQ8 в RAM + FP32 на диске)."""
    return models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE,
        on_disk=True,  # Исходные FP32 хранятся на NVMe для финального rescore=True
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=20000,  # КБ: переключение узких срезов (<0.5%) в SIMD full-scan
            on_disk=False,  # Граф HNSW закреплен в RAM
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,  # SQ8 в RAM
                quantile=0.99,
                always_ram=True,
            )
        ),
    )


def build_acorn_search_params(
    hnsw_ef: int = 128,
    max_selectivity: float = 0.4,
    oversampling: float = 2.5,
) -> models.SearchParams:
    """Возвращает параметры поиска с ACORN-1 (2-hop обход) и SQ8 rescore."""
    return models.SearchParams(
        hnsw_ef=hnsw_ef,
        exact=False,
        quantization=models.QuantizationSearchParams(
            ignore=False,
            rescore=True,
            oversampling=oversampling,
        ),
        acorn=models.AcornSearchParams(
            enable=True,
            max_selectivity=max_selectivity,
        ),
    )


def create_required_payload_indexes(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Создает все payload-индексы с фиксацией в RAM строго до загрузки векторов."""
    # 1. Индекс фильтрации по списку подписок channel_id IN (30..500)
    client.create_payload_index(
        collection_name=collection_name,
        field_name="channel_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            on_disk=False,  # Битовые маски зафиксированы в RAM
        ),
        wait=True,
    )

    # 2. Временной индекс окна поиска и скользящего окна дедупликации (72 ч)
    client.create_payload_index(
        collection_name=collection_name,
        field_name="published_at",
        field_schema=models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME,
            is_principal=True,  # Оптимизация диапазонных сканов по времени
            on_disk=False,
        ),
        wait=True,
    )

    # 3. Булевый фильтр отсечения нативной рекламы (is_ad = false)
    client.create_payload_index(
        collection_name=collection_name,
        field_name="is_ad",
        field_schema=models.BoolIndexParams(
            type=models.BoolIndexType.BOOL,
            on_disk=False,
        ),
        wait=True,
    )

    # 4. Целочисленный индекс для group_by="post_id"
    client.create_payload_index(
        collection_name=collection_name,
        field_name="post_id",
        field_schema=models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            lookup=True,
            range=False,
            on_disk=False,
        ),
        wait=True,
    )

    # 5. Ключевой индекс кластера событий
    client.create_payload_index(
        collection_name=collection_name,
        field_name="cluster_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            on_disk=False,
        ),
        wait=True,
    )

    # 6. Полнотекстовый индекс payload со стеммером Russian Snowball
    client.create_payload_index(
        collection_name=collection_name,
        field_name="text_searchable",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=40,
            lowercase=True,
            stemmer=models.SnowballParams(
                type=models.Snowball.SNOWBALL,
                language=models.SnowballLanguage.RUSSIAN,
            ),
            stopwords=models.Language.RUSSIAN,
            on_disk=False,
        ),
        wait=True,
    )


def init_public_channels_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
) -> bool:
    """Создает коллекцию `public_channels_v1` и payload-индексы до вставки векторов.

    Возвращает True, если коллекция была создана, и False, если она уже существовала.
    """
    exists = client.collection_exists(collection_name)
    if exists:
        if not recreate:
            # Убеждаемся, что payload-индексы присутствуют, если коллекция еще пуста
            info = client.get_collection(collection_name)
            if (info.points_count or 0) == 0:
                create_required_payload_indexes(client, collection_name)
            return False
        client.delete_collection(collection_name=collection_name)

    dense_params = build_dense_vector_params()

    client.create_collection(
        collection_name=collection_name,
        vectors_config={name: dense_params for name in DENSE_VECTOR_NAMES},
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=False),
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=20000,
            on_disk=False,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )

    # Критический инвариант: payload-индексы создаются строго до загрузки любых векторов
    create_required_payload_indexes(client, collection_name)
    return True


def verify_collection_schema(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    is_local: bool = False,
) -> models.CollectionInfo:
    """Проверяет соответствие конфигурации коллекции и индексов спецификации."""
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict):
        raise AssertionError(f"Expected named vectors dict, got {type(vectors)}")

    for name in DENSE_VECTOR_NAMES:
        if name not in vectors:
            raise AssertionError(f"Missing dense vector '{name}' in {collection_name}")
        vec_cfg = vectors[name]
        if vec_cfg.size != VECTOR_SIZE:
            raise AssertionError(f"Vector '{name}' size={vec_cfg.size}, expected {VECTOR_SIZE}")
        if vec_cfg.distance != models.Distance.COSINE:
            raise AssertionError(f"Vector '{name}' distance={vec_cfg.distance}, expected Cosine")
        if vec_cfg.on_disk is not True:
            raise AssertionError(f"Vector '{name}' on_disk={vec_cfg.on_disk}, expected True")

    sparse = info.config.params.sparse_vectors or {}
    if "bm25" not in sparse:
        raise AssertionError("Missing sparse vector 'bm25'")
    if sparse["bm25"].modifier != models.Modifier.IDF:
        raise AssertionError(f"BM25 modifier={sparse['bm25'].modifier}, expected IDF")

    if not is_local:
        payload_schema = info.payload_schema or {}
        for field_name in REQUIRED_PAYLOAD_INDEXES:
            if field_name not in payload_schema:
                raise AssertionError(f"Missing payload index '{field_name}' in {collection_name}")

    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Initialize Qdrant collection public_channels_v1 and payload indexes."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant HTTP URL (default: env QDRANT_URL or http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", COLLECTION_NAME),
        help=f"Collection name (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Run against local QdrantClient(':memory:') instead of remote server",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection if it already exists",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify collection vectors and payload indexes after initialization",
    )
    args = parser.parse_args(argv)

    client = QdrantClient(":memory:") if args.in_memory else QdrantClient(url=args.url)
    created = init_public_channels_collection(
        client=client,
        collection_name=args.collection,
        recreate=args.recreate,
    )
    status = "created" if created else "already exists"
    print(f"Collection '{args.collection}' {status}.")

    if args.verify:
        info = verify_collection_schema(
            client=client,
            collection_name=args.collection,
            is_local=args.in_memory,
        )
        print(
            f"Verified '{args.collection}': "
            f"vectors={sorted(info.config.params.vectors.keys())}, "
            f"sparse={sorted((info.config.params.sparse_vectors or {}).keys())}, "
            f"payload_indexes={sorted((info.payload_schema or {}).keys())}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
