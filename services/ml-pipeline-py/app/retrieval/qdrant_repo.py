"""Запись и поиск в Qdrant (`public_channels_v1`): дробление лонгридов и именованные векторы."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import Qwen3EmbeddingClient
from app.preprocessing.entities_ner import has_exact_lexical_triggers
from app.preprocessing.normalizer import normalize_text_nfkc

try:
    from deploy.qdrant.init_public_channels_v1 import (
        COLLECTION_NAME as DEFAULT_COLLECTION_NAME,
        init_public_channels_collection,
    )
except ImportError:  # pragma: no cover - fallback при запуске вне корня репо
    DEFAULT_COLLECTION_NAME = "public_channels_v1"

    def init_public_channels_collection(client: QdrantClient, *, recreate: bool = False) -> None:
        if recreate or not client.collection_exists(DEFAULT_COLLECTION_NAME):
            if recreate and client.collection_exists(DEFAULT_COLLECTION_NAME):
                client.delete_collection(DEFAULT_COLLECTION_NAME)
            client.create_collection(
                collection_name=DEFAULT_COLLECTION_NAME,
                vectors_config={
                    name: models.VectorParams(size=512, distance=models.Distance.COSINE)
                    for name in ("dense_body", "dense_code", "dense_ocr", "dense_comments")
                },
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )


_POINT_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_BM25_TOKEN_RE = re.compile(r"[a-zа-яё0-9%_-]{2,}", re.IGNORECASE)


def deterministic_point_id(post_id: int, chunk_id: int) -> str:
    """Генерирует стабильный UUIDv5 для точки Qdrant из `(post_id, chunk_id)`."""
    return str(uuid.uuid5(_POINT_UUID_NAMESPACE, f"tg_post:{post_id}:chunk:{chunk_id}"))


def bm25_text_to_sparse_vector(text: str) -> models.SparseVector:
    """Детерминированный sparse-вектор TF-весов для локального Qdrant `:memory:` (без fastembed)."""
    tokens = _BM25_TOKEN_RE.findall(normalize_text_nfkc(text).lower())
    if not tokens:
        return models.SparseVector(indices=[], values=[])

    counts = Counter(tokens)
    bucket_weights: dict[int, float] = {}
    for tok, tf in counts.items():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
        idx = int.from_bytes(digest, "little") & 0x7FFFFFFF
        # Sublinear TF weighting
        weight = 1.0 + math.log(tf)
        bucket_weights[idx] = bucket_weights.get(idx, 0.0) + weight

    sorted_items = sorted(bucket_weights.items(), key=lambda kv: kv[0])
    return models.SparseVector(
        indices=[k for k, _ in sorted_items],
        values=[float(v) for _, v in sorted_items],
    )


def _derive_post_title(text: str, explicit_title: str | None = None) -> str:
    if explicit_title and explicit_title.strip():
        return explicit_title.strip()[:80]
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())
    first_sent = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0].strip()
    if len(first_sent) <= 72:
        return first_sent
    return first_sent[:69].rstrip() + "..."


def _split_text_into_n_parts(text: str, n_parts: int) -> list[str]:
    """Делит текст на `n_parts` сбалансированных фрагментов по границам абзацев/предложений."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= max(len(text) // n_parts + 100, 400):
            units.append(para)
        else:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            units.extend(sents if len(sents) > 1 else [para])

    if len(units) >= n_parts:
        target_len = len(text) / n_parts
        chunks: list[list[str]] = [[] for _ in range(n_parts)]
        cur_idx = 0
        cur_len = 0
        for unit_i, unit in enumerate(units):
            remaining_units = len(units) - unit_i
            remaining_buckets = n_parts - cur_idx - 1
            if (
                cur_idx < n_parts - 1
                and chunks[cur_idx]
                and (cur_len >= target_len or remaining_units <= remaining_buckets)
            ):
                if remaining_units > remaining_buckets and cur_len < target_len * 0.75:
                    pass
                else:
                    cur_idx += 1
                    cur_len = 0
            chunks[cur_idx].append(unit)
            cur_len += len(unit) + 1
        joined = ["\n".join(bucket).strip() for bucket in chunks if bucket]
        if len(joined) == n_parts:
            return joined

    # Фолбэк: режем по ближайшим пробелам
    total = len(text)
    parts: list[str] = []
    start = 0
    for i in range(1, n_parts):
        approx = int(total * i / n_parts)
        cut = text.rfind(" ", start + 50, min(approx + 80, total - 20))
        if cut <= start:
            cut = approx
        parts.append(text[start:cut].strip())
        start = cut
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def split_longread_with_header(
    text: str,
    *,
    channel_label: str,
    published_at: datetime,
    post_title: str | None = None,
    longread_threshold: int = 2000,
    max_chunks: int = 3,
) -> list[str]:
    """Условное дробление лонгридов > 2000 символов (раздел 7.3.1 SPEC.md).

    - Если `len(text) <= longread_threshold`, возвращается 1 чанк без модификации.
    - Если `len(text) > longread_threshold`, пост делится на 2–3 чанка и в начало
      каждого чанка добавляется контекстная шапка ``[Канал | Заголовок | Дата]``.
    """
    clean = normalize_text_nfkc(text).strip()
    if len(clean) <= longread_threshold:
        return [clean]

    title = _derive_post_title(clean, explicit_title=post_title)
    pub_utc = (
        published_at.replace(tzinfo=timezone.utc)
        if published_at.tzinfo is None
        else published_at.astimezone(timezone.utc)
    )
    date_str = pub_utc.strftime("%Y-%m-%d")
    header = f"[{channel_label or 'channel'} | {title} | {date_str}]"

    target_chunks = 2 if len(clean) <= int(longread_threshold * 1.75) else max(2, max_chunks)
    target_chunks = min(max(2, target_chunks), max(2, max_chunks))
    body_parts = _split_text_into_n_parts(clean, target_chunks)
    return [f"{header}\n{part}" for part in body_parts]


@dataclass(slots=True)
class PostChunkRecord:
    """Метаданные и векторы одного чанка поста, записанного в Qdrant."""

    point_id: str
    post_id: int
    chunk_id: int
    total_chunks: int
    text: str
    vector_names: list[str]
    payload: dict[str, Any] = field(default_factory=dict)


class QdrantPublicChannelsRepo:
    """Обертка для записи/удаления/поиска точек коллекции `public_channels_v1`."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        *,
        settings: Settings | None = None,
        embedder: Qwen3EmbeddingClient | None = None,
        collection_name: str | None = None,
        use_remote_bm25_document: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name or self.settings.qdrant_collection or DEFAULT_COLLECTION_NAME
        self.client = client or QdrantClient(url=self.settings.qdrant_url)
        self.embedder = embedder or Qwen3EmbeddingClient(self.settings)
        if use_remote_bm25_document is None:
            # Локальный :memory: клиент не имеет встроенного fastembed-инференса для Qdrant/bm25
            self.use_remote_bm25_document = "Local" not in type(self.client._client).__name__
        else:
            self.use_remote_bm25_document = use_remote_bm25_document

    def ensure_collection(self, *, recreate: bool = False) -> None:
        init_public_channels_collection(self.client, recreate=recreate)

    def _make_bm25_vector(self, text: str) -> models.Document | models.SparseVector:
        if self.use_remote_bm25_document:
            return models.Document(
                text=text,
                model="Qdrant/bm25",
                options={"language": "russian", "avg_len": 256.0},
            )
        return bm25_text_to_sparse_vector(text)

    def upsert_post_chunks(
        self,
        *,
        post_id: int,
        channel_id: int | str,
        channel_username: str,
        channel_title: str,
        text: str,
        published_at: datetime,
        cluster_id: str | None = None,
        origin_post_id: int | None = None,
        origin_channel_id: int | str | None = None,
        is_ad: bool = False,
        spam_class: str = "organic",
        code_text: str = "",
        ocr_text: str = "",
        comments_digest_text: str = "",
        exact_entities: dict[str, list[str]] | None = None,
        lang: str = "ru",
        post_title: str | None = None,
    ) -> list[PostChunkRecord]:
        """Разбивает длинный пост, кодирует непустые пространства и сохраняет в Qdrant."""
        pub_utc = (
            published_at.replace(tzinfo=timezone.utc)
            if published_at.tzinfo is None
            else published_at.astimezone(timezone.utc)
        )
        channel_label = channel_title or channel_username or str(channel_id)
        chunks = split_longread_with_header(
            text,
            channel_label=channel_label,
            published_at=pub_utc,
            post_title=post_title,
            longread_threshold=self.settings.longread_char_threshold,
            max_chunks=self.settings.longread_max_chunks,
        )

        # Кодируем все текстовые пространства строго как документы (без Instruct-префикса)
        body_vecs = self.embedder.encode_documents(chunks, reject_instruct_prefix=False)
        code_vec = (
            self.embedder.encode_document(code_text, reject_instruct_prefix=False).tolist()
            if code_text.strip()
            else None
        )
        ocr_vec = (
            self.embedder.encode_document(ocr_text, reject_instruct_prefix=False).tolist()
            if ocr_text.strip()
            else None
        )
        comments_vec = (
            self.embedder.encode_document(comments_digest_text, reject_instruct_prefix=False).tolist()
            if comments_digest_text.strip()
            else None
        )

        # Перед записью новых чанков удаляем прежние чанки этого же post_id (если пост редактировался)
        self.delete_post(post_id)

        points: list[models.PointStruct] = []
        records: list[PostChunkRecord] = []
        total_chunks = len(chunks)

        for chunk_id, chunk_text in enumerate(chunks):
            bm25_input = chunk_text
            if chunk_id == 0:
                extra_bm25_parts = [p.strip() for p in (code_text, ocr_text) if p and p.strip()]
                if extra_bm25_parts:
                    bm25_input = "\n".join([chunk_text, *extra_bm25_parts])
            vec_dict: dict[str, Any] = {
                "dense_body": np.asarray(body_vecs[chunk_id], dtype=np.float32).tolist(),
                "bm25": self._make_bm25_vector(bm25_input),
            }
            # Неинформативные модальности в нулевых векторах не заполняются;
            # опциональные пространства прикрепляются к основному чанку (или каждому при наличии).
            if chunk_id == 0:
                if code_vec is not None:
                    vec_dict["dense_code"] = code_vec
                if ocr_vec is not None:
                    vec_dict["dense_ocr"] = ocr_vec
                if comments_vec is not None:
                    vec_dict["dense_comments"] = comments_vec

            point_id = deterministic_point_id(post_id, chunk_id)
            source_url = (
                f"https://t.me/{channel_username.lstrip('@')}/{post_id}"
                if channel_username
                else ""
            )
            payload: dict[str, Any] = {
                "post_id": post_id,
                "chunk_id": chunk_id,
                "total_chunks": total_chunks,
                "channel_id": str(channel_id),
                "channel_username": channel_username.lstrip("@"),
                "channel_title": channel_label,
                "cluster_id": cluster_id,
                "origin_post_id": origin_post_id if origin_post_id is not None else post_id,
                "origin_channel_id": str(origin_channel_id if origin_channel_id is not None else channel_id),
                "is_origin": origin_post_id is None or origin_post_id == post_id,
                "is_ad": is_ad,
                "spam_class": spam_class,
                "published_at": pub_utc.timestamp(),
                "published_at_iso": pub_utc.isoformat(),
                "lang": lang,
                "has_code": bool(code_vec is not None),
                "has_ocr": bool(ocr_vec is not None),
                "has_comments": bool(comments_vec is not None),
                "text": chunk_text,
                "full_text": normalize_text_nfkc(text).strip(),
                "code_text": normalize_text_nfkc(code_text).strip() if code_text.strip() else "",
                "ocr_text": normalize_text_nfkc(ocr_text).strip() if ocr_text.strip() else "",
                "comments_digest_text": (
                    normalize_text_nfkc(comments_digest_text).strip()
                    if comments_digest_text.strip()
                    else ""
                ),
                "exact_entities": exact_entities or {},
                "source_url": source_url,
            }
            points.append(models.PointStruct(id=point_id, vector=vec_dict, payload=payload))
            records.append(
                PostChunkRecord(
                    point_id=point_id,
                    post_id=post_id,
                    chunk_id=chunk_id,
                    total_chunks=total_chunks,
                    text=chunk_text,
                    vector_names=sorted(vec_dict.keys()),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return records

    def update_post_comments_vector(
        self,
        post_id: int,
        comments_vector: Sequence[float],
        *,
        comments_digest_text: str | None = None,
    ) -> None:
        """Обновляет вектор `dense_comments` у корневой точки (`chunk_id=0`) поста."""
        point_id = deterministic_point_id(post_id, 0)
        self.client.update_vectors(
            collection_name=self.collection_name,
            points=[
                models.PointVectors(
                    id=point_id,
                    vector={"dense_comments": [float(x) for x in comments_vector]},
                )
            ],
            wait=True,
        )
        payload_upd: dict[str, Any] = {"has_comments": True}
        if comments_digest_text is not None:
            payload_upd["comments_digest_text"] = normalize_text_nfkc(comments_digest_text).strip()
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload_upd,
            points=[point_id],
            wait=True,
        )

    def build_hybrid_query_args(
        self,
        *,
        query_text: str,
        query_dense_512d: Sequence[float] | np.ndarray,
        subscribed_channel_ids: Sequence[int | str],
        published_after: datetime,
        has_exact_entities: bool | None = None,
        group_limit: int | None = None,
        group_size: int = 1,
        cluster_id: str | None = None,
    ) -> dict[str, Any]:
        """Формирует параметры единого 5-веточного `query_points_groups` с Weighted RRF и ACORN-1."""
        pub_utc = (
            published_after.replace(tzinfo=timezone.utc)
            if published_after.tzinfo is None
            else published_after.astimezone(timezone.utc)
        )
        min_ts = pub_utc.timestamp()
        channel_keys = [str(cid) for cid in subscribed_channel_ids]

        must_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="channel_id",
                match=models.MatchAny(any=channel_keys),
            ),
            models.FieldCondition(
                key="published_at",
                range=models.Range(gte=min_ts),
            ),
            models.FieldCondition(
                key="is_ad",
                match=models.MatchValue(value=False),
            ),
        ]
        if cluster_id:
            must_conditions.append(
                models.FieldCondition(
                    key="cluster_id",
                    match=models.MatchValue(value=str(cluster_id)),
                )
            )
        common_filter = models.Filter(must=must_conditions)

        acorn_search_params = models.SearchParams(
            hnsw_ef=self.settings.acorn_hnsw_ef,
            exact=False,
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=self.settings.quantization_oversampling,
            ),
            acorn=models.AcornSearchParams(
                enable=True,
                max_selectivity=self.settings.acorn_max_selectivity,
            ),
        )

        exact_flag = (
            has_exact_lexical_triggers(query_text)
            if has_exact_entities is None
            else bool(has_exact_entities)
        )
        w_bm25 = (
            self.settings.rrf_w_bm25_exact
            if exact_flag
            else self.settings.rrf_w_bm25_base
        )
        q_vec_list = [float(x) for x in np.asarray(query_dense_512d, dtype=np.float32).tolist()]

        prefetches = [
            models.Prefetch(
                query=q_vec_list,
                using="dense_body",
                filter=common_filter,
                params=acorn_search_params,
                score_threshold=self.settings.threshold_dense_body,
                limit=self.settings.prefetch_limit_body,
            ),
            models.Prefetch(
                query=self._make_bm25_vector(query_text),
                using="bm25",
                filter=common_filter,
                limit=self.settings.prefetch_limit_bm25,
            ),
            models.Prefetch(
                query=q_vec_list,
                using="dense_code",
                filter=common_filter,
                params=acorn_search_params,
                score_threshold=self.settings.threshold_dense_code,
                limit=self.settings.prefetch_limit_code,
            ),
            models.Prefetch(
                query=q_vec_list,
                using="dense_ocr",
                filter=common_filter,
                params=acorn_search_params,
                score_threshold=self.settings.threshold_dense_ocr,
                limit=self.settings.prefetch_limit_ocr,
            ),
            models.Prefetch(
                query=q_vec_list,
                using="dense_comments",
                filter=common_filter,
                params=acorn_search_params,
                score_threshold=self.settings.threshold_dense_comments,
                limit=self.settings.prefetch_limit_comments,
            ),
        ]

        weights = [
            self.settings.rrf_w_body,
            w_bm25,
            self.settings.rrf_w_code,
            self.settings.rrf_w_ocr,
            self.settings.rrf_w_comments,
        ]

        return {
            "collection_name": self.collection_name,
            "prefetch": prefetches,
            "query": models.RrfQuery(
                rrf=models.Rrf(
                    k=self.settings.rrf_k,
                    weights=weights,
                )
            ),
            "query_filter": common_filter,
            "search_params": acorn_search_params,
            "group_by": "post_id",
            "group_size": group_size,
            "limit": group_limit or self.settings.hybrid_group_limit,
            "with_payload": True,
        }

    def query_hybrid_groups(
        self,
        *,
        query_text: str,
        query_dense_512d: Sequence[float] | np.ndarray,
        subscribed_channel_ids: Sequence[int | str],
        published_after: datetime,
        has_exact_entities: bool | None = None,
        group_limit: int | None = None,
        group_size: int = 1,
        cluster_id: str | None = None,
    ) -> models.GroupsResult:
        """Выполняет единый 5-веточный гибридный поиск с группировкой по `post_id`."""
        if not subscribed_channel_ids:
            return models.GroupsResult(groups=[])
        kwargs = self.build_hybrid_query_args(
            query_text=query_text,
            query_dense_512d=query_dense_512d,
            subscribed_channel_ids=subscribed_channel_ids,
            published_after=published_after,
            has_exact_entities=has_exact_entities,
            group_limit=group_limit,
            group_size=group_size,
            cluster_id=cluster_id,
        )
        return self.client.query_points_groups(**kwargs)

    def search_cluster_candidates(
        self,
        query_vector: np.ndarray,
        min_published_at: datetime,
        *,
        limit: int = 20,
    ) -> list[str]:
        """Ищет Top-K близких кластеров за последние 72 часа по `dense_body`."""
        min_ts = (
            min_published_at.replace(tzinfo=timezone.utc)
            if min_published_at.tzinfo is None
            else min_published_at.astimezone(timezone.utc)
        ).timestamp()
        q_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="is_ad",
                    match=models.MatchValue(value=False),
                ),
                models.FieldCondition(
                    key="published_at",
                    range=models.Range(gte=min_ts),
                ),
            ]
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(query_vector, dtype=np.float32).tolist(),
            using="dense_body",
            query_filter=q_filter,
            limit=limit,
            with_payload=True,
        )
        cluster_ids: list[str] = []
        for point in response.points:
            cid = (point.payload or {}).get("cluster_id")
            if cid and cid not in cluster_ids:
                cluster_ids.append(str(cid))
        return cluster_ids

    def delete_post(self, post_id: int) -> None:
        """Удаляет все чанки поста из Qdrant по `post_id`."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="post_id",
                            match=models.MatchValue(value=post_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
