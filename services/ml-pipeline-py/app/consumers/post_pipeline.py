"""Полный end-to-end конвейер обработки постов: предобработка -> спам -> кластеризация -> Qdrant."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field

from app.clustering.single_pass import ClusterAssignmentResult, SinglePassClusterer
from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import Qwen3EmbeddingClient
from app.embeddings.qwen3_reranker import Qwen3RerankerClient
from app.preprocessing.entities_ner import ExtractedEntities, extract_entities
from app.preprocessing.normalizer import PreprocessedPost, preprocess_post
from app.retrieval.qdrant_repo import PostChunkRecord, QdrantPublicChannelsRepo
from app.spam_filter.cascade import SpamFilterCascade, SpamVerdict


class TelegramEntity(BaseModel):
    """Telegram MessageEntity в JSON-представлении Go ingestion-сервиса."""

    type: str
    offset: int
    length: int
    url: str | None = None
    language: str | None = None


class IngestPostInput(BaseModel):
    """Входная нагрузка события ``IngestPostEventV1`` для ML-конвейера."""

    event_id: str = ""
    post_id: int
    channel_id: int | str
    channel_username: str = ""
    channel_title: str = ""
    published_at: datetime
    edit_date: datetime | None = None
    text_raw: str = ""
    entities: list[TelegramEntity] = Field(default_factory=list)
    ocr_text: str | None = None
    media_phashes: list[int | str] = Field(default_factory=list)
    has_informative_media: bool = False
    explicit_cite_score: float = 0.0
    channel_authority: float = 0.0
    originality_score: float = 1.0
    has_primary_evidence: bool = False
    lang: str = "ru"


def parse_redis_stream_payload(fields: Mapping[str, Any]) -> IngestPostInput:
    """Декодирует сообщение из Redis Stream `stream:ingest:posts` (Go producer shape)."""
    raw_payload = fields.get("payload", fields)
    if isinstance(raw_payload, (bytes, bytearray)):
        raw_payload = raw_payload.decode("utf-8")
    if isinstance(raw_payload, str):
        data = json.loads(raw_payload)
    else:
        data = dict(raw_payload)

    # Маппинг полей схемы Go (`tg_post_id`/`tg_channel_id`/`text`) на внутренний Post ID
    post_id = data.get("post_id") or data.get("tg_post_id") or int(data["event_id"].split(":")[-1])
    channel_id = data.get("channel_id") or data.get("tg_channel_id") or data.get("channel_username", "")
    published_raw = data.get("published_at") or data.get("created_at") or datetime.now(timezone.utc)
    if isinstance(published_raw, (int, float)):
        published_at = datetime.fromtimestamp(published_raw, tz=timezone.utc)
    else:
        published_at = published_raw

    return IngestPostInput(
        event_id=str(data.get("event_id", fields.get("event_id", ""))),
        post_id=int(post_id),
        channel_id=channel_id,
        channel_username=str(data.get("channel_username", "")),
        channel_title=str(data.get("channel_title", "")),
        published_at=published_at,
        edit_date=data.get("edit_date"),
        text_raw=str(data.get("text_raw", data.get("text", ""))),
        entities=[TelegramEntity(**e) for e in data.get("entities", [])],
        ocr_text=data.get("ocr_text"),
        media_phashes=list(data.get("media_phashes", [])),
        has_informative_media=bool(data.get("has_informative_media", bool(data.get("media")))),
        lang=str(data.get("lang", "ru")),
    )


@dataclass(slots=True)
class PipelineProcessResult:
    """Агрегированный результат обработки поста всеми ступенями ML-конвейера."""

    post_id: int
    preprocessed: PreprocessedPost
    extracted_entities: ExtractedEntities
    spam_verdict: SpamVerdict
    cluster_result: ClusterAssignmentResult | None = None
    qdrant_chunks: list[PostChunkRecord] = field(default_factory=list)


class PostPipelineService:
    """Координатор обработки, фильтрации, кластеризации и индексации постов."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Qwen3EmbeddingClient | None = None,
        reranker: Qwen3RerankerClient | None = None,
        spam_cascade: SpamFilterCascade | None = None,
        clusterer: SinglePassClusterer | None = None,
        qdrant_repo: QdrantPublicChannelsRepo | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Qwen3EmbeddingClient(self.settings)
        self.reranker = reranker or Qwen3RerankerClient(self.settings)
        self.spam_cascade = spam_cascade or SpamFilterCascade(self.settings)
        self.qdrant_repo = qdrant_repo

        def _qdrant_fetcher(vec: Any, now: datetime, k: int) -> list[str]:
            if self.qdrant_repo is None:
                return []
            min_ts = now - _window_delta(self.settings.sliding_window_hours)
            return self.qdrant_repo.search_cluster_candidates(vec, min_ts, limit=k)

        self.clusterer = clusterer or SinglePassClusterer(
            self.settings,
            embedder=self.embedder,
            reranker=self.reranker,
            qdrant_candidate_fetcher=_qdrant_fetcher if self.qdrant_repo is not None else None,
        )

    def process_post(self, item: IngestPostInput) -> PipelineProcessResult:
        entities_dicts = [e.model_dump() for e in item.entities]

        # 1. Раздел 7.4.1 (шаг 1): NFKC, предложения [P{post_id}_S{sent_id}], карта смещений UTF-16
        preprocessed = preprocess_post(
            post_id=item.post_id,
            text_raw=item.text_raw,
            entities=entities_dicts,
            ocr_text=item.ocr_text,
        )
        extracted_entities = extract_entities(preprocessed.text_nfkc)

        # 2. Раздел 7.2.2: Каскад фильтрации спама L0 -> L1 -> L2
        verdict = self.spam_cascade.evaluate(
            preprocessed.text_nfkc,
            entities=entities_dicts,
            source_channel=item.channel_username or str(item.channel_id),
            published_at=item.published_at,
            media_phashes=item.media_phashes,
            has_informative_media=item.has_informative_media,
        )

        if not verdict.should_persist:
            return PipelineProcessResult(
                post_id=item.post_id,
                preprocessed=preprocessed,
                extracted_entities=extracted_entities,
                spam_verdict=verdict,
            )

        # 3. Раздел 7.2.3: Кластеризация органических постов (рекламные сохраняются с is_ad=True без слияния в события)
        cluster_res: ClusterAssignmentResult | None = None
        if not verdict.is_ad:
            doc_vec = self.embedder.encode_document(
                preprocessed.text_nfkc,
                reject_instruct_prefix=False,
            )
            cluster_res = self.clusterer.ingest_post(
                post_id=item.post_id,
                channel_id=item.channel_id,
                text=preprocessed.text_nfkc,
                published_at=item.published_at,
                embedding=doc_vec,
                entities=set(extracted_entities.all_canonical),
                edit_date=item.edit_date,
                explicit_cite_score=item.explicit_cite_score,
                channel_authority=item.channel_authority,
                originality_score=item.originality_score,
                has_primary_evidence=item.has_primary_evidence,
            )

        # 4. Раздел 7.3.1: Запись в Qdrant (условный split > 2000 символов + именованные векторы)
        qdrant_chunks: list[PostChunkRecord] = []
        if self.qdrant_repo is not None:
            qdrant_chunks = self.qdrant_repo.upsert_post_chunks(
                post_id=item.post_id,
                channel_id=item.channel_id,
                channel_username=item.channel_username,
                channel_title=item.channel_title,
                text=preprocessed.text_nfkc,
                published_at=item.published_at,
                cluster_id=cluster_res.cluster_id if cluster_res else None,
                origin_post_id=cluster_res.origin_post_id if cluster_res else item.post_id,
                origin_channel_id=cluster_res.origin_channel_id if cluster_res else item.channel_id,
                is_ad=verdict.is_ad,
                spam_class=verdict.spam_class,
                code_text=preprocessed.code_blocks_text or "",
                ocr_text=preprocessed.ocr_text or "",
                exact_entities=extracted_entities.to_json_dict(),
                lang=item.lang,
            )

        return PipelineProcessResult(
            post_id=item.post_id,
            preprocessed=preprocessed,
            extracted_entities=extracted_entities,
            spam_verdict=verdict,
            cluster_result=cluster_res,
            qdrant_chunks=qdrant_chunks,
        )

    def process_post_edit(
        self,
        *,
        post_id: int,
        new_text: str,
        edit_date: datetime,
    ) -> ClusterAssignmentResult | None:
        preprocessed = preprocess_post(post_id=post_id, text_raw=new_text)
        res = self.clusterer.handle_post_edited(
            post_id=post_id,
            new_text=preprocessed.text_nfkc,
            edit_date=edit_date,
        )
        return res

    def process_post_deletion(self, post_id: int) -> None:
        if self.qdrant_repo is not None:
            self.qdrant_repo.delete_post(post_id)


def _window_delta(hours: int):
    from datetime import timedelta

    return timedelta(hours=hours)
