"""Репозиторий векторного индекса Qdrant (`public_channels_v1`, раздел 7.3.1) и гибридный поиск (7.3.2)."""

from app.retrieval.hybrid_search import (
    DEFAULT_SEARCH_RERANK_INSTRUCTION,
    HybridSearchResult,
    HybridSearchService,
    RetrievedPostCandidate,
)
from app.retrieval.qdrant_repo import (
    PostChunkRecord,
    QdrantPublicChannelsRepo,
    bm25_text_to_sparse_vector,
    deterministic_point_id,
    split_longread_with_header,
)

__all__ = [
    "DEFAULT_SEARCH_RERANK_INSTRUCTION",
    "HybridSearchResult",
    "HybridSearchService",
    "PostChunkRecord",
    "QdrantPublicChannelsRepo",
    "RetrievedPostCandidate",
    "bm25_text_to_sparse_vector",
    "deterministic_point_id",
    "split_longread_with_header",
]
