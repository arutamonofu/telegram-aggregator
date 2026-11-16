"""Клиенты эмбеддингов и реранкинга семейства Qwen3."""

from app.embeddings.qwen3_embedder import (
    DEFAULT_QUERY_INSTRUCTION,
    Qwen3EmbeddingClient,
    TransformersQwen3EmbeddingBackbone,
    format_query_with_instruction,
    last_token_pool,
    mrl_truncate_and_normalize,
)
from app.embeddings.qwen3_reranker import (
    DEFAULT_DEDUP_RERANK_INSTRUCTION,
    Qwen3RerankerClient,
    TransformersQwen3RerankerBackbone,
)

__all__ = [
    "DEFAULT_DEDUP_RERANK_INSTRUCTION",
    "DEFAULT_QUERY_INSTRUCTION",
    "Qwen3EmbeddingClient",
    "Qwen3RerankerClient",
    "TransformersQwen3EmbeddingBackbone",
    "TransformersQwen3RerankerBackbone",
    "format_query_with_instruction",
    "last_token_pool",
    "mrl_truncate_and_normalize",
]
