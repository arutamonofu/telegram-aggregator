"""Конфигурация сервиса `services/ml-pipeline-py` (Стадия 1: Dev)."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Параметры ML-конвейера из спецификации 7.2, 7.3, 7.4.1 и 8.1."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # Инфраструктура
    database_url: str = "postgresql://tg_agg:dev_password_change_me@localhost:5432/tg_agg_dev"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_grpc_url: str = "localhost:6334"
    qdrant_collection: str = "public_channels_v1"
    ingestion_internal_url: str = "http://localhost:8081"

    # Модели (раздел 8.1)
    embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim_mrl: int = 512
    embedding_onnx_path: str | None = None
    embedding_tokenizer_path: str | None = None
    embedding_vllm_url: str | None = None

    reranker_model_id: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_onnx_path: str | None = None
    reranker_tokenizer_path: str | None = None
    reranker_vllm_url: str | None = None

    coldstart_spam_model_id: str = "deepvk/GeRaCl-USER2-base"
    modernbert_spam_model_id: str = "deepvk/RuModernBERT-base"
    comment_rescue_model_id: str = "deepvk/RuModernBERT-small"
    spam_onnx_path: str | None = None
    spam_tokenizer_path: str | None = None
    comment_rescue_onnx_path: str | None = None
    comment_rescue_tokenizer_path: str | None = None

    nli_gate_model_id: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    # Пороги и веса потоковой кластеризации (раздел 7.2.3)
    w_cos: float = 0.55
    w_bm25: float = 0.20
    w_ent: float = 0.25
    lambda_time_decay: float = 0.003

    clustering_tau_low: float = 0.72
    clustering_tau_high: float = 0.88
    rerank_merge_threshold: float = 0.85
    sliding_window_hours: int = 72
    ring_buffer_minutes: int = 15
    ring_buffer_max_size: int = 15000
    qdrant_top_k_candidates: int = 20

    # Параметры OriginScore (раздел 7.2.3)
    origin_alpha: float = 0.50
    origin_beta: float = 0.25
    origin_delta: float = 0.20
    origin_eta: float = 0.25
    origin_gamma_per_min: float = 0.50
    stub_edit_change_threshold: float = 0.30

    # Каскад спам-фильтрации (раздел 7.2.2)
    spam_l1_low_threshold: float = 0.25
    spam_l1_high_threshold: float = 0.80
    phash_hamming_threshold: int = 6
    min_organic_text_length: int = 15

    # Воронка комментариев (раздел 7.2.4)
    comment_graph_wd: float = 0.45
    comment_graph_wu: float = 0.30
    comment_graph_wp: float = 0.60
    comment_ancestor_max_chars: int = 200
    comment_mmr_lambda: float = 0.70
    comment_mmr_dedup_threshold: float = 0.82
    comment_buckets: tuple[int, ...] = (20, 50, 150, 500)

    # Дробление лонгридов в Qdrant (раздел 7.3.1)
    longread_char_threshold: int = 2000
    longread_max_chunks: int = 3

    # Гибридный поиск и Weighted RRF (раздел 7.3.2)
    rrf_k: int = 60
    rrf_w_body: float = 1.0
    rrf_w_bm25_base: float = 0.9
    rrf_w_bm25_exact: float = 1.6
    rrf_w_code: float = 0.6
    rrf_w_ocr: float = 0.5
    rrf_w_comments: float = 0.4
    prefetch_limit_body: int = 50
    prefetch_limit_bm25: int = 50
    prefetch_limit_code: int = 25
    prefetch_limit_ocr: int = 25
    prefetch_limit_comments: int = 25
    threshold_dense_body: float = 0.42
    threshold_dense_code: float = 0.55
    threshold_dense_ocr: float = 0.52
    threshold_dense_comments: float = 0.50
    hybrid_group_limit: int = 30
    rerank_top_k_green: int = 10
    rerank_top_k_yellow: int = 7
    rerank_top_k_red: int = 6
    rerank_top_k_guardrail: int = 4
    acorn_hnsw_ef: int = 128
    acorn_max_selectivity: float = 0.4
    quantization_oversampling: float = 2.5

    # Протокол верифицируемости Glass-box RAG (раздел 7.4.1)
    sw_match_score: float = 2.0
    sw_mismatch_score: float = -1.0
    sw_gap_open_score: float = -2.0
    sw_gap_extend_score: float = -0.5
    sw_similarity_threshold: float = 0.85
    sw_jaccard_fallback_threshold: float = 0.70
    nli_entailment_threshold: float = 0.75
    nli_onnx_path: str | None = None
    nli_tokenizer_path: str | None = None

    # Двухуровневый семантический кэш (разделы 7.3.2, 7.8.5)
    cache_tau_green: float = 0.95
    cache_tau_yellow: float = 0.90
    cache_tau_red: float = 0.88
    cache_news_ttl_sec: int = 7200
    cache_retro_ttl_sec: int = 86400

    # Маршрутизатор моделей, Stance, дайджесты и FinOps (разделы 7.3.3, 7.4.2, 7.4.3, 7.8.6)
    stance_max_k_per_side: int = 8
    stance_max_per_channel: int = 2
    digest_min_rubrics: int = 4
    digest_max_rubrics: int = 7
    digest_max_cards: int = 30
    core_channel_subscribers_threshold: int = 5
    trial_and_basic_rag_limit: int = 30
    finops_yellow_threshold_rub: float = 180.0
    finops_yellow_lowered_threshold_rub: float = 150.0
    finops_red_threshold_rub: float = 260.0
    finops_tech_cogs_cap_rub: float = 332.0
    finops_guardrail_daily_queries: int = 120
    finops_guardrail_rpm: int = 5
    finops_guardrail_throttle_sec: float = 3.0
    sse_keepalive_interval_sec: float = 15.0
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model_id: str | None = None
    llm_strict_vllm_extra_body: bool = False
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"


@lru_cache
def get_settings() -> Settings:
    if "pytest" not in sys.modules and "app.calibration_qa" not in sys.modules:
        resolved_parents = Path(__file__).resolve().parents
        candidates = [Path.cwd() / ".env"]
        if len(resolved_parents) > 3:
            candidates.append(resolved_parents[3] / ".env")
        for env_path in candidates:
            if env_path.is_file():
                return Settings(_env_file=env_path, _env_file_encoding="utf-8")
    return Settings()
