"""FastAPI-приложение сервиса `services/ml-pipeline-py`."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.api.routes_rag import build_rag_router
from app.cache.semantic_cache import TwoLevelSemanticCache
from app.comments.graph_pruner import RawComment
from app.comments.summarizer import CommentFunnelSummarizer
from app.config import Settings, get_settings
from app.consumers.post_pipeline import IngestPostInput, PostPipelineService
from app.rag.digest_builder import ClusteredDigestService
from app.rag.nli_gate import NliVerificationGate
from app.rag.query_planner import FinOpsBudgetTracker
from app.rag.stance_matrix import StanceMatrixSynthesizer
from app.rag.synthesizer import GlassBoxRagSynthesizer, IndexedSourcePost, RagLlmBackend
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.qdrant_repo import QdrantPublicChannelsRepo


class SummarizeCommentsRequest(BaseModel):
    post_id: int
    comments: list[dict[str, Any]] = Field(default_factory=list)
    max_selected: int = 8
    force_refresh: bool = False


class EditPostRequest(BaseModel):
    post_id: int
    new_text: str
    edit_date: datetime


def create_app(
    settings: Settings | None = None,
    *,
    pipeline: PostPipelineService | None = None,
    comment_summarizer: CommentFunnelSummarizer | None = None,
    semantic_cache: TwoLevelSemanticCache | None = None,
    hybrid_search: HybridSearchService | None = None,
    nli_gate: NliVerificationGate | None = None,
    finops_tracker: FinOpsBudgetTracker | None = None,
    llm_backend: RagLlmBackend | None = None,
    rag_synthesizer: GlassBoxRagSynthesizer | None = None,
    digest_service: ClusteredDigestService | None = None,
    stance_synthesizer: StanceMatrixSynthesizer | None = None,
    clean_posts_store: Sequence[IndexedSourcePost] | None = None,
) -> FastAPI:
    cfg = settings or get_settings()
    app = FastAPI(title="ml-pipeline-py", version="0.1.0")
    svc = pipeline or PostPipelineService(cfg)
    comments_svc = comment_summarizer or CommentFunnelSummarizer(cfg, embedder=svc.embedder)

    cache_svc = semantic_cache or TwoLevelSemanticCache(cfg)
    qdrant_repo = getattr(svc, "qdrant_repo", None) or QdrantPublicChannelsRepo(
        settings=cfg,
        embedder=svc.embedder,
    )
    hybrid_svc = hybrid_search or HybridSearchService(
        qdrant_repo,
        settings=cfg,
        embedder=svc.embedder,
        reranker=svc.reranker,
        semantic_cache=cache_svc,
    )
    nli_svc = nli_gate or NliVerificationGate(cfg)
    finops_svc = finops_tracker or FinOpsBudgetTracker(cfg)
    rag_svc = rag_synthesizer or GlassBoxRagSynthesizer(
        settings=cfg,
        hybrid_search=hybrid_svc,
        semantic_cache=cache_svc,
        nli_gate=nli_svc,
        finops_tracker=finops_svc,
        llm_backend=llm_backend,
    )
    digest_svc = digest_service or ClusteredDigestService(
        settings=cfg,
        semantic_cache=cache_svc,
        nli_gate=nli_svc,
        embedder=svc.embedder,
        finops_tracker=finops_svc,
    )
    stance_svc = stance_synthesizer or StanceMatrixSynthesizer(
        settings=cfg,
        nli_gate=nli_svc,
        embedder=svc.embedder,
    )

    app.state.semantic_cache = cache_svc
    app.state.hybrid_search = hybrid_svc
    app.state.nli_gate = nli_svc
    app.state.finops_tracker = finops_svc
    app.state.rag_synthesizer = rag_svc
    app.state.digest_service = digest_svc
    app.state.stance_synthesizer = stance_svc

    app.include_router(
        build_rag_router(
            settings=cfg,
            rag_synthesizer=rag_svc,
            digest_service=digest_svc,
            stance_synthesizer=stance_svc,
            finops_tracker=finops_svc,
            clean_posts_store=clean_posts_store,
        )
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "embedding_backend": svc.embedder.backend_name,
            "reranker_backend": svc.reranker.backend_name,
        }

    @app.post("/v1/posts/process")
    def process_post_endpoint(req: IngestPostInput) -> dict[str, Any]:
        result = svc.process_post(req)
        return {
            "post_id": result.post_id,
            "sentences": [s.to_dict() for s in result.preprocessed.offset_map.sentences],
            "cp_to_utf16_len": len(result.preprocessed.offset_map.cp_to_utf16),
            "spam": {
                "spam_class": result.spam_verdict.spam_class,
                "is_ad": result.spam_verdict.is_ad,
                "is_noise": result.spam_verdict.is_noise,
                "should_persist": result.spam_verdict.should_persist,
                "stage_decided": result.spam_verdict.stage_decided,
                "reasons": result.spam_verdict.reasons,
            },
            "cluster": (
                {
                    "cluster_id": result.cluster_result.cluster_id,
                    "is_new_cluster": result.cluster_result.is_new_cluster,
                    "decision": result.cluster_result.decision,
                    "origin_post_id": result.cluster_result.origin_post_id,
                    "origin_channel_id": result.cluster_result.origin_channel_id,
                }
                if result.cluster_result is not None
                else None
            ),
            "qdrant_chunks_count": len(result.qdrant_chunks),
        }

    @app.post("/v1/posts/edit")
    def edit_post_endpoint(req: EditPostRequest) -> dict[str, Any]:
        res = svc.process_post_edit(
            post_id=req.post_id,
            new_text=req.new_text,
            edit_date=req.edit_date,
        )
        if res is None:
            return {"updated": False}
        return {
            "updated": True,
            "cluster_id": res.cluster_id,
            "decision": res.decision,
            "origin_post_id": res.origin_post_id,
        }

    @app.post("/v1/comments/summarize")
    def summarize_comments_endpoint(req: SummarizeCommentsRequest) -> dict[str, Any]:
        raw_comments = [RawComment(**c) for c in req.comments]
        res = comments_svc.summarize(
            post_id=req.post_id,
            comments=raw_comments,
            max_selected=req.max_selected,
            force_refresh=req.force_refresh,
        )
        return {
            "post_id": res.post_id,
            "bucket": res.bucket,
            "cache_hit": res.cache_hit,
            "kept_after_pruning": res.kept_after_pruning,
            "digest_text": res.digest_text,
            "selected": [
                {
                    "comment_id": item.comment_id,
                    "reply_to_comment_id": item.reply_to_comment_id,
                    "graph_score": item.graph_score,
                    "is_rescued_bug_report": item.is_rescued_bug_report,
                    "confirmations_count": item.confirmations_count,
                    "formatted_text": item.formatted_text,
                }
                for item in res.selected_comments
            ],
        }

    return app


app = create_app()
