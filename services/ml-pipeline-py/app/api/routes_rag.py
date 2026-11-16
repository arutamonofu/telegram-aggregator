"""SSE-маршрут Glass-box RAG / Stance (`/api/v1/rag/query`) и лента дайджестов (`/api/v1/feed`)."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Literal, Sequence

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.preprocessing.offsets import build_utf16_offset_map
from app.rag.digest_builder import CanonicalEventCard, ClusteredDigestService
from app.rag.nli_gate import NO_VERIFIED_DATA_STUB
from app.rag.query_planner import FinOpsBudgetTracker
from app.rag.stance_matrix import StanceMatrixSynthesizer, StancePostInput
from app.rag.synthesizer import GlassBoxRagSynthesizer, IndexedSourcePost


class RegisterChannelRequest(BaseModel):
    """Запрос на регистрацию публичного Telegram-канала в ingestion-go."""

    username: str = Field(min_length=1)
    tier: str = "standard"


class RagQueryRequest(BaseModel):
    """Запрос на синтез ответа (Q&A / дайджест / матрица ЗА vs ПРОТИВ) в потоковом SSE."""

    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    mode: Literal["qa", "digest", "stance_matrix"] = "qa"
    channel_ids: list[int | str] = Field(default_factory=list)
    time_window_days: int = Field(default=90, ge=1, le=3650)
    cluster_id: str | None = None
    plan_tier: str | None = None
    deep_analysis: bool = False
    is_private_contour: bool = False
    tenant_id: str | None = None
    tenant_salt: str | None = None
    preloaded_posts: list[dict[str, Any]] | None = None


def format_sse_event(event_name: str, payload: Any) -> str:
    """Форматирует единичный SSE-кадр (`event: ...\ndata: ...\n\n`)."""
    data_str = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_name}\ndata: {data_str}\n\n"


def _parse_iso_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        dt = datetime.fromisoformat(value.strip())
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _split_csv_params(raw_values: Sequence[str] | None) -> list[str]:
    if not raw_values:
        return []
    out: list[str] = []
    for item in raw_values:
        for part in str(item).split(","):
            cleaned = part.strip()
            if cleaned:
                out.append(cleaned)
    return out


def _coerce_indexed_post(raw: dict[str, Any]) -> IndexedSourcePost:
    post_id = int(raw["post_id"])
    ch_id = str(raw.get("channel_id", ""))
    ch_user = str(raw.get("channel_username") or "").lstrip("@")
    ch_title = str(raw.get("channel_title") or ch_user or ch_id)
    msg_id = int(raw.get("msg_id", post_id))
    text = str(raw.get("text") or raw.get("text_raw") or "")
    omap = build_utf16_offset_map(text, post_id=post_id)
    deeplink = str(raw.get("deeplink") or "")
    if not deeplink:
        deeplink = f"https://t.me/{ch_user}/{msg_id}" if ch_user else f"https://t.me/c/{ch_id}/{msg_id}"
    return IndexedSourcePost(
        post_id=post_id,
        channel_id=ch_id,
        channel_username=ch_user,
        channel_title=ch_title,
        msg_id=msg_id,
        published_at=_parse_iso_dt(raw.get("published_at")),
        text=omap.text_nfkc,
        deeplink=deeplink,
        offset_map=omap,
        compressed_text=raw.get("compressed_text"),
        cluster_id=str(raw["cluster_id"]) if raw.get("cluster_id") else None,
        entities=list(raw.get("entities", [])),
        media_manifest=list(raw.get("media_manifest", [])),
        replies_count=int(raw.get("replies_count", 0)),
        also_in_channels_count=int(raw.get("also_in_channels_count", 0)),
        neutral_headline=raw.get("neutral_headline"),
    )


def _coerce_stance_post(raw: dict[str, Any]) -> StancePostInput:
    post_id = int(raw["post_id"])
    ch_id = str(raw.get("channel_id", ""))
    ch_user = str(raw.get("channel_username") or "").lstrip("@")
    ch_title = str(raw.get("channel_title") or ch_user or ch_id)
    msg_id = int(raw.get("msg_id", post_id))
    text = str(raw.get("text") or raw.get("text_raw") or "")
    return StancePostInput(
        post_id=post_id,
        channel_id=ch_id,
        channel_username=ch_user,
        channel_title=ch_title,
        msg_id=msg_id,
        published_at=_parse_iso_dt(raw.get("published_at")),
        text=text,
        deeplink=str(raw.get("deeplink") or ""),
        is_ad=bool(raw.get("is_ad", False)),
        stance_hint=raw.get("stance_hint"),
    )


def build_rag_router(
    *,
    settings: Settings | None = None,
    rag_synthesizer: GlassBoxRagSynthesizer,
    digest_service: ClusteredDigestService,
    stance_synthesizer: StanceMatrixSynthesizer,
    finops_tracker: FinOpsBudgetTracker,
    clean_posts_store: Sequence[IndexedSourcePost] | None = None,
) -> APIRouter:
    """Создает APIRouter с эндпоинтами `POST /api/v1/rag/query` и `GET /api/v1/feed`."""
    cfg = settings or get_settings()
    router = APIRouter()
    originals_store = list(clean_posts_store or [])

    @router.post("/api/v1/rag/query")
    async def rag_query_sse(req: RagQueryRequest) -> StreamingResponse:
        # Предварительная проверка квоты и FinOps-маршрута до открытия SSE-потока
        routing = finops_tracker.plan_rag_route(
            req.user_id,
            query=req.query,
            mode=req.mode,
            plan_tier=req.plan_tier,
            deep_analysis=req.deep_analysis,
            subscribed_channels_count=len(req.channel_ids),
            is_private_contour=req.is_private_contour,
        )
        if not routing.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": routing.rejection_reason or "rag_quota_exceeded",
                    "finops_zone": routing.finops_zone,
                    "quota_remaining": routing.quota_remaining,
                },
            )

        preloaded_rag: list[IndexedSourcePost] | None = None
        preloaded_stance: list[StancePostInput] | None = None
        if req.preloaded_posts is not None:
            preloaded_rag = [_coerce_indexed_post(item) for item in req.preloaded_posts]
            preloaded_stance = [_coerce_stance_post(item) for item in req.preloaded_posts]

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _execute_pipeline() -> None:
            try:
                await queue.put(
                    format_sse_event(
                        "pipeline_stage",
                        {
                            "stage": "routing",
                            "status": "completed",
                            "finops_zone": routing.finops_zone,
                            "model_tier": routing.model_tier,
                            "provider": routing.provider,
                            "model_id": routing.model_id,
                            "challenge_required": routing.challenge_required,
                        },
                    )
                )

                if routing.throttle_delay_sec > 0:
                    await asyncio.sleep(routing.throttle_delay_sec)

                if req.mode == "stance_matrix":
                    def _run_stance() -> Any:
                        posts_for_stance: list[StancePostInput]
                        if preloaded_stance is not None:
                            posts_for_stance = preloaded_stance
                        elif rag_synthesizer.hybrid_search is not None:
                            published_after = datetime.now(timezone.utc) - timedelta(days=req.time_window_days)
                            search_res = rag_synthesizer.hybrid_search.search_and_rerank(
                                query=req.query,
                                subscribed_channel_ids=req.channel_ids,
                                published_after=published_after,
                                top_k=routing.top_k_docs,
                                group_limit=cfg.hybrid_group_limit,
                                enable_reranker=routing.enable_gpu_reranker,
                                cluster_id=req.cluster_id,
                            )
                            posts_for_stance = [
                                StancePostInput(
                                    post_id=cand.post_id,
                                    channel_id=cand.channel_id,
                                    channel_username=cand.channel_username,
                                    channel_title=cand.channel_title,
                                    msg_id=cand.post_id,
                                    published_at=cand.published_at,
                                    text=cand.text,
                                    deeplink=cand.source_url,
                                )
                                for cand in search_res.reranked_posts
                            ]
                        else:
                            posts_for_stance = []
                        return stance_synthesizer.build_verified_matrix(
                            query=req.query,
                            posts=posts_for_stance,
                        )

                    stance_res = await asyncio.to_thread(_run_stance)
                    finops_tracker.record_query(
                        req.user_id,
                        used_frontier=(routing.model_tier == "frontier"),
                        cache_hit=False,
                    )
                    await queue.put(
                        format_sse_event(
                            "pipeline_stage",
                            {
                                "stage": "stance_verification",
                                "status": "completed",
                                "stance_counts": stance_res.stance_counts,
                                "selected_pro_count": stance_res.selected_pro_count,
                                "selected_con_count": stance_res.selected_con_count,
                                "rejected_nli_count": stance_res.rejected_nli_count,
                                "rejected_polarity_count": stance_res.rejected_polarity_count,
                            },
                        )
                    )
                    for row in stance_res.rows:
                        await queue.put(format_sse_event("stance_row", row.to_dict()))
                    await queue.put(
                        format_sse_event(
                            "source_posts",
                            {"source_posts": stance_res.source_posts_payload},
                        )
                    )
                    has_rows = bool(stance_res.rows)
                    await queue.put(
                        format_sse_event(
                            "done",
                            {
                                "status": "ok" if has_rows else "no_verified_data",
                                "mode": req.mode,
                                "finops_zone": routing.finops_zone,
                                "model_id": routing.model_id,
                                "rows_count": len(stance_res.rows),
                                "fallback_message": None if has_rows else NO_VERIFIED_DATA_STUB,
                            },
                        )
                    )
                else:
                    rag_res = await asyncio.to_thread(
                        rag_synthesizer.run_rag_query,
                        user_id=req.user_id,
                        query=req.query,
                        channel_ids=req.channel_ids,
                        mode=req.mode,
                        time_window_days=req.time_window_days,
                        cluster_id=req.cluster_id,
                        plan_tier=req.plan_tier,
                        deep_analysis=req.deep_analysis,
                        is_private_contour=req.is_private_contour,
                        tenant_id=req.tenant_id,
                        tenant_salt=req.tenant_salt,
                        preloaded_posts=preloaded_rag,
                    )
                    for st in rag_res.pipeline_stages:
                        await queue.put(format_sse_event("pipeline_stage", st))
                    if rag_res.direct_answer is not None:
                        await queue.put(
                            format_sse_event("direct_answer", rag_res.direct_answer.to_dict())
                        )
                    for claim in rag_res.verified_claims:
                        await queue.put(format_sse_event("verified_claim", claim.to_dict()))
                    await queue.put(
                        format_sse_event(
                            "source_posts",
                            {"source_posts": rag_res.source_posts_payload},
                        )
                    )
                    answer_status = (
                        rag_res.direct_answer.status
                        if rag_res.direct_answer is not None
                        else ("direct" if rag_res.verified_claims else "not_found")
                    )
                    await queue.put(
                        format_sse_event(
                            "done",
                            {
                                "status": "ok" if rag_res.verified_claims else "no_verified_data",
                                "mode": req.mode,
                                "answer_status": answer_status,
                                "direct_answer": (
                                    rag_res.direct_answer.to_dict()
                                    if rag_res.direct_answer is not None
                                    else None
                                ),
                                "cache_hit": rag_res.cache_hit,
                                "cache_key": rag_res.cache_key,
                                "finops_zone": rag_res.routing.finops_zone,
                                "model_id": rag_res.routing.model_id,
                                "verified_claims_count": len(rag_res.verified_claims),
                                "rejected_claims_count": rag_res.rejected_claims_count,
                                "fallback_message": rag_res.fallback_message,
                                "total_latency_ms": rag_res.total_latency_ms,
                            },
                        )
                    )
            except Exception as exc:  # pragma: no cover - защитный SSE кадр при неожиданной ошибке
                await queue.put(format_sse_event("error", {"detail": str(exc)}))
            finally:
                await queue.put(None)

        async def event_stream() -> AsyncIterator[str]:
            worker = asyncio.create_task(_execute_pipeline())
            keepalive_sec = max(0.01, float(cfg.sse_keepalive_interval_sec))
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=keepalive_sec)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if chunk is None:
                        break
                    yield chunk
            finally:
                if not worker.done():
                    worker.cancel()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.get("/api/v1/feed")
    def get_feed(
        user_id: str = Query(default="anonymous"),
        mode: Literal["digest", "clean_originals"] = Query(default="digest"),
        channel_ids: list[str] | None = Query(default=None),
        cluster_ids: list[str] | None = Query(default=None),
        plan_tier: str | None = Query(default=None),
        limit: int = Query(default=30, ge=1, le=50),
    ) -> dict[str, Any]:
        sub_channels = _split_csv_params(channel_ids)
        target_clusters = _split_csv_params(cluster_ids)

        if mode == "clean_originals":
            sub_set = set(sub_channels)
            items = [
                post.to_clean_post_item()
                for post in originals_store
                if not sub_set or str(post.channel_id) in sub_set
            ]
            items.sort(key=lambda x: str(x.get("published_at", "")), reverse=True)
            sliced = items[:limit]
            return {
                "mode": "clean_originals",
                "user_id": user_id,
                "total": len(sliced),
                "items": sliced,
            }

        if not target_clusters:
            target_clusters = list(digest_service.semantic_cache._l2_event_cards.keys())

        if sub_channels:
            card_dicts = digest_service.semantic_cache.get_event_cards(
                target_clusters,
                subscribed_channel_ids=sub_channels,
            )
            effective_channels = sub_channels
        else:
            card_dicts = digest_service.semantic_cache.get_event_cards(
                target_clusters,
                subscribed_channel_ids=None,
            )
            all_ch: set[str] = set()
            for cd in card_dicts:
                all_ch.update(str(c) for c in cd.get("channel_ids", []))
            effective_channels = sorted(all_ch)

        cards = [CanonicalEventCard.from_dict(cd) for cd in card_dicts]
        digest_res = digest_service.build_personal_digest(
            user_id=user_id,
            subscribed_channel_ids=effective_channels,
            cards=cards,
            plan_tier=plan_tier,
            max_cards=limit,
        )
        return {
            "mode": "digest",
            **digest_res.to_dict(),
        }

    @router.post("/api/v1/channels/register")
    def register_channel(req: RegisterChannelRequest) -> dict[str, Any]:
        clean_username = req.username.strip().lstrip("@")
        if not clean_username:
            raise HTTPException(status_code=400, detail="username is required")

        target_url = f"{cfg.ingestion_internal_url.rstrip('/')}/internal/v1/channels/register"
        body = json.dumps({"username": clean_username, "tier": req.tier}).encode("utf-8")
        http_req = urllib.request.Request(
            target_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=10.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip() or exc.reason
            raise HTTPException(status_code=502, detail=f"ingestion-go error: {detail}") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"ingestion-go unreachable: {exc}") from exc

    return router
