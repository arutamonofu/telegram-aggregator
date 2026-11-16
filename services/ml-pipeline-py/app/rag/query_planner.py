"""Маршрутизатор моделей (раздел 7.3.3) и 4-зонный FinOps-каскад Graceful Degradation (7.8.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from app.config import Settings, get_settings

PlanTier = Literal["TRIAL", "BASIC", "PROSUMER", "B2B_WORKSPACE"]
FinOpsZone = Literal["GREEN", "YELLOW", "RED", "GUARDRAIL"]
TaskClass = Literal[
    "STREAMING_MAP_AND_COMMENTS",
    "ASYNC_DIGEST_BATCH",
    "INTERACTIVE_RAG",
    "STANCE_AND_NLI",
]
ModelTier = Literal["fast", "mid", "frontier", "self_host", "local_onnx"]

_MSK_TZ = timezone(timedelta(hours=3))


def is_deepseek_peak_window(now: datetime | None = None) -> bool:
    """Проверяет будние пиковые интервалы DeepSeek (04:00-07:00 и 09:00-13:00 МСК, 2x тариф)."""
    dt = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    msk_dt = dt.astimezone(_MSK_TZ)
    if msk_dt.weekday() >= 5:
        return False
    hour = msk_dt.hour
    return (4 <= hour < 7) or (9 <= hour < 13)


@dataclass(slots=True)
class UserUsageState:
    """Текущее потребление и квоты пользователя (соответствует tenant_rf.user_finops_quotas)."""

    user_id: str
    plan_tier: str = "PROSUMER"
    rag_queries_used_period: int = 0
    rag_queries_today: int = 0
    rag_queries_last_minute: int = 0
    frontier_queries_today: int = 0
    accumulated_tech_cogs_rub: float = 0.0
    subscribed_channels_count: int = 50
    is_private_contour: bool = False
    red_zone_cohort_share: float = 0.03


@dataclass(slots=True)
class RoutingDecision:
    """Итоговое решение маршрутизатора моделей и параметров контекста для запроса."""

    allowed: bool
    rejection_reason: str | None
    finops_zone: FinOpsZone
    model_tier: ModelTier
    provider: str
    model_id: str
    use_citations_api: bool
    use_llguidance_schema: bool
    disable_thinking: bool
    top_k_docs: int
    use_compressed_event_cards: bool
    enable_gpu_reranker: bool
    enable_multihop_rewrite: bool
    tau_cache: float
    digest_reduce_model: str
    throttle_delay_sec: float
    challenge_required: bool
    quota_remaining: int
    weight_quantization: str = "fp8"
    kv_cache_dtype: str = "bf16"
    prompt_cache_enabled: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskRouteSpec:
    """Конфигурация эшелона модели для заданного класса задач (раздел 7.3.3 SPEC)."""

    task_class: TaskClass
    provider: str
    model_id: str
    fallback_provider: str | None = None
    fallback_model_id: str | None = None
    disable_thinking: bool = True
    prompt_cache_enabled: bool = True
    use_citations_api: bool = False
    use_llguidance_schema: bool = False
    radix_attention: bool = False
    weight_quantization: str = "fp8"
    kv_cache_dtype: str = "bf16"
    deepseek_blocked_for_peak: bool = False


def plan_task_route(
    task_class: TaskClass,
    *,
    is_private_contour: bool = False,
    prefer_batch: bool = False,
    model_tier: ModelTier = "fast",
    now: datetime | None = None,
) -> TaskRouteSpec:
    """Возвращает маршрут модели по 4 классам задач из матрицы 7.3.3 SPEC.md."""
    peak_ds = is_deepseek_peak_window(now)

    if task_class == "STANCE_AND_NLI":
        return TaskRouteSpec(
            task_class=task_class,
            provider="local_onnx",
            model_id="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            disable_thinking=True,
            prompt_cache_enabled=False,
        )

    if is_private_contour:
        if task_class == "ASYNC_DIGEST_BATCH":
            return TaskRouteSpec(
                task_class=task_class,
                provider="sglang_self_host",
                model_id="Qwen/Qwen3.8-27B-FP8",
                fallback_provider="sglang_self_host",
                fallback_model_id="Qwen/Qwen3-72B-FP8",
                radix_attention=True,
                weight_quantization="fp8",
                kv_cache_dtype="bf16",
            )
        if task_class == "INTERACTIVE_RAG":
            model_id = (
                "Qwen/Qwen3-235B-A22B-FP8"
                if model_tier == "frontier"
                else "Qwen/Qwen3.8-27B-FP8"
            )
            return TaskRouteSpec(
                task_class=task_class,
                provider="vllm_self_host",
                model_id=model_id,
                fallback_provider="vllm_self_host",
                fallback_model_id="Qwen/Qwen3-72B-FP8",
                use_llguidance_schema=True,
                weight_quantization="fp8",
                kv_cache_dtype="bf16",
            )
        return TaskRouteSpec(
            task_class=task_class,
            provider="vllm_self_host",
            model_id="Qwen/Qwen3.8-27B-FP8",
            fallback_provider="vllm_self_host",
            fallback_model_id="T-Pro-2.0",
            weight_quantization="fp8",
            kv_cache_dtype="bf16",
        )

    if task_class == "STREAMING_MAP_AND_COMMENTS":
        fallback_prov = "mistral" if peak_ds else "deepseek"
        fallback_mod = "mistral-small-4" if peak_ds else "deepseek-v4.1-flash"
        return TaskRouteSpec(
            task_class=task_class,
            provider="openai",
            model_id="gpt-6-luna",
            fallback_provider=fallback_prov,
            fallback_model_id=fallback_mod,
            deepseek_blocked_for_peak=peak_ds,
        )

    if task_class == "ASYNC_DIGEST_BATCH":
        return TaskRouteSpec(
            task_class=task_class,
            provider="openai_batch" if prefer_batch else "openai",
            model_id="gpt-6-luna-batch" if prefer_batch else "gpt-6-luna",
            fallback_provider="mistral_batch",
            fallback_model_id="mistral-small-4-batch",
            deepseek_blocked_for_peak=peak_ds,
        )

    # INTERACTIVE_RAG
    if model_tier == "frontier":
        return TaskRouteSpec(
            task_class=task_class,
            provider="anthropic",
            model_id="claude-sonnet-5-5",
            fallback_provider="openai",
            fallback_model_id="gpt-6.1-sol",
            use_citations_api=True,
            use_llguidance_schema=False,
        )
    if model_tier == "mid":
        return TaskRouteSpec(
            task_class=task_class,
            provider="anthropic",
            model_id="claude-3-5-haiku",
            fallback_provider="openai",
            fallback_model_id="gpt-6-luna",
            use_citations_api=True,
            use_llguidance_schema=False,
        )
    return TaskRouteSpec(
        task_class=task_class,
        provider="openai",
        model_id="gpt-6-luna",
        fallback_provider="mistral" if peak_ds else "deepseek",
        fallback_model_id="mistral-small-4" if peak_ds else "deepseek-v4.1-flash",
        use_llguidance_schema=True,
        deepseek_blocked_for_peak=peak_ds,
    )


class FinOpsBudgetTracker:
    """Учет тарифных квот и выбор зоны Graceful Degradation (Green / Yellow / Red / Guardrail)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._users: dict[str, UserUsageState] = {}

    def get_state(self, user_id: str, *, plan_tier: str = "PROSUMER") -> UserUsageState:
        state = self._users.get(user_id)
        if state is None:
            state = UserUsageState(user_id=user_id, plan_tier=plan_tier.upper())
            self._users[user_id] = state
        return state

    def set_state(self, state: UserUsageState) -> None:
        state.plan_tier = state.plan_tier.upper()
        self._users[state.user_id] = state

    def record_query(
        self,
        user_id: str,
        *,
        tech_cogs_rub: float = 0.129,
        used_frontier: bool = False,
        cache_hit: bool = False,
    ) -> UserUsageState:
        state = self.get_state(user_id)
        state.rag_queries_used_period += 1
        state.rag_queries_today += 1
        state.rag_queries_last_minute += 1
        if used_frontier:
            state.frontier_queries_today += 1
        effective_cost = 0.0001 if cache_hit else tech_cogs_rub
        state.accumulated_tech_cogs_rub = round(state.accumulated_tech_cogs_rub + effective_cost, 4)
        return state

    def resolve_zone(self, state: UserUsageState) -> FinOpsZone:
        """Определяет зону FinOps по накопленному TechCOGS, нагрузке и триггеру 7.8.8."""
        if (
            state.rag_queries_today > self.settings.finops_guardrail_daily_queries
            or state.rag_queries_last_minute > self.settings.finops_guardrail_rpm
        ):
            return "GUARDRAIL"

        if (
            state.accumulated_tech_cogs_rub > self.settings.finops_red_threshold_rub
            or state.rag_queries_today > 40
            or state.subscribed_channels_count > 150
        ):
            return "RED"

        yellow_threshold = (
            self.settings.finops_yellow_lowered_threshold_rub
            if state.red_zone_cohort_share > 0.07
            else self.settings.finops_yellow_threshold_rub
        )
        if (
            state.accumulated_tech_cogs_rub > yellow_threshold
            or state.rag_queries_today > 20
        ):
            return "YELLOW"

        return "GREEN"

    def plan_rag_route(
        self,
        user_id: str,
        *,
        query: str = "",
        mode: str = "qa",
        plan_tier: str | None = None,
        deep_analysis: bool = False,
        subscribed_channels_count: int | None = None,
        is_private_contour: bool | None = None,
        now: datetime | None = None,
    ) -> RoutingDecision:
        """Вычисляет квоту, зону FinOps и маршрут LLM для RAG-запроса."""
        state = self.get_state(user_id, plan_tier=plan_tier or "PROSUMER")
        if plan_tier is not None:
            state.plan_tier = plan_tier.upper()
        if subscribed_channels_count is not None:
            state.subscribed_channels_count = subscribed_channels_count
        if is_private_contour is not None:
            state.is_private_contour = is_private_contour

        tier = state.plan_tier
        limit_30 = self.settings.trial_and_basic_rag_limit

        # Лимит 30 RAG-запросов для TRIAL и тарифа BASIC
        if tier in ("TRIAL", "BASIC"):
            remaining = max(0, limit_30 - state.rag_queries_used_period)
            if state.rag_queries_used_period >= limit_30:
                return RoutingDecision(
                    allowed=False,
                    rejection_reason="rag_quota_exceeded_30",
                    finops_zone=self.resolve_zone(state),
                    model_tier="fast",
                    provider="openai",
                    model_id="gpt-6-luna",
                    use_citations_api=False,
                    use_llguidance_schema=True,
                    disable_thinking=True,
                    top_k_docs=0,
                    use_compressed_event_cards=False,
                    enable_gpu_reranker=False,
                    enable_multihop_rewrite=False,
                    tau_cache=self.settings.cache_tau_green,
                    digest_reduce_model="gpt-6-luna",
                    throttle_delay_sec=0.0,
                    challenge_required=False,
                    quota_remaining=0,
                )
            if tier == "BASIC" and state.rag_queries_today >= 1:
                return RoutingDecision(
                    allowed=False,
                    rejection_reason="basic_daily_limit_1_exceeded",
                    finops_zone=self.resolve_zone(state),
                    model_tier="fast",
                    provider="openai",
                    model_id="gpt-6-luna",
                    use_citations_api=False,
                    use_llguidance_schema=True,
                    disable_thinking=True,
                    top_k_docs=0,
                    use_compressed_event_cards=False,
                    enable_gpu_reranker=False,
                    enable_multihop_rewrite=False,
                    tau_cache=self.settings.cache_tau_green,
                    digest_reduce_model="gpt-6-luna",
                    throttle_delay_sec=0.0,
                    challenge_required=False,
                    quota_remaining=remaining,
                )
            quota_remaining = remaining - 1
        else:
            quota_remaining = -1

        zone = self.resolve_zone(state)
        peak_ds = is_deepseek_peak_window(now)
        notes: list[str] = []
        if peak_ds:
            notes.append("deepseek_off_peak_blocked")

        # Сложность запроса для выбора Fast / Mid / Frontier
        is_complex = mode == "stance_matrix" or deep_analysis or len(query) > 160
        is_medium = not is_complex and (len(query) > 70 or "почему" in query.lower() or "сравни" in query.lower())

        # Ограничение тарифа BASIC: строго эшелон Fast
        if tier == "BASIC":
            chosen_tier: ModelTier = "fast"
        elif zone == "GREEN":
            if is_complex:
                chosen_tier = "frontier"
            elif is_medium:
                chosen_tier = "mid"
            else:
                chosen_tier = "fast"
        elif zone == "YELLOW":
            if deep_analysis and state.frontier_queries_today < 3:
                chosen_tier = "frontier"
            elif is_medium:
                chosen_tier = "mid"
            else:
                chosen_tier = "fast"
        else:
            # RED / GUARDRAIL: полное отключение Frontier (0%)
            chosen_tier = "fast"

        if state.is_private_contour:
            task_route = plan_task_route(
                "INTERACTIVE_RAG",
                is_private_contour=True,
                model_tier=chosen_tier,
                now=now,
            )
            provider = task_route.provider
            model_id = task_route.model_id
            use_citations = False
            use_llguidance = True
            model_tier_out: ModelTier = "self_host" if zone in ("RED", "GUARDRAIL") else chosen_tier
        elif zone == "GUARDRAIL":
            provider = "openai"
            model_id = "gpt-5-nano"
            use_citations = False
            use_llguidance = True
            model_tier_out = "fast"
        elif zone == "RED":
            provider = "openai"
            model_id = "gpt-6-luna"
            use_citations = False
            use_llguidance = True
            model_tier_out = "fast"
        else:
            task_route = plan_task_route(
                "INTERACTIVE_RAG",
                is_private_contour=False,
                model_tier=chosen_tier,
                now=now,
            )
            provider = task_route.provider
            model_id = task_route.model_id
            use_citations = task_route.use_citations_api
            use_llguidance = task_route.use_llguidance_schema
            model_tier_out = chosen_tier

        if zone == "GREEN":
            top_k = self.settings.rerank_top_k_green
            compressed = False
            gpu_rerank = True
            multihop = True
            tau_cache = self.settings.cache_tau_green
            reduce_model = "gpt-6-luna" if peak_ds else "deepseek-v4.1-flash"
            throttle = 0.0
            challenge = False
        elif zone == "YELLOW":
            top_k = self.settings.rerank_top_k_yellow
            compressed = False
            gpu_rerank = True
            multihop = False
            tau_cache = self.settings.cache_tau_yellow
            reduce_model = "gpt-6-luna-batch"
            throttle = 0.0
            challenge = False
        elif zone == "RED":
            top_k = self.settings.rerank_top_k_red
            compressed = True
            gpu_rerank = True
            multihop = False
            tau_cache = self.settings.cache_tau_red
            reduce_model = "gpt-5-nano"
            throttle = 0.0
            challenge = False
        else:  # GUARDRAIL
            top_k = self.settings.rerank_top_k_guardrail
            compressed = True
            gpu_rerank = False
            multihop = False
            tau_cache = self.settings.cache_tau_red
            reduce_model = "gpt-5-nano"
            throttle = self.settings.finops_guardrail_throttle_sec
            challenge = True

        return RoutingDecision(
            allowed=True,
            rejection_reason=None,
            finops_zone=zone,
            model_tier=model_tier_out,
            provider=provider,
            model_id=model_id,
            use_citations_api=use_citations,
            use_llguidance_schema=use_llguidance,
            disable_thinking=True,
            top_k_docs=top_k,
            use_compressed_event_cards=compressed,
            enable_gpu_reranker=gpu_rerank,
            enable_multihop_rewrite=multihop,
            tau_cache=tau_cache,
            digest_reduce_model=reduce_model,
            throttle_delay_sec=throttle,
            challenge_required=challenge,
            quota_remaining=quota_remaining,
            notes=notes,
        )
