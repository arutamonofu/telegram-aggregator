"""Синтез ответов Glass-box RAG (Anthropic Citations API, OpenAI/vLLM llguidance JSON Schema) и верификация."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol, Sequence

import httpx

from app.cache.semantic_cache import (
    TwoLevelSemanticCache,
    build_private_cache_key,
    build_public_cache_key,
    normalize_time_bucket,
)
from app.config import Settings, get_settings
from app.preprocessing.entities_ner import extract_entities
from app.preprocessing.offsets import PostUtf16OffsetMap, build_utf16_offset_map
from app.rag.nli_gate import NO_VERIFIED_DATA_STUB, NliVerificationGate, _content_stems
from app.rag.query_planner import FinOpsBudgetTracker, RoutingDecision
from app.rag.smith_waterman import SmithWatermanAlignment, smith_waterman_align
from app.retrieval.hybrid_search import HybridSearchService, RetrievedPostCandidate

PROMPT_CACHE_SYSTEM_HEADER = (
    "You are an analytical, strictly factual Telegram Glass-box RAG QA synthesizer. Answer strictly in "
    "Russian based ONLY on the provided numbered post clauses [P{post_id}_S{sentence_id}]. "
    "First classify answer_status for the user's query: "
    "'direct' if the sources directly answer the query; "
    "'indirect' if the sources do not confirm the query's specific event/premise (e.g., a yes/no question "
    "whose event is not confirmed) yet provide relevant facts on the same topic/subject; "
    "'not_found' if the sources have no relevant facts for the query (in which case return claims: []). "
    "Second, write direct_answer (1 to 3 concise sentences in Russian) directly addressing the user's "
    "question using ONLY facts from claims and inline refs like [c1], [c2]. If answer_status is 'indirect', "
    "explicitly state first that there is no direct confirmation in the sources, then summarize what is "
    "actually reported on the topic. "
    "Third, synthesize 1 to 5 concrete, fact-dense claims covering distinct key events, locations, decisions, "
    "or figures directly relevant to the query. Skip empty video/stream announcements, channel promos, "
    "link stubs (e.g. 'смотрите обзор по ссылке'), and vague meta-filler without concrete facts. "
    "Each claim's statement MUST be fully supported by its attached citations, and every citation's "
    "exact_quote MUST be copied verbatim from a single clause without the [P..._S...] tag and without adding '...'. "
    "Never invent facts, numbers, ports, tickers, or citations. Output strict JSON matching RagSynthesisOutput."
)

RAG_SYNTHESIS_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RagSynthesisOutput",
    "type": "object",
    "additionalProperties": False,
    "required": ["mode", "answer_status", "direct_answer", "claims", "stance_comparison_matrix"],
    "properties": {
        "mode": {"type": "string", "enum": ["qa", "digest", "stance_matrix"]},
        "answer_status": {"type": "string", "enum": ["direct", "indirect", "not_found"]},
        "direct_answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_id", "topic_group", "statement", "citations"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "topic_group": {"type": "string"},
                    "statement": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["post_id", "sentence_id", "exact_quote"],
                            "properties": {
                                "post_id": {"type": "integer"},
                                "sentence_id": {"type": "integer"},
                                "exact_quote": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "stance_comparison_matrix": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["aspect", "pro_points", "con_points"],
                "properties": {
                    "aspect": {"type": "string"},
                    "pro_points": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/StanceArgument"},
                    },
                    "con_points": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/StanceArgument"},
                    },
                },
            },
        },
    },
    "$defs": {
        "StanceArgument": {
            "type": "object",
            "additionalProperties": False,
            "required": ["claim_id", "statement", "citations"],
            "properties": {
                "claim_id": {"type": "string"},
                "statement": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["post_id", "sentence_id", "exact_quote"],
                        "properties": {
                            "post_id": {"type": "integer"},
                            "sentence_id": {"type": "integer"},
                            "exact_quote": {"type": "string"},
                        },
                    },
                },
            },
        }
    },
}


@dataclass(slots=True)
class DraftCitation:
    """Сырая ссылка из ответа LLM до проверки Смитом — Ватерманом и NLI."""

    post_id: int
    sentence_id: int | None
    exact_quote: str


@dataclass(slots=True)
class DraftClaim:
    """Сырой тезис из ответа LLM до верификации."""

    claim_id: str
    topic_group: str
    statement: str
    citations: list[DraftCitation]


@dataclass(slots=True)
class VerifiedCitation:
    """Верифицированная цитата с точными UTF-16 координатами для браузерного DOM."""

    post_id: int
    channel_id: str
    channel_title: str
    msg_id: int
    published_at: str
    deeplink: str
    utf16_range: list[int]
    verbatim_text: str
    nli_entailment_score: float
    sw_alignment_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "channel_id": int(self.channel_id) if self.channel_id.isdigit() else self.channel_id,
            "channel_title": self.channel_title,
            "msg_id": self.msg_id,
            "published_at": self.published_at,
            "deeplink": self.deeplink,
            "utf16_range": list(self.utf16_range),
            "verbatim_text": self.verbatim_text,
            "nli_entailment_score": self.nli_entailment_score,
            "sw_alignment_score": self.sw_alignment_score,
        }


@dataclass(slots=True)
class VerifiedClaim:
    """Тезис, прошедший Смита — Ватермана (>=0.85) и NLI-гейт ENTAILMENT (>=0.75)."""

    claim_id: str
    topic_group: str
    statement: str
    citations: list[VerifiedCitation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "topic_group": self.topic_group,
            "statement": self.statement,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass(slots=True)
class IndexedSourcePost:
    """Контекстный документ с предрасчитанной картой UTF-16 смещений и клаузами."""

    post_id: int
    channel_id: str
    channel_username: str
    channel_title: str
    msg_id: int
    published_at: datetime
    text: str
    deeplink: str
    offset_map: PostUtf16OffsetMap
    compressed_text: str | None = None
    cluster_id: str | None = None
    entities: list[dict[str, Any]] = field(default_factory=list)
    media_manifest: list[dict[str, Any]] = field(default_factory=list)
    replies_count: int = 0
    also_in_channels_count: int = 0
    neutral_headline: str | None = None

    def format_clauses_block(self, *, use_compressed: bool = False) -> str:
        if use_compressed and self.compressed_text:
            return f"[P{self.post_id}_S1] {self.compressed_text}"
        if self.offset_map.sentences:
            return "\n".join(s.formatted for s in self.offset_map.sentences)
        return f"[P{self.post_id}_S1] {self.offset_map.text_nfkc}"

    def to_clean_post_item(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "channel_id": int(self.channel_id) if self.channel_id.isdigit() else self.channel_id,
            "channel_username": self.channel_username or None,
            "channel_title": self.channel_title,
            "msg_id": self.msg_id,
            "published_at": self.published_at.isoformat(),
            "neutral_headline": self.neutral_headline,
            "text_raw": self.offset_map.text_nfkc,
            "entities": self.entities,
            "media_manifest": self.media_manifest,
            "replies_count": self.replies_count,
            "cluster_id": self.cluster_id,
            "also_in_channels_count": self.also_in_channels_count,
            "deeplink": self.deeplink or f"https://t.me/c/{self.channel_id}/{self.msg_id}",
        }


@dataclass(slots=True)
class VerifiedDirectAnswer:
    """Верифицированный прямой ответ на вопрос пользователя с привязкой к источникам."""

    status: str  # "direct" | "indirect" | "not_found"
    text: str
    citations: list[VerifiedCitation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass(slots=True)
class RagSynthesisResult:
    """Финальный верифицированный ответ Glass-box RAG."""

    query: str
    mode: str
    cache_hit: bool
    cache_key: str
    routing: RoutingDecision
    verified_claims: list[VerifiedClaim]
    rejected_claims_count: int
    sw_alignment_failures: int
    fallback_message: str | None
    source_posts_payload: list[dict[str, Any]]
    direct_answer: VerifiedDirectAnswer | None = None
    pipeline_stages: list[dict[str, Any]] = field(default_factory=list)
    stance_rows: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def to_cache_payload(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "verified_claims": [c.to_dict() for c in self.verified_claims],
            "direct_answer": self.direct_answer.to_dict() if self.direct_answer is not None else None,
            "rejected_claims_count": self.rejected_claims_count,
            "sw_alignment_failures": self.sw_alignment_failures,
            "fallback_message": self.fallback_message,
            "source_posts_payload": self.source_posts_payload,
            "stance_rows": self.stance_rows,
        }


class RagLlmBackend(Protocol):
    """Протокол генерации черновика тезисов и цитат по списку документов."""

    def synthesize(
        self,
        *,
        query: str,
        documents: Sequence[IndexedSourcePost],
        routing: RoutingDecision,
        mode: str = "qa",
    ) -> list[DraftClaim]: ...


def build_anthropic_citations_payload(
    query: str,
    documents: Sequence[IndexedSourcePost],
    *,
    model_id: str = "claude-sonnet-5-5",
    max_tokens: int = 2048,
    use_compressed: bool = False,
) -> dict[str, Any]:
    """Формирует запрос Anthropic Messages API с включенным `citations: {enabled: True}`."""
    content_blocks: list[dict[str, Any]] = []
    for doc in documents:
        doc_text = (
            doc.compressed_text
            if (use_compressed and doc.compressed_text)
            else doc.offset_map.text_nfkc
        )
        content_blocks.append(
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": doc_text,
                },
                "title": f"post_id={doc.post_id} | {doc.channel_title}",
                "context": f"channel_id={doc.channel_id}; published_at={doc.published_at.isoformat()}",
                "citations": {"enabled": True},
            }
        )
    content_blocks.append(
        {
            "type": "text",
            "text": (
                f"Вопрос пользователя: {query}\n"
                "Сформулируй краткие и точные фактологические тезисы на русском языке с опорой на документы."
            ),
        }
    )
    return {
        "model": model_id,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": PROMPT_CACHE_SYSTEM_HEADER,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": content_blocks}],
    }


def _find_sentence_for_char_index(
    offset_map: PostUtf16OffsetMap,
    char_idx: int,
) -> int | None:
    for sent in offset_map.sentences:
        if sent.start_cp <= char_idx < sent.end_cp:
            return sent.sentence_id
    return offset_map.sentences[0].sentence_id if offset_map.sentences else None


def parse_anthropic_citations_response(
    response_json: Mapping[str, Any],
    documents: Sequence[IndexedSourcePost],
) -> list[DraftClaim]:
    """Разбирает ответ Anthropic Citations API (`char_location`, `cited_text`) в список DraftClaim."""
    blocks = response_json.get("content", [])
    claims: list[DraftClaim] = []
    claim_counter = 1

    for block in blocks:
        if not isinstance(block, Mapping) or block.get("type") != "text":
            continue
        text = str(block.get("text", "")).strip()
        raw_cites = block.get("citations") or []
        if not text or not raw_cites:
            continue

        draft_cites: list[DraftCitation] = []
        for cite in raw_cites:
            if not isinstance(cite, Mapping):
                continue
            doc_idx = int(cite.get("document_index", -1))
            if doc_idx < 0 or doc_idx >= len(documents):
                continue
            doc = documents[doc_idx]
            cited_text = str(cite.get("cited_text", "")).strip()
            if not cited_text:
                continue
            start_char = cite.get("start_char_index")
            sent_id = (
                _find_sentence_for_char_index(doc.offset_map, int(start_char))
                if start_char is not None
                else None
            )
            draft_cites.append(
                DraftCitation(
                    post_id=doc.post_id,
                    sentence_id=sent_id,
                    exact_quote=cited_text,
                )
            )

        if draft_cites:
            claims.append(
                DraftClaim(
                    claim_id=f"c{claim_counter}",
                    topic_group="Основное",
                    statement=text,
                    citations=draft_cites[:3],
                )
            )
            claim_counter += 1

    return claims


def build_openai_llguidance_payload(
    query: str,
    documents: Sequence[IndexedSourcePost],
    *,
    model_id: str = "gpt-6.1-sol",
    mode: str = "qa",
    use_compressed: bool = False,
    disable_thinking: bool = True,
    include_vllm_extra_body: bool = True,
) -> dict[str, Any]:
    """Формирует запрос к OpenAI / OpenRouter / vLLM со строгой JSON-схемой (`response_format`)."""
    context_sections = [
        f"--- POST {doc.post_id} ({doc.channel_title}, {doc.published_at:%Y-%m-%d}) ---\n"
        f"{doc.format_clauses_block(use_compressed=use_compressed)}"
        for doc in documents
    ]
    user_prompt = (
        f"Mode: {mode}\n"
        f"Query: {query}\n\n"
        "Sources:\n" + "\n\n".join(context_sections)
    )
    payload: dict[str, Any] = {
        "model": model_id,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": PROMPT_CACHE_SYSTEM_HEADER},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "RagSynthesisOutput",
                "strict": True,
                "schema": RAG_SYNTHESIS_JSON_SCHEMA,
            },
        },
    }
    if include_vllm_extra_body:
        payload["extra_body"] = {
            "structured_outputs": {"json": RAG_SYNTHESIS_JSON_SCHEMA},
            "guided_decoding_backend": "llguidance",
        }
        if disable_thinking:
            payload["reasoning_effort"] = "none"
            payload["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def _strip_json_fences(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def parse_structured_json_synthesis(
    raw_output: str | Mapping[str, Any],
) -> tuple[list[DraftClaim], str | None, str | None]:
    """Парсит строгий JSON-вывод OpenAI / vLLM (`claims`, `answer_status`, `direct_answer`)."""
    if isinstance(raw_output, str):
        data = json.loads(_strip_json_fences(raw_output))
    else:
        data = dict(raw_output)

    # Поддержка OpenAI Chat Completions envelope
    if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
        msg = data["choices"][0].get("message", {})
        content = msg.get("content", "{}")
        data = json.loads(_strip_json_fences(content)) if isinstance(content, str) else dict(content)

    raw_status = data.get("answer_status")
    answer_status = (
        str(raw_status).strip().lower()
        if isinstance(raw_status, str) and raw_status.strip().lower() in {"direct", "indirect", "not_found"}
        else None
    )
    raw_direct = data.get("direct_answer")
    direct_answer = str(raw_direct).strip() if isinstance(raw_direct, str) and raw_direct.strip() else None

    claims_raw = data.get("claims") or []
    parsed: list[DraftClaim] = []
    for idx, item in enumerate(claims_raw, start=1):
        if not isinstance(item, Mapping):
            continue
        statement = str(item.get("statement", "")).strip()
        raw_cites = item.get("citations") or []
        cites: list[DraftCitation] = []
        for c in raw_cites:
            if not isinstance(c, Mapping):
                continue
            quote = str(c.get("exact_quote", "")).strip()
            pid = c.get("post_id")
            if not quote or pid is None:
                continue
            sid = c.get("sentence_id")
            cites.append(
                DraftCitation(
                    post_id=int(pid),
                    sentence_id=int(sid) if sid is not None else None,
                    exact_quote=quote,
                )
            )
        if statement and cites:
            parsed.append(
                DraftClaim(
                    claim_id=str(item.get("claim_id") or f"c{idx}"),
                    topic_group=str(item.get("topic_group") or "Факты"),
                    statement=statement,
                    citations=cites[:3],
                )
            )
    return parsed, answer_status, direct_answer


def parse_structured_json_response(raw_output: str | Mapping[str, Any]) -> list[DraftClaim]:
    """Парсит строгий JSON-вывод OpenAI / vLLM (`claims`: [{statement, citations: [...]}])."""
    claims, _, _ = parse_structured_json_synthesis(raw_output)
    return claims


class HttpOpenAILlguidanceBackend:
    """HTTP-клиент для OpenAI / OpenRouter / vLLM с поддержкой `response_format`."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_sec: float = 30.0,
        model_override: str | None = None,
        include_vllm_extra_body: bool = True,
    ) -> None:
        self.endpoint = base_url.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint = f"{self.endpoint}/chat/completions"
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.model_override = model_override
        self.include_vllm_extra_body = include_vllm_extra_body
        self.last_draft_answer: tuple[str | None, str | None] = (None, None)

    def synthesize(
        self,
        *,
        query: str,
        documents: Sequence[IndexedSourcePost],
        routing: RoutingDecision,
        mode: str = "qa",
    ) -> list[DraftClaim]:
        self.last_draft_answer = (None, None)
        payload = build_openai_llguidance_payload(
            query,
            documents,
            model_id=self.model_override or routing.model_id,
            mode=mode,
            use_compressed=routing.use_compressed_event_cards,
            disable_thinking=routing.disable_thinking,
            include_vllm_extra_body=self.include_vllm_extra_body,
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(self.endpoint, json=payload, headers=headers)
                if resp.status_code == 400 and "response_format" in payload:
                    fallback_payload = dict(payload)
                    fallback_payload["response_format"] = {"type": "json_object"}
                    fallback_payload.pop("extra_body", None)
                    resp = client.post(self.endpoint, json=fallback_payload, headers=headers)
                resp.raise_for_status()
                parsed, answer_status, direct_answer = parse_structured_json_synthesis(resp.json())
                self.last_draft_answer = (answer_status, direct_answer)
                if parsed or answer_status == "not_found":
                    return parsed
        except Exception:
            pass
        self.last_draft_answer = (None, None)
        return ExtractiveGlassBoxBackend().synthesize(
            query=query,
            documents=documents,
            routing=routing,
            mode=mode,
        )


class HttpAnthropicCitationsBackend:
    """HTTP-клиент для Anthropic Messages API с `citations: {enabled: True}`."""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com", *, timeout_sec: float = 30.0) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/v1/messages"
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.last_draft_answer: tuple[str | None, str | None] = (None, None)

    def synthesize(
        self,
        *,
        query: str,
        documents: Sequence[IndexedSourcePost],
        routing: RoutingDecision,
        mode: str = "qa",
    ) -> list[DraftClaim]:
        self.last_draft_answer = (None, None)
        payload = build_anthropic_citations_payload(
            query,
            documents,
            model_id=routing.model_id,
            use_compressed=routing.use_compressed_event_cards,
        )
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(self.endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                parsed = parse_anthropic_citations_response(resp.json(), documents)
                if parsed:
                    return parsed
        except Exception:
            pass
        return ExtractiveGlassBoxBackend().synthesize(
            query=query,
            documents=documents,
            routing=routing,
            mode=mode,
        )


_WORD_TOK_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_OPEN_WH_QUERY_RE = re.compile(
    r"(?:^|\b)(?:как(?:ой|ая|ое|ие|их|им|ую|ом|ого)|что|кто|где|куда|откуда|когда|"
    r"сколько|насколько|почему|зачем|чем|в\s+чем|расскажи|покажи|найди|опиши|назови|"
    r"перечисли|сводк[аеиу]|новост[ьиейях]+|обзор|итог[иов]*|причин[аы]|услови[яй]|"
    r"what|which|who|where|when|why|how|tell|show|find|list|news|summary)\b",
    re.IGNORECASE,
)
_POLAR_QUERY_CUE_RE = re.compile(
    r"(?:\?|\b(?:ли|правда|действительно|реально|подтвержд[а-я]*|состоял[а-я]*|произош[её]л[а-я]*|случил[а-я]*)\b)",
    re.IGNORECASE,
)
_POLAR_GENERIC_STEMS = frozenset(
    {
        "сего", "вчер", "сейч", "вооб", "данн", "моме", "итог", "случ",
        "прав", "дейс", "реал", "скаж", "подт", "факт", "прои",
        "войн", "ситу", "обст", "тема", "теме", "ново", "посл",
    }
)
_INDIRECT_ACK_RE = re.compile(
    r"(?:нет\s+(?:прям[а-я]+\s+)?(?:подтвержден|сведени|сообщени|данн|информаци)|"
    r"не\s+(?:подтвержда|сообща|упомина)|не\s+найден)",
    re.IGNORECASE,
)
_CLAIM_REF_TOKEN_RE = re.compile(r"\[([^\]]+)\]")
_VIDEO_PROMO_CUE_RE = re.compile(
    r"(?:ссылк[аеиу]\s+на\s+видео|смотреть\s+видео|смотрите\s+(?:видео|обзор|в\s+вк|на\s+vk|на\s+youtube|на\s+rutube)|"
    r"vkvideo\.ru|rutube\.ru|youtube\.com|youtu\.be|подписывайт[еи]сь\s+на)",
    re.IGNORECASE,
)
_EXTRACTIVE_QUERY_STOPWORDS = frozenset(
    {
        "какие", "какой", "какая", "какое", "каких", "каким", "есть", "что", "как", "где",
        "когда", "почему", "зачем", "кто", "это", "этот", "эта", "эти", "все", "всё",
        "или", "для", "при", "под", "над", "про", "без", "через", "между", "после",
        "перед", "новости", "новость", "новостей", "новостям", "новостях", "сводка",
        "сводку", "сводки", "расскажи", "покажи", "найди", "скажи",
        "последние", "последних", "последняя", "последнее", "свежие", "свежих", "свежая",
        "актуальные", "актуальных", "главные", "главных", "важные", "важных",
        "сегодня", "вчера", "неделю", "недели", "ситуация", "ситуации", "обстановка", "обстановке",
        "the", "and", "for", "are", "was", "were", "what", "which", "how", "who",
        "where", "when", "why", "news", "latest", "recent", "about", "from", "with",
    }
)
_TEASER_QUERY_ALLOW_STEMS = ("видео", "ролик", "стрим", "эфир", "ссылк", "vkvideo", "youtube", "rutube")


def _compute_teaser_quality_multiplier(text: str, query_lower: str) -> float:
    """Понижает приоритет пустых видео-анонсов и рекламных заглушек, если запрос не про видео/ссылку."""
    if any(stem in query_lower for stem in _TEASER_QUERY_ALLOW_STEMS):
        return 1.0
    stripped = text.strip()
    if not stripped:
        return 0.5
    has_url = bool(_URL_RE.search(stripped))
    has_promo_cue = bool(_VIDEO_PROMO_CUE_RE.search(stripped))
    non_url_len = len(_URL_RE.sub(" ", stripped).strip())
    ends_with_ellipsis = stripped.rstrip().endswith(("...", "…"))
    if has_promo_cue and (non_url_len < 280 or has_url):
        return 0.25 if non_url_len < 260 else 0.6
    if has_url and non_url_len < 200 and ends_with_ellipsis:
        return 0.45
    return 1.0


def _rerank_preloaded_posts(
    query: str,
    posts: Sequence[IndexedSourcePost],
    *,
    top_k: int,
    time_window_days: int,
    now: datetime,
) -> list[IndexedSourcePost]:
    """Быстрый (<15 мс) детерминированный реранкер для `preloaded_posts` на CPU."""
    if len(posts) <= 1:
        return list(posts)[: max(1, top_k)]

    raw_q_tokens = [w.lower().replace("ё", "е") for w in _WORD_TOK_RE.findall(query) if len(w) >= 3]
    filtered_q = [w for w in raw_q_tokens if w not in _EXTRACTIVE_QUERY_STOPWORDS]
    effective_q = filtered_q if filtered_q else raw_q_tokens
    q_stems = {w[:5] for w in effective_q}

    q_ent = extract_entities(query)
    q_nums = set(q_ent.numbers) | set(q_ent.tickers) | set(q_ent.law_ids)
    q_lower = query.lower()

    scored_candidates: list[tuple[float, int, IndexedSourcePost]] = []
    window_days = max(1, int(time_window_days))

    for idx, doc in enumerate(posts):
        body_text = doc.offset_map.text_nfkc or doc.text
        body_words = [w.lower().replace("ё", "е") for w in _WORD_TOK_RE.findall(body_text) if len(w) >= 3]
        body_stems = {w[:5] for w in body_words}
        headline_stems = {
            w.lower().replace("ё", "е")[:5]
            for w in _WORD_TOK_RE.findall(doc.neutral_headline or "")
            if len(w) >= 3
        }

        matched_stems = q_stems & (body_stems | headline_stems)
        stem_recall = len(matched_stems) / max(1, len(q_stems))
        headline_hits = len(q_stems & headline_stems)
        tf_sum = sum(min(3, sum(1 for bw in body_words if bw[:5] == qs)) for qs in q_stems)
        tf_score = tf_sum / max(1, 3 * len(q_stems))

        doc_ent = extract_entities(body_text)
        doc_nums = set(doc_ent.numbers) | set(doc_ent.tickers) | set(doc_ent.law_ids)
        ent_score = 0.0
        if q_nums:
            if q_nums & doc_nums:
                ent_score += 1.5 * len(q_nums & doc_nums)
            elif doc_nums:
                ent_score -= 1.2

        pub = doc.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - pub).total_seconds() / 86400.0)
        recency_bonus = 0.18 * max(0.0, 1.0 - (age_days / float(window_days)))

        non_url_len = len(_URL_RE.sub(" ", body_text).strip())
        density_bonus = 0.12 if non_url_len >= 220 else (-0.08 if non_url_len < 90 else 0.0)
        quality_mult = _compute_teaser_quality_multiplier(body_text, q_lower)
        pos_tie = 0.03 / (idx + 1)

        base_relevance = (1.6 * stem_recall) + (0.45 * tf_score) + (0.35 * min(2, headline_hits)) + ent_score
        if base_relevance <= 0.0:
            total_score = quality_mult * (0.05 + recency_bonus + pos_tie)
        else:
            total_score = (base_relevance + recency_bonus + density_bonus + pos_tie) * quality_mult

        scored_candidates.append((total_score, -idx, doc))

    scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    # Мягкое разнообразие по каналам при отборе top_k, чтобы один канал не вытеснил остальные
    target_k = max(1, top_k)
    channel_counts: dict[str, int] = {}
    adjusted: list[tuple[float, int, IndexedSourcePost]] = []
    for base_score, neg_idx, doc in scored_candidates:
        cnt = channel_counts.get(doc.channel_id, 0)
        channel_counts[doc.channel_id] = cnt + 1
        div_penalty = 0.06 * max(0, cnt - 2)
        adjusted.append((base_score - div_penalty, neg_idx, doc))

    adjusted.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in adjusted[:target_k]]


class ExtractiveGlassBoxBackend:
    """Детерминированный локальный синтезатор для Dev и автотестов без внешних ключей LLM."""

    def synthesize(
        self,
        *,
        query: str,
        documents: Sequence[IndexedSourcePost],
        routing: RoutingDecision,
        mode: str = "qa",
    ) -> list[DraftClaim]:
        _ = routing, mode
        if not documents:
            return []

        raw_q_tokens = [w.lower().replace("ё", "е") for w in _WORD_TOK_RE.findall(query) if len(w) >= 3]
        filtered_q_tokens = [w for w in raw_q_tokens if w not in _EXTRACTIVE_QUERY_STOPWORDS]
        effective_q_tokens = filtered_q_tokens if filtered_q_tokens else raw_q_tokens
        q_words = {w[:5] for w in effective_q_tokens}

        q_ent = extract_entities(query)
        q_nums = set(q_ent.numbers) | set(q_ent.tickers) | set(q_ent.law_ids)
        q_code_ids = {
            t.lower() for t in q_ent.ner if "." in t or any(c.isupper() for c in t[1:])
        } | {t.lower() for t in q_ent.tickers if not t.isupper()}
        q_lower = query.lower()

        scored_clauses: list[tuple[float, IndexedSourcePost, int, str]] = []
        for doc_idx, doc in enumerate(documents):
            doc_rank_bonus = 0.15 / (doc_idx + 1)
            clauses = doc.offset_map.sentences
            if not clauses and doc.offset_map.text_nfkc:
                clauses_text = [(1, doc.offset_map.text_nfkc)]
            else:
                clauses_text = [(c.sentence_id, c.text) for c in clauses]

            headline_words = {
                w.lower().replace("ё", "е")[:5]
                for w in _WORD_TOK_RE.findall(doc.neutral_headline or "")
                if len(w) >= 3
            }
            doc_words = {
                w.lower().replace("ё", "е")[:5]
                for w in _WORD_TOK_RE.findall(doc.offset_map.text_nfkc)
                if len(w) >= 3
            } | headline_words
            doc_ent = extract_entities(doc.offset_map.text_nfkc)
            doc_nums = set(doc_ent.numbers) | set(doc_ent.tickers) | set(doc_ent.law_ids)
            doc_has_match = bool(
                (q_words & doc_words)
                or (q_nums and (q_nums & doc_nums))
                or (q_code_ids and any(cid in doc.offset_map.text_nfkc.lower() for cid in q_code_ids))
            )

            for sid, sent_text in clauses_text:
                s_words = {
                    w.lower().replace("ё", "е")[:5]
                    for w in _WORD_TOK_RE.findall(sent_text)
                    if len(w) >= 3
                }
                if not s_words:
                    continue
                overlap = len(q_words & s_words)
                s_ent = extract_entities(sent_text)
                s_nums = set(s_ent.numbers) | set(s_ent.tickers) | set(s_ent.law_ids)
                s_code_ids = {
                    t.lower() for t in s_ent.ner if "." in t or any(c.isupper() for c in t[1:])
                } | {t.lower() for t in s_ent.tickers if not t.isupper()}
                ent_bonus = 2.0 * len(q_nums & s_nums)
                # Штраф, если запрос содержит точное число/порт, а предложение упоминает другое число
                if q_nums and s_nums and q_nums.isdisjoint(s_nums):
                    ent_bonus -= 2.5
                code_err_bonus = 0.5 * len(q_code_ids & s_code_ids)
                if code_err_bonus == 0.0 and ("ошибк" in q_lower or "error" in q_lower):
                    if any("." in tok for tok in s_ent.ner):
                        code_err_bonus = 0.5

                doc_context_bonus = 0.25 if (doc_has_match and len(s_words) >= 4) else 0.0
                stripped_sent = sent_text.strip()
                teaser_penalty = (
                    -0.20
                    if len(clauses_text) > 1 and stripped_sent.endswith(("...", "…", ":"))
                    else 0.0
                )
                if _VIDEO_PROMO_CUE_RE.search(stripped_sent) and not any(
                    stem in q_lower for stem in _TEASER_QUERY_ALLOW_STEMS
                ):
                    teaser_penalty -= 0.45
                score = (
                    (overlap / max(1, len(q_words)))
                    + ent_bonus
                    + code_err_bonus
                    + doc_rank_bonus
                    + doc_context_bonus
                    + teaser_penalty
                )
                doc_quality = _compute_teaser_quality_multiplier(doc.offset_map.text_nfkc, q_lower)
                score *= doc_quality
                if overlap > 0 or ent_bonus > 0 or code_err_bonus > 0 or (doc_has_match and len(s_words) >= 5):
                    scored_clauses.append((score, doc, sid, sent_text))

        scored_clauses.sort(key=lambda item: item[0], reverse=True)

        claims: list[DraftClaim] = []
        post_claim_counts: dict[int, int] = {}
        seen_clause_keys: set[tuple[int, int]] = set()

        def _append_claim(doc: IndexedSourcePost, sid: int, sent_text: str) -> None:
            seen_clause_keys.add((doc.post_id, sid))
            post_claim_counts[doc.post_id] = post_claim_counts.get(doc.post_id, 0) + 1
            quote = sent_text.strip()
            if len(quote) > 360:
                quote = quote[:360].rsplit(" ", 1)[0].strip() or quote[:360]
            claims.append(
                DraftClaim(
                    claim_id=f"c{len(claims) + 1}",
                    topic_group=doc.channel_title or "Источники",
                    statement=quote,
                    citations=[
                        DraftCitation(
                            post_id=doc.post_id,
                            sentence_id=sid,
                            exact_quote=quote,
                        )
                    ],
                )
            )

        # Первый проход: по 1 наиболее релевантному тезису с поста
        for score, doc, sid, sent_text in scored_clauses:
            if score <= 0.10 or post_claim_counts.get(doc.post_id, 0) >= 1:
                continue
            _append_claim(doc, sid, sent_text)
            if len(claims) >= 5:
                return claims

        # Второй проход для сводных ответов: добираем до 2 тезисов с поста, если набралось < 4
        if len(claims) < 4:
            for score, doc, sid, sent_text in scored_clauses:
                if score <= 0.15 or (doc.post_id, sid) in seen_clause_keys:
                    continue
                if post_claim_counts.get(doc.post_id, 0) >= 2:
                    continue
                _append_claim(doc, sid, sent_text)
                if len(claims) >= 5:
                    break

        return claims


def _candidate_to_indexed_post(cand: RetrievedPostCandidate) -> IndexedSourcePost:
    omap = build_utf16_offset_map(cand.text, post_id=cand.post_id)
    msg_id = int(cand.payload.get("msg_id", cand.post_id))
    deeplink = cand.source_url
    if not deeplink:
        if cand.channel_username:
            deeplink = f"https://t.me/{cand.channel_username}/{msg_id}"
        else:
            deeplink = f"https://t.me/c/{cand.channel_id}/{msg_id}"
    compressed = None
    if cand.text:
        compressed = cand.text[:600].rstrip() + ("..." if len(cand.text) > 600 else "")
    return IndexedSourcePost(
        post_id=cand.post_id,
        channel_id=cand.channel_id,
        channel_username=cand.channel_username,
        channel_title=cand.channel_title,
        msg_id=msg_id,
        published_at=cand.published_at,
        text=cand.text,
        deeplink=deeplink,
        offset_map=omap,
        compressed_text=compressed,
        cluster_id=cand.cluster_id,
    )


class GlassBoxRagSynthesizer:
    """Сквозной оркестратор Glass-box RAG: Кэш -> Поиск -> Реранк -> LLM -> Смит-Ватерман -> NLI."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        hybrid_search: HybridSearchService | None = None,
        semantic_cache: TwoLevelSemanticCache | None = None,
        nli_gate: NliVerificationGate | None = None,
        finops_tracker: FinOpsBudgetTracker | None = None,
        llm_backend: RagLlmBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.semantic_cache = semantic_cache or TwoLevelSemanticCache(self.settings)
        self.hybrid_search = hybrid_search
        self.nli_gate = nli_gate or NliVerificationGate(self.settings)
        self.finops_tracker = finops_tracker or FinOpsBudgetTracker(self.settings)
        self.llm_backend = llm_backend or self._init_default_llm_backend()

    def _init_default_llm_backend(self) -> RagLlmBackend:
        if self.settings.llm_api_key and self.settings.llm_base_url:
            return HttpOpenAILlguidanceBackend(
                self.settings.llm_base_url,
                self.settings.llm_api_key,
                model_override=self.settings.llm_model_id,
                include_vllm_extra_body=self.settings.llm_strict_vllm_extra_body,
            )
        if self.settings.anthropic_api_key:
            return HttpAnthropicCitationsBackend(
                self.settings.anthropic_api_key,
                self.settings.anthropic_base_url,
            )
        return ExtractiveGlassBoxBackend()

    @staticmethod
    def _build_local_clause_context(
        doc: IndexedSourcePost,
        alignment: SmithWatermanAlignment,
        sentence_id: int | None,
    ) -> str:
        sentences = doc.offset_map.sentences
        if not sentences:
            return doc.offset_map.text_nfkc
        target_sid = sentence_id
        if target_sid is None:
            for sent in sentences:
                if sent.start_cp <= alignment.start_cp < sent.end_cp:
                    target_sid = sent.sentence_id
                    break
        if target_sid is None:
            return alignment.verbatim_text

        selected_texts: list[str] = []
        seen_sids: set[int] = set()
        # Включаем первое предложение поста (обычно заголовок/тема) и окно соседних предложений
        target_ids = [1, target_sid - 1, target_sid, target_sid + 1]
        by_sid = {s.sentence_id: s.text for s in sentences}
        for sid in target_ids:
            if sid in by_sid and sid not in seen_sids:
                seen_sids.add(sid)
                selected_texts.append(by_sid[sid])
        return " ".join(selected_texts).strip() or alignment.verbatim_text

    def _align_citation(
        self,
        cite: DraftCitation,
        indexed_posts: Mapping[int, IndexedSourcePost],
        ordered_docs: Sequence[IndexedSourcePost],
    ) -> tuple[IndexedSourcePost | None, SmithWatermanAlignment | None]:
        doc = indexed_posts.get(cite.post_id)
        if doc is None and 1 <= cite.post_id <= len(ordered_docs):
            doc = ordered_docs[cite.post_id - 1]

        if doc is not None:
            alignment = smith_waterman_align(
                doc.offset_map.text_nfkc,
                cite.exact_quote,
                offset_map=doc.offset_map,
                post_id=doc.post_id,
                sentence_id=cite.sentence_id,
                match_score=self.settings.sw_match_score,
                mismatch_score=self.settings.sw_mismatch_score,
                gap_open=self.settings.sw_gap_open_score,
                gap_extend=self.settings.sw_gap_extend_score,
                similarity_threshold=self.settings.sw_similarity_threshold,
                jaccard_threshold=self.settings.sw_jaccard_fallback_threshold,
            )
            if alignment is not None:
                return doc, alignment

        # Фоллбэк поиска цитаты по другим постам выборки, если LLM перепутала post_id
        for other_doc in ordered_docs:
            if doc is not None and other_doc.post_id == doc.post_id:
                continue
            alignment = smith_waterman_align(
                other_doc.offset_map.text_nfkc,
                cite.exact_quote,
                offset_map=other_doc.offset_map,
                post_id=other_doc.post_id,
                sentence_id=None,
                match_score=self.settings.sw_match_score,
                mismatch_score=self.settings.sw_mismatch_score,
                gap_open=self.settings.sw_gap_open_score,
                gap_extend=self.settings.sw_gap_extend_score,
                similarity_threshold=self.settings.sw_similarity_threshold,
                jaccard_threshold=self.settings.sw_jaccard_fallback_threshold,
            )
            if alignment is not None:
                return other_doc, alignment

        return doc, None

    def verify_draft_claims(
        self,
        draft_claims: Sequence[DraftClaim],
        indexed_posts: Mapping[int, IndexedSourcePost],
    ) -> tuple[list[VerifiedClaim], int, int, list[dict[str, Any]]]:
        """Применяет Смита — Ватермана + UTF-16 маппинг + NLI-гейт к списку сырых тезисов."""
        verified_claims: list[VerifiedClaim] = []
        rejected_claims = 0
        sw_failures = 0
        highlights_by_post: dict[int, list[dict[str, Any]]] = {}
        ordered_docs = list(indexed_posts.values())

        for draft in draft_claims:
            aligned_cites: list[tuple[IndexedSourcePost, DraftCitation, SmithWatermanAlignment]] = []
            for cite in draft.citations:
                doc, alignment = self._align_citation(cite, indexed_posts, ordered_docs)
                if doc is None or alignment is None:
                    sw_failures += 1
                    continue
                aligned_cites.append((doc, cite, alignment))

            if not aligned_cites:
                rejected_claims += 1
                continue

            combined_verbatim = " ".join(aln.verbatim_text for _, _, aln in aligned_cites).strip()
            stmt_stems = _content_stems(draft.statement.lower().replace("ё", "е"))
            verified_cites: list[VerifiedCitation] = []

            for doc, cite, alignment in aligned_cites:
                nli_res = self.nli_gate.check_pair(
                    statement=draft.statement,
                    verbatim_text=alignment.verbatim_text,
                )
                if not nli_res.is_entailed:
                    if nli_res.label == "CONTRADICTS":
                        continue
                    cite_stems = _content_stems(alignment.verbatim_text.lower().replace("ё", "е"))
                    if not (stmt_stems & cite_stems):
                        continue

                    candidate_contexts: list[str] = []
                    if len(aligned_cites) > 1 and combined_verbatim and combined_verbatim != alignment.verbatim_text:
                        candidate_contexts.append(combined_verbatim)
                    local_ctx = self._build_local_clause_context(doc, alignment, cite.sentence_id)
                    if local_ctx and local_ctx != alignment.verbatim_text:
                        candidate_contexts.append(local_ctx)

                    accepted_ctx_res = None
                    for ctx_text in candidate_contexts:
                        ctx_res = self.nli_gate.check_pair(
                            statement=draft.statement,
                            verbatim_text=ctx_text,
                        )
                        if ctx_res.label == "CONTRADICTS":
                            accepted_ctx_res = None
                            break
                        if ctx_res.is_entailed:
                            accepted_ctx_res = ctx_res
                            break
                    if accepted_ctx_res is None:
                        continue
                    nli_res = accepted_ctx_res

                v_cite = VerifiedCitation(
                    post_id=doc.post_id,
                    channel_id=doc.channel_id,
                    channel_title=doc.channel_title,
                    msg_id=doc.msg_id,
                    published_at=doc.published_at.isoformat(),
                    deeplink=doc.deeplink,
                    utf16_range=[alignment.utf16_range[0], alignment.utf16_range[1]],
                    verbatim_text=alignment.verbatim_text,
                    nli_entailment_score=nli_res.entailment_prob,
                    sw_alignment_score=alignment.similarity_score,
                )
                verified_cites.append(v_cite)
                highlights_by_post.setdefault(doc.post_id, []).append(
                    {
                        "claimId": draft.claim_id,
                        "startUtf16": alignment.utf16_range[0],
                        "endUtf16": alignment.utf16_range[1],
                    }
                )

            if verified_cites:
                verified_claims.append(
                    VerifiedClaim(
                        claim_id=draft.claim_id,
                        topic_group=draft.topic_group,
                        statement=draft.statement,
                        citations=verified_cites,
                    )
                )
            else:
                rejected_claims += 1

        cited_order: list[int] = []
        cited_set: set[int] = set()
        for claim in verified_claims:
            for v_cite in claim.citations:
                if v_cite.post_id in indexed_posts and v_cite.post_id not in cited_set:
                    cited_set.add(v_cite.post_id)
                    cited_order.append(v_cite.post_id)

        ordered_source_ids = cited_order + [pid for pid in indexed_posts if pid not in cited_set]
        source_posts_payload: list[dict[str, Any]] = []
        for pid in ordered_source_ids:
            doc = indexed_posts[pid]
            if pid in highlights_by_post or verified_claims:
                source_posts_payload.append(
                    {
                        "post": doc.to_clean_post_item(),
                        "rag_highlights": highlights_by_post.get(pid, []),
                    }
                )

        return verified_claims, rejected_claims, sw_failures, source_posts_payload

    @staticmethod
    def _resolve_direct_answer_citations(
        draft_text: str,
        verified_claims: Sequence[VerifiedClaim],
        source_posts_payload: Sequence[Mapping[str, Any]],
    ) -> tuple[str | None, list[VerifiedCitation]]:
        claims_by_id = {c.claim_id.strip().lower(): c for c in verified_claims}
        source_idx_by_post_id: dict[int, int] = {}
        post_id_by_source_idx: dict[int, int] = {}
        for idx, item in enumerate(source_posts_payload, start=1):
            post_obj = item.get("post") or {}
            pid = post_obj.get("post_id")
            if pid is not None:
                pid_int = int(pid)
                source_idx_by_post_id[pid_int] = idx
                post_id_by_source_idx[idx] = pid_int

        first_cite_by_post: dict[int, VerifiedCitation] = {}
        for cl in verified_claims:
            for vc in cl.citations:
                first_cite_by_post.setdefault(vc.post_id, vc)

        collected: list[VerifiedCitation] = []
        collected_keys: set[tuple[int, int, int]] = set()
        invalid_ref = False

        def _add_cite(vc: VerifiedCitation) -> None:
            key = (vc.post_id, vc.utf16_range[0], vc.utf16_range[1])
            if key not in collected_keys:
                collected_keys.add(key)
                collected.append(vc)

        def _replace_match(match: re.Match[str]) -> str:
            nonlocal invalid_ref
            inner = match.group(1).strip()
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            if not parts:
                return match.group(0)
            resolved_nums: list[int] = []
            for part in parts:
                low = part.lower()
                if re.fullmatch(r"c\d+", low):
                    cl = claims_by_id.get(low)
                    if cl is None or not cl.citations:
                        invalid_ref = True
                        return match.group(0)
                    for vc in cl.citations:
                        _add_cite(vc)
                        s_idx = source_idx_by_post_id.get(vc.post_id)
                        if s_idx is not None and s_idx not in resolved_nums:
                            resolved_nums.append(s_idx)
                elif re.fullmatch(r"\d+", low):
                    num = int(low)
                    pid = post_id_by_source_idx.get(num)
                    if pid is None or pid not in first_cite_by_post:
                        invalid_ref = True
                        return match.group(0)
                    _add_cite(first_cite_by_post[pid])
                    if num not in resolved_nums:
                        resolved_nums.append(num)
                else:
                    return match.group(0)
            if not resolved_nums:
                return match.group(0)
            return "[" + ", ".join(str(n) for n in resolved_nums) + "]"

        resolved_text = _CLAIM_REF_TOKEN_RE.sub(_replace_match, draft_text).strip()
        if invalid_ref:
            return None, []

        if not collected:
            for cl in verified_claims[:2]:
                for vc in cl.citations[:1]:
                    _add_cite(vc)

        return resolved_text, collected

    @staticmethod
    def _build_deterministic_direct_answer(
        status: str,
        verified_claims: Sequence[VerifiedClaim],
        source_posts_payload: Sequence[Mapping[str, Any]],
    ) -> VerifiedDirectAnswer | None:
        if not verified_claims:
            return None

        source_idx_by_post_id: dict[int, int] = {}
        for idx, item in enumerate(source_posts_payload, start=1):
            post_obj = item.get("post") or {}
            pid = post_obj.get("post_id")
            if pid is not None:
                source_idx_by_post_id[int(pid)] = idx

        # Берём 1–2 наиболее информативных верифицированных тезиса (сначала из разных постов)
        selected_claims: list[VerifiedClaim] = []
        used_post_ids: set[int] = set()
        for cl in verified_claims:
            pid = cl.citations[0].post_id if cl.citations else None
            if pid is not None and pid not in used_post_ids:
                selected_claims.append(cl)
                used_post_ids.add(pid)
            if len(selected_claims) >= 2:
                break
        if len(selected_claims) < 2:
            for cl in verified_claims:
                if cl not in selected_claims:
                    selected_claims.append(cl)
                if len(selected_claims) >= 2:
                    break

        parts: list[str] = []
        citations: list[VerifiedCitation] = []
        seen_post_cites: set[int] = set()
        for cl in selected_claims:
            ref_nums: list[int] = []
            for vc in cl.citations:
                s_idx = source_idx_by_post_id.get(vc.post_id)
                if s_idx is not None and s_idx not in ref_nums:
                    ref_nums.append(s_idx)
                if vc.post_id not in seen_post_cites:
                    seen_post_cites.add(vc.post_id)
                    citations.append(vc)
            stmt = cl.statement.strip().rstrip(".!?")
            if ref_nums:
                refs_str = ", ".join(str(n) for n in ref_nums)
                parts.append(f"{stmt} [{refs_str}].")
            else:
                parts.append(f"{stmt}.")

        summary_body = " ".join(parts).strip()
        if status == "indirect":
            text = (
                "Прямого подтверждения этому в подключённых каналах нет. "
                f"По теме в источниках сообщается следующее. {summary_body}"
            )
        else:
            text = summary_body

        return VerifiedDirectAnswer(
            status=status,
            text=text,
            citations=citations,
        )

    def _build_verified_direct_answer(
        self,
        *,
        query: str,
        verified_claims: Sequence[VerifiedClaim],
        source_posts_payload: Sequence[Mapping[str, Any]],
        draft_status: str | None = None,
        draft_text: str | None = None,
    ) -> VerifiedDirectAnswer | None:
        if not verified_claims:
            return None

        verified_corpus = " ".join(
            [c.statement for c in verified_claims]
            + [ct.verbatim_text for c in verified_claims for ct in c.citations]
        )
        verified_stems = _content_stems(verified_corpus)
        raw_q_tokens = [w.lower().replace("ё", "е") for w in _WORD_TOK_RE.findall(query) if len(w) >= 3]
        meaningful_q = [w for w in raw_q_tokens if w not in _EXTRACTIVE_QUERY_STOPWORDS]
        q_stems = {w[:4] if len(w) >= 4 else w for w in meaningful_q}
        matched_q_stems = q_stems & verified_stems
        missing_q_stems = q_stems - verified_stems - _POLAR_GENERIC_STEMS
        is_polar_query = bool(_POLAR_QUERY_CUE_RE.search(query)) and not bool(_OPEN_WH_QUERY_RE.search(query))
        inferred_status = (
            "indirect"
            if (is_polar_query and bool(matched_q_stems) and bool(missing_q_stems))
            else "direct"
        )
        effective_status = (
            "indirect"
            if (draft_status == "indirect" or inferred_status == "indirect")
            else "direct"
        )

        if draft_text:
            resolved_text, resolved_cites = self._resolve_direct_answer_citations(
                draft_text,
                verified_claims,
                source_posts_payload,
            )
            if resolved_text:
                if effective_status == "indirect" and not _INDIRECT_ACK_RE.search(resolved_text):
                    resolved_text = (
                        "Прямого подтверждения этому в подключённых каналах нет. "
                        + resolved_text
                    )
                nli_res = self.nli_gate.verify_direct_answer(
                    query=query,
                    direct_answer=resolved_text,
                    answer_status=effective_status,
                    verified_statements=[c.statement for c in verified_claims],
                    verified_quotes=[ct.verbatim_text for c in verified_claims for ct in c.citations],
                )
                if nli_res.is_entailed:
                    return VerifiedDirectAnswer(
                        status=effective_status,
                        text=resolved_text,
                        citations=resolved_cites,
                    )

        return self._build_deterministic_direct_answer(
            effective_status,
            verified_claims,
            source_posts_payload,
        )

    def run_rag_query(
        self,
        *,
        user_id: str,
        query: str,
        channel_ids: Sequence[int | str],
        mode: str = "qa",
        time_window_days: int = 90,
        cluster_id: str | None = None,
        plan_tier: str | None = None,
        deep_analysis: bool = False,
        is_private_contour: bool = False,
        tenant_id: str | None = None,
        tenant_salt: str | None = None,
        preloaded_posts: Sequence[IndexedSourcePost] | None = None,
        now: datetime | None = None,
    ) -> RagSynthesisResult:
        """Выполняет полный цикл Glass-box RAG с проверкой квот, кэша, поиска и NLI."""
        t0 = time.perf_counter()
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        routing = self.finops_tracker.plan_rag_route(
            user_id,
            query=query,
            mode=mode,
            plan_tier=plan_tier,
            deep_analysis=deep_analysis,
            subscribed_channels_count=len(channel_ids),
            is_private_contour=is_private_contour,
            now=current_time,
        )
        if not routing.allowed:
            raise PermissionError(routing.rejection_reason or "rag_quota_exceeded")

        # 1. Композитный ключ двухуровневого семантического кэша
        time_bucket = normalize_time_bucket(time_window_days, reference_time=current_time)
        if is_private_contour:
            if not tenant_id or not tenant_salt:
                raise ValueError("tenant_id and tenant_salt required for private contour RAG")
            cache_key = build_private_cache_key(
                tenant_id=tenant_id,
                tenant_salt=tenant_salt,
                channel_aliases=[str(c) for c in channel_ids],
                time_bucket=time_bucket,
                exact_entities=query,
            )
        else:
            cache_key = build_public_cache_key(
                channel_ids=channel_ids,
                time_bucket=time_bucket,
                exact_entities=query,
            )

        stages: list[dict[str, Any]] = []
        t_cache = time.perf_counter()
        query_emb = (
            self.hybrid_search.embedder.encode_query(query)
            if self.hybrid_search is not None
            else None
        )
        cache_hit = None
        if query_emb is not None:
            cache_hit = self.semantic_cache.get_level1(
                cache_key,
                query_emb,
                tau_cache=routing.tau_cache,
                now=current_time,
                is_private=is_private_contour,
                tenant_id=tenant_id,
            )
        cache_ms = round((time.perf_counter() - t_cache) * 1000.0, 2)
        stages.append(
            {
                "stage": "cache_check",
                "status": "hit" if cache_hit is not None else "miss",
                "latency_ms": cache_ms,
                "cache_key": cache_key,
            }
        )

        if cache_hit is not None:
            cached = cache_hit.entry.response_payload
            verified_claims = [
                VerifiedClaim(
                    claim_id=c["claim_id"],
                    topic_group=c["topic_group"],
                    statement=c["statement"],
                    citations=[VerifiedCitation(**ct) for ct in c["citations"]],
                )
                for c in cached.get("verified_claims", [])
            ]
            cached_da = cached.get("direct_answer")
            restored_da = (
                VerifiedDirectAnswer(
                    status=str(cached_da["status"]),
                    text=str(cached_da["text"]),
                    citations=[VerifiedCitation(**ct) for ct in cached_da.get("citations", [])],
                )
                if isinstance(cached_da, Mapping) and cached_da.get("text")
                else None
            )
            self.finops_tracker.record_query(user_id, cache_hit=True)
            return RagSynthesisResult(
                query=query,
                mode=mode,
                cache_hit=True,
                cache_key=cache_key,
                routing=routing,
                verified_claims=verified_claims,
                direct_answer=restored_da,
                rejected_claims_count=int(cached.get("rejected_claims_count", 0)),
                sw_alignment_failures=int(cached.get("sw_alignment_failures", 0)),
                fallback_message=cached.get("fallback_message"),
                source_posts_payload=list(cached.get("source_posts_payload", [])),
                pipeline_stages=stages,
                stance_rows=list(cached.get("stance_rows", [])),
                total_latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # 2. Гибридный поиск + кросс-энкодерный реранкинг
        indexed_list: list[IndexedSourcePost]
        if preloaded_posts is not None:
            raw_preloaded = list(preloaded_posts)
            t_rr0 = time.perf_counter()
            if (
                len(raw_preloaded) > 1
                and self.hybrid_search is not None
                and routing.enable_gpu_reranker
                and getattr(self.hybrid_search.reranker, "backend_name", "") == "vllm"
            ):
                docs_text = [doc.offset_map.text_nfkc or doc.text for doc in raw_preloaded]
                scores = self.hybrid_search.reranker.score_batch(query, docs_text)
                ranked_pairs = sorted(
                    zip(scores, range(len(raw_preloaded)), raw_preloaded),
                    key=lambda item: (item[0], -item[1]),
                    reverse=True,
                )
                indexed_list = [doc for _, _, doc in ranked_pairs][: routing.top_k_docs]
                rerank_status = "completed"
            elif len(raw_preloaded) > 1:
                indexed_list = _rerank_preloaded_posts(
                    query,
                    raw_preloaded,
                    top_k=routing.top_k_docs,
                    time_window_days=time_window_days,
                    now=current_time,
                )
                rerank_status = "completed_preloaded"
            else:
                indexed_list = raw_preloaded[: routing.top_k_docs]
                rerank_status = "skipped_preloaded"
            rerank_ms = round((time.perf_counter() - t_rr0) * 1000.0, 2)
            stages.append(
                {
                    "stage": "hybrid_retrieval",
                    "status": "completed",
                    "latency_ms": 0.0,
                    "candidates_count": len(raw_preloaded),
                }
            )
            stages.append(
                {
                    "stage": "reranking",
                    "status": rerank_status,
                    "latency_ms": rerank_ms,
                    "selected_count": len(indexed_list),
                }
            )
        elif self.hybrid_search is not None:
            published_after = current_time - timedelta(days=time_window_days)
            search_res = self.hybrid_search.search_and_rerank(
                query=query,
                subscribed_channel_ids=channel_ids,
                published_after=published_after,
                top_k=routing.top_k_docs,
                group_limit=self.settings.hybrid_group_limit,
                enable_reranker=routing.enable_gpu_reranker,
                cluster_id=cluster_id,
            )
            stages.append(
                {
                    "stage": "hybrid_retrieval",
                    "status": "completed",
                    "latency_ms": search_res.retrieval_latency_ms,
                    "candidates_count": search_res.rrf_candidates_count,
                }
            )
            stages.append(
                {
                    "stage": "reranking",
                    "status": "completed" if routing.enable_gpu_reranker else "disabled_guardrail",
                    "latency_ms": search_res.rerank_latency_ms,
                    "selected_count": len(search_res.reranked_posts),
                }
            )
            indexed_list = [_candidate_to_indexed_post(c) for c in search_res.reranked_posts]
        else:
            indexed_list = []

        indexed_map = {doc.post_id: doc for doc in indexed_list}

        # 3. Генерация утверждений LLM
        t_syn = time.perf_counter()
        draft_claims = (
            self.llm_backend.synthesize(
                query=query,
                documents=indexed_list,
                routing=routing,
                mode=mode,
            )
            if indexed_list
            else []
        )
        draft_answer_pair = getattr(self.llm_backend, "last_draft_answer", (None, None))
        draft_answer_status, draft_answer_text = (
            draft_answer_pair
            if isinstance(draft_answer_pair, tuple) and len(draft_answer_pair) == 2
            else (None, None)
        )
        stages.append(
            {
                "stage": "llm_synthesis",
                "status": "completed",
                "latency_ms": round((time.perf_counter() - t_syn) * 1000.0, 2),
                "draft_claims_count": len(draft_claims),
            }
        )

        # 4. Детерминированное выравнивание Смита — Ватермана + NLI-гейт
        t_nli = time.perf_counter()
        verified_claims, rejected_count, sw_failures, source_posts_payload = (
            self.verify_draft_claims(draft_claims, indexed_map)
        )
        if (
            not verified_claims
            and draft_answer_status != "not_found"
            and indexed_list
            and isinstance(self.llm_backend, (HttpOpenAILlguidanceBackend, HttpAnthropicCitationsBackend))
        ):
            fallback_drafts = ExtractiveGlassBoxBackend().synthesize(
                query=query,
                documents=indexed_list,
                routing=routing,
                mode=mode,
            )
            verified_claims, rejected_count, sw_failures, source_posts_payload = (
                self.verify_draft_claims(fallback_drafts, indexed_map)
            )
            draft_answer_status, draft_answer_text = (None, None)
        direct_answer = self._build_verified_direct_answer(
            query=query,
            verified_claims=verified_claims,
            source_posts_payload=source_posts_payload,
            draft_status=draft_answer_status,
            draft_text=draft_answer_text,
        )
        stages.append(
            {
                "stage": "nli_verification",
                "status": "completed",
                "latency_ms": round((time.perf_counter() - t_nli) * 1000.0, 2),
                "verified_claims_count": len(verified_claims),
                "rejected_claims_count": rejected_count,
                "answer_status": direct_answer.status if direct_answer else "not_found",
            }
        )

        fallback_message = NO_VERIFIED_DATA_STUB if not verified_claims else None
        result = RagSynthesisResult(
            query=query,
            mode=mode,
            cache_hit=False,
            cache_key=cache_key,
            routing=routing,
            verified_claims=verified_claims,
            direct_answer=direct_answer,
            rejected_claims_count=rejected_count,
            sw_alignment_failures=sw_failures,
            fallback_message=fallback_message,
            source_posts_payload=source_posts_payload,
            pipeline_stages=stages,
            total_latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

        if query_emb is not None and verified_claims:
            self.semantic_cache.put_level1(
                cache_key=cache_key,
                query_text=query,
                query_embedding=query_emb,
                response_payload=result.to_cache_payload(),
                is_news_query=(time_window_days <= 7),
                now=current_time,
                is_private=is_private_contour,
                tenant_id=tenant_id,
            )

        self.finops_tracker.record_query(
            user_id,
            used_frontier=(routing.model_tier == "frontier"),
            cache_hit=False,
        )
        return result
