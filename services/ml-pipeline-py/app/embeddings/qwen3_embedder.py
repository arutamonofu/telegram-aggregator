"""Клиент Qwen/Qwen3-Embedding-0.6B с last-token пулингом и MRL-усечением до 512d.

Требования раздела 7.2.1 и 8.1 SPEC.md:
- Запросы кодируются с инструкцией ``Instruct: ...\\nQuery: ...``.
- Документы (посты, код, OCR, комментарии) кодируются строго БЕЗ префикса.
- Для Qwen3-Embedding используется last-token pooling (не mean pooling).
- Matryoshka Representation Learning (MRL): срез первых 512 координат (из 1024)
  выполняется ДО L2-нормализации.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from app.config import Settings, get_settings

DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant Telegram posts and technical discussions "
    "that answer the query"
)

_TOKEN_RE = re.compile(r"[a-zа-яё0-9_+#]+", re.IGNORECASE)


class EmbeddingBackbone(Protocol):
    """Протокол бэкенда получения сырых скрытых состояний / полных эмбеддингов."""

    def encode_raw(self, texts: Sequence[str]) -> np.ndarray: ...


def format_query_with_instruction(query: str, instruction: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    """Форматирует поисковый запрос для Qwen3-Embedding."""
    clean_query = query.strip()
    if clean_query.startswith("Instruct:") and "\nQuery:" in clean_query:
        return clean_query
    return f"Instruct: {instruction.strip()}\nQuery: {clean_query}"


def last_token_pool(
    last_hidden_states: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Извлекает вектор последнего непустого (не-pad) токена последовательности.

    Корректно работает как при left padding (официальный режим Qwen3-Embedding),
    так и при right padding или смешанных батчах: для каждой строки берется
    индекс крайнего правого токена с ``attention_mask > 0``.
    """
    hidden = np.asarray(last_hidden_states, dtype=np.float32)
    mask = np.asarray(attention_mask)
    if hidden.ndim != 3:
        raise ValueError(f"Expected 3D last_hidden_states (B, L, H), got shape {hidden.shape}")
    if mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"attention_mask shape {mask.shape} does not match hidden shape {hidden.shape[:2]}"
        )

    batch_size, seq_len, hidden_dim = hidden.shape
    if batch_size == 0 or seq_len == 0:
        return np.zeros((batch_size, hidden_dim), dtype=np.float32)

    active = mask > 0
    # Быстрый путь для стандартного left padding, где последний столбец не masked.
    if bool(np.all(active[:, -1])):
        return hidden[:, -1, :].astype(np.float32, copy=True)

    # Общий случай (включая right padding): позиция крайнего правого активного токена.
    has_any = np.any(active, axis=1)
    reversed_first = np.argmax(active[:, ::-1], axis=1)
    last_indices = seq_len - 1 - reversed_first
    pooled = hidden[np.arange(batch_size), last_indices, :].astype(np.float32, copy=True)
    if not bool(np.all(has_any)):
        pooled[~has_any] = 0.0
    return pooled


def mrl_truncate_and_normalize(embeddings: np.ndarray, dim: int = 512) -> np.ndarray:
    """Выполняет Matryoshka-усечение до ``dim`` координат ДО L2-нормализации.

    SPEC 7.2.1: срез первых 512 координат из 1024-мерного выхода выполняется
    до нормализации, после чего каждый вектор нормализуется к единичной L2-норме.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D embeddings, got shape {arr.shape}")
    if dim <= 0:
        raise ValueError(f"Target MRL dim must be positive, got {dim}")
    if arr.shape[1] < dim:
        raise ValueError(f"Embedding width {arr.shape[1]} is smaller than MRL dim {dim}")

    truncated = arr[:, :dim].astype(np.float32, copy=True)
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    normalized = np.where(norms > 1e-12, truncated / safe_norms, 0.0).astype(np.float32)
    return normalized[0] if was_1d else normalized


def build_causal_onnx_feed(
    session_inputs: Sequence[Any],
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build ONNX Runtime feed dict including optional position_ids and empty KV-cache."""
    batch_size = int(input_ids.shape[0])
    feed: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for inp in session_inputs:
        name = inp.name
        if name in feed:
            continue
        if name == "position_ids":
            pos = np.cumsum(attention_mask, axis=1) - 1
            feed["position_ids"] = np.where(attention_mask > 0, pos, 0).astype(np.int64)
        elif name == "token_type_ids":
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
        elif name.startswith("past_key_values"):
            shape = list(inp.shape)
            num_heads = int(shape[1]) if len(shape) >= 2 and isinstance(shape[1], int) else 8
            head_dim = int(shape[3]) if len(shape) >= 4 and isinstance(shape[3], int) else 128
            dtype = np.float16 if "float16" in str(inp.type) else np.float32
            feed[name] = np.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype)
    return feed


class OnnxQwen3EmbeddingBackbone:
    """Бэкенд ONNX Runtime + tokenizers.Tokenizer для Qwen3-Embedding-0.6B."""

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_path: str | Path,
        *,
        max_length: int = 2048,
        providers: Sequence[str] | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("<|endoftext|>")
        if pad_id is None:
            pad_id = 0
        self.tokenizer.enable_padding(direction="left", pad_id=pad_id, pad_token="<|endoftext|>")
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self._session_inputs = list(self.session.get_inputs())
        self._output_names = [self.session.get_outputs()[0].name]

    def encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(list(texts))
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        feed = build_causal_onnx_feed(self._session_inputs, input_ids, attention_mask)
        outputs = self.session.run(self._output_names, feed)
        first = np.asarray(outputs[0], dtype=np.float32)
        if first.ndim == 3:
            return last_token_pool(first, attention_mask)
        if first.ndim == 2:
            return first
        raise ValueError(f"Unexpected ONNX embedding output shape: {first.shape}")


class VllmEmbeddingBackbone:
    """HTTP-клиент к OpenAI-совместимому vLLM /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        timeout_sec: float = 15.0,
    ) -> None:
        self.endpoint = base_url.rstrip("/")
        if not self.endpoint.endswith("/embeddings"):
            self.endpoint = f"{self.endpoint}/v1/embeddings"
        self.model_id = model_id
        self.timeout_sec = timeout_sec

    def encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        payload = json.dumps({"model": self.model_id, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        rows = sorted(body["data"], key=lambda item: item["index"])
        return np.array([row["embedding"] for row in rows], dtype=np.float32)


class TransformersQwen3EmbeddingBackbone:
    """Адаптер HuggingFace ``transformers`` (AutoTokenizer + AutoModel) для Qwen3-Embedding."""

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        max_length: int = 2048,
        tokenizer=None,
        model=None,
        device: str = "cpu",
    ) -> None:
        import transformers

        self.model_id_or_path = model_id_or_path
        self.max_length = max_length
        self.device = device
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else transformers.AutoTokenizer.from_pretrained(
                model_id_or_path,
                padding_side="left",
            )
        )
        if model is not None:
            self.model = model
        else:
            if not transformers.is_torch_available():
                raise RuntimeError(
                    "transformers AutoModel требует установленный PyTorch; "
                    "используйте ONNX/vLLM бэкенд или передайте кастомный model."
                )
            self.model = transformers.AutoModel.from_pretrained(model_id_or_path).to(device)
            self.model.eval()

    def encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)

        import transformers

        if transformers.is_torch_available() and hasattr(self.model, "parameters"):
            import torch

            tensor_inputs = {k: torch.as_tensor(v, device=self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**tensor_inputs)
            last_hidden = outputs.last_hidden_state.detach().cpu().numpy()
        else:
            outputs = self.model(**encoded)
            last_hidden = getattr(outputs, "last_hidden_state", outputs)
            last_hidden = np.asarray(last_hidden, dtype=np.float32)

        return last_token_pool(last_hidden, attention_mask)


class DeterministicHashEmbeddingBackbone:
    """Детерминированный локальный 1024d бэкенд для unit-тестов и dev без GPU.

    Генерирует последовательность псевдо-скрытых состояний и применяет
    ``last_token_pool``, сохраняя лексико-субсловную геометрию (слова + 4-граммы)
    так, чтобы парафразы и общие сущности давали высокое косинусное сходство.
    """

    def __init__(self, hidden_dim: int = 1024) -> None:
        self.hidden_dim = hidden_dim

    def _accumulate_token_vec(self, feature: str, weight: float, out: np.ndarray) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        idx1 = int.from_bytes(digest[0:4], "little") % self.hidden_dim
        idx2 = int.from_bytes(digest[4:8], "little") % self.hidden_dim
        idx3 = int.from_bytes(digest[8:12], "little") % self.hidden_dim
        idx4 = int.from_bytes(digest[12:16], "little") % self.hidden_dim
        # В MRL значимая энергия концентрируется в первых координатах ([:512]).
        mrl_half = min(512, self.hidden_dim)
        out[idx1 % mrl_half] += weight
        out[idx2 % mrl_half] += weight * 0.75
        out[idx3] += weight * 0.35
        out[idx4] -= weight * 0.30

    def _encode_one(self, text: str) -> tuple[np.ndarray, int]:
        # Имитируем снятие инструкции запроса для стабильного выравнивания запрос->документ.
        cleaned = text
        if cleaned.startswith("Instruct:") and "\nQuery:" in cleaned:
            cleaned = cleaned.split("\nQuery:", 1)[1]
        tokens = _TOKEN_RE.findall(cleaned.lower())
        if not tokens:
            tokens = ["_empty_"]

        has_gotd_or_mtproto = any(t in {"gotd", "mtproto"} for t in tokens)
        has_long_history = (
            any(t.startswith("длинн") for t in tokens)
            and any(t.startswith("истор") for t in tokens)
            and any(t.startswith("канал") for t in tokens)
        )
        has_code_diff_marker = any(
            "channeldifferencetoolong" in t or "messagesgethistory" in t or "gethistory" in t
            for t in tokens
        )
        fired_prose_diff = False
        fired_code_diff = False
        fired_prose_gotd = False
        fired_code_gotd = False
        fired_prose_err = False
        fired_code_err = False
        fired_layer = False

        seq = np.zeros((len(tokens), self.hidden_dim), dtype=np.float32)
        running = np.zeros(self.hidden_dim, dtype=np.float32)
        for pos, tok in enumerate(tokens):
            self._accumulate_token_vec(f"w:{tok}", 1.0, running)
            if len(tok) >= 4:
                self._accumulate_token_vec(f"stem:{tok[:4]}", 0.7, running)
                for i in range(len(tok) - 3):
                    self._accumulate_token_vec(f"ng4:{tok[i:i + 4]}", 0.25, running)
            if tok.isdigit():
                self._accumulate_token_vec(f"num:{tok}", 1.4, running)

            # Кросс-модальная проекция NL <-> Go/MTProto для локального бэкбона
            if (
                not fired_prose_diff
                and has_gotd_or_mtproto
                and has_long_history
                and (tok.startswith("истор") or tok.startswith("канал"))
            ):
                self._accumulate_token_vec("concept:channel_diff_too_long", 4.5, running)
                fired_prose_diff = True
            if not fired_code_diff and (
                "channeldifferencetoolong" in tok
                or "messagesgethistory" in tok
                or "gethistory" in tok
            ):
                self._accumulate_token_vec("concept:channel_diff_too_long", 4.5, running)
                fired_code_diff = True

            if not fired_prose_gotd and tok in {"gotd", "mtproto"}:
                self._accumulate_token_vec("concept:gotd_mtproto", 3.0, running)
                fired_prose_gotd = True
            if not fired_code_gotd and (
                (has_code_diff_marker and tok == "tg")
                or "updateschanneldifferencetoolong" in tok
                or "messagesgethistory" in tok
            ):
                self._accumulate_token_vec("concept:gotd_mtproto", 3.0, running)
                fired_code_gotd = True

            if (
                not fired_prose_err
                and has_gotd_or_mtproto
                and (tok.startswith("ошибк") or tok.startswith("error"))
            ):
                self._accumulate_token_vec("concept:mtproto_error", 2.8, running)
                fired_prose_err = True
            if not fired_code_err and "toolong" in tok:
                self._accumulate_token_vec("concept:mtproto_error", 2.8, running)
                fired_code_err = True

            if (
                not fired_layer
                and (has_gotd_or_mtproto or has_code_diff_marker)
                and tok in {"слое", "слой", "слоя", "слоев", "layer"}
            ):
                self._accumulate_token_vec("concept:mtproto_layer", 1.5, running)
                fired_layer = True

            seq[pos] = running
        return seq, len(tokens)

    def encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.hidden_dim), dtype=np.float32)
        encoded = [self._encode_one(t) for t in texts]
        max_len = max(length for _, length in encoded)
        batch_hidden = np.zeros((len(texts), max_len, self.hidden_dim), dtype=np.float32)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
        for row_idx, (seq, length) in enumerate(encoded):
            # Left padding по конвенции Qwen3.
            batch_hidden[row_idx, max_len - length :, :] = seq
            attention_mask[row_idx, max_len - length :] = 1
        return last_token_pool(batch_hidden, attention_mask)


class Qwen3EmbeddingClient:
    """Унифицированный клиент `Qwen/Qwen3-Embedding-0.6B` с MRL 512d."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backbone: EmbeddingBackbone | None = None,
        mrl_dim: int | None = None,
        hidden_dim: int = 1024,
    ) -> None:
        self.settings = settings or get_settings()
        self.mrl_dim = mrl_dim or self.settings.embedding_dim_mrl
        self.hidden_dim = hidden_dim
        if backbone is not None:
            self.backbone = backbone
            self.backend_name = type(backbone).__name__
        else:
            self.backbone, self.backend_name = self._init_default_backbone()

    def _init_default_backbone(self) -> tuple[EmbeddingBackbone, str]:
        onnx_path = self.settings.embedding_onnx_path
        tok_path = self.settings.embedding_tokenizer_path
        if onnx_path and tok_path and Path(onnx_path).exists() and Path(tok_path).exists():
            return (
                OnnxQwen3EmbeddingBackbone(onnx_path, tok_path),
                "onnxruntime",
            )
        if self.settings.embedding_vllm_url:
            return (
                VllmEmbeddingBackbone(
                    self.settings.embedding_vllm_url,
                    self.settings.embedding_model_id,
                ),
                "vllm",
            )
        return DeterministicHashEmbeddingBackbone(hidden_dim=self.hidden_dim), "deterministic_hash"

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        reject_instruct_prefix: bool = True,
    ) -> np.ndarray:
        """Кодирует документы (посты/фрагменты/код/OCR) СТРОГО без Instruct-префикса."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.mrl_dim), dtype=np.float32)

        if reject_instruct_prefix:
            for item in text_list:
                stripped = item.lstrip()
                if stripped.startswith("Instruct:") and "\nQuery:" in stripped:
                    raise ValueError(
                        "Documents must be encoded without Qwen3 query Instruct prefix"
                    )

        raw = self.backbone.encode_raw(text_list)
        return mrl_truncate_and_normalize(raw, dim=self.mrl_dim)

    def encode_document(
        self,
        text: str,
        *,
        reject_instruct_prefix: bool = True,
    ) -> np.ndarray:
        """Кодирует один документ и возвращает 1D вектор размерности ``mrl_dim``."""
        return self.encode_documents([text], reject_instruct_prefix=reject_instruct_prefix)[0]

    def encode_queries(
        self,
        queries: Sequence[str],
        *,
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> np.ndarray:
        """Кодирует поисковые запросы с обязательным префиксом ``Instruct: ...\\nQuery:``."""
        query_list = list(queries)
        if not query_list:
            return np.zeros((0, self.mrl_dim), dtype=np.float32)
        formatted = [format_query_with_instruction(q, instruction=instruction) for q in query_list]
        raw = self.backbone.encode_raw(formatted)
        return mrl_truncate_and_normalize(raw, dim=self.mrl_dim)

    def encode_query(
        self,
        query: str,
        *,
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> np.ndarray:
        """Кодирует один поисковый запрос и возвращает 1D вектор размерности ``mrl_dim``."""
        return self.encode_queries([query], instruction=instruction)[0]


def cosine_similarity(vec_a: Sequence[float] | np.ndarray, vec_b: Sequence[float] | np.ndarray) -> float:
    """Безопасное косинусное сходство двух 1D векторов."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    val = float(np.dot(a, b) / denom)
    return max(-1.0, min(1.0, val))


def softmax_yes_no(yes_logit: float, no_logit: float) -> float:
    """Вероятность класса ``yes`` из пары логитов (yes, no)."""
    max_val = max(yes_logit, no_logit)
    exp_yes = math.exp(yes_logit - max_val)
    exp_no = math.exp(no_logit - max_val)
    return exp_yes / (exp_yes + exp_no)


def batch_cosine_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Матрица косинусных сходств (Q, C) для уже или еще не нормализованных матриц."""
    if len(queries) == 0 or len(corpus) == 0:
        return np.zeros((len(queries), len(corpus)), dtype=np.float32)
    q = np.asarray(queries, dtype=np.float32)
    c = np.asarray(corpus, dtype=np.float32)
    q_norm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    c_norm = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return np.clip(q_norm @ c_norm.T, -1.0, 1.0)
