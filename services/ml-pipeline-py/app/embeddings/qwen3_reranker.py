"""Клиент кросс-энкодера `Qwen/Qwen3-Reranker-0.6B` для серой зоны и реранкинга.

Раздел 7.2.3 SPEC.md:
Если композитное сходство $S(P_{\\text{new}}, C_j)$ попадает в серую зону
$[\\tau_{\\text{low}}, \\tau_{\\text{high}}) = [0.72, 0.88)$, $P_{\\text{new}}$ сравнивается
строго с **медоидом** $M_j$ через ``Qwen3-Reranker-0.6B``. Если реранкер выдает
оценку $\\ge 0.85$, пост объединяется с кластером $C_j$.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.embeddings.qwen3_embedder import softmax_yes_no
from app.preprocessing.entities_ner import extract_entities, jaccard_similarity

DEFAULT_DEDUP_RERANK_INSTRUCTION = (
    "Determine whether the two Telegram news posts describe the same specific news event, "
    "decision, or fact with matching entities and quantitative figures (for example rates, "
    "amounts, dates, or versions)."
)

_TOKEN_RE = re.compile(r"[a-zа-яё0-9%._-]+", re.IGNORECASE)
_NUM_UNIT_RE = re.compile(r"\d+(?:[.,]\d+)?%?")


class RerankerBackbone(Protocol):
    """Протокол вычисления релевантности (вероятности [0, 1]) пары текстов."""

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        instruction: str,
    ) -> list[float]: ...


def format_reranker_prompt(query: str, document: str, instruction: str) -> str:
    """Форматирует chat-шаблон Qwen3-Reranker для оценки yes/no."""
    return (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
        f"<Instruct>: {instruction.strip()}\n"
        f"<Query>: {query.strip()}\n"
        f"<Document>: {document.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


class OnnxQwen3RerankerBackbone:
    """ONNX Runtime бэкенд для Qwen3-Reranker-0.6B по логитам токенов ``yes``/``no``."""

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
        self.yes_id = self.tokenizer.token_to_id("yes")
        self.no_id = self.tokenizer.token_to_id("no")
        if self.yes_id is None or self.no_id is None:
            raise ValueError("Tokenizer does not contain 'yes'/'no' token ids required for Qwen3-Reranker")
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self._session_inputs = list(self.session.get_inputs())
        self._output_names = [self.session.get_outputs()[0].name]

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        instruction: str,
    ) -> list[float]:
        from app.embeddings.qwen3_embedder import build_causal_onnx_feed

        prompts = [format_reranker_prompt(q, d, instruction) for q, d in pairs]
        encodings = self.tokenizer.encode_batch(prompts)
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        feed = build_causal_onnx_feed(self._session_inputs, input_ids, attention_mask)
        outputs = self.session.run(self._output_names, feed)
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim == 3:
            last_logits = logits[:, -1, :]
            scores: list[float] = []
            for row in last_logits:
                scores.append(softmax_yes_no(float(row[self.yes_id]), float(row[self.no_id])))
            return scores
        if logits.ndim == 2 and logits.shape[1] == 2:
            return [softmax_yes_no(float(r[1]), float(r[0])) for r in logits]
        if logits.ndim == 2 and logits.shape[1] == 1:
            return [1.0 / (1.0 + math.exp(-float(r[0]))) for r in logits]
        raise ValueError(f"Unexpected reranker ONNX logits shape: {logits.shape}")


class VllmRerankerBackbone:
    """HTTP-клиент к vLLM /score или /v1/rerank."""

    def __init__(self, base_url: str, model_id: str, *, timeout_sec: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_sec = timeout_sec

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        instruction: str,
    ) -> list[float]:
        scores: list[float] = []
        endpoint = f"{self.base_url}/score"
        for q, d in pairs:
            payload = json.dumps(
                {
                    "model": self.model_id,
                    "text_1": f"<Instruct>: {instruction}\n<Query>: {q}",
                    "text_2": d,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            score = float(body["data"][0]["score"])
            scores.append(max(0.0, min(1.0, score)))
        return scores


class TransformersQwen3RerankerBackbone:
    """Адаптер HuggingFace ``transformers`` для `Qwen/Qwen3-Reranker-0.6B` (логиты yes/no)."""

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen3-Reranker-0.6B",
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
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else {}
        self.yes_id = vocab.get("yes") if isinstance(vocab, dict) else None
        self.no_id = vocab.get("no") if isinstance(vocab, dict) else None
        if self.yes_id is None:
            self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.no_id is None:
            self.no_id = self.tokenizer.convert_tokens_to_ids("no")

        if model is not None:
            self.model = model
        else:
            if not transformers.is_torch_available():
                raise RuntimeError(
                    "transformers AutoModelForCausalLM требует установленный PyTorch; "
                    "используйте ONNX/vLLM бэкенд или передайте кастомный model."
                )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id_or_path).to(device)
            self.model.eval()

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        instruction: str,
    ) -> list[float]:
        prompts = [format_reranker_prompt(q, d, instruction) for q, d in pairs]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        import transformers

        if transformers.is_torch_available() and hasattr(self.model, "parameters"):
            import torch

            tensor_inputs = {k: torch.as_tensor(v, device=self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.model(**tensor_inputs).logits[:, -1, :].detach().cpu().numpy()
        else:
            outputs = self.model(**encoded)
            logits = np.asarray(getattr(outputs, "logits", outputs), dtype=np.float32)[:, -1, :]

        return [softmax_yes_no(float(row[self.yes_id]), float(row[self.no_id])) for row in logits]


class HeuristicFactRerankerBackbone:
    """Детерминированный локальный скорер для unit-тестов и dev без GPU.

    Сравнивает сущности, количественные значения (проценты/ставки/числа) и
    лемматизированные основы слов, штрафуя расхождение ключевых чисел (например 21% vs 19%).
    """

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        instruction: str,
    ) -> list[float]:
        _ = instruction
        results: list[float] = []
        for query, doc in pairs:
            ent_q_obj = extract_entities(query)
            ent_q = ent_q_obj.all_canonical
            ent_d = extract_entities(doc).all_canonical
            j_ent = jaccard_similarity(ent_q, ent_d)

            nums_q = set(_NUM_UNIT_RE.findall(query))
            nums_d = set(_NUM_UNIT_RE.findall(doc))
            if nums_q and nums_d and nums_q != nums_d and nums_q.isdisjoint(nums_d):
                # Прямой конфликт количественных значений (например 21% против 19%).
                results.append(round(0.18 + 0.25 * j_ent, 4))
                continue

            stems_q = {t[:4] for t in _TOKEN_RE.findall(query.lower()) if len(t) >= 3}
            stems_d = {t[:4] for t in _TOKEN_RE.findall(doc.lower()) if len(t) >= 3}
            j_stem = jaccard_similarity(stems_q, stems_d)
            q_recall = len(stems_q & stems_d) / max(1, len(stems_q))
            effective_stem = max(j_stem, 0.75 * q_recall)

            if nums_q and nums_q == nums_d:
                num_bonus = 0.25
            elif nums_q and nums_d and (nums_q.issubset(nums_d) or nums_d.issubset(nums_q)):
                num_bonus = 0.22
            elif nums_q and nums_d:
                num_bonus = 0.15 * jaccard_similarity(nums_q, nums_d)
            else:
                num_bonus = 0.0

            q_lower = query.lower()
            has_tech_trigger = bool(
                ent_q_obj.tickers
                or ent_q_obj.law_ids
                or ent_q_obj.cli_flags
                or ent_q_obj.numbers
                or any("." in n for n in ent_q_obj.ner)
                or "ошибк" in q_lower
                or "error" in q_lower
            )
            code_bonus = 0.12 if ("[Code]" in doc and has_tech_trigger) else 0.0

            raw_score = 0.45 * j_ent + 0.35 * effective_stem + num_bonus + code_bonus + 0.12
            results.append(round(max(0.0, min(0.99, raw_score)), 4))
        return results


class Qwen3RerankerClient:
    """Клиент `Qwen/Qwen3-Reranker-0.6B`."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backbone: RerankerBackbone | None = None,
        scorer_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._scorer_fn = scorer_fn
        if backbone is not None:
            self.backbone = backbone
            self.backend_name = type(backbone).__name__
        elif scorer_fn is not None:
            self.backbone = HeuristicFactRerankerBackbone()
            self.backend_name = "custom_callable"
        else:
            self.backbone, self.backend_name = self._init_default_backbone()

    def _init_default_backbone(self) -> tuple[RerankerBackbone, str]:
        onnx_path = self.settings.reranker_onnx_path
        tok_path = self.settings.reranker_tokenizer_path
        if onnx_path and tok_path and Path(onnx_path).exists() and Path(tok_path).exists():
            return OnnxQwen3RerankerBackbone(onnx_path, tok_path), "onnxruntime"
        if self.settings.reranker_vllm_url:
            return (
                VllmRerankerBackbone(
                    self.settings.reranker_vllm_url,
                    self.settings.reranker_model_id,
                ),
                "vllm",
            )
        return HeuristicFactRerankerBackbone(), "heuristic_fact"

    def score_batch(
        self,
        query_text: str,
        documents: Sequence[str],
        *,
        instruction: str = DEFAULT_DEDUP_RERANK_INSTRUCTION,
    ) -> list[float]:
        docs = list(documents)
        if not docs:
            return []
        if self._scorer_fn is not None:
            return [max(0.0, min(1.0, float(self._scorer_fn(query_text, doc)))) for doc in docs]
        pairs = [(query_text, doc) for doc in docs]
        return self.backbone.score_pairs(pairs, instruction=instruction)

    def score_pair(
        self,
        query_text: str,
        document_text: str,
        *,
        instruction: str = DEFAULT_DEDUP_RERANK_INSTRUCTION,
    ) -> float:
        """Оценивает вероятность [0, 1], что два поста относятся к одному событию."""
        return self.score_batch(query_text, [document_text], instruction=instruction)[0]
