"""Ступень 2.5 воронки комментариев: спасение одиночных баг-репортов и корректировок.

Раздел 7.2.4 SPEC.md:
Если одиночный комментарий-лист получил ``GraphScore == 0`` (на него никто не ответил),
он не отбрасывается вслепую: регулярные маркеры ошибок и легкий классификатор
``deepvk/RuModernBERT-small`` проверяют, не является ли он важным баг-репортом,
предупреждением о регрессии, воркэраундом или фактической поправкой к посту.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from app.comments.graph_pruner import ScoredComment
from app.config import Settings, get_settings

_ERROR_STACKTRACE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:traceback\s*\(most recent call|file\s+\"[^\"]+\",\s*line\s+\d+)", re.IGNORECASE), "stacktrace"),
    (re.compile(r"\b(?:http\s*)?(?:500|502|503|504|429|403)\b(?:\s+(?:error|bad gateway|timeout))?", re.IGNORECASE), "http_status_error"),
    (re.compile(r"\b(?:oom|out of memory|segfault|sigsegv|deadlock|panic|exception|nullpointer|keyerror|valueerror|typeerror)\b", re.IGNORECASE), "runtime_exception"),
    (re.compile(r"(?:не\s+работает|падает\s+с|крашит(?:ся)?|сломал(?:и|ось)|утечк[аи]\s+памяти|регресси[яи]|отвалил(?:ось|ась)|баг\b|фикс\b|workaround|обходной\s+путь|откат(?:ились|ить))", re.IGNORECASE), "bug_or_workaround_ru"),
    (re.compile(r"(?:опечатк[аеу]|в\s+посте\s+ошибк|автор\s+перепутал|на\s+самом\s+деле\s+(?:не|это))", re.IGNORECASE), "factual_correction"),
)


class RescueClassifierBackend(Protocol):
    """Протокол классификатора спасения комментариев (0..1)."""

    def score_bug_report(self, texts: Sequence[str]) -> list[float]: ...


class OnnxRuModernBertSmallBackend:
    """ONNX Runtime инференс для `deepvk/RuModernBERT-small`."""

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_path: str | Path,
        *,
        max_length: int = 256,
        providers: Sequence[str] | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(direction="right", pad_id=0)
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )

    def score_bug_report(self, texts: Sequence[str]) -> list[float]:
        encodings = self.tokenizer.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        logits = np.asarray(
            self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0],
            dtype=np.float32,
        )
        scores: list[float] = []
        for row in logits:
            if row.shape[0] >= 2:
                # Бинарная голова [noise, insightful_bug_or_correction]
                m = float(max(row[0], row[1]))
                e0 = math.exp(float(row[0]) - m)
                e1 = math.exp(float(row[1]) - m)
                scores.append(round(e1 / (e0 + e1), 4))
            else:
                scores.append(round(1.0 / (1.0 + math.exp(-float(row[0]))), 4))
        return scores


class TransformersBugRescueBackend:
    """Адаптер ``transformers`` для `deepvk/RuModernBERT-small` (спасение баг-репортов)."""

    def __init__(
        self,
        model_id_or_path: str = "deepvk/RuModernBERT-small",
        *,
        max_length: int = 256,
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
            else transformers.AutoTokenizer.from_pretrained(model_id_or_path)
        )
        if model is not None:
            self.model = model
        else:
            if not transformers.is_torch_available():
                raise RuntimeError(
                    "transformers AutoModelForSequenceClassification требует PyTorch; "
                    "используйте ONNX/coldstart бэкенд или передайте кастомный model."
                )
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_id_or_path
            ).to(device)
            self.model.eval()

    def score_bug_report(self, texts: Sequence[str]) -> list[float]:
        encoded = self.tokenizer(
            list(texts),
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
                logits = self.model(**tensor_inputs).logits.detach().cpu().numpy()
        else:
            outputs = self.model(**encoded)
            logits = np.asarray(getattr(outputs, "logits", outputs), dtype=np.float32)

        scores: list[float] = []
        for row in logits:
            if row.shape[0] >= 2:
                m = float(max(row[0], row[1]))
                e0 = math.exp(float(row[0]) - m)
                e1 = math.exp(float(row[1]) - m)
                scores.append(round(e1 / (e0 + e1), 4))
            else:
                scores.append(round(1.0 / (1.0 + math.exp(-float(row[0]))), 4))
        return scores


class LexicalRuModernBertSmallFallback:
    """Офлайн cold-start адаптер для `deepvk/RuModernBERT-small` без загрузки весов."""

    _POSITIVE_CUES = (
        "воспроизвел",
        "проверил",
        "баг",
        "ошибк",
        "флаг",
        "конфиг",
        "обновил",
        "откат",
        "workaround",
        "помогло",
        "упал",
        "висит",
        "memory",
        "timeout",
        "версии",
        "после обновления",
    )

    def score_bug_report(self, texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for txt in texts:
            lower = txt.lower()
            hits = sum(1 for cue in self._POSITIVE_CUES if cue in lower)
            logit = -1.1 + 0.85 * hits
            prob = 1.0 / (1.0 + math.exp(-logit))
            scores.append(round(prob, 4))
        return scores


@dataclass(slots=True)
class RescueVerdict:
    rescued: bool
    score: float
    reasons: list[str] = field(default_factory=list)


class BugReportRescuer:
    """Спасение полезных листьев с ``GraphScore == 0`` (регулярные маркеры + RuModernBERT-small)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: RescueClassifierBackend | None = None,
        model_threshold: float = 0.55,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_threshold = model_threshold
        if backend is not None:
            self.backend = backend
            self.model_used = type(backend).__name__
        else:
            onnx_path = self.settings.comment_rescue_onnx_path
            tok_path = self.settings.comment_rescue_tokenizer_path
            if onnx_path and tok_path and Path(onnx_path).exists() and Path(tok_path).exists():
                self.backend = OnnxRuModernBertSmallBackend(onnx_path, tok_path)
                self.model_used = self.settings.comment_rescue_model_id
            else:
                self.backend = LexicalRuModernBertSmallFallback()
                self.model_used = f"{self.settings.comment_rescue_model_id}:coldstart"

    def evaluate_texts(self, texts: Sequence[str]) -> list[RescueVerdict]:
        if not texts:
            return []
        model_scores = self.backend.score_bug_report(texts)
        verdicts: list[RescueVerdict] = []
        for text, model_score in zip(texts, model_scores):
            reasons: list[str] = []
            for pattern, label in _ERROR_STACKTRACE_PATTERNS:
                if pattern.search(text):
                    reasons.append(f"regex:{label}")
            if model_score >= self.model_threshold:
                reasons.append(f"model:{self.model_used}")
            rescued = bool(reasons)
            final_score = max(model_score, 0.90 if any(r.startswith("regex:") for r in reasons) else model_score)
            verdicts.append(
                RescueVerdict(
                    rescued=rescued,
                    score=round(final_score, 4),
                    reasons=reasons,
                )
            )
        return verdicts

    def rescue_zero_graph_comments(
        self,
        scored_comments: Sequence[ScoredComment],
    ) -> list[ScoredComment]:
        """Оставляет комментарии с ``GraphScore > 0`` или спасенные баг-репорты."""
        zero_indices: list[int] = [
            idx for idx, c in enumerate(scored_comments) if c.graph_score <= 0.0
        ]
        zero_texts = [scored_comments[idx].text for idx in zero_indices]
        verdicts = self.evaluate_texts(zero_texts)
        verdict_by_idx = dict(zip(zero_indices, verdicts))

        retained: list[ScoredComment] = []
        for idx, item in enumerate(scored_comments):
            if item.graph_score > 0.0:
                retained.append(item)
                continue
            verdict = verdict_by_idx[idx]
            if verdict.rescued:
                item.is_rescued_bug_report = True
                item.rescue_score = verdict.score
                item.rescue_reasons = verdict.reasons
                retained.append(item)
        return retained
