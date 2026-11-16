"""Ступень L2: нейросетевой классификатор cold-start `GeRaCl-USER2-base` / `RuModernBERT-base`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.spam_filter.features import StructuralSpamFeatures

SPAM_CLASSES: tuple[str, ...] = (
    "organic",
    "native_ad",
    "mutual_pr",
    "clickbait",
    "crypto_scam",
)

# Описания классов для zero-shot / linear-probe вербализатора GeRaCl-USER2-base (раздел 8.1)
CLASS_VERBALIZATIONS: dict[str, str] = {
    "organic": "Информационный технический пост, новостной факт, релиз или аналитика без рекламы",
    "native_ad": "Нативная реклама продукта, курса, сервиса, скидки или платной подписки с призывом перейти",
    "mutual_pr": "Взаимный пиар и партнерская рекомендация другого Telegram-канала с переходом",
    "clickbait": "Кликбейт с недосказанностью и интригой в начале текста и призывом нажать на ссылку",
    "crypto_scam": "Крипто-скам, сомнительные аирдропы, обещание гарантированного дохода и реферальные схемы",
}


class L2ClassifierBackend(Protocol):
    """Протокол классификатора 5 классов спама."""

    def predict_proba(
        self,
        text: str,
        features: StructuralSpamFeatures,
    ) -> dict[str, float]: ...


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    max_v = max(logits.values())
    exps = {k: math.exp(v - max_v) for k, v in logits.items()}
    total = sum(exps.values()) or 1.0
    return {k: round(v / total, 4) for k, v in exps.items()}


class OnnxSequenceClassifierBackend:
    """ONNX Runtime инференс для `deepvk/GeRaCl-USER2-base` или дообученного `RuModernBERT-base`."""

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_path: str | Path,
        *,
        labels: Sequence[str] = SPAM_CLASSES,
        max_length: int = 512,
        providers: Sequence[str] | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.labels = tuple(labels)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(direction="right", pad_id=0)
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )

    def predict_proba(
        self,
        text: str,
        features: StructuralSpamFeatures,
    ) -> dict[str, float]:
        _ = features
        enc = self.tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        outputs = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        raw_logits = np.asarray(outputs[0], dtype=np.float32)[0]
        logit_map = {
            label: float(raw_logits[idx])
            for idx, label in enumerate(self.labels)
            if idx < len(raw_logits)
        }
        return _softmax(logit_map)


class TransformersSequenceClassifierBackend:
    """Адаптер ``transformers`` для `deepvk/GeRaCl-USER2-base` и `RuModernBERT-base`."""

    def __init__(
        self,
        model_id_or_path: str = "deepvk/GeRaCl-USER2-base",
        *,
        labels: Sequence[str] = SPAM_CLASSES,
        max_length: int = 512,
        tokenizer=None,
        model=None,
        device: str = "cpu",
    ) -> None:
        import transformers

        self.model_id_or_path = model_id_or_path
        self.labels = tuple(labels)
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

    def predict_proba(
        self,
        text: str,
        features: StructuralSpamFeatures,
    ) -> dict[str, float]:
        _ = features
        encoded = self.tokenizer(
            [text],
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
                logits = self.model(**tensor_inputs).logits[0].detach().cpu().numpy()
        else:
            outputs = self.model(**encoded)
            logits = np.asarray(getattr(outputs, "logits", outputs), dtype=np.float32)[0]

        logit_map = {
            label: float(logits[idx])
            for idx, label in enumerate(self.labels)
            if idx < len(logits)
        }
        return _softmax(logit_map)


class ColdStartFeatureAndLexicalBackend:
    """Локальный cold-start классификатор для Dev без скачивания весов с HuggingFace.

    Сочетает структурно-графовые признаки L1 с лексическими весами подклассов
    (`native_ad`, `mutual_pr`, `clickbait`, `crypto_scam`, `organic`).
    """

    def predict_proba(
        self,
        text: str,
        features: StructuralSpamFeatures,
    ) -> dict[str, float]:
        lower = text.lower()
        logits: dict[str, float] = {cls: 0.0 for cls in SPAM_CLASSES}

        # Базовый приоритет organic при отсутствии аномалий
        logits["organic"] = 1.15 - 1.3 * features.l1_score

        # mutual_pr: взаимный пиар и рекомендации каналов в хвосте
        if features.is_reciprocal_48h:
            logits["mutual_pr"] += 2.4
        if features.external_targets and features.link_in_last_20_percent:
            logits["mutual_pr"] += 1.05
        if any(w in lower for w in ("рекомендую канал", "подписывайтесь на канал", "коллеги из @")):
            logits["mutual_pr"] += 1.35

        # native_ad: коммерческие офферы, промокоды, курсы, партнерские сервисы
        ad_hits = sum(
            1
            for cue in (
                "скидк",
                "промокод",
                "курс",
                "вебинар",
                "тариф",
                "пробный период",
                "бесплатная консультация",
                "рекламодатель",
                "партнер",
                "регистрируйтесь",
                "заказывай",
            )
            if cue in lower
        )
        logits["native_ad"] += 0.9 * ad_hits
        if features.has_pragmatic_shift:
            logits["native_ad"] += 1.0
        if features.link_in_last_20_percent or features.has_hidden_tme_text_url:
            logits["native_ad"] += 0.75

        # clickbait: катафора в первом абзаце + недосказанность + призыв
        if features.has_first_para_cataphora:
            logits["clickbait"] += 1.85
        if features.link_in_last_20_percent and features.has_first_para_cataphora:
            logits["clickbait"] += 0.85

        # crypto_scam
        scam_hits = sum(
            1
            for cue in ("крипт", "пассивный доход", "гарантир", "сигнал", "токен", "прибыль 100%")
            if cue in lower
        )
        if scam_hits >= 2:
            logits["crypto_scam"] += 2.2

        return _softmax(logits)


@dataclass(slots=True)
class L2ClassifierResult:
    """Результат ступени L2 (`GeRaCl-USER2-base` / `RuModernBERT-base`)."""

    spam_class: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    model_used: str = ""


class L2SpamClassifier:
    """Обертка выбора ONNX / cold-start L2 классификатора."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: L2ClassifierBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if backend is not None:
            self.backend = backend
            self.model_used = type(backend).__name__
        else:
            onnx_path = self.settings.spam_onnx_path
            tok_path = self.settings.spam_tokenizer_path
            if onnx_path and tok_path and Path(onnx_path).exists() and Path(tok_path).exists():
                self.backend = OnnxSequenceClassifierBackend(onnx_path, tok_path)
                self.model_used = self.settings.modernbert_spam_model_id
            else:
                self.backend = ColdStartFeatureAndLexicalBackend()
                self.model_used = f"{self.settings.coldstart_spam_model_id}:coldstart"

    def classify(self, text: str, features: StructuralSpamFeatures) -> L2ClassifierResult:
        probs = self.backend.predict_proba(text, features)
        best_class = max(probs, key=lambda k: probs[k])
        return L2ClassifierResult(
            spam_class=best_class,
            confidence=probs[best_class],
            probabilities=probs,
            model_used=self.model_used,
        )
