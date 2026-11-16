"""Независимый NLI-гейт логического следования (mDeBERTa-v3-xnli / PsiloQA, раздел 7.4.1 SPEC.md)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.preprocessing.entities_ner import extract_entities

NO_VERIFIED_DATA_STUB = "В выбранных каналах нет подтвержденных данных по запросу"

NLI_LABELS: tuple[str, ...] = ("ENTAILMENT", "NEUTRAL", "CONTRADICTS")

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_NUM_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_EMBEDDED_DIGITS_RE = re.compile(r"(?<=[a-zа-яё_])\d+|\d+(?=[a-zа-яё_])", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:не|нет|без|ни|отсутств[а-я]*|невозможн[а-я]*|not|no|never|without)\b", re.IGNORECASE)
_NON_PREDICATE_NEG_IDIOMS_RE = re.compile(
    r"\b(?:"
    r"несмотря\s+на|тем\s+не\s+менее|без\s+ограничени[йя]|без\s+изменени[йя]|"
    r"без\s+проблем|без\s+потерь|без\s+труда|как\s+ни\s+крути|не\s+только|"
    r"не\s+менее|не\s+более|не\s+говоря|не\s+секрет|не\s+случайно|не\s+редко|не\s+раз"
    r")\b",
    re.IGNORECASE,
)
_NEG_FOLLOWING_RE = re.compile(
    r"\b(?:не|нет|без|ни|отсутств[а-я]*|невозможн[а-я]*|not|no|never|without)\s+"
    r"([a-zа-яё0-9]+(?:\s+[a-zа-яё0-9]+)?)",
    re.IGNORECASE,
)
_CITATION_MARKER_RE = re.compile(r"\[(?:c?\d+(?:\s*,\s*c?\d+)*)\]", re.IGNORECASE)
_INDIRECT_META_CLAUSE_RE = re.compile(
    r"^(?:[^.!?]*?\b(?:нет\s+(?:прям[а-я]+\s+)?(?:подтвержден[а-я]*|сведени[а-я]*|сообщени[а-я]*|данн[а-я]*|информаци[а-я]*)|"
    r"не\s+(?:подтвержда[а-я]*|сообща[а-я]*|упомина[а-я]*))[^.!?]*[.!?:;]\s*)+",
    re.IGNORECASE,
)
_DIRECT_ANSWER_META_STEMS = frozenset(
    {
        "исто", "кана", "подк", "пост", "сооб", "данн", "инфо", "подт",
        "прям", "одна", "напр", "теме", "согл", "след", "указ", "отме",
        "прив", "выбр", "осно", "факт", "этом", "этог",
    }
)

_RU_EN_STOPWORDS = frozenset(
    {
        "в", "во", "на", "из", "за", "по", "при", "что", "как", "для", "это", "от", "до",
        "и", "или", "а", "но", "о", "об", "с", "со", "к", "ко", "у", "же", "бы", "ли",
        "то", "так", "уже", "все", "всё", "под", "над", "между", "через", "после", "перед",
        "the", "a", "an", "in", "on", "at", "of", "to", "for", "with", "by", "from", "and", "or", "is", "are",
    }
)

_ANTONYM_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("повыс", "подня", "увелич", "вырос", "рост"), ("сниз", "опуст", "уменьш", "упал", "паден", "сократ")),
    (("ускор", "быстр", "оптимиз"), ("замедл", "тормоз", "деград")),
    (("поддерж", "одобр", "выгод", "полез", "эффектив", "успеш", "эконом"), ("критик", "опасн", "риск", "провал", "вред", "разруш", "убыт")),
    (("безопас", "защищ", "стабиль"), ("уязвим", "сбой", "отказ", "падает", "утеч")),
    (("включ", "разреш", "открыт"), ("отключ", "запрещ", "закрыт", "блокир")),
)

_PRO_POLARITY_CUES = (
    "поддерж", "одобр", "ускор", "эконом", "снижает расход", "снижает затрат", "выгод",
    "эффектив", "улучш", "преимущ", "рекоменд", "удобн", "полезн", "успеш", "стабиль",
    "прозрачн", "защищ", "масштабир", "плюс", "позитив",
)
_CON_POLARITY_CUES = (
    "критик", "риск", "опасн", "замедл", "увеличивает расход", "дорог", "деград",
    "уязвим", "утеч", "недостат", "минус", "проблем", "слома", "падает", "ошибк",
    "сбой", "запрещ", "штраф", "блокир", "негатив", "проигрыв", "вымыван",
)


class NliGateBackend(Protocol):
    """Протокол классификатора логического следования (premise -> hypothesis)."""

    def predict_nli(self, premise: str, hypothesis: str) -> dict[str, float]: ...


@dataclass(slots=True)
class NliCheckResult:
    """Результат проверки одного тезиса относительно дословного спана цитаты."""

    label: str
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    is_entailed: bool
    polarity_valid: bool = True


def _softmax_probs(logits: Sequence[float], labels: Sequence[str]) -> dict[str, float]:
    arr = [float(x) for x in logits]
    max_v = max(arr) if arr else 0.0
    exps = [math.exp(v - max_v) for v in arr]
    total = sum(exps) or 1.0
    out = {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTS": 0.0}
    for idx, val in enumerate(exps):
        if idx >= len(labels):
            continue
        raw_lbl = labels[idx].upper()
        if "ENTAIL" in raw_lbl:
            out["ENTAILMENT"] += val / total
        elif "CONTRA" in raw_lbl:
            out["CONTRADICTS"] += val / total
        else:
            out["NEUTRAL"] += val / total
    return {k: round(v, 4) for k, v in out.items()}


class OnnxNliGateBackend:
    """ONNX Runtime бэкенд для `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`."""

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_path: str | Path,
        *,
        labels: Sequence[str] = NLI_LABELS,
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
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    def predict_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        enc = self.tokenizer.encode(premise, hypothesis)
        feed: dict[str, np.ndarray] = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        logits = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)[0]
        return _softmax_probs(logits, self.labels)


class TransformersNliGateBackend:
    """Адаптер HuggingFace `transformers` для mDeBERTa-v3-xnli."""

    def __init__(
        self,
        model_id_or_path: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        *,
        max_length: int = 512,
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
                    "используйте ONNX или локальный эвристический NLI-бэкенд."
                )
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_id_or_path
            ).to(device)
            self.model.eval()

        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if isinstance(id2label, dict) and id2label:
            self.labels = tuple(str(id2label[i]) for i in sorted(id2label.keys()))
        else:
            self.labels = NLI_LABELS

    def predict_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        encoded = self.tokenizer(
            premise,
            hypothesis,
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

        return _softmax_probs(logits, self.labels)


def _content_stems(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower().replace("ё", "е"))
    stems: set[str] = set()
    for tok in tokens:
        if tok in _RU_EN_STOPWORDS or len(tok) <= 1:
            continue
        if tok.isdigit():
            stems.add(tok)
        elif len(tok) >= 4:
            stems.add(tok[:4])
        else:
            stems.add(tok)
    return stems


def _extract_numeric_tokens(text_norm: str) -> set[str]:
    nums = {m.replace(",", ".") for m in _NUM_TOKEN_RE.findall(text_norm)}
    nums.update(_EMBEDDED_DIGITS_RE.findall(text_norm))
    return nums


def _negated_following_stems(text_clean: str) -> set[str]:
    stems: set[str] = set()
    for match in _NEG_FOLLOWING_RE.finditer(text_clean):
        stems.update(_content_stems(match.group(1)))
    return stems


class HeuristicPsiloQaNliBackend:
    """Детерминированный локальный NLI-оценщик для Dev и unit-тестов без ONNX-весов.

    Проверяет:
    1) Совпадение количественных фактов, процентов, портов, законов и тикеров;
    2) Инверсию отрицаний и антонимических предикатов (CONTRADICTS);
    3) Полноту покрытия ключевых основ и сущностей тезиса цитатой (ENTAILMENT vs NEUTRAL).
    """

    def predict_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        prem_norm = premise.lower().replace("ё", "е").strip()
        hyp_norm = hypothesis.lower().replace("ё", "е").strip()

        if not prem_norm or not hyp_norm:
            return {"ENTAILMENT": 0.05, "NEUTRAL": 0.90, "CONTRADICTS": 0.05}

        # 1. Прямая или почти дословная проверка числовых/сущностных фактов
        prem_nums = _extract_numeric_tokens(prem_norm)
        hyp_nums = _extract_numeric_tokens(hyp_norm)
        if hyp_nums:
            missing_nums = hyp_nums - prem_nums
            if missing_nums:
                # Если тезис приписывает иное число/процент/порт, которого нет в спане
                if prem_nums:
                    return {"ENTAILMENT": 0.04, "NEUTRAL": 0.08, "CONTRADICTS": 0.88}
                return {"ENTAILMENT": 0.18, "NEUTRAL": 0.70, "CONTRADICTS": 0.12}

        ent_prem = extract_entities(premise)
        ent_hyp = extract_entities(hypothesis)
        if ent_hyp.law_ids and not set(ent_hyp.law_ids).issubset(set(ent_prem.law_ids)):
            return {"ENTAILMENT": 0.06, "NEUTRAL": 0.16, "CONTRADICTS": 0.78}
        if ent_hyp.tickers and prem_norm and not set(ent_hyp.tickers).issubset(set(ent_prem.tickers)):
            # Проверим на уровне нижнего регистра для верхнерегистровых тикеров/аббревиатур (напр. BGP);
            # строчные технические токены (gotd и т.п.) не блокируют перефраз без упоминания стека.
            unresolved = [t for t in ent_hyp.tickers if t.isupper() and t.lower() not in prem_norm]
            if unresolved:
                return {"ENTAILMENT": 0.12, "NEUTRAL": 0.68, "CONTRADICTS": 0.20}

        # 2. Проверка антонимов
        for pos_cues, neg_cues in _ANTONYM_PAIRS:
            prem_pos = any(c in prem_norm for c in pos_cues)
            prem_neg = any(c in prem_norm for c in neg_cues)
            hyp_pos = any(c in hyp_norm for c in pos_cues)
            hyp_neg = any(c in hyp_norm for c in neg_cues)
            if (prem_pos and hyp_neg and not prem_neg) or (prem_neg and hyp_pos and not prem_pos):
                return {"ENTAILMENT": 0.03, "NEUTRAL": 0.07, "CONTRADICTS": 0.90}

        # 3. Проверка инверсии отрицания при совпадении темы
        prem_stems = _content_stems(prem_norm)
        hyp_stems = _content_stems(hyp_norm)
        if not hyp_stems:
            return {"ENTAILMENT": 0.10, "NEUTRAL": 0.85, "CONTRADICTS": 0.05}

        shared_stems = hyp_stems & prem_stems
        coverage = len(shared_stems) / len(hyp_stems)

        prem_neg_clean = _NON_PREDICATE_NEG_IDIOMS_RE.sub(" ", prem_norm)
        hyp_neg_clean = _NON_PREDICATE_NEG_IDIOMS_RE.sub(" ", hyp_norm)
        prem_neg_count = len(_NEGATION_RE.findall(prem_neg_clean))
        hyp_neg_count = len(_NEGATION_RE.findall(hyp_neg_clean))
        if coverage >= 0.40 and (prem_neg_count % 2) != (hyp_neg_count % 2):
            introduces_negation = hyp_neg_count > 0 and prem_neg_count == 0
            short_premise = len(prem_stems) <= 14
            negated_shared = bool(
                (_negated_following_stems(prem_neg_clean) | _negated_following_stems(hyp_neg_clean))
                & shared_stems
            )
            if introduces_negation or short_premise or negated_shared:
                return {"ENTAILMENT": 0.05, "NEUTRAL": 0.10, "CONTRADICTS": 0.85}

        # 4. Оценка ENTAILMENT vs NEUTRAL по покрытию основ и сущностей
        if coverage >= 0.55:
            entail_p = round(min(0.97, 0.78 + 0.19 * coverage), 4)
            rem = round(1.0 - entail_p, 4)
            return {
                "ENTAILMENT": entail_p,
                "NEUTRAL": round(rem * 0.85, 4),
                "CONTRADICTS": round(rem * 0.15, 4),
            }
        if coverage >= 0.25:
            return {"ENTAILMENT": 0.34, "NEUTRAL": 0.60, "CONTRADICTS": 0.06}
        return {"ENTAILMENT": 0.10, "NEUTRAL": 0.82, "CONTRADICTS": 0.08}


class NliVerificationGate:
    """Локальный NLI-гейт: пропускает строго `ENTAILMENT` с `p_entail >= 0.75`."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: NliGateBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._heuristic_guard = HeuristicPsiloQaNliBackend()
        if backend is not None:
            self.backend = backend
            self.backend_name = type(backend).__name__
        else:
            onnx_path = self.settings.nli_onnx_path
            tok_path = self.settings.nli_tokenizer_path
            if onnx_path and tok_path and Path(onnx_path).exists() and Path(tok_path).exists():
                self.backend = OnnxNliGateBackend(onnx_path, tok_path)
                self.backend_name = "onnxruntime"
            else:
                self.backend = HeuristicPsiloQaNliBackend()
                self.backend_name = "heuristic_psiloqa"

    def check_pair(self, statement: str, verbatim_text: str) -> NliCheckResult:
        """Проверяет, следует ли тезис `statement` логически из дословной цитаты `verbatim_text`."""
        probs = self.backend.predict_nli(premise=verbatim_text, hypothesis=statement)
        if not isinstance(self.backend, HeuristicPsiloQaNliBackend):
            heur_probs = self._heuristic_guard.predict_nli(premise=verbatim_text, hypothesis=statement)
            if float(heur_probs.get("CONTRADICTS", 0.0)) >= 0.75:
                probs = heur_probs
            elif (
                float(heur_probs.get("ENTAILMENT", 0.0)) >= self.settings.nli_entailment_threshold
                and float(probs.get("CONTRADICTS", 0.0)) < 0.25
            ):
                p_ent_cal = max(
                    float(probs.get("ENTAILMENT", 0.0)),
                    float(heur_probs["ENTAILMENT"]),
                )
                rem = max(0.0, 1.0 - p_ent_cal)
                raw_neu = float(probs.get("NEUTRAL", 0.0))
                raw_con = float(probs.get("CONTRADICTS", 0.0))
                raw_other = raw_neu + raw_con
                if raw_other > 0.0:
                    p_neu_cal = rem * (raw_neu / raw_other)
                    p_con_cal = rem * (raw_con / raw_other)
                else:
                    p_neu_cal = rem * 0.85
                    p_con_cal = rem * 0.15
                probs = {
                    "ENTAILMENT": p_ent_cal,
                    "NEUTRAL": p_neu_cal,
                    "CONTRADICTS": p_con_cal,
                }

        p_ent = float(probs.get("ENTAILMENT", 0.0))
        p_neu = float(probs.get("NEUTRAL", 0.0))
        p_con = float(probs.get("CONTRADICTS", 0.0))

        threshold = self.settings.nli_entailment_threshold
        if p_ent >= threshold and p_ent >= p_neu and p_ent >= p_con:
            label = "ENTAILMENT"
            is_entailed = True
        elif p_con >= p_neu and p_con > p_ent:
            label = "CONTRADICTS"
            is_entailed = False
        else:
            label = "NEUTRAL"
            is_entailed = False

        return NliCheckResult(
            label=label,
            entailment_prob=round(p_ent, 4),
            neutral_prob=round(p_neu, 4),
            contradiction_prob=round(p_con, 4),
            is_entailed=is_entailed,
        )

    def verify_direct_answer(
        self,
        *,
        query: str,
        direct_answer: str,
        answer_status: str,
        verified_statements: Sequence[str],
        verified_quotes: Sequence[str],
    ) -> NliCheckResult:
        """Ступень 2 NLI + Entity Guard: проверяет прямой ответ относительно верифицированных фактов."""
        clean_ans = _CITATION_MARKER_RE.sub(" ", direct_answer).strip()
        if not clean_ans or answer_status == "not_found" or not verified_quotes:
            return NliCheckResult(
                label="NEUTRAL",
                entailment_prob=0.0,
                neutral_prob=1.0,
                contradiction_prob=0.0,
                is_entailed=False,
            )

        combined_premise = " ".join([*verified_quotes, *verified_statements]).strip()
        prem_norm = combined_premise.lower().replace("ё", "е")
        ans_norm = clean_ans.lower().replace("ё", "е")
        q_norm = query.lower().replace("ё", "е")

        # Zero-Hallucination Entity & Number Guard: запрещаем новые числа, законы и тикеры
        allowed_nums = _extract_numeric_tokens(prem_norm) | _extract_numeric_tokens(q_norm)
        ans_nums = _extract_numeric_tokens(ans_norm)
        if ans_nums - allowed_nums:
            return NliCheckResult(
                label="CONTRADICTS",
                entailment_prob=0.04,
                neutral_prob=0.08,
                contradiction_prob=0.88,
                is_entailed=False,
            )

        ent_prem = extract_entities(combined_premise)
        ent_query = extract_entities(query)
        ent_ans = extract_entities(clean_ans)

        allowed_laws = set(ent_prem.law_ids) | set(ent_query.law_ids)
        if ent_ans.law_ids and not set(ent_ans.law_ids).issubset(allowed_laws):
            return NliCheckResult(
                label="CONTRADICTS",
                entailment_prob=0.04,
                neutral_prob=0.08,
                contradiction_prob=0.88,
                is_entailed=False,
            )

        allowed_tickers = {t.lower() for t in ent_prem.tickers} | {t.lower() for t in ent_query.tickers}
        unresolved_tickers = [
            t
            for t in ent_ans.tickers
            if t.isupper() and t.lower() not in allowed_tickers and t.lower() not in prem_norm
        ]
        if unresolved_tickers:
            return NliCheckResult(
                label="CONTRADICTS",
                entailment_prob=0.04,
                neutral_prob=0.08,
                contradiction_prob=0.88,
                is_entailed=False,
            )

        target_hypothesis = clean_ans
        if answer_status == "indirect":
            factual_tail = _INDIRECT_META_CLAUSE_RE.sub("", clean_ans).strip()
            if factual_tail:
                target_hypothesis = factual_tail

        res = self.check_pair(statement=target_hypothesis, verbatim_text=combined_premise)
        if res.is_entailed:
            return res

        if res.label != "CONTRADICTS" and res.contradiction_prob < 0.25:
            prem_stems = _content_stems(prem_norm)
            hyp_stems = _content_stems(target_hypothesis.lower().replace("ё", "е"))
            q_stems = _content_stems(q_norm)
            factual_stems = hyp_stems - _DIRECT_ANSWER_META_STEMS - (q_stems if answer_status == "indirect" else set())
            if not factual_stems:
                factual_stems = hyp_stems - _DIRECT_ANSWER_META_STEMS
            if factual_stems:
                factual_coverage = len(factual_stems & prem_stems) / len(factual_stems)
                if factual_coverage >= 0.55:
                    ent_p = round(min(0.95, 0.76 + 0.18 * factual_coverage), 4)
                    rem = round(max(0.0, 1.0 - ent_p), 4)
                    return NliCheckResult(
                        label="ENTAILMENT",
                        entailment_prob=max(res.entailment_prob, ent_p),
                        neutral_prob=round(rem * 0.85, 4),
                        contradiction_prob=round(rem * 0.15, 4),
                        is_entailed=True,
                    )

        return res

    def check_stance_polarity(self, verbatim_text: str, column_polarity: str) -> bool:
        """Второй инвариант 7.4.3: проверяет отсутствие инверсии полярности относительно колонки PRO/CON."""
        side = column_polarity.strip().upper()
        if side not in ("PRO", "CON"):
            return True

        norm = verbatim_text.lower().replace("ё", "е")
        has_pro_cue = any(cue in norm for cue in _PRO_POLARITY_CUES)
        has_con_cue = any(cue in norm for cue in _CON_POLARITY_CUES)

        # Проверяем на прямую инверсию по лексическим маркерам
        if side == "PRO" and has_con_cue and not has_pro_cue:
            return False
        if side == "CON" and has_pro_cue and not has_con_cue:
            return False

        # Если подключен нейросетевой бэкенд (ONNX/Transformers), сверяем NLI-противоречие с эталонной гипотезой
        if not isinstance(self.backend, HeuristicPsiloQaNliBackend):
            opposite_hyp = (
                "Данный аргумент критикует решение и указывает на негативные последствия."
                if side == "PRO"
                else "Данный аргумент одобряет решение и указывает на положительные преимущества."
            )
            opp_probs = self.backend.predict_nli(premise=verbatim_text, hypothesis=opposite_hyp)
            if float(opp_probs.get("ENTAILMENT", 0.0)) >= self.settings.nli_entailment_threshold:
                return False

        return True
