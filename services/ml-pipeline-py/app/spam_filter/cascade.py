"""Оркестратор трехступенчатого каскада спам-фильтрации L0 -> L1 -> L2 (раздел 7.2.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from app.config import Settings, get_settings
from app.spam_filter.features import (
    MentionGraph14d,
    StructuralSpamFeatures,
    extract_structural_features,
)
from app.spam_filter.l0_deterministic import (
    MediaPhashIndex,
    MinHashLshCreativeIndex,
    SimhashBlocklistIndex,
    run_l0_filter,
)
from app.spam_filter.l2_geracl_nli import L2SpamClassifier

AD_CLASSES = frozenset({"native_ad", "mutual_pr", "crypto_scam"})


@dataclass(slots=True)
class SpamVerdict:
    """Итоговое решение каскада спам-фильтрации для поста.

    Рекламные посты (`is_ad=True`) обязательно сохраняются в БД и индексе
    (`should_persist=True`), но помечаются флагом `is_ad=True` для фильтрации
    в пользовательских дайджестах.
    """

    spam_class: str
    is_ad: bool
    is_noise: bool
    should_persist: bool
    stage_decided: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    l1_features: StructuralSpamFeatures | None = None
    l2_probabilities: dict[str, float] = field(default_factory=dict)
    text_simhash: int = 0
    homoglyph_folded_text: str = ""


class SpamFilterCascade:
    """Каскадный фильтр спама/рекламы/ВП (L0 -> L1 -> L2)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        phash_index: MediaPhashIndex | None = None,
        simhash_index: SimhashBlocklistIndex | None = None,
        minhash_index: MinHashLshCreativeIndex | None = None,
        mention_graph: MentionGraph14d | None = None,
        l2_classifier: L2SpamClassifier | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.phash_index = phash_index if phash_index is not None else MediaPhashIndex()
        self.simhash_index = simhash_index if simhash_index is not None else SimhashBlocklistIndex()
        self.minhash_index = minhash_index
        self.mention_graph = mention_graph if mention_graph is not None else MentionGraph14d()
        self.l2_classifier = l2_classifier if l2_classifier is not None else L2SpamClassifier(self.settings)

    def evaluate(
        self,
        text: str,
        *,
        entities: Sequence[Mapping[str, Any]] = (),
        source_channel: str = "",
        published_at: datetime | None = None,
        media_phashes: Sequence[int | str] = (),
        has_informative_media: bool = False,
        record_in_graph: bool = True,
    ) -> SpamVerdict:
        # Ступень L0: regex (erid/ОРД/UTM/реф-боты), омоглифы, длина < 15, pHash, SimHash
        l0 = run_l0_filter(
            text,
            min_text_length=self.settings.min_organic_text_length,
            media_phashes=media_phashes,
            phash_index=self.phash_index,
            phash_hamming_threshold=self.settings.phash_hamming_threshold,
            simhash_index=self.simhash_index,
            minhash_index=self.minhash_index,
            has_informative_media=has_informative_media,
        )
        if l0.decided:
            spam_class = l0.spam_class or "native_ad"
            is_ad = l0.is_ad or (spam_class in AD_CLASSES)
            return SpamVerdict(
                spam_class=spam_class,
                is_ad=is_ad,
                is_noise=l0.is_noise,
                should_persist=not l0.is_noise,
                stage_decided="L0",
                confidence=1.0,
                reasons=l0.reasons,
                text_simhash=l0.text_simhash,
                homoglyph_folded_text=l0.homoglyph_folded_text,
            )

        # Ступень L1: структурные + графовые (14д, +-48ч) признаки
        l1 = extract_structural_features(
            text,
            entities=entities,
            source_channel=source_channel,
            published_at=published_at,
            mention_graph=self.mention_graph,
            record_in_graph=record_in_graph,
        )

        # Порог быстрого пропуска L1 < 0.25 -> чистый органический контент
        if l1.l1_score < self.settings.spam_l1_low_threshold:
            return SpamVerdict(
                spam_class="organic",
                is_ad=False,
                is_noise=False,
                should_persist=True,
                stage_decided="L1",
                confidence=round(1.0 - l1.l1_score, 4),
                reasons=l1.reasons,
                l1_features=l1,
                text_simhash=l0.text_simhash,
                homoglyph_folded_text=l0.homoglyph_folded_text,
            )

        # Порог явного L1 >= 0.80 (например, взаимный пиар за 48ч + хвост + призыв)
        if l1.l1_score >= self.settings.spam_l1_high_threshold:
            high_class = "mutual_pr" if l1.is_reciprocal_48h else "native_ad"
            return SpamVerdict(
                spam_class=high_class,
                is_ad=True,
                is_noise=False,
                should_persist=True,
                stage_decided="L1",
                confidence=l1.l1_score,
                reasons=l1.reasons,
                l1_features=l1,
                text_simhash=l0.text_simhash,
                homoglyph_folded_text=l0.homoglyph_folded_text,
            )

        # Серая зона L1 [0.25, 0.80) -> нейросетевая классификация L2 (`GeRaCl-USER2-base` / `RuModernBERT-base`)
        l2 = self.l2_classifier.classify(text, l1)
        is_ad = l2.spam_class in AD_CLASSES
        reasons = list(l1.reasons)
        reasons.append(f"l2:{l2.model_used}:{l2.spam_class}")
        return SpamVerdict(
            spam_class=l2.spam_class,
            is_ad=is_ad,
            is_noise=False,
            should_persist=True,
            stage_decided="L2",
            confidence=l2.confidence,
            reasons=reasons,
            l1_features=l1,
            l2_probabilities=l2.probabilities,
            text_simhash=l0.text_simhash,
            homoglyph_folded_text=l0.homoglyph_folded_text,
        )
