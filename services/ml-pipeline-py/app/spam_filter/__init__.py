"""Каскад фильтрации спама, нативной рекламы и ВП (разделы 7.2.2 и 8.1)."""

from app.spam_filter.cascade import SpamFilterCascade, SpamVerdict
from app.spam_filter.features import (
    MentionGraph14d,
    StructuralSpamFeatures,
    extract_structural_features,
)
from app.spam_filter.l0_deterministic import (
    L0CheckResult,
    MediaPhashIndex,
    MinHashLshCreativeIndex,
    SimhashBlocklistIndex,
    build_text_minhash,
    compute_image_phash,
    compute_text_simhash,
    normalize_homoglyphs,
    phash_hamming_distance,
    run_l0_filter,
)
from app.spam_filter.l2_geracl_nli import (
    L2ClassifierResult,
    L2SpamClassifier,
    TransformersSequenceClassifierBackend,
)

__all__ = [
    "L0CheckResult",
    "L2ClassifierResult",
    "L2SpamClassifier",
    "TransformersSequenceClassifierBackend",
    "MediaPhashIndex",
    "MentionGraph14d",
    "MinHashLshCreativeIndex",
    "SimhashBlocklistIndex",
    "SpamFilterCascade",
    "SpamVerdict",
    "StructuralSpamFeatures",
    "build_text_minhash",
    "compute_image_phash",
    "compute_text_simhash",
    "extract_structural_features",
    "normalize_homoglyphs",
    "phash_hamming_distance",
    "run_l0_filter",
]
