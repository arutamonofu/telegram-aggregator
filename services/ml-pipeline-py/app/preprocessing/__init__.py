"""Модули предобработки текста: NFKC-нормализация, UTF-16 смещения, сегментация и NER."""

from app.preprocessing.entities_ner import (
    ExtractedEntities,
    extract_entities,
    format_exact_entities_signature,
    has_exact_lexical_triggers,
    jaccard_similarity,
)
from app.preprocessing.normalizer import PreprocessedPost, normalize_text_nfkc, preprocess_post
from app.preprocessing.offsets import (
    PostUtf16OffsetMap,
    SentenceClause,
    build_cp_to_utf16,
    build_utf16_offset_map,
    segment_into_sentences,
)

__all__ = [
    "ExtractedEntities",
    "PostUtf16OffsetMap",
    "PreprocessedPost",
    "SentenceClause",
    "build_cp_to_utf16",
    "build_utf16_offset_map",
    "extract_entities",
    "format_exact_entities_signature",
    "has_exact_lexical_triggers",
    "jaccard_similarity",
    "normalize_text_nfkc",
    "preprocess_post",
    "segment_into_sentences",
]
