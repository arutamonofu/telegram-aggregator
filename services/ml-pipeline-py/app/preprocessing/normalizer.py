"""NFKC-нормализация поста, выделение блоков кода (messageEntityPre) и SHA-256."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.preprocessing.offsets import PostUtf16OffsetMap, build_utf16_offset_map


@dataclass(slots=True)
class PreprocessedPost:
    """Результат шага нормализации и индексации смещений UTF-16."""

    post_id: int
    text_raw: str
    text_nfkc: str
    content_sha256: bytes
    offset_map: PostUtf16OffsetMap
    code_blocks_text: str | None = None
    ocr_text: str | None = None
    entities: list[dict[str, Any]] = field(default_factory=list)


def normalize_text_nfkc(text: str, *, rstrip_trailing: bool = True) -> str:
    """Приводит текст к канонической форме NFKC с удалением замыкающих пробелов."""
    cleaned = text.rstrip() if rstrip_trailing else text
    return unicodedata.normalize("NFKC", cleaned)


def extract_code_blocks_from_entities(
    text_raw: str,
    entities: list[dict[str, Any]] | None,
) -> str | None:
    """Извлекает блоки кода (`pre` и `code`) по UTF-16 координатам Telegram MessageEntity."""
    if not text_raw or not entities:
        return None

    raw_trimmed = text_raw.rstrip()
    raw_map = build_utf16_offset_map(
        raw_trimmed,
        post_id=0,
        normalize_nfkc=False,
        rstrip_trailing=False,
        segment=False,
    )

    chunks: list[str] = []
    # Сначала собираем `pre` блоки, а inline `code` только если они не внутри `pre`
    pre_ranges: list[tuple[int, int]] = []
    for ent in entities:
        if ent.get("type") != "pre":
            continue
        offset = int(ent.get("offset", 0))
        length = int(ent.get("length", 0))
        if length <= 0 or offset < 0:
            continue
        end = min(offset + length, raw_map.utf16_length)
        if offset >= end:
            continue
        pre_ranges.append((offset, end))
        snippet = raw_map.slice_by_utf16(offset, end).strip()
        if snippet:
            lang = (ent.get("language") or "").strip()
            chunks.append(f"[{lang}]\n{snippet}" if lang else snippet)

    for ent in entities:
        if ent.get("type") != "code":
            continue
        offset = int(ent.get("offset", 0))
        length = int(ent.get("length", 0))
        if length <= 0 or offset < 0:
            continue
        end = min(offset + length, raw_map.utf16_length)
        if offset >= end:
            continue
        if any(ps <= offset and end <= pe for ps, pe in pre_ranges):
            continue
        snippet = raw_map.slice_by_utf16(offset, end).strip()
        if len(snippet) >= 3:
            chunks.append(snippet)

    if not chunks:
        return None
    return normalize_text_nfkc("\n\n".join(chunks))


def preprocess_post(
    *,
    post_id: int,
    text_raw: str,
    entities: list[dict[str, Any]] | None = None,
    ocr_text: str | None = None,
) -> PreprocessedPost:
    """Выполняет NFKC-нормализацию, строит UTF-16 карту, клаузы и выделяет код/OCR."""
    text_trimmed = text_raw.rstrip()
    offset_map = build_utf16_offset_map(
        text_trimmed,
        post_id=post_id,
        normalize_nfkc=True,
        rstrip_trailing=False,
        segment=True,
    )
    code_blocks = extract_code_blocks_from_entities(text_trimmed, entities)
    ocr_norm = normalize_text_nfkc(ocr_text) if ocr_text and ocr_text.strip() else None
    content_sha256 = hashlib.sha256(offset_map.text_nfkc.encode("utf-8")).digest()

    return PreprocessedPost(
        post_id=post_id,
        text_raw=text_trimmed,
        text_nfkc=offset_map.text_nfkc,
        content_sha256=content_sha256,
        offset_map=offset_map,
        code_blocks_text=code_blocks,
        ocr_text=ocr_norm or None,
        entities=list(entities or []),
    )
