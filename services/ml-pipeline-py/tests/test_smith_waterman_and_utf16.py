"""Unit-тесты локального выравнивания Смита — Ватермана и маппинга UTF-16 (раздел 7.4.1)."""

from __future__ import annotations

from app.preprocessing.offsets import build_utf16_offset_map
from app.rag.smith_waterman import reversible_normalize, smith_waterman_align


def test_spec_741_rocket_emoji_and_em_dash_alignment() -> None:
    post_text = "🚀 Сбой в ЦОД: потеряно 14% пакетов из-за BGP."
    quote_from_llm = "потеряно 14% пакетов из—за BGP"  # длинное тире U+2014

    omap = build_utf16_offset_map(post_text, post_id=701)
    assert omap.cp_to_utf16[0] == 0
    assert omap.cp_to_utf16[1] == 2  # 🚀 (U+1F680) занимает 2 UTF-16 code units

    alignment = smith_waterman_align(
        post_text,
        quote_from_llm,
        offset_map=omap,
        post_id=701,
        sentence_id=1,
    )
    assert alignment is not None
    assert alignment.post_id == 701
    assert alignment.sentence_id == 1
    assert alignment.start_cp == 14
    assert alignment.end_cp == 44
    assert alignment.utf16_range == (15, 45)
    assert alignment.similarity_score >= 0.85
    assert alignment.verbatim_text == "потеряно 14% пакетов из-за BGP"
    assert omap.slice_by_utf16(*alignment.utf16_range) == "потеряно 14% пакетов из-за BGP"


def test_reversible_normalize_yo_quotes_and_affine_gap() -> None:
    post_text = "🔥 В «СберБанке»   зафиксирован жесткий рост трафика — до 240 Гбит/с на ядре."
    # Цитата с «ё», без кавычек-елочек и с коротким пропуском слова при аффинном штрафе
    quote_from_llm = "В СберБанке зафиксирован жёсткий трафика - до 240 Гбит/с"

    norm = reversible_normalize(post_text)
    assert "е" in norm.norm_text
    assert "«" not in norm.norm_text
    assert "—" not in norm.norm_text

    omap = build_utf16_offset_map(post_text, post_id=701)
    alignment = smith_waterman_align(post_text, quote_from_llm, offset_map=omap, post_id=702)
    assert alignment is not None
    assert alignment.similarity_score >= 0.85
    assert "СберБанке" in alignment.verbatim_text
    assert "240 Гбит/с" in alignment.verbatim_text
    assert omap.slice_by_utf16(*alignment.utf16_range) == alignment.verbatim_text


def test_multiple_astral_emojis_shift_utf16_offsets_accurately() -> None:
    post_text = "🚀🦀🛑 Авария на узле Frankfurt-1 полностью устранена инженерами."
    quote_from_llm = "Авария на узле Frankfurt-1 полностью устранена"

    omap = build_utf16_offset_map(post_text, post_id=701)
    alignment = smith_waterman_align(post_text, quote_from_llm, offset_map=omap, post_id=703)
    assert alignment is not None
    assert alignment.exact_match is True
    assert alignment.utf16_range[0] - alignment.start_cp == 3
    assert alignment.utf16_range[1] - alignment.end_cp == 3
    assert omap.slice_by_utf16(*alignment.utf16_range) == quote_from_llm


def test_fabricated_quote_is_rejected() -> None:
    post_text = "🚀 Сбой в ЦОД: потеряно 14% пакетов из-за BGP."
    fabricated_quote = "уничтожено 99% серверов в Токио из-за DNS"

    alignment = smith_waterman_align(post_text, fabricated_quote, post_id=704)
    assert alignment is None
