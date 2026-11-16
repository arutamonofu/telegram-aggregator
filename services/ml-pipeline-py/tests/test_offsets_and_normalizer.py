"""Тесты NFKC-нормализации, сегментации [P{post_id}_S{sent_id}] и карты UTF-16 смещений."""

from __future__ import annotations

from app.preprocessing import (
    build_utf16_offset_map,
    normalize_text_nfkc,
    preprocess_post,
)


def test_surrogate_pair_rocket_offset_shift_spec_example() -> None:
    """Проверяет пример из раздела 7.4.1 SPEC с символом 🚀 (U+1F680): [14, 44] -> [15, 45]."""
    # Позиция 0 = 🚀 (1 codepoint, 2 UTF-16 units); позиции 1..13 = ' Новый релиз ' (13 chars)
    # Итого префикс длиной 14 codepoints -> 15 UTF-16 code units.
    # Интервал [14, 44) (длина 30 chars) в UTF-16 становится [15, 45).
    text = "🚀 Новый релиз Python версии 3.14 ускорил сбор"
    assert len(text) == 45
    assert text[14:44] == "Python версии 3.14 ускорил сбо"

    offset_map = build_utf16_offset_map(text, post_id=42)
    assert offset_map.has_non_bmp is True
    assert offset_map.cp_span_to_utf16(14, 44) == (15, 45)
    assert offset_map.utf16_span_to_cp(15, 45) == (14, 44)
    assert offset_map.slice_by_utf16(15, 45) == text[14:44]


def test_bidirectional_utf16_with_multiple_non_bmp_emojis() -> None:
    """Проверяет двунаправленность карты при нескольких эмодзи и ZWJ."""
    text = "Старт 🚀 сервиса 🎉 прошел успешно!"
    offset_map = build_utf16_offset_map(text, post_id=7)
    utf16_raw = text.encode("utf-16-le")
    assert offset_map.utf16_length == len(utf16_raw) // 2

    idx_passed_cp = text.index("прошел")
    u_start, u_end = offset_map.cp_span_to_utf16(idx_passed_cp, idx_passed_cp + len("прошел"))
    # Перед "прошел" два суррогатных эмодзи (🚀 и 🎉) -> сдвиг +2
    assert (u_start, u_end) == (idx_passed_cp + 2, idx_passed_cp + len("прошел") + 2)
    assert offset_map.utf16_span_to_cp(u_start, u_end) == (idx_passed_cp, idx_passed_cp + len("прошел"))

    # Попадание внутрь второй половины суррогатной пары для 🚀 (UTF-16 индекс = text.index('🚀') + 1)
    rocket_cp = text.index("🚀")
    assert offset_map.utf16_to_cp_index(rocket_cp + 1, snap_end=False) == rocket_cp
    assert offset_map.utf16_to_cp_index(rocket_cp + 1, snap_end=True) == rocket_cp + 1


def test_nfkc_normalization_sentence_tags_and_code_blocks() -> None:
    """Проверяет NFKC-нормализацию, теги [P{id}_S{n}] и выделение блоков кода (pre)."""
    # Используем лигатуру ﬁ (U+FB01) и полноширинную цифру ２ (U+FF12),
    # плюс сокращение "г." (не должно разрывать предложение) и суррогатный эмодзи перед code-блоком.
    raw = "🔥 Релиз v2026 г. и ﬁx ２.\nСмотрите код:\nprint('hi')"
    code_start_cp = raw.index("print('hi')")
    # Включаем суррогатный сдвиг +1 для UTF-16 offset в Telegram MessageEntity
    entities = [
        {
            "type": "pre",
            "offset": code_start_cp + 1,
            "length": len("print('hi')"),
            "language": "python",
        }
    ]
    pre = preprocess_post(
        post_id=104,
        text_raw=raw,
        entities=entities,
        ocr_text="Err\uff1a 502",
    )

    assert "fix 2" in pre.text_nfkc
    assert pre.ocr_text == normalize_text_nfkc("Err\uff1a 502")
    assert pre.code_blocks_text == "[python]\nprint('hi')"

    clauses = pre.offset_map.sentences
    assert len(clauses) >= 2
    assert clauses[0].tag == "P104_S1"
    assert clauses[1].tag == "P104_S2"
    assert clauses[0].formatted.startswith("[P104_S1] ")
