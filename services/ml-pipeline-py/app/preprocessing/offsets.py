"""Построение двунаправленной карты смещений Python Codepoints <-> UTF-16 и сегментация на клаузы.

Реализует шаг 1 раздела 7.4.1 docs/SPEC.md и п.1 раздела 2.3 docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# Сокращения и паттерны, после точки в которых предложение не разрывается
_ABBREV_BEFORE_DOT_RE = re.compile(
    r"(?:^|[\s(\[«\"])(?:"
    r"г|гг|руб|коп|млн|млрд|трлн|тыс|шт|экз|"
    r"т\.?\s*д|т\.?\s*п|т\.?\s*е|т\.?\s*к|т\.?\s*о|"
    r"б\.?\s*п|п\.?\s*п|и\.?\s*о|в\.?\s*в|н\.?\s*э|"
    r"им|ул|пер|пр|д|кв|стр|корп|ст|ч|п|пп|абз|рис|табл|см|ср|др|"
    r"проф|акад|доц|канд|тел|факс|офис|обл|р-н|пос|с|дер|"
    r"[А-ЯЁA-Z]"
    r")$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class SentenceClause:
    """Пронумерованная клауза/предложение `[P{post_id}_S{sent_id}]` с координатами."""

    sentence_id: int
    tag: str
    start_cp: int
    end_cp: int
    start_utf16: int
    end_utf16: int
    text: str

    @property
    def formatted(self) -> str:
        return f"[{self.tag}] {self.text}"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "sentence_id": self.sentence_id,
            "tag": self.tag,
            "start_cp": self.start_cp,
            "end_cp": self.end_cp,
            "start_utf16": self.start_utf16,
            "end_utf16": self.end_utf16,
            "text": self.text,
        }


@dataclass(slots=True)
class PostUtf16OffsetMap:
    """Двунаправленная карта смещений `char_idx` (Python Unicode Codepoint) <-> `utf16_offset`."""

    post_id: int
    text_nfkc: str
    cp_length: int
    utf16_length: int
    has_non_bmp: bool
    cp_to_utf16: list[int]
    utf16_to_cp: list[int]
    non_bmp_anchor_points: list[dict[str, int]] = field(default_factory=list)
    sentences: list[SentenceClause] = field(default_factory=list)

    def cp_to_utf16_index(self, cp_idx: int) -> int:
        """Преобразует индекс кодовой точки Python в смещение UTF-16 (JS DOM / Telegram)."""
        if cp_idx < 0 or cp_idx > self.cp_length:
            raise IndexError(f"cp_idx={cp_idx} out of bounds [0, {self.cp_length}]")
        return self.cp_to_utf16[cp_idx]

    def utf16_to_cp_index(self, utf16_offset: int, *, snap_end: bool = False) -> int:
        """Преобразует смещение UTF-16 в индекс кодовой точки Python.

        Если `utf16_offset` попадает на вторую половинку суррогатной пары (low surrogate):
        - при `snap_end=False` возвращает начало символа (`cp_idx`);
        - при `snap_end=True` возвращает `cp_idx + 1` (конец символа).
        """
        if utf16_offset < 0 or utf16_offset > self.utf16_length:
            raise IndexError(
                f"utf16_offset={utf16_offset} out of bounds [0, {self.utf16_length}]"
            )
        if utf16_offset == self.utf16_length:
            return self.cp_length
        cp_idx = self.utf16_to_cp[utf16_offset]
        if snap_end and utf16_offset > 0 and self.utf16_to_cp[utf16_offset - 1] == cp_idx:
            return cp_idx + 1
        return cp_idx

    def cp_span_to_utf16(self, start_cp: int, end_cp: int) -> tuple[int, int]:
        """Транслирует полуинтервал `[start_cp, end_cp)` в UTF-16 `[start_utf16, end_utf16)`."""
        if start_cp > end_cp:
            raise ValueError(f"Invalid span: start_cp={start_cp} > end_cp={end_cp}")
        return (self.cp_to_utf16_index(start_cp), self.cp_to_utf16_index(end_cp))

    def utf16_span_to_cp(self, start_utf16: int, end_utf16: int) -> tuple[int, int]:
        """Транслирует UTF-16 полуинтервал `[start_utf16, end_utf16)` в `[start_cp, end_cp)`."""
        if start_utf16 > end_utf16:
            raise ValueError(f"Invalid UTF-16 span: {start_utf16} > {end_utf16}")
        return (
            self.utf16_to_cp_index(start_utf16, snap_end=False),
            self.utf16_to_cp_index(end_utf16, snap_end=True),
        )

    def slice_by_utf16(self, start_utf16: int, end_utf16: int) -> str:
        """Возвращает срез `text_nfkc` по координатам UTF-16."""
        start_cp, end_cp = self.utf16_span_to_cp(start_utf16, end_utf16)
        return self.text_nfkc[start_cp:end_cp]


def build_cp_to_utf16(text: str) -> tuple[list[int], bool]:
    """Строит таблицу `cp_to_utf16` длиной `len(text) + 1` (ARCHITECTURE.md 2.3)."""
    cp_to_utf16 = [0] * (len(text) + 1)
    utf16_pos = 0
    has_non_bmp = False
    for idx, ch in enumerate(text):
        cp_to_utf16[idx] = utf16_pos
        if ord(ch) > 0xFFFF:
            utf16_pos += 2
            has_non_bmp = True
        else:
            utf16_pos += 1
    cp_to_utf16[len(text)] = utf16_pos
    return cp_to_utf16, has_non_bmp


def _is_sentence_boundary(text: str, dot_idx: int) -> bool:
    """Определяет, является ли знак препинания в `dot_idx` концом предложения."""
    ch = text[dot_idx]
    n = len(text)

    if ch in ("!", "?"):
        # Пропускаем серию знаков (?!, !!)
        if dot_idx + 1 < n and text[dot_idx + 1] in ("!", "?", "."):
            return False
        return True

    if ch != ".":
        return False

    # Многоточие `...` — разрываем только на последней точке
    if dot_idx + 1 < n and text[dot_idx + 1] == ".":
        return False

    # Десятичные дроби / IP / версии (21.5%, v0.145.1)
    if (
        dot_idx > 0
        and dot_idx + 1 < n
        and text[dot_idx - 1].isdigit()
        and text[dot_idx + 1].isdigit()
    ):
        return False

    # Домены и пути (t.me/name, example.com) без пробела после точки
    if dot_idx + 1 < n and not text[dot_idx + 1].isspace() and text[dot_idx + 1] not in ('"', "'", "»", ")", "]"):
        return False

    # Сокращения ("б.п.", "руб.", "т.д.", "г.")
    prefix = text[max(0, dot_idx - 24) : dot_idx]
    if _ABBREV_BEFORE_DOT_RE.search(prefix):
        # Если после точки идет слово со строчной буквы или число — точно не конец предложения
        lookahead = text[dot_idx + 1 :].lstrip(" \t")
        if lookahead and (lookahead[0].islower() or lookahead[0].isdigit() or lookahead[0] in ("—", "-", ",")):
            return False
        # Для сокращений вроде "200 б.п. — до 21%." или "2026 г. в РФ" также не режем перед строчной/тире
        if re.search(r"(?:б\.?\s*п|т\.?\s*д|т\.?\s*п|т\.?\s*е|млн|млрд|тыс|руб|г|гг)$", prefix, re.IGNORECASE):
            if not lookahead or not lookahead[0].isupper():
                return False

    return True


def segment_into_sentences(
    text_nfkc: str,
    post_id: int,
    offset_map: PostUtf16OffsetMap | None = None,
) -> list[SentenceClause]:
    """Разбивает текст на пронумерованные предложения `[P{post_id}_S{sent_id}]`."""
    if not text_nfkc:
        return []

    if offset_map is None:
        offset_map = build_utf16_offset_map(text_nfkc, post_id=post_id, segment=False)

    n = len(text_nfkc)
    raw_spans: list[tuple[int, int]] = []
    start = 0

    idx = 0
    while idx < n:
        ch = text_nfkc[idx]
        if ch == "\n":
            if idx > start:
                raw_spans.append((start, idx))
            while idx < n and text_nfkc[idx] == "\n":
                idx += 1
            start = idx
            continue

        if ch in (".", "!", "?") and _is_sentence_boundary(text_nfkc, idx):
            end = idx + 1
            while end < n and text_nfkc[end] in ('"', "'", "»", ")", "]"):
                end += 1
            raw_spans.append((start, end))
            start = end
            idx = end
            continue

        idx += 1

    if start < n:
        raw_spans.append((start, n))

    clauses: list[SentenceClause] = []
    sent_counter = 1
    for raw_start, raw_end in raw_spans:
        # Подрезаем ведущие и замыкающие пробелы без смещения глобальных координат
        s = raw_start
        e = raw_end
        while s < e and text_nfkc[s].isspace():
            s += 1
        while e > s and text_nfkc[e - 1].isspace():
            e -= 1
        if s >= e:
            continue
        start_utf16, end_utf16 = offset_map.cp_span_to_utf16(s, e)
        clauses.append(
            SentenceClause(
                sentence_id=sent_counter,
                tag=f"P{post_id}_S{sent_counter}",
                start_cp=s,
                end_cp=e,
                start_utf16=start_utf16,
                end_utf16=end_utf16,
                text=text_nfkc[s:e],
            )
        )
        sent_counter += 1

    return clauses


def build_utf16_offset_map(
    text: str,
    post_id: int,
    *,
    normalize_nfkc: bool = True,
    rstrip_trailing: bool = True,
    segment: bool = True,
) -> PostUtf16OffsetMap:
    """Нормализует текст (NFKC), строит двунаправленную карту `cp <-> utf16` и клаузы."""
    cleaned = text.rstrip() if rstrip_trailing else text
    text_nfkc = unicodedata.normalize("NFKC", cleaned) if normalize_nfkc else cleaned

    cp_length = len(text_nfkc)
    cp_to_utf16 = [0] * (cp_length + 1)
    utf16_to_cp: list[int] = []
    non_bmp_anchor_points: list[dict[str, int]] = []

    utf16_pos = 0
    cum_shift = 0
    has_non_bmp = False

    for idx, ch in enumerate(text_nfkc):
        cp_to_utf16[idx] = utf16_pos
        if ord(ch) > 0xFFFF:
            has_non_bmp = True
            cum_shift += 1
            non_bmp_anchor_points.append({"cp_idx": idx, "cum_shift": cum_shift})
            utf16_to_cp.append(idx)
            utf16_to_cp.append(idx)
            utf16_pos += 2
        else:
            utf16_to_cp.append(idx)
            utf16_pos += 1

    cp_to_utf16[cp_length] = utf16_pos
    utf16_to_cp.append(cp_length)

    offset_map = PostUtf16OffsetMap(
        post_id=post_id,
        text_nfkc=text_nfkc,
        cp_length=cp_length,
        utf16_length=utf16_pos,
        has_non_bmp=has_non_bmp,
        cp_to_utf16=cp_to_utf16,
        utf16_to_cp=utf16_to_cp,
        non_bmp_anchor_points=non_bmp_anchor_points,
    )
    if segment:
        offset_map.sentences = segment_into_sentences(text_nfkc, post_id=post_id, offset_map=offset_map)
    return offset_map
