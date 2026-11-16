"""Детерминированное восстановление подстроки (Смит — Ватерман) и UTF-16 маппинг.

Реализует шаги 3 и 4 раздела 7.4.1 SPEC.md:
- Обратимая нормализация («ё -> е», ««» -> ""», «—/– -> -», сжатие пробелов).
- Локальное выравнивание Смита — Ватермана с аффинным штрафом Гото
  (match=+2, mismatch=-1, gap_open=-2, gap_extend=-0.5) и порогом >= 0.85
  (либо пословным Жаккаром >= 0.70 по методологии FullCite).
- O(1) трансляция координат [start_cp, end_cp] -> utf16_range [start_utf16, end_utf16]
  через предвычисленную карту `cp_to_utf16`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.preprocessing.offsets import PostUtf16OffsetMap, build_utf16_offset_map

_DASH_CHARS = frozenset({"—", "–", "―", "−", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"})
_QUOTE_CHARS = frozenset({"«", "»", "„", "“", "”", "‘", "’", "'"})
_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_CLAUSE_PREFIX_RE = re.compile(r"^\[P\d+_S\d+\]\s*", re.IGNORECASE)
_EDGE_ELLIPSIS_START_RE = re.compile(r"^(?:\.{2,}|…)\s*")
_EDGE_ELLIPSIS_END_RE = re.compile(r"\s*(?:\.{2,}|…)$")
_INNER_ELLIPSIS_SPLIT_RE = re.compile(r"(?:\.{3}|…)")


@dataclass(slots=True)
class ReversibleNormalizedText:
    """Обратимо нормализованная строка с картой соответствия в исходные кодовые точки."""

    orig_text: str
    norm_text: str
    norm_to_orig_start: list[int]
    norm_to_orig_end: list[int]


def reversible_normalize(text: str) -> ReversibleNormalizedText:
    """Выполняет обратимую нормализацию с сохранением точных индексов кодовых точек."""
    norm_chars: list[str] = []
    norm_to_orig_start: list[int] = []
    norm_to_orig_end: list[int] = []

    idx = 0
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch.isspace():
            if not norm_chars or norm_chars[-1] != " ":
                norm_chars.append(" ")
                norm_to_orig_start.append(idx)
                norm_to_orig_end.append(idx + 1)
            else:
                norm_to_orig_end[-1] = idx + 1
            idx += 1
            continue

        low = ch.lower()
        if low == "ё":
            mapped = "е"
        elif low in _DASH_CHARS:
            mapped = "-"
        elif low in _QUOTE_CHARS:
            mapped = '"'
        else:
            mapped = low

        norm_chars.append(mapped)
        norm_to_orig_start.append(idx)
        norm_to_orig_end.append(idx + 1)
        idx += 1

    # Подрезаем ведущие/замыкающие пробелы, сохраняя точную карту индексов
    left = 0
    right = len(norm_chars)
    while left < right and norm_chars[left] == " ":
        left += 1
    while right > left and norm_chars[right - 1] == " ":
        right -= 1

    return ReversibleNormalizedText(
        orig_text=text,
        norm_text="".join(norm_chars[left:right]),
        norm_to_orig_start=norm_to_orig_start[left:right],
        norm_to_orig_end=norm_to_orig_end[left:right],
    )


@dataclass(slots=True)
class SmithWatermanAlignment:
    """Результат восстановления цитатного спана и трансляции в UTF-16."""

    post_id: int
    sentence_id: int | None
    start_cp: int
    end_cp: int
    utf16_range: tuple[int, int]
    verbatim_text: str
    similarity_score: float
    exact_match: bool

    def to_frontend_dict(self) -> dict[str, object]:
        return {
            "post_id": self.post_id,
            "utf16_range": [self.utf16_range[0], self.utf16_range[1]],
            "verbatim_text": self.verbatim_text,
        }


def _word_jaccard(text_a: str, text_b: str) -> float:
    words_a = {w.lower().replace("ё", "е") for w in _WORD_RE.findall(text_a)}
    words_b = {w.lower().replace("ё", "е") for w in _WORD_RE.findall(text_b)}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _run_sw_on_slice(
    *,
    slice_text: str,
    slice_offset_cp: int,
    quote_clean: str,
    post_id: int,
    sentence_id: int | None,
    offset_map: PostUtf16OffsetMap,
    match_score: float,
    mismatch_score: float,
    gap_open: float,
    gap_extend: float,
    similarity_threshold: float,
    jaccard_threshold: float,
) -> SmithWatermanAlignment | None:
    if not slice_text or not quote_clean:
        return None

    # 1. Быстрый путь: прямое точное вхождение подстроки
    direct_pos = slice_text.find(quote_clean)
    if direct_pos >= 0:
        start_cp = slice_offset_cp + direct_pos
        end_cp = start_cp + len(quote_clean)
        u_start, u_end = offset_map.cp_span_to_utf16(start_cp, end_cp)
        return SmithWatermanAlignment(
            post_id=post_id,
            sentence_id=sentence_id,
            start_cp=start_cp,
            end_cp=end_cp,
            utf16_range=(u_start, u_end),
            verbatim_text=offset_map.text_nfkc[start_cp:end_cp],
            similarity_score=1.0,
            exact_match=True,
        )

    # 2. Обратимая нормализация и локальное выравнивание Смита — Ватермана (Gotoh)
    norm_target = reversible_normalize(slice_text)
    norm_quote = reversible_normalize(quote_clean)
    t_str = norm_target.norm_text
    q_str = norm_quote.norm_text
    n = len(t_str)
    m = len(q_str)
    if n == 0 or m == 0:
        return None

    neg_inf = float("-inf")
    h_mat = [[0.0] * (m + 1) for _ in range(n + 1)]
    e_mat = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    f_mat = [[neg_inf] * (m + 1) for _ in range(n + 1)]

    # 0=stop, 1=diag(H), 2=from E, 3=from F
    ptr_h = [[0] * (m + 1) for _ in range(n + 1)]
    # 1=open from H, 2=extend
    ptr_e = [[0] * (m + 1) for _ in range(n + 1)]
    ptr_f = [[0] * (m + 1) for _ in range(n + 1)]

    best_score = 0.0
    best_i = 0
    best_j = 0

    for i in range(1, n + 1):
        t_ch = t_str[i - 1]
        h_row = h_mat[i]
        h_prev = h_mat[i - 1]
        e_row = e_mat[i]
        e_prev = e_mat[i - 1]
        f_row = f_mat[i]
        ph_row = ptr_h[i]
        pe_row = ptr_e[i]
        pf_row = ptr_f[i]

        for j in range(1, m + 1):
            # Gap in Q (пропуск символа в целевом посте T)
            e_open = h_prev[j] + gap_open
            e_ext = e_prev[j] + gap_extend
            if e_open >= e_ext:
                e_val = e_open
                pe_row[j] = 1
            else:
                e_val = e_ext
                pe_row[j] = 2
            e_row[j] = e_val

            # Gap in T (лишний символ в цитате Q)
            f_open = h_row[j - 1] + gap_open
            f_ext = f_row[j - 1] + gap_extend
            if f_open >= f_ext:
                f_val = f_open
                pf_row[j] = 1
            else:
                f_val = f_ext
                pf_row[j] = 2
            f_row[j] = f_val

            diag = h_prev[j - 1] + (match_score if t_ch == q_str[j - 1] else mismatch_score)

            val = 0.0
            p = 0
            if diag > val:
                val = diag
                p = 1
            if e_val > val:
                val = e_val
                p = 2
            if f_val > val:
                val = f_val
                p = 3

            h_row[j] = val
            ph_row[j] = p

            if val > best_score:
                best_score = val
                best_i = i
                best_j = j

    if best_score <= 0.0 or best_i == 0 or best_j == 0:
        return None

    # Обратный проход (traceback) для восстановления точных границ [i_start, best_i)
    i = best_i
    j = best_j
    state = "H"
    aligned_t_indices: list[int] = []
    aligned_q_indices: list[int] = []
    aligned_raw_diffs = 0

    while i > 0 and j > 0:
        if state == "H":
            if h_mat[i][j] <= 0.0 or ptr_h[i][j] == 0:
                break
            direction = ptr_h[i][j]
            if direction == 1:
                ti = i - 1
                qj = j - 1
                aligned_t_indices.append(ti)
                aligned_q_indices.append(qj)
                orig_t_pos = norm_target.norm_to_orig_start[ti]
                orig_q_pos = norm_quote.norm_to_orig_start[qj]
                if slice_text[orig_t_pos].lower() != quote_clean[orig_q_pos].lower():
                    aligned_raw_diffs += 1
                i -= 1
                j -= 1
            elif direction == 2:
                state = "E"
            else:
                state = "F"
        elif state == "E":
            aligned_t_indices.append(i - 1)
            prev_state = "H" if ptr_e[i][j] == 1 else "E"
            i -= 1
            state = prev_state
        else:  # "F"
            aligned_q_indices.append(j - 1)
            prev_state = "H" if ptr_f[i][j] == 1 else "F"
            j -= 1
            state = prev_state

    if not aligned_t_indices:
        return None

    t_norm_start = min(aligned_t_indices)
    t_norm_end = max(aligned_t_indices)

    raw_start_cp = slice_offset_cp + norm_target.norm_to_orig_start[t_norm_start]
    raw_end_cp = slice_offset_cp + norm_target.norm_to_orig_end[t_norm_end]

    # Подрезаем граничные пробелы без изменения смысла
    full_text = offset_map.text_nfkc
    while raw_start_cp < raw_end_cp and full_text[raw_start_cp].isspace():
        raw_start_cp += 1
    while raw_end_cp > raw_start_cp and full_text[raw_end_cp - 1].isspace():
        raw_end_cp -= 1
    if raw_start_cp >= raw_end_cp:
        return None

    verbatim = full_text[raw_start_cp:raw_end_cp]
    max_possible_score = m * match_score
    r_sw = min(1.0, best_score / max_possible_score) if max_possible_score > 0 else 0.0
    if aligned_raw_diffs > 0:
        sim_score = round(r_sw * (m / (m + aligned_raw_diffs)), 3)
    else:
        sim_score = round(r_sw, 3)

    if sim_score < similarity_threshold:
        jaccard = _word_jaccard(verbatim, quote_clean)
        if jaccard >= jaccard_threshold and r_sw >= 0.60:
            sim_score = round(max(sim_score, similarity_threshold), 3)
        else:
            return None

    u_start, u_end = offset_map.cp_span_to_utf16(raw_start_cp, raw_end_cp)
    return SmithWatermanAlignment(
        post_id=post_id,
        sentence_id=sentence_id,
        start_cp=raw_start_cp,
        end_cp=raw_end_cp,
        utf16_range=(u_start, u_end),
        verbatim_text=verbatim,
        similarity_score=sim_score,
        exact_match=False,
    )


def smith_waterman_align(
    post_text: str,
    exact_quote: str,
    *,
    offset_map: PostUtf16OffsetMap | None = None,
    post_id: int = 0,
    sentence_id: int | None = None,
    match_score: float = 2.0,
    mismatch_score: float = -1.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    similarity_threshold: float = 0.85,
    jaccard_threshold: float = 0.70,
) -> SmithWatermanAlignment | None:
    """Восстанавливает дословный спан `[start_cp, end_cp]` и `utf16_range` в оригинальном посте."""
    quote_clean = unicodedata.normalize("NFKC", exact_quote).strip()
    if not post_text or not quote_clean:
        return None

    omap = offset_map or build_utf16_offset_map(post_text, post_id=post_id)

    def _align_candidate(candidate: str) -> SmithWatermanAlignment | None:
        if not candidate:
            return None
        # При наличии sentence_id сначала ищем внутри целевого предложения
        if sentence_id is not None and omap.sentences:
            target_clause = next(
                (s for s in omap.sentences if s.sentence_id == sentence_id),
                None,
            )
            if target_clause is not None:
                clause_match = _run_sw_on_slice(
                    slice_text=target_clause.text,
                    slice_offset_cp=target_clause.start_cp,
                    quote_clean=candidate,
                    post_id=post_id,
                    sentence_id=sentence_id,
                    offset_map=omap,
                    match_score=match_score,
                    mismatch_score=mismatch_score,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    similarity_threshold=similarity_threshold,
                    jaccard_threshold=jaccard_threshold,
                )
                if clause_match is not None:
                    return clause_match

        # Фоллбэк на полный NFKC-текст поста (на случай сбоя номера предложения у LLM)
        return _run_sw_on_slice(
            slice_text=omap.text_nfkc,
            slice_offset_cp=0,
            quote_clean=candidate,
            post_id=post_id,
            sentence_id=sentence_id,
            offset_map=omap,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_open=gap_open,
            gap_extend=gap_extend,
            similarity_threshold=similarity_threshold,
            jaccard_threshold=jaccard_threshold,
        )

    direct_match = _align_candidate(quote_clean)
    if direct_match is not None:
        return direct_match

    # Очищаем от скопированных тегов `[P123_S4]` и добавленных моделью многоточий по краям
    stripped_quote = _CLAUSE_PREFIX_RE.sub("", quote_clean)
    stripped_quote = _EDGE_ELLIPSIS_START_RE.sub("", stripped_quote)
    stripped_quote = _EDGE_ELLIPSIS_END_RE.sub("", stripped_quote).strip()
    if stripped_quote and stripped_quote != quote_clean:
        cleaned_match = _align_candidate(stripped_quote)
        if cleaned_match is not None:
            return cleaned_match

    # Если LLM склеила фрагменты длинного предложения через `...`/`…`, выравниваем самый длинный фрагмент
    source_for_split = stripped_quote or quote_clean
    if _INNER_ELLIPSIS_SPLIT_RE.search(source_for_split):
        parts = [p.strip() for p in _INNER_ELLIPSIS_SPLIT_RE.split(source_for_split) if len(p.strip()) >= 15]
        if parts:
            longest_part = max(parts, key=len)
            return _align_candidate(longest_part)

    return None
