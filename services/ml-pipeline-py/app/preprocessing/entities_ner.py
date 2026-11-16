"""Извлечение и канонизация именованных и числовых сущностей для J_ent и CacheKey.

Реализует защиту от слияния постов с разными числовыми фактами (раздел 7.2.3 SPEC)
и построение ExactEntities для семантического кэша (раздел 7.3.2 SPEC).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# Канонические алиасы ключевых институтов и организаций
_ORG_ALIAS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:банк[аеу]?\s+росси[ия]|центральн(?:ый|ого|ому|ым|ом)\s+банк[аеу]?|"
            r"центробанк[аеу]?|цб\s+рф|цб)\b",
            re.IGNORECASE,
        ),
        "ЦБ",
    ),
    (re.compile(r"\b(?:роскомнадзор[аеу]?|ркн)\b", re.IGNORECASE), "РКН"),
    (re.compile(r"\bминцифр[ыаеу]?\b", re.IGNORECASE), "Минцифры"),
    (re.compile(r"\bминфин[аеу]?\b", re.IGNORECASE), "Минфин"),
    (re.compile(r"\bфнс(?:\s+рф)?\b", re.IGNORECASE), "ФНС"),
    (re.compile(r"\bфас(?:\s+рф)?\b", re.IGNORECASE), "ФАС"),
    (re.compile(r"\bгосдум[аыеу]?\b", re.IGNORECASE), "Госдума"),
)

# Стоп-слова с заглавной буквы, не являющиеся персонами или организациями
_RU_CAPITALIZED_STOPWORDS: frozenset[str] = frozenset(
    {
        "Заявление",
        "Прогноз",
        "Ставка",
        "Инфляция",
        "Вчера",
        "Сегодня",
        "Завтра",
        "Однако",
        "При",
        "Для",
        "Если",
        "Когда",
        "После",
        "Перед",
        "Между",
        "Через",
        "Также",
        "Кроме",
        "Итог",
        "Итого",
        "Важно",
        "Срочно",
        "Обновление",
        "Релиз",
        "Версия",
        "Ошибка",
        "Россия",
        "России",
        "Москва",
        "Москве",
        "РФ",
        "МСК",
        "UTC",
        "GMT",
        "ЦОД",
    }
)

# Падежные суффиксы русских фамилий (жен. на -ина/-ова/-ева/-ская и муж. на -ин/-ов/-ев/-ский)
_SURNAME_FEM_RE = re.compile(r"^([А-ЯЁ][а-яё]+?(?:ин|ов|ев))(?:а|ой|у|е|ую)$")
_SURNAME_MASC_RE = re.compile(r"^([А-ЯЁ][а-яё]+?(?:ин|ов|ев))(?:а|у|ом|е|ы)$")
_SURNAME_SKAYA_RE = re.compile(r"^([А-ЯЁ][а-яё]+?ск)(?:ая|ой|ую)$")
_SURNAME_SKIY_RE = re.compile(r"^([А-ЯЁ][а-яё]+?ск)(?:ий|ого|ому|им|ом)$")

# Извлечение чисел и числовых фактов
_PERCENT_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*%")
_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2})(?:\s*(?:МСК|UTC|GMT))?\b", re.IGNORECASE)
_LAW_ID_RE = re.compile(r"\b(\d{1,4}\s*-\s*ФЗ)\b", re.IGNORECASE)
_VERSION_RE = re.compile(r"\b(v\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE)
_PORT_EXPLICIT_RE = re.compile(
    r"(?:\bпорт[а-я]*\s+|(?:localhost|127\.0\.0\.1|0\.0\.0\.0|[a-z0-9_.-]+):)(\d{2,5})\b",
    re.IGNORECASE,
)
_CLI_FLAG_RE = re.compile(r"(?:^|\s)(--[a-zA-Z0-9][a-zA-Z0-9_-]{1,32}(?:=[a-zA-Z0-9._:/-]+)?)")
_TICKER_RE = re.compile(r"\b([A-Z]{2,6})\b")
_DATE_YEAR_OR_NUM_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_CURRENCY_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(?:₽|руб\.?|USD|EUR|\$|млрд|млн|тыс\.?)\b",
    re.IGNORECASE,
)
_STANDALONE_NUM_RE = re.compile(
    r"(?<![a-zA-Zа-яА-ЯёЁ0-9_])(\d+(?:[.,]\d+)?)(?![a-zA-Zа-яА-ЯёЁ0-9_%])"
)
_TECH_STACK_TOKEN_RE = re.compile(
    r"\b(gotd|qdrant|vllm|sglang|bbolt|llguidance|xgrammar|mtproto|acorn|grammers)\b",
    re.IGNORECASE,
)
_QUALIFIED_API_OR_ERR_RE = re.compile(
    r"\b((?:updates|messages|channels|auth|account|users|upload|contacts|chatlists|stories|bots|payments|phone|stats|stickers|help)\.[a-zA-Z_][a-zA-Z0-9_.]*)\b"
)
_CODE_OR_ERROR_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"[A-Z][A-Z0-9_]{3,}"
    r"|[a-z]+[A-Z][A-Za-z0-9]*"
    r"|[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_.]*"
    r"|gotd|qdrant|vllm|sglang|bbolt|llguidance|xgrammar|mtproto|acorn"
    r")\b"
)

# Относительные дельты в базисных пунктах ("на 200 б.п.") не являются уровнем ставки
_BPS_CONTEXT_RE = re.compile(r"\b(?:на\s+)?\d+(?:[.,]\d+)?\s*(?:б\.?\s*п\.?|п\.?\s*п\.?)", re.IGNORECASE)

# Имена и фамилии (1 или 2 слова с заглавной кириллической буквы)
_PERSON_TWO_WORD_RE = re.compile(r"\b([А-ЯЁ][а-яё]{2,})\s+([А-ЯЁ][а-яё]{3,})\b")
_PERSON_SURNAME_CANDIDATE_RE = re.compile(
    r"\b([А-ЯЁ][а-яё]{3,}(?:ин|ов|ев|ск)(?:а|ой|у|е|ую|ом|ы|ая|ий|ого|ому|им))\b"
)

_LATIN_STOP_ACRONYMS: frozenset[str] = frozenset(
    {"UTC", "GMT", "MSK", "HTTP", "HTTPS", "JSON", "HTML", "DOM", "RAM", "CPU", "GPU", "API", "SQL"}
)


@dataclass(slots=True)
class ExtractedEntities:
    """Структурированное представление извлеченных сущностей поста/запроса."""

    ner: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    law_ids: list[str] = field(default_factory=list)
    cli_flags: list[str] = field(default_factory=list)
    all_canonical: set[str] = field(default_factory=set)

    def to_json_dict(self) -> dict[str, list[str]]:
        return {
            "ner": self.ner,
            "numbers": self.numbers,
            "tickers": self.tickers,
            "law_ids": self.law_ids,
            "cli_flags": self.cli_flags,
        }


def _canonicalize_surname(token: str) -> str:
    if m := _SURNAME_SKAYA_RE.match(token):
        return f"{m.group(1)}ая"
    if m := _SURNAME_SKIY_RE.match(token):
        return f"{m.group(1)}ий"
    if m := _SURNAME_FEM_RE.match(token):
        return f"{m.group(1)}а"
    if m := _SURNAME_MASC_RE.match(token):
        return m.group(1)
    return token


def extract_entities(text: str) -> ExtractedEntities:
    """Извлекает канонические сущности E(P) для J_ent и ExactEntities."""
    if not text:
        return ExtractedEntities()

    norm_text = unicodedata.normalize("NFKC", text)
    ner_set: set[str] = set()
    numbers_set: set[str] = set()
    tickers_set: set[str] = set()
    law_ids_set: set[str] = set()
    cli_flags_set: set[str] = set()

    # 1. Канонические организации/институты (ЦБ, РКН, Минцифры...)
    masked_spans: list[tuple[int, int]] = []
    for pattern, canonical in _ORG_ALIAS_PATTERNS:
        for m in pattern.finditer(norm_text):
            ner_set.add(canonical)
            masked_spans.append((m.start(), m.end()))

    def _in_masked(pos: int) -> bool:
        return any(s <= pos < e for s, e in masked_spans)

    # 2. Персоны: "Имя Фамилия" -> каноническая фамилия, а также одиночные фамилии в падеже
    for m in _PERSON_TWO_WORD_RE.finditer(norm_text):
        if _in_masked(m.start()):
            continue
        first, second = m.group(1), m.group(2)
        if first in _RU_CAPITALIZED_STOPWORDS or second in _RU_CAPITALIZED_STOPWORDS:
            continue
        if _SURNAME_FEM_RE.match(second) or _SURNAME_MASC_RE.match(second) or _SURNAME_SKAYA_RE.match(second) or _SURNAME_SKIY_RE.match(second):
            ner_set.add(_canonicalize_surname(second))
        else:
            ner_set.add(_canonicalize_surname(second))

    for m in _PERSON_SURNAME_CANDIDATE_RE.finditer(norm_text):
        if _in_masked(m.start()):
            continue
        tok = m.group(1)
        if tok in _RU_CAPITALIZED_STOPWORDS:
            continue
        # Исключаем притяжательные/прилагательные вроде "ключевой", "инфляционной" (строчная) —
        # регулярка требует заглавную букву. Дополнительно отсеиваем очевидные слова в начале предложения,
        # если слово уже в _RU_CAPITALIZED_STOPWORDS.
        canon = _canonicalize_surname(tok)
        ner_set.add(canon)

    covered_num_spans: list[tuple[int, int]] = []

    def _in_num_spans(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in covered_num_spans)

    # 3. Проценты (21%, 19%, 14%)
    for m in _PERCENT_RE.finditer(norm_text):
        val = m.group(1).replace(",", ".")
        numbers_set.add(f"{val}%")
        covered_num_spans.append((m.start(), m.end()))

    # 4. Время (15:00)
    for m in _TIME_RE.finditer(norm_text):
        numbers_set.add(m.group(1))
        covered_num_spans.append((m.start(), m.end()))

    # 5. Законы (152-ФЗ, 23-ФЗ)
    for m in _LAW_ID_RE.finditer(norm_text):
        law = re.sub(r"\s+", "", m.group(1)).upper()
        law_ids_set.add(law)
        covered_num_spans.append((m.start(), m.end()))

    # 6. CLI-флаги
    for m in _CLI_FLAG_RE.finditer(norm_text):
        cli_flags_set.add(m.group(1))
        covered_num_spans.append((m.start(1), m.end(1)))

    # 7. Порты / версии / суммы валют / числа (с исключением относительных "200 б.п.")
    bps_spans = [(m.start(), m.end()) for m in _BPS_CONTEXT_RE.finditer(norm_text)]

    def _in_bps(pos: int) -> bool:
        return any(s <= pos < e for s, e in bps_spans)

    for m in _PORT_EXPLICIT_RE.finditer(norm_text):
        if _in_bps(m.start()):
            continue
        numbers_set.add(m.group(1).strip())
        covered_num_spans.append((m.start(1), m.end(1)))

    for m in _VERSION_RE.finditer(norm_text):
        if _in_bps(m.start()):
            continue
        numbers_set.add(m.group(1).lower().strip())
        covered_num_spans.append((m.start(), m.end()))

    for m in _CURRENCY_RE.finditer(norm_text):
        if _in_bps(m.start()):
            continue
        numbers_set.add(re.sub(r"\s+", "", m.group(0).lower()))
        covered_num_spans.append((m.start(), m.end()))

    for m in _DATE_YEAR_OR_NUM_RE.finditer(norm_text):
        if _in_bps(m.start()) or _in_num_spans(m.start(), m.end()):
            continue
        numbers_set.add(m.group(1))
        covered_num_spans.append((m.start(), m.end()))

    for m in _STANDALONE_NUM_RE.finditer(norm_text):
        if _in_bps(m.start()) or _in_num_spans(m.start(), m.end()):
            continue
        numbers_set.add(m.group(1).replace(",", "."))
        covered_num_spans.append((m.start(), m.end()))

    # 8. Латинские тикеры (SBER, VTBR, BGP...)
    for m in _TICKER_RE.finditer(norm_text):
        tok = m.group(1)
        if tok in _LATIN_STOP_ACRONYMS:
            continue
        tickers_set.add(tok)

    # 9. Стек и инфраструктурные идентификаторы (gotd, qdrant, mtproto...)
    for m in _TECH_STACK_TOKEN_RE.finditer(norm_text):
        tech = m.group(1).lower()
        tickers_set.add(tech)
        ner_set.add(tech)

    # 10. Квалифицированные имена API/ошибок/структур (updates.channelDifferenceTooLong и т.п.)
    for m in _QUALIFIED_API_OR_ERR_RE.finditer(norm_text):
        ner_set.add(m.group(1))

    all_canonical = ner_set | numbers_set | tickers_set | law_ids_set | cli_flags_set
    return ExtractedEntities(
        ner=sorted(ner_set),
        numbers=sorted(numbers_set),
        tickers=sorted(tickers_set),
        law_ids=sorted(law_ids_set),
        cli_flags=sorted(cli_flags_set),
        all_canonical=all_canonical,
    )


def has_exact_lexical_triggers(
    text: str,
    entities: ExtractedEntities | None = None,
) -> bool:
    """Определяет наличие точных сущностей (тикеры, ФЗ, порты, CLI-флаги, коды ошибок) для буста BM25."""
    ent = entities or extract_entities(text)
    if ent.tickers or ent.law_ids or ent.cli_flags or ent.numbers:
        return True
    norm = unicodedata.normalize("NFKC", text)
    return bool(_CODE_OR_ERROR_TRIGGER_RE.search(norm))


def format_exact_entities_signature(entities: ExtractedEntities) -> str:
    """Формирует каноническую строку ExactEntities(numbers, tickers, law_ids, cli_flags) для CacheKey."""
    nums = ",".join(sorted(set(entities.numbers)))
    tickers = ",".join(sorted(set(entities.tickers)))
    laws = ",".join(sorted(set(entities.law_ids)))
    flags = ",".join(sorted(set(entities.cli_flags)))
    return f"numbers={nums}|tickers={tickers}|law_ids={laws}|cli_flags={flags}"


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Коэффициент Жаккара J_ent(E(P_new), E(M_j))."""
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
