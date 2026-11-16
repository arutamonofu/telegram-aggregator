"""Ступень L1: структурные, прагматические и 14-дневные графовые признаки спама/ВП."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from app.preprocessing.normalizer import normalize_text_nfkc

_TME_URL_RE = re.compile(r"(?:https?://)?t\.me/([a-z0-9_]{4,})", re.IGNORECASE)
_AT_MENTION_RE = re.compile(r"(?<![a-z0-9_.])@([a-z0-9_]{4,})", re.IGNORECASE)

_CATAPHORA_HOOK_RE = re.compile(
    r"(?:этот\s+(?:метод|сервис|инструмент|бот|канал|способ|прием)|"
    r"один\s+(?:секрет|канал|инструмент|лайфхак)|"
    r"эта\s+(?:находка|схема|ссылка|платформа)|"
    r"мало\s+кто\s+знает|никто\s+не\s+ожидал|вы\s+не\s+поверите|"
    r"почему\s+все\s+переходят|вот\s+что\s+(?:случилось|нашли)|"
    r"сделает\s+за\s+вас\s+вс[её])",
    re.IGNORECASE,
)

_CTA_TAIL_RE = re.compile(
    r"(?:подписывай\w*|переходи\w*|жми\w*|забирай\w*|успей\w*|"
    r"ссылка\s+(?:ниже|здесь|внутри|тут)|подробнее\s+(?:в\s+канале|тут|по\s+ссылке)|"
    r"читай(?:те)?\s+в\s+(?:источнике|канале)|бесплатно\s+по\s+ссылке|"
    r"промокод|скидк[аи]\s+\d+%|регистрируй\w*)",
    re.IGNORECASE,
)

# Для проверки наличия в 1-м абзаце конкретных сущностей (тогда катафора не считается пустым кликбейтом)
_CONCRETE_NAME_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9_-]{2,}|v?\d+\.\d+(?:\.\d+)?)\b")


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _normalize_channel_handle(raw: str | None) -> str:
    if not raw:
        return ""
    handle = raw.strip().lstrip("@").lower()
    if "t.me/" in handle:
        handle = handle.split("t.me/", 1)[1].split("/", 1)[0].split("?", 1)[0]
    return handle


@dataclass(slots=True)
class MentionEdge:
    source: str
    target: str
    timestamp: datetime


class MentionGraph14d:
    """Направленный граф перекрестных упоминаний каналов за скользящее окно 14 дней.

    Раздел 7.2.2 SPEC.md:
    Отслеживает взаимный пиар (mutual PR) — взаимные репосты/упоминания двух каналов
    в окне +/- 48 часов, а также аномальную частоту исходящих упоминаний за 14 дней.
    """

    def __init__(
        self,
        *,
        window_days: int = 14,
        reciprocity_hours: int = 48,
    ) -> None:
        self.window = timedelta(days=window_days)
        self.reciprocity_window = timedelta(hours=reciprocity_hours)
        self._edges_by_pair: dict[tuple[str, str], list[datetime]] = defaultdict(list)

    def prune(self, now: datetime) -> None:
        now_utc = _ensure_utc(now)
        cutoff = now_utc - self.window
        empty_keys: list[tuple[str, str]] = []
        for pair, timestamps in self._edges_by_pair.items():
            kept = [ts for ts in timestamps if ts >= cutoff]
            if kept:
                self._edges_by_pair[pair] = kept
            else:
                empty_keys.append(pair)
        for key in empty_keys:
            del self._edges_by_pair[key]

    def record_mention(
        self,
        source_channel: str,
        target_channel: str,
        timestamp: datetime,
    ) -> None:
        src = _normalize_channel_handle(source_channel)
        dst = _normalize_channel_handle(target_channel)
        if not src or not dst or src == dst:
            return
        ts = _ensure_utc(timestamp)
        self.prune(ts)
        self._edges_by_pair[(src, dst)].append(ts)

    def is_reciprocal_within_48h(
        self,
        source_channel: str,
        target_channel: str,
        timestamp: datetime,
    ) -> bool:
        """Проверяет, упоминал ли `target_channel` канал `source_channel` в пределах +-48 ч."""
        src = _normalize_channel_handle(source_channel)
        dst = _normalize_channel_handle(target_channel)
        if not src or not dst or src == dst:
            return False
        ts = _ensure_utc(timestamp)
        reverse_timestamps = self._edges_by_pair.get((dst, src), [])
        return any(abs(ts - rev_ts) <= self.reciprocity_window for rev_ts in reverse_timestamps)

    def outbound_stats_14d(
        self,
        source_channel: str,
        timestamp: datetime,
    ) -> tuple[int, int]:
        """Возвращает `(total_outbound_mentions, unique_targets)` за 14 дней."""
        src = _normalize_channel_handle(source_channel)
        ts = _ensure_utc(timestamp)
        self.prune(ts)
        total = 0
        targets: set[str] = set()
        for (edge_src, edge_dst), timestamps in self._edges_by_pair.items():
            if edge_src == src and timestamps:
                total += len(timestamps)
                targets.add(edge_dst)
        return total, len(targets)


@dataclass(slots=True)
class StructuralSpamFeatures:
    """Набор структурно-графовых и дискурсивных признаков ступени L1."""

    external_targets: list[str] = field(default_factory=list)
    has_hidden_tme_text_url: bool = False
    last_link_relative_pos: float | None = None
    link_in_last_20_percent: bool = False
    has_first_para_cataphora: bool = False
    has_pragmatic_shift: bool = False
    is_reciprocal_48h: bool = False
    outbound_mentions_14d: int = 0
    unique_outbound_targets_14d: int = 0
    l1_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def extract_structural_features(
    text: str,
    *,
    entities: Sequence[Mapping[str, Any]] = (),
    source_channel: str = "",
    published_at: datetime | None = None,
    mention_graph: MentionGraph14d | None = None,
    record_in_graph: bool = True,
) -> StructuralSpamFeatures:
    """Извлекает признаки смещения `t.me/` в хвост, катафор в 1-м абзаце и 14-дневного графа."""
    nfkc = normalize_text_nfkc(text)
    total_len = max(len(nfkc), 1)
    self_handle = _normalize_channel_handle(source_channel)
    now_ts = _ensure_utc(published_at or datetime.now(timezone.utc))

    external_targets: list[str] = []
    link_positions: list[int] = []
    has_hidden_text_url = False

    for match in _TME_URL_RE.finditer(nfkc):
        handle = _normalize_channel_handle(match.group(1))
        if handle and handle != self_handle and handle not in {"joinchat", "addstickers", "s"}:
            if handle not in external_targets:
                external_targets.append(handle)
            link_positions.append(match.start())

    for match in _AT_MENTION_RE.finditer(nfkc):
        handle = _normalize_channel_handle(match.group(1))
        if handle and handle != self_handle:
            if handle not in external_targets:
                external_targets.append(handle)
            link_positions.append(match.start())

    # Анализ скрытых ссылок text_url внутри Telegram MessageEntity
    for ent in entities:
        ent_type = str(ent.get("type", "")).lower()
        if ent_type in {"text_url", "messageentitytexturl"}:
            url = str(ent.get("url", ""))
            m = _TME_URL_RE.search(url)
            if m:
                handle = _normalize_channel_handle(m.group(1))
                if handle and handle != self_handle:
                    has_hidden_text_url = True
                    if handle not in external_targets:
                        external_targets.append(handle)
                    # offset может быть в UTF-16, но для относительной доли позиции это близко
                    offset = int(ent.get("offset", total_len - 1))
                    link_positions.append(min(offset, total_len - 1))

    # Проверяем положение внешней ссылки относительно длины текста (последние 20% = >= 0.80)
    last_link_rel: float | None = None
    in_last_20 = False
    if link_positions:
        last_link_pos = max(link_positions)
        last_link_rel = round(last_link_pos / total_len, 4)
        if last_link_rel >= 0.80:
            in_last_20 = True

    # Первый абзац и катафора без конкретной сущности
    paragraphs = [p.strip() for p in re.split(r"\n{1,2}", nfkc) if p.strip()]
    first_para = paragraphs[0] if paragraphs else nfkc
    has_cataphora = bool(_CATAPHORA_HOOK_RE.search(first_para)) and not bool(
        _CONCRETE_NAME_RE.search(first_para)
    )

    # Прагматический сдвиг: наличие призыва к действию в финальных 25% текста
    tail_slice = nfkc[int(total_len * 0.70) :]
    has_pragmatic_shift = bool(_CTA_TAIL_RE.search(tail_slice))

    # Проверка и обновление 14-дневного графа упоминаний
    is_reciprocal = False
    outbound_14d = 0
    unique_targets_14d = 0
    if mention_graph is not None and self_handle:
        for target in external_targets:
            if mention_graph.is_reciprocal_within_48h(self_handle, target, now_ts):
                is_reciprocal = True
            if record_in_graph:
                mention_graph.record_mention(self_handle, target, now_ts)
        outbound_14d, unique_targets_14d = mention_graph.outbound_stats_14d(self_handle, now_ts)

    # Расчет композитного L1-риска [0, 1]
    score = 0.0
    reasons: list[str] = []
    if is_reciprocal:
        score += 0.50
        reasons.append("l1_reciprocal_mention_48h")
    if in_last_20:
        score += 0.28
        reasons.append("l1_link_in_last_20_percent")
    elif external_targets:
        score += 0.12
        reasons.append("l1_external_tme_link")
    if has_hidden_text_url:
        score += 0.15
        reasons.append("l1_hidden_tme_text_url")
    if has_cataphora:
        score += 0.22
        reasons.append("l1_first_para_cataphora")
    if has_pragmatic_shift:
        score += 0.22
        reasons.append("l1_tail_cta_pragmatic_shift")
    if unique_targets_14d >= 5:
        score += 0.18
        reasons.append("l1_high_outbound_mentions_14d")

    score = round(min(1.0, score), 4)
    return StructuralSpamFeatures(
        external_targets=external_targets,
        has_hidden_tme_text_url=has_hidden_text_url,
        last_link_relative_pos=last_link_rel,
        link_in_last_20_percent=in_last_20,
        has_first_para_cataphora=has_cataphora,
        has_pragmatic_shift=has_pragmatic_shift,
        is_reciprocal_48h=is_reciprocal,
        outbound_mentions_14d=outbound_14d,
        unique_outbound_targets_14d=unique_targets_14d,
        l1_score=score,
        reasons=reasons,
    )
