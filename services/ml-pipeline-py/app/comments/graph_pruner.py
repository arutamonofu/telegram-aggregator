"""Ступень 1 и 2 воронки комментариев: детерминированная очистка и топологический GraphScore."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from app.preprocessing.normalizer import normalize_text_nfkc

_URL_RE = re.compile(r"https?://\S+|t\.me/\S+", re.IGNORECASE)
_ALNUM_RE = re.compile(r"[a-zа-яё0-9]", re.IGNORECASE)
_SPAM_COMMENT_RE = re.compile(
    r"(?:переходи\s+в\s+профиль|ищу\s+сотрудник|заработок\s+от\s+\d+|"
    r"подпишись\s+на\s+мой\s+канал|казино|ставки\s+на\s+спорт|"
    r"обналич|крипто.?бот\b|слив\s+курс)",
    re.IGNORECASE,
)
_NOISE_PHRASES: frozenset[str] = frozenset(
    {
        "первый",
        "второй",
        "третий",
        "1",
        "2",
        "3",
        "+",
        "+1",
        "++",
        "ок",
        "ok",
        "ага",
        "да",
        "нет",
        "лол",
        "lol",
        "кек",
        "жиза",
        "спс",
        "спасибо",
        "класс",
        "огонь",
        "топ",
        "база",
        "круто",
        "up",
        "f",
    }
)
_TECH_SHORT_TOKEN_RE = re.compile(
    r"\b(?:oom|502|503|504|500|429|404|403|401|bug|fix|err|tls|dns|sql|cpu|ram)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RawComment:
    """Сырой комментарий из обсуждения под постом."""

    comment_id: int
    post_id: int
    author_hash: str
    text: str
    reply_to_comment_id: int | None = None
    reactions_count: int = 0
    created_at: datetime | None = None
    is_service: bool = False
    is_spam_bot: bool = False


@dataclass(slots=True)
class ScoredComment:
    """Очищенный комментарий с вычисленными метриками графа `reply_to`."""

    comment_id: int
    post_id: int
    author_hash: str
    text: str
    reply_to_comment_id: int | None
    reactions_count: int = 0
    out_degree: int = 0
    unique_authors_subtree: int = 0
    ping_pong_depth: int = 0
    graph_score: float = 0.0
    is_rescued_bug_report: bool = False
    rescue_score: float = 0.0
    rescue_reasons: list[str] = field(default_factory=list)
    confirmations_count: int = 0


def clean_comment_text(text: str) -> str:
    """NFKC-нормализация и схлопывание лишних пробелов в тексте комментария."""
    nfkc = normalize_text_nfkc(text)
    return re.sub(r"\s+", " ", nfkc).strip()


def is_noise_or_spam_comment(comment: RawComment, cleaned_text: str | None = None) -> bool:
    """Ступень 1: отсев сервисных сообщений, ботов, спама и беспредметных реакций."""
    if comment.is_service or comment.is_spam_bot:
        return True
    txt = cleaned_text if cleaned_text is not None else clean_comment_text(comment.text)
    if not txt:
        return True
    if _SPAM_COMMENT_RE.search(txt):
        return True

    stripped_lower = re.sub(r"[!?.,:;)\](]+", "", txt).strip().lower()
    if stripped_lower in _NOISE_PHRASES:
        return True

    alnum_chars = _ALNUM_RE.findall(txt)
    if not alnum_chars:
        return True

    # Если одни только URL или слишком короткая эмоциональная реакция без технического токена
    no_url = _URL_RE.sub("", txt).strip()
    if not _ALNUM_RE.findall(no_url):
        return True
    if len(_ALNUM_RE.findall(no_url)) < 4 and not _TECH_SHORT_TOKEN_RE.search(no_url):
        return True
    return False


def _compute_ping_pong_depths(
    node_ids: Sequence[int],
    children: dict[int, list[int]],
    parent_of: dict[int, int | None],
    author_of: dict[int, str],
) -> dict[int, int]:
    """Детектирует цепочки пинг-понга (`A -> B -> A` и длиннее, depth >= 3) в дереве reply_to."""
    depths: dict[int, int] = {nid: 0 for nid in node_ids}
    roots = [nid for nid in node_ids if parent_of.get(nid) is None or parent_of.get(nid) not in author_of]

    paths: list[list[int]] = []

    def _dfs_paths(curr: int, path: list[int]) -> None:
        kids = children.get(curr, [])
        path.append(curr)
        if not kids:
            paths.append(list(path))
        else:
            for kid in kids:
                if kid not in path:  # защита от аномального цикла
                    _dfs_paths(kid, path)
        path.pop()

    for root in roots:
        _dfs_paths(root, [])

    # В каждом пути от корня к листу ищем максимальные участки чередующихся 2-х авторов длиной >= 3
    for path in paths:
        n = len(path)
        if n < 3:
            continue
        start = 0
        while start < n - 2:
            a1 = author_of[path[start]]
            a2 = author_of[path[start + 1]]
            if a1 == a2:
                start += 1
                continue
            end = start + 1
            while end + 1 < n:
                expected = a1 if ((end + 1 - start) % 2 == 0) else a2
                if author_of[path[end + 1]] == expected:
                    end += 1
                else:
                    break
            seg_len = end - start + 1
            if seg_len >= 3:
                for idx in range(start, end + 1):
                    nid = path[idx]
                    if seg_len > depths[nid]:
                        depths[nid] = seg_len
                start = end
            else:
                start += 1

    return depths


def filter_and_score_comment_tree(
    comments: Sequence[RawComment],
    *,
    wd: float = 0.45,
    wu: float = 0.30,
    wp: float = 0.60,
) -> tuple[list[ScoredComment], dict[int, str]]:
    """Выполняет ступени 1 и 2 для списка комментариев поста.

    Возвращает `(scored_kept_comments, all_cleaned_texts_by_id)`.
    Очищенный словарь всех комментариев по `comment_id` нужен на шаге
    Context-Ancestor Closure, даже если родитель был отфильтрован.
    """
    all_cleaned: dict[int, str] = {}
    kept_raw: list[RawComment] = []

    for c in comments:
        cleaned = clean_comment_text(c.text)
        all_cleaned[c.comment_id] = cleaned
        if not is_noise_or_spam_comment(c, cleaned):
            kept_raw.append(c)

    if not kept_raw:
        return [], all_cleaned

    kept_ids = {c.comment_id for c in kept_raw}
    children: dict[int, list[int]] = defaultdict(list)
    parent_of: dict[int, int | None] = {}
    author_of: dict[int, str] = {}
    by_id: dict[int, RawComment] = {}

    for c in kept_raw:
        by_id[c.comment_id] = c
        author_of[c.comment_id] = c.author_hash
        pid = c.reply_to_comment_id
        parent_of[c.comment_id] = pid
        if pid is not None and pid in kept_ids and pid != c.comment_id:
            children[pid].append(c.comment_id)

    # Строгие потомки (Subtree без самого узла u), чтобы изолированный лист имел GraphScore == 0
    subtree_authors_cache: dict[int, set[str]] = {}

    def _descendant_authors(nid: int, visiting: set[int]) -> set[str]:
        if nid in subtree_authors_cache:
            return subtree_authors_cache[nid]
        if nid in visiting:
            return set()
        visiting.add(nid)
        authors: set[str] = set()
        for kid in children.get(nid, []):
            authors.add(author_of[kid])
            authors.update(_descendant_authors(kid, visiting))
        visiting.remove(nid)
        subtree_authors_cache[nid] = authors
        return authors

    ping_depths = _compute_ping_pong_depths(
        [c.comment_id for c in kept_raw],
        children,
        parent_of,
        author_of,
    )

    scored: list[ScoredComment] = []
    for c in kept_raw:
        cid = c.comment_id
        out_deg = len(children.get(cid, []))
        desc_authors = _descendant_authors(cid, set())
        uniq_authors = len(desc_authors)
        pp_depth = ping_depths.get(cid, 0)

        g_score = (
            wd * math.log1p(out_deg)
            + wu * math.log1p(uniq_authors)
            + wp * float(pp_depth)
        )
        scored.append(
            ScoredComment(
                comment_id=cid,
                post_id=c.post_id,
                author_hash=c.author_hash,
                text=all_cleaned[cid],
                reply_to_comment_id=c.reply_to_comment_id,
                reactions_count=c.reactions_count,
                out_degree=out_deg,
                unique_authors_subtree=uniq_authors,
                ping_pong_depth=pp_depth,
                graph_score=round(g_score, 6),
            )
        )

    return scored, all_cleaned
