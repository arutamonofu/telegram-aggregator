"""Ступень L0 спам-фильтрации: детерминированные правила, омоглифы, pHash и SimHash."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import imagehash
from datasketch import MinHash, MinHashLSH
from PIL import Image
from simhash import Simhash, SimhashIndex

from app.preprocessing.normalizer import normalize_text_nfkc

# Соответствие русско-латинских омоглифов (приводим кириллические двойники к латинице для regex).
_CYRILLIC_TO_LATIN_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "А": "A",
        "в": "b",
        "В": "B",
        "е": "e",
        "Е": "E",
        "ё": "e",
        "Ё": "E",
        "з": "3",
        "З": "3",
        "к": "k",
        "К": "K",
        "м": "m",
        "М": "M",
        "н": "h",
        "Н": "H",
        "о": "o",
        "О": "O",
        "р": "p",
        "Р": "P",
        "с": "c",
        "С": "C",
        "т": "t",
        "Т": "T",
        "у": "y",
        "У": "Y",
        "х": "x",
        "Х": "X",
        "і": "i",
        "І": "I",
        "ї": "i",
        "Ї": "I",
        "ј": "j",
        "Ј": "J",
    }
)

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_MULTISPACE_RE = re.compile(r"\s+")

# Регулярные выражения для L0 (на нормализованном к латинице/NFKC тексте и исходном NFKC)
_ERID_RE = re.compile(r"\berid\s*[:=#\-]?\s*[a-z0-9_-]{5,}\b", re.IGNORECASE)
_ORD_TOKEN_RE = re.compile(
    r"(?:\bтоке[нн]\s+орд\b|\bорд\s*[:=#]\s*[a-zа-я0-9_-]{4,}|"
    r"\bорд\s+(?:вк|яндекс|озон|мегафон|билайн|мтс|сбер)\b|"
    r"маркировк[аи]\s+реклам|на\s+правах\s+реклам|#\s*(?:реклама|ad|promo|partner)\b)",
    re.IGNORECASE,
)
_UTM_RE = re.compile(r"(?:\b|[?&])utm_(?:source|medium|campaign|content|term)\s*=", re.IGNORECASE)
_REF_BOT_RE = re.compile(
    r"(?:https?://)?t\.me/[a-z0-9_]{3,}bot\?(?:start|startapp)=[a-z0-9_-]+|"
    r"[?&](?:ref|ref_id|invite|aff)=[a-z0-9_-]{3,}",
    re.IGNORECASE,
)
_CRYPTO_SCAM_RE = re.compile(
    r"(?:airdrop|аирдроп|удвоить\s+депозит|гарантирован\w+\s+доход|сигналы\s+крипт|"
    r"connect\s+wallet|seed\s*phrase)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def normalize_homoglyphs(text: str) -> str:
    """Схлопывает NFKC + удаляет zero-width символы + заменяет кириллические омоглифы на латиницу.

    Также устраняет обфускацию вида ``e.r.i.d:`` или ``e r i d :``? Не ломаем
    обычные слова, но отдельно детектируем zero-width обфускацию.
    """
    nfkc = normalize_text_nfkc(text)
    cleaned = _ZERO_WIDTH_RE.sub("", nfkc)
    return cleaned.translate(_CYRILLIC_TO_LATIN_HOMOGLYPHS)


def compute_image_phash(image_input: Image.Image | bytes | str | Path) -> int:
    """Вычисляет 64-битный перцептивный хэш (DCT pHash) через ``imagehash.phash``."""
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    else:
        img = Image.open(Path(image_input))

    img_hash = imagehash.phash(img, hash_size=8, highfreq_factor=4)
    return int(str(img_hash), 16)


def phash_hamming_distance(hash_a: int, hash_b: int) -> int:
    """Расстояние Хэмминга между двумя 64-битными pHash."""
    return (hash_a ^ hash_b).bit_count()


def _simhash_features(text: str) -> list[str]:
    nfkc = normalize_text_nfkc(text).lower()
    tokens = _WORD_RE.findall(nfkc)
    if not tokens:
        return []
    features = list(tokens)
    features.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    return features


def compute_text_simhash(text: str) -> int:
    """Вычисляет 64-битный SimHash текста через библиотеку ``simhash``."""
    features = _simhash_features(text)
    if not features:
        return 0
    return int(Simhash(features, f=64).value)


class MediaPhashIndex:
    """Реестр 64-битных pHash известных рекламных/спам-креативов."""

    def __init__(self, known_hashes: Iterable[int | str | imagehash.ImageHash] = ()) -> None:
        self._hashes: list[int] = []
        for item in known_hashes:
            self.add(item)

    @staticmethod
    def _coerce(value: int | str | imagehash.ImageHash) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, imagehash.ImageHash):
            return int(str(value), 16)
        return int(value, 16)

    def add(self, phash: int | str | imagehash.ImageHash) -> None:
        val = self._coerce(phash)
        if val not in self._hashes:
            self._hashes.append(val)

    def match(
        self,
        phash: int | str | imagehash.ImageHash,
        *,
        max_distance: int = 6,
    ) -> tuple[bool, int | None]:
        query = self._coerce(phash)
        best_dist: int | None = None
        for known in self._hashes:
            dist = phash_hamming_distance(query, known)
            if best_dist is None or dist < best_dist:
                best_dist = dist
            if dist <= max_distance:
                return True, dist
        return False, best_dist


class SimhashBlocklistIndex:
    """Реестр 64-битных SimHash известных спам-шаблонов на базе ``simhash.SimhashIndex``."""

    def __init__(self, known_hashes: Iterable[int] = (), *, default_k: int = 4) -> None:
        self.default_k = default_k
        self._hashes: list[int] = []
        self._index = SimhashIndex([], f=64, k=default_k)
        for h in known_hashes:
            self.add(h)

    def add(self, simhash: int) -> None:
        if simhash not in self._hashes:
            obj_id = str(len(self._hashes))
            self._hashes.append(simhash)
            self._index.add(obj_id, Simhash(simhash, f=64))

    def match(self, simhash: int, *, max_distance: int = 4) -> bool:
        if max_distance == self.default_k and self._hashes:
            if self._index.get_near_dups(Simhash(simhash, f=64)):
                return True
        return any(phash_hamming_distance(simhash, h) <= max_distance for h in self._hashes)


def build_text_minhash(text: str, *, num_perm: int = 128) -> MinHash:
    """Строит 128-перестановочный ``datasketch.MinHash`` для шаблонного спама."""
    mh = MinHash(num_perm=num_perm)
    for feat in _simhash_features(text):
        mh.update(feat.encode("utf-8"))
    return mh


class MinHashLshCreativeIndex:
    """LSH-индекс на базе ``datasketch.MinHashLSH`` для близких текстовых креативов."""

    def __init__(self, *, threshold: float = 0.75, num_perm: int = 128) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._next_id = 0

    def add_text(self, text: str, *, key: str | None = None) -> str:
        item_key = key or f"tmpl_{self._next_id}"
        self._next_id += 1
        mh = build_text_minhash(text, num_perm=self.num_perm)
        self._lsh.insert(item_key, mh)
        return item_key

    def query_text(self, text: str) -> list[str]:
        mh = build_text_minhash(text, num_perm=self.num_perm)
        return [str(k) for k in self._lsh.query(mh)]


@dataclass(slots=True)
class L0CheckResult:
    """Результат ступени L0 каскада фильтрации."""

    decided: bool
    spam_class: str | None
    is_ad: bool
    is_noise: bool
    reasons: list[str] = field(default_factory=list)
    homoglyph_folded_text: str = ""
    text_simhash: int = 0
    matched_phash_distance: int | None = None


def run_l0_filter(
    text: str,
    *,
    min_text_length: int = 15,
    media_phashes: Sequence[int | str] = (),
    phash_index: MediaPhashIndex | None = None,
    phash_hamming_threshold: int = 6,
    simhash_index: SimhashBlocklistIndex | None = None,
    minhash_index: MinHashLshCreativeIndex | None = None,
    has_informative_media: bool = False,
) -> L0CheckResult:
    """Выполняет детерминированную L0-проверку поста.

    Проверяет:
    1. Минимальную длину текста (< 15 символов без информативного медиа -> шум).
    2. Омоглиф-нормализованные регулярные маркеры (`erid`, токены ОРД, UTM, реф-боты, криптоскам).
    3. Перцептивные хэши медиа (`pHash`, расстояние Хэмминга <= 6) и текстовый SimHash.
    """
    nfkc = normalize_text_nfkc(text)
    folded = normalize_homoglyphs(nfkc)
    compact_text = _MULTISPACE_RE.sub(" ", _ZERO_WIDTH_RE.sub("", nfkc)).strip()
    simhash = compute_text_simhash(compact_text)

    reasons: list[str] = []
    is_ad = False
    is_noise = False
    spam_class: str | None = None
    matched_dist: int | None = None

    # Проверка явных рекламных маркеров (с учетом омоглифов, например кириллической 'е' в erid)
    if _ERID_RE.search(folded) or _ERID_RE.search(nfkc):
        reasons.append("l0_erid_token")
        is_ad = True
        spam_class = "native_ad"

    if _ORD_TOKEN_RE.search(nfkc):
        reasons.append("l0_ord_ad_marker")
        is_ad = True
        spam_class = spam_class or "native_ad"

    if _UTM_RE.search(folded):
        reasons.append("l0_utm_tracking")
        is_ad = True
        spam_class = spam_class or "native_ad"

    if _REF_BOT_RE.search(folded):
        reasons.append("l0_referral_bot")
        is_ad = True
        spam_class = spam_class or "native_ad"

    if _CRYPTO_SCAM_RE.search(nfkc):
        reasons.append("l0_crypto_scam_regex")
        spam_class = "crypto_scam"

    if phash_index is not None and media_phashes:
        for ph in media_phashes:
            matched, dist = phash_index.match(ph, max_distance=phash_hamming_threshold)
            if dist is not None and (matched_dist is None or dist < matched_dist):
                matched_dist = dist
            if matched:
                reasons.append("l0_phash_ad_creative")
                is_ad = True
                spam_class = spam_class or "native_ad"
                break

    if simhash_index is not None and compact_text and simhash_index.match(simhash):
        reasons.append("l0_simhash_duplicate_ad")
        is_ad = True
        spam_class = spam_class or "native_ad"

    if minhash_index is not None and compact_text and minhash_index.query_text(compact_text):
        reasons.append("l0_minhash_duplicate_ad")
        is_ad = True
        spam_class = spam_class or "native_ad"

    # Проверка длины (< 15 символов)
    if len(compact_text) < min_text_length and not is_ad:
        reasons.append("l0_short_text_lt_15")
        if not has_informative_media:
            is_noise = True
            spam_class = spam_class or "clickbait"

    decided = bool(reasons)
    return L0CheckResult(
        decided=decided,
        spam_class=spam_class,
        is_ad=is_ad,
        is_noise=is_noise,
        reasons=reasons,
        homoglyph_folded_text=folded,
        text_simhash=simhash,
        matched_phash_distance=matched_dist,
    )
