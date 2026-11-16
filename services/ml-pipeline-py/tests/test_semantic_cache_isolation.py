"""Unit-тесты изоляции двухуровневого семантического кэша (разделы 7.3.2 и 7.8.5)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app.cache.semantic_cache import (
    TwoLevelSemanticCache,
    build_private_cache_key,
    build_public_cache_key,
)
from app.preprocessing.entities_ner import has_exact_lexical_triggers


def _unit_vec(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_port_8080_vs_8081_cache_key_isolation() -> None:
    cache = TwoLevelSemanticCache()
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    channels = [101, 202]
    bucket = "24h:2026-11-15"

    q_8080 = "Как настроить прокси nginx на порту 8080?"
    q_8081 = "Как настроить прокси nginx на порту 8081?"

    key_8080 = build_public_cache_key(channels, bucket, q_8080)
    key_8081 = build_public_cache_key(channels, bucket, q_8081)
    assert key_8080 != key_8081

    emb_8080 = _unit_vec([1.0, 0.0, 0.0, 0.0])
    # Почти идентичный вектор (cos > 0.999), имитирующий высокую плотность эмбеддингов
    emb_8081 = _unit_vec([0.9999, 0.01, 0.0, 0.0])

    cache.put_level1(
        cache_key=key_8080,
        query_text=q_8080,
        query_embedding=emb_8080,
        response_payload={"answer": "Прокси слушает порт 8080"},
        now=now,
    )

    # Даже при пороге tau_cache=0.88 попадания между 8080 и 8081 быть НЕ ДОЛЖНО (False Hit Rate = 0%)
    miss = cache.get_level1(key_8081, emb_8081, tau_cache=0.88, now=now)
    assert miss is None


def test_same_hash_bucket_paraphrase_hit_and_channel_order_invariance() -> None:
    cache = TwoLevelSemanticCache()
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    bucket = "24h:2026-11-15"

    q_orig = "Как настроить прокси nginx на порту 8080?"
    q_para = "Как правильно настроить nginx proxy для порта 8080"

    key_1 = build_public_cache_key([202, 101], bucket, q_orig)
    key_2 = build_public_cache_key([101, 202], bucket, q_para)
    assert key_1 == key_2

    emb_orig = _unit_vec([0.9, 0.4, 0.1, 0.0])
    emb_para = _unit_vec([0.89, 0.42, 0.11, 0.0])

    cache.put_level1(
        cache_key=key_1,
        query_text=q_orig,
        query_embedding=emb_orig,
        response_payload={"answer": "Откройте порт 8080 в конфиге nginx"},
        ttl_sec=3600,
        now=now,
    )

    hit = cache.get_level1(key_2, emb_para, tau_cache=0.95, now=now + timedelta(minutes=10))
    assert hit is not None
    assert hit.similarity >= 0.95
    assert hit.entry.response_payload["answer"] == "Откройте порт 8080 в конфиге nginx"

    # По истечении TTL запись протухает
    expired = cache.get_level1(key_2, emb_para, tau_cache=0.95, now=now + timedelta(hours=2))
    assert expired is None


def test_private_cache_key_tenant_and_salt_isolation() -> None:
    cache = TwoLevelSemanticCache()
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    aliases = ["sec_chat", "devops_internal"]
    q = "Инцидент SBER --runner=pooling 152-ФЗ"

    key_t1 = build_private_cache_key("tenant-alpha", "salt-A", aliases, "7d:2026-11-15", q)
    key_t2 = build_private_cache_key("tenant-beta", "salt-B", aliases, "7d:2026-11-15", q)
    key_salt_changed = build_private_cache_key("tenant-alpha", "salt-A-v2", aliases, "7d:2026-11-15", q)

    assert len({key_t1, key_t2, key_salt_changed}) == 3

    emb = _unit_vec([0.5, 0.5, 0.5, 0.5])
    cache.put_level1(
        key_t1,
        q,
        emb,
        {"secret": "alpha-only"},
        is_private=True,
        tenant_id="tenant-alpha",
        now=now,
    )
    assert cache.get_level1(key_t1, emb, tau_cache=0.95, now=now, is_private=True, tenant_id="tenant-alpha") is not None
    assert cache.get_level1(key_t1, emb, tau_cache=0.95, now=now, is_private=True, tenant_id="tenant-beta") is None
    assert cache.get_level1(key_t2, emb, tau_cache=0.95, now=now, is_private=True, tenant_id="tenant-beta") is None


def test_exact_lexical_trigger_detection() -> None:
    assert has_exact_lexical_triggers("Ошибка порта 8080 в конфиге") is True
    assert has_exact_lexical_triggers("Отчет SBER по 152-ФЗ") is True
    assert has_exact_lexical_triggers("Флаг --runner=pooling в vLLM") is True
    assert has_exact_lexical_triggers("Что думают аналитики о текущем тренде рынка?") is False
