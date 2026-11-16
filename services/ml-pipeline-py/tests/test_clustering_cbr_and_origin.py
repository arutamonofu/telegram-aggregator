"""Тесты Single-Pass кластеризации: кейс ЦБ РФ (21% vs 19%) и пересчет OriginScore."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app.clustering import (
    ClusterMember,
    EventClusterState,
    SinglePassClusterer,
    compute_bm25_norm,
    compute_composite_similarity,
    resolve_effective_published_at,
)
from app.embeddings.qwen3_embedder import Qwen3EmbeddingClient
from app.embeddings.qwen3_reranker import Qwen3RerankerClient
from app.preprocessing.entities_ner import extract_entities, jaccard_similarity


class _ControlledEmbeddingBackbone:
    """Бэкенд для точной фиксации косинусного сходства между тестовыми постами."""

    def __init__(self, target_cos_map: dict[str, np.ndarray], dim: int = 1024) -> None:
        self.dim = dim
        self.map = target_cos_map

    def encode_raw(self, texts):
        rows = []
        for t in texts:
            if t in self.map:
                rows.append(self.map[t])
            else:
                v = np.zeros(self.dim, dtype=np.float32)
                v[0] = 1.0
                rows.append(v)
        return np.stack(rows, axis=0)


def _unit_pair_with_cosine(cos_val: float, dim: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    e1 = np.zeros(dim, dtype=np.float32)
    e1[0] = 1.0
    e2 = np.zeros(dim, dtype=np.float32)
    e2[0] = cos_val
    e2[1] = float(np.sqrt(max(0.0, 1.0 - cos_val * cos_val)))
    return e1, e2


def test_composite_similarity_cbr_21_vs_19_percent_spec_example() -> None:
    """Точная проверка примера со ставкой ЦБ РФ (21% против 19%) из раздела 7.2.3 SPEC.md."""
    t0 = datetime(2026, 3, 15, 13, 30, tzinfo=timezone.utc)

    medoid_text = "ЦБ РФ повысил ключевую ставку до 21%, отметила Эльвира Набиуллина"
    post_19_text = "ЦБ РФ поднял ключевую ставку до 19%, сообщил аналитик Иван Сидоров"
    post_21_paraphrase = "Банк России поднял ключевую ставку до 21%, заявила Эльвира Набиуллина"

    ent_m = extract_entities(medoid_text).all_canonical
    ent_19 = extract_entities(post_19_text).all_canonical
    ent_21_p = extract_entities(post_21_paraphrase).all_canonical

    assert ent_m == {"ЦБ", "21%", "Набиуллина"}
    assert ent_19 == {"ЦБ", "19%", "Сидоров"}
    # Intersection = {"ЦБ"} (1), Union = {"ЦБ", "21%", "Набиуллина", "19%", "Сидоров"} (5) -> J_ent = 0.20
    j_ent_19 = jaccard_similarity(ent_19, ent_m)
    assert abs(j_ent_19 - 0.20) < 1e-6

    # Проверяем прямое вычисление формулы SPEC 7.2.3:
    # S = 0.55 * 0.90 + 0.20 * 0.85 + 0.25 * 0.20 - 0.003 * 0 = 0.495 + 0.170 + 0.050 = 0.715 < 0.72
    s_exact = compute_composite_similarity(
        cosine_centroid=0.90,
        bm25_norm=0.85,
        j_ent=j_ent_19,
        delta_hours=0.0,
    )
    assert abs(s_exact - 0.715) < 1e-6
    assert s_exact < 0.72

    # Интеграционная проверка через SinglePassClusterer с Qwen3-Reranker-0.6B:
    # M_104 в кластере C_104; P_19 не должен слиться (< 0.72), P_21 в серой зоне проходит через реранкер.
    v_base, v_90 = _unit_pair_with_cosine(0.90, dim=1024)
    _, v_76 = _unit_pair_with_cosine(0.76, dim=1024)
    embedder = Qwen3EmbeddingClient(
        backbone=_ControlledEmbeddingBackbone(
            {
                medoid_text: v_base,
                post_19_text: v_90,
                post_21_paraphrase: v_76,
            }
        ),
        mrl_dim=512,
    )

    # Логируем вызовы кросс-энкодера, чтобы удостовериться: сравнение идет СТРОГО с медоидом M_104
    rerank_calls: list[tuple[str, str]] = []

    def _mock_reranker(query: str, doc: str) -> float:
        rerank_calls.append((query, doc))
        if "21%" in query and "21%" in doc:
            return 0.93
        return 0.12

    reranker = Qwen3RerankerClient(scorer_fn=_mock_reranker)
    clusterer = SinglePassClusterer(
        embedder=embedder,
        reranker=reranker,
        bm25_fn=lambda q, d: 0.85 if "19%" in q else compute_bm25_norm(q, d),
    )

    v_m_512 = v_base[:512]
    c104 = EventClusterState(
        cluster_id="C_104",
        centroid=v_m_512.copy(),
        embedding_sum=v_m_512.copy(),
        medoid_post_id=104,
        medoid_text=medoid_text,
        medoid_entities=ent_m,
        medoid_embedding=v_m_512.copy(),
        last_updated_at=t0,
        origin_post_id=104,
        origin_channel_id="ch_origin",
        members=[
            ClusterMember(
                post_id=104,
                channel_id="ch_origin",
                first_version_text=medoid_text,
                current_text=medoid_text,
                embedding=v_m_512.copy(),
                entities=ent_m,
                published_at=t0,
            )
        ],
    )
    clusterer.register_existing_cluster(c104)

    # 1) Пост про ставку 19% (S = 0.715 < tau_low = 0.72) -> НЕ попадает в C_104, даже без реранкера
    res_19 = clusterer.ingest_post(
        post_id=201,
        channel_id="ch_other",
        text=post_19_text,
        published_at=t0,
    )
    assert res_19.is_new_cluster is True
    assert res_19.cluster_id != "C_104"
    assert res_19.decision == "reject_below_tau_low"
    assert res_19.best_candidate is not None
    assert abs(res_19.best_candidate.composite_score - 0.715) < 1e-6
    assert len(rerank_calls) == 0

    # 2) Пересказ про ставку 21% попадает в серую зону [0.72, 0.88) -> Qwen3-Reranker на медоиде M_104 -> слияние
    # Проверяем J_ent между paraphrase и medoid: {"ЦБ", "21%", "Набиуллина"} -> 1.0!
    assert jaccard_similarity(ent_21_p, ent_m) == 1.0
    res_21 = clusterer.ingest_post(
        post_id=202,
        channel_id="ch_second",
        text=post_21_paraphrase,
        published_at=t0,
    )
    assert res_21.is_new_cluster is False
    assert res_21.cluster_id == "C_104"
    assert res_21.decision == "merge_reranked"
    assert rerank_calls == [(post_21_paraphrase, medoid_text)]


def test_origin_score_recalculation_when_stub_post_is_edited_over_30_percent() -> None:
    """Проверяет защиту от грабберов: правка > 30% переключает effective ts на edit_date."""
    t_b_stub = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    t_a_orig = datetime(2026, 3, 15, 12, 0, 30, tzinfo=timezone.utc)
    t_b_edit = datetime(2026, 3, 15, 12, 4, 0, tzinfo=timezone.utc)

    # Одинаковый вектор, чтобы оба поста попали в один кластер C
    vec = np.zeros(1024, dtype=np.float32)
    vec[0] = 1.0
    backbone = _ControlledEmbeddingBackbone({})
    embedder = Qwen3EmbeddingClient(backbone=backbone, mrl_dim=512)
    clusterer = SinglePassClusterer(embedder=embedder, bm25_fn=lambda a, b: 0.95)

    stub_text = "⚡️ ЦБ 21% скоро"
    full_a_text = "ЦБ РФ повысил ключевую ставку до 21%, опубликовав официальный пресс-релиз регулятора."
    rewritten_b_text = (
        "Банк России сегодня принял решение повысить ключевую ставку сразу до 21% "
        "из-за высокого инфляционного давления в экономике и роста кредитования."
    )

    # Шаг 1: Канал B (высокий авторитет 150) вбросил заглушку раньше на 30 сек
    res_b = clusterer.ingest_post(
        post_id=5001,
        channel_id="channel_B_grabber",
        text=stub_text,
        published_at=t_b_stub,
        embedding=vec[:512],
        entities={"ЦБ", "21%"},
        channel_authority=150.0,
        originality_score=0.6,
        has_primary_evidence=False,
    )
    assert res_b.origin_post_id == 5001
    assert res_b.origin_channel_id == "channel_B_grabber"

    # Шаг 2: Канал A (обычный авторитет 20, но первичный источник) публикует полный пост в 12:00:30
    res_a = clusterer.ingest_post(
        post_id=5002,
        channel_id="channel_A_original",
        text=full_a_text,
        published_at=t_a_orig,
        embedding=vec[:512],
        entities={"ЦБ", "21%"},
        channel_authority=20.0,
        originality_score=1.0,
        has_primary_evidence=True,
    )
    assert res_a.cluster_id == res_b.cluster_id
    # До правки канал B опережает за счет Authority и мнимого t_min=12:00:00
    assert res_a.origin_post_id == 5001

    # Шаг 3: В 12:04:00 канал B дописывает заглушку (изменение > 30% символов)
    edit_res = clusterer.handle_post_edited(
        post_id=5001,
        new_text=rewritten_b_text,
        edit_date=t_b_edit,
        new_embedding=vec[:512],
    )
    assert edit_res is not None
    assert edit_res.decision == "edit_switched_to_edit_date"
    # t_B_effective теперь 12:04:00, t_min = 12:00:30 (канал A), штраф B = 0.50 * 3.5 = 1.75 -> побеждает канал A!
    assert edit_res.origin_post_id == 5002
    assert edit_res.origin_channel_id == "channel_A_original"

    # Небольшая правка опечатки (<= 30%) НЕ должна переключать effective timestamp на edit_date
    eff_minor, ratio_minor, switched_minor = resolve_effective_published_at(
        t_a_orig,
        t_a_orig + timedelta(minutes=10),
        full_a_text,
        full_a_text.replace("регулятора.", "регулятора!"),
    )
    assert ratio_minor <= 0.30
    assert switched_minor is False
    assert eff_minor == t_a_orig
