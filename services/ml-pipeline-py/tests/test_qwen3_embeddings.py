"""Тесты last-token пулинга, MRL-усечения до L2-нормы и разделения doc/query Qwen3."""

from __future__ import annotations

import numpy as np
import pytest

from app.embeddings.qwen3_embedder import (
    Qwen3EmbeddingClient,
    last_token_pool,
    mrl_truncate_and_normalize,
)


class _RecordingBackbone:
    """Фиктивный 1024d бэкенд для проверки точности вызова и прохождения текста."""

    def __init__(self, width: int = 1024) -> None:
        self.width = width
        self.seen_texts: list[str] = []

    def encode_raw(self, texts):
        self.seen_texts.extend(texts)
        rows = []
        for idx, _ in enumerate(texts):
            vec = np.linspace(0.5, 2.0, self.width, dtype=np.float32) + idx * 0.1
            rows.append(vec)
        return np.stack(rows, axis=0)


def test_last_token_pool_left_and_right_padding() -> None:
    """Проверяет last_token_pool на left padding и right padding."""
    # Batch=2, Seq=4, Hidden=3
    hidden = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [9, 9, 9]],
            [[4, 4, 4], [7, 7, 7], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.float32,
    )
    # Left padding: для обеих строк последний токен — индекс 3
    left_mask = np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int64)
    pooled_left = last_token_pool(hidden, left_mask)
    np.testing.assert_allclose(pooled_left[0], [9, 9, 9])
    np.testing.assert_allclose(pooled_left[1], [0, 0, 0])

    # Right padding: строка 0 имеет 3 токена (индекс 2 -> [3,3,3]), строка 1 имеет 2 токена (индекс 1 -> [7,7,7])
    right_mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int64)
    pooled_right = last_token_pool(hidden, right_mask)
    np.testing.assert_allclose(pooled_right[0], [3, 3, 3])
    np.testing.assert_allclose(pooled_right[1], [7, 7, 7])


def test_mrl_truncation_512_happens_before_l2_normalization() -> None:
    """Проверяет срез первых 512 координат ДО L2-нормализации (критично для MRL)."""
    # Создаем 1024-мерный вектор, у которого первые 512 компонент малы (0.1), а [512:] велики (10.0).
    # Если сначала L2-нормализовать 1024d, а потом отрезать [:512] БЕЗ ренормализации, норма != 1.0!
    raw = np.zeros((2, 1024), dtype=np.float32)
    raw[:, :512] = 0.1
    raw[:, 512:] = 10.0

    mrl = mrl_truncate_and_normalize(raw, dim=512)
    assert mrl.shape == (2, 512)
    norms = np.linalg.norm(mrl, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    # Ошибочный порядок (сначала полная нормализация, потом срез) НЕ дал бы единичной нормы
    wrong = (raw / np.linalg.norm(raw, axis=1, keepdims=True))[:, :512]
    assert np.linalg.norm(wrong[0]) < 0.05


def test_encode_documents_forbids_instruct_while_queries_require_it() -> None:
    """Проверяет отсутствие префикса для документов и наличие Instruct: для запросов."""
    backbone = _RecordingBackbone(width=1024)
    client = Qwen3EmbeddingClient(backbone=backbone, mrl_dim=512)

    doc_vec = client.encode_document("ЦБ РФ повысил ключевую ставку до 21%")
    assert doc_vec.shape == (512,)
    np.testing.assert_allclose(np.linalg.norm(doc_vec), 1.0, atol=1e-6)
    assert backbone.seen_texts[-1] == "ЦБ РФ повысил ключевую ставку до 21%"
    assert not backbone.seen_texts[-1].startswith("Instruct:")

    # Документный метод должен отвергать случайно переданный query-шаблон
    with pytest.raises(ValueError, match="without Qwen3 query Instruct prefix"):
        client.encode_document("Instruct: retrieve\nQuery: ставка ЦБ")

    query_vec = client.encode_query("какая ключевая ставка ЦБ")
    assert query_vec.shape == (512,)
    np.testing.assert_allclose(np.linalg.norm(query_vec), 1.0, atol=1e-6)
    assert backbone.seen_texts[-1].startswith("Instruct: ")
    assert "\nQuery: какая ключевая ставка ЦБ" in backbone.seen_texts[-1]


def _make_local_hf_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<|endoftext|>": 0,
        "[UNK]": 1,
        "yes": 2,
        "no": 3,
        "ЦБ": 4,
        "ставка": 5,
        "21": 6,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<|endoftext|>",
        unk_token="[UNK]",
        padding_side="left",
    )


def test_transformers_qwen3_embedding_and_reranker_backends() -> None:
    """Проверяет интеграцию `transformers` (PreTrainedTokenizerFast + адаптеры Qwen3)."""
    from app.embeddings import (
        Qwen3RerankerClient,
        TransformersQwen3EmbeddingBackbone,
        TransformersQwen3RerankerBackbone,
    )

    hf_tok = _make_local_hf_tokenizer()

    def _dummy_embed_model(input_ids, attention_mask, **kwargs):
        _ = kwargs
        bsz, seq_len = input_ids.shape
        hidden = np.zeros((bsz, seq_len, 1024), dtype=np.float32)
        for row_idx in range(bsz):
            for pos in range(seq_len):
                if attention_mask[row_idx, pos] > 0:
                    hidden[row_idx, pos, :512] = float(pos + row_idx + 1)
                    hidden[row_idx, pos, 512:] = 50.0
        return hidden

    emb_backbone = TransformersQwen3EmbeddingBackbone(
        "Qwen/Qwen3-Embedding-0.6B",
        tokenizer=hf_tok,
        model=_dummy_embed_model,
    )
    emb_client = Qwen3EmbeddingClient(backbone=emb_backbone, mrl_dim=512)
    vecs = emb_client.encode_documents(["ЦБ ставка", "ЦБ ставка 21"])
    assert vecs.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), [1.0, 1.0], atol=1e-6)

    def _dummy_rerank_model(input_ids, attention_mask, **kwargs):
        _ = attention_mask, kwargs
        bsz, seq_len = input_ids.shape
        logits = np.zeros((bsz, seq_len, 10), dtype=np.float32)
        logits[:, -1, 2] = 4.0  # 'yes'
        logits[:, -1, 3] = 1.0  # 'no'
        return logits

    rerank_backbone = TransformersQwen3RerankerBackbone(
        "Qwen/Qwen3-Reranker-0.6B",
        tokenizer=hf_tok,
        model=_dummy_rerank_model,
    )
    reranker = Qwen3RerankerClient(backbone=rerank_backbone)
    prob = reranker.score_pair("ставка 21", "ЦБ ставка 21")
    # softmax([4.0, 1.0])[0] = exp(4) / (exp(4) + exp(1)) ≈ 0.95257
    assert abs(prob - 0.95257) < 1e-4
