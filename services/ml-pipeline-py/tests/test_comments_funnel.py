"""Тесты трехступенчатой воронки комментариев: GraphScore, пинг-понг, bug rescue, MMR, closure, бакеты."""

from __future__ import annotations

from app.comments import (
    CommentFunnelSummarizer,
    RawComment,
    resolve_comment_bucket,
)


def test_three_stage_comment_funnel_ping_pong_rescue_mmr_closure_and_buckets() -> None:
    summarizer = CommentFunnelSummarizer()

    # Проверка разрешения бакетов 20 / 50 / 150 / 500
    assert resolve_comment_bucket(5) == 20
    assert resolve_comment_bucket(20) == 20
    assert resolve_comment_bucket(21) == 50
    assert resolve_comment_bucket(50) == 50
    assert resolve_comment_bucket(51) == 150
    assert resolve_comment_bucket(500) == 500

    comments: list[RawComment] = [
        # Шум и боты (ступень 1 отбрасывает)
        RawComment(comment_id=1, post_id=77, author_hash="u_noise1", text="Первый!"),
        RawComment(comment_id=2, post_id=77, author_hash="u_noise2", text="🔥🔥🔥"),
        RawComment(
            comment_id=3,
            post_id=77,
            author_hash="u_bot",
            text="Заработок от 50000 в день, переходи в профиль!",
            is_spam_bot=True,
        ),
        # Длинная невыбранная или родительская реплика (#c98 > 200 символов для проверки Context-Ancestor Closure)
        RawComment(
            comment_id=98,
            post_id=77,
            author_hash="u_arch",
            text=(
                "Мы тестировали переход на новый драйвер балансировки соединений в продакшене "
                "при высокой конкурентной нагрузке, и в целом среднее время ответа снизилось на двадцать процентов "
                "после настройки пула воркеров и уменьшения таймаутов чтения сокета до двух секунд."
            ),
        ),
        # Цепочка пинг-понга длиной 3 (>= 3 чередующихся ответов разных авторов: u_arch -> u_dev -> u_arch)
        RawComment(
            comment_id=102,
            post_id=77,
            author_hash="u_dev",
            reply_to_comment_id=98,
            text="Но под нагрузкой пул блокирует event loop и растет задержка.",
        ),
        RawComment(
            comment_id=103,
            post_id=77,
            author_hash="u_arch",
            reply_to_comment_id=102,
            text="Да, если keepalive_timeout > 60s — воспроизвел локально.",
        ),
        # Одиночный лист без ответов (GraphScore == 0), НО с баг-репортом (должен спастись RuModernBERT-small / regex)
        RawComment(
            comment_id=200,
            post_id=77,
            author_hash="u_sre1",
            text="После обновления миграция падает с ошибкой HTTP 502 и OOM kill в воркере.",
        ),
        # Повторные жалобы на ту же ошибку (схлопнутся в [Подтверждено в N аналогичных комментариях])
        RawComment(
            comment_id=201,
            post_id=77,
            author_hash="u_sre2",
            text="После обновления миграция тоже падает с ошибкой HTTP 502 и OOM kill в воркере!",
        ),
        RawComment(
            comment_id=202,
            post_id=77,
            author_hash="u_sre3",
            text="После обновления миграция у нас тоже падает с ошибкой HTTP 502 и OOM kill в воркере.",
        ),
        # Одиночный бесполезный комментарий без ответов (GraphScore == 0, НЕ баг-репорт -> не спасается)
        RawComment(
            comment_id=300,
            post_id=77,
            author_hash="u_chat",
            text="Интересная заметка, почитаю завтра вечером за чаем.",
        ),
    ]

    res = summarizer.summarize(post_id=77, comments=comments, max_selected=4)
    assert res.bucket == 20
    assert res.cache_hit is False
    assert res.dense_comments_vector is not None
    assert len(res.dense_comments_vector) == 512

    selected_ids = [item.comment_id for item in res.selected_comments]
    # Шум #1, #2, #3 и пустая болтовня #300 не должны попасть в выборку
    assert 1 not in selected_ids
    assert 2 not in selected_ids
    assert 3 not in selected_ids
    assert 300 not in selected_ids

    # Проверяем пинг-понг (c98/c102/c103) и ОДИН спасенный баг-репорт из группы {200, 201, 202}
    # со счетчиком 2 аналогичных подтверждений (остальные два схлопнуты MMR-дедупликацией)
    bug_group = {200, 201, 202}
    selected_bugs = [it for it in res.selected_comments if it.comment_id in bug_group]
    assert len(selected_bugs) == 1
    assert selected_bugs[0].is_rescued_bug_report is True
    assert selected_bugs[0].graph_score == 0.0
    assert selected_bugs[0].confirmations_count == 2
    assert "[Подтверждено в 2 аналогичных комментариях]" in res.digest_text

    # Проверяем отдельным вызовом max_selected=1 на c102 с принудительно длинным предком c98 -> Context-Ancestor Closure <= 200
    res_closure = summarizer.summarize(
        post_id=78,
        comments=[
            comments[3],  # #98 (> 200 символов)
            RawComment(
                comment_id=102,
                post_id=78,
                author_hash="u_dev",
                reply_to_comment_id=98,
                text="Воспроизвел регрессию: здесь deadlock и traceback в модуле пула!",
            ),
            RawComment(
                comment_id=103,
                post_id=78,
                author_hash="u_arch",
                reply_to_comment_id=102,
                text="Подтверждаю, нужно откатить флаг.",
            ),
        ],
        max_selected=3,
    )
    items_by_id = {it.comment_id: it for it in res_closure.selected_comments}
    assert 98 in items_by_id
    # Если предок не выбран (например max_selected=1 для одиночного ответа на длинный удаленный/отфильтрованный корень):
    res_orphan_reply = summarizer.summarize(
        post_id=79,
        comments=[
            RawComment(comment_id=98, post_id=79, author_hash="u0", text="А" * 240, is_service=True),
            RawComment(
                comment_id=102,
                post_id=79,
                author_hash="u1",
                reply_to_comment_id=98,
                text="В логах Traceback exception и HTTP 500 на POST /v1/auth!",
            ),
        ],
        max_selected=2,
    )
    assert len(res_orphan_reply.selected_comments) == 1
    item_102 = res_orphan_reply.selected_comments[0]
    assert item_102.ancestor_preview is not None
    assert len(item_102.ancestor_preview) <= 200
    assert "[#c102 -> ответ на #c98 " in item_102.formatted_text

    # Проверяем кэширование в бакете 20: повторный запрос при N=12 (<20) дает cache_hit=True,
    # а при N=25 (переход в бакет 50) пересчитывает кэш.
    res_hit = summarizer.summarize(post_id=77, comments=comments + comments[:2], max_selected=4)
    assert res_hit.bucket == 20
    assert res_hit.cache_hit is True

    comments_25 = list(comments) + [
        RawComment(comment_id=1000 + i, post_id=77, author_hash=f"extra_{i}", text=f"Коммент #{i}")
        for i in range(15)
    ]
    res_next_bucket = summarizer.summarize(post_id=77, comments=comments_25, max_selected=4)
    assert res_next_bucket.bucket == 50
    assert res_next_bucket.cache_hit is False


def test_transformers_bug_rescue_backend() -> None:
    """Проверяет TransformersBugRescueBackend (RuModernBERT-small) при GraphScore == 0."""
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from app.comments import BugReportRescuer, RawComment, TransformersBugRescueBackend, filter_and_score_comment_tree

    tok = Tokenizer(WordLevel(vocab={"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")

    def _dummy_rescue_model(input_ids, attention_mask, **kwargs):
        _ = attention_mask, kwargs
        bsz = input_ids.shape[0]
        logits = np.zeros((bsz, 2), dtype=np.float32)
        logits[:, 1] = 4.0  # высокий логит полезного репорта
        return logits

    backend = TransformersBugRescueBackend(
        "deepvk/RuModernBERT-small",
        tokenizer=hf_tok,
        model=_dummy_rescue_model,
    )
    rescuer = BugReportRescuer(backend=backend, model_threshold=0.55)
    scored, _ = filter_and_score_comment_tree(
        [
            RawComment(
                comment_id=501,
                post_id=88,
                author_hash="u_leaf",
                text="Наблюдаем деградацию латентности базы при пиковой записи после деплоя.",
            )
        ]
    )
    assert scored[0].graph_score == 0.0
    retained = rescuer.rescue_zero_graph_comments(scored)
    assert len(retained) == 1
    assert retained[0].is_rescued_bug_report is True
    assert retained[0].rescue_score > 0.90
