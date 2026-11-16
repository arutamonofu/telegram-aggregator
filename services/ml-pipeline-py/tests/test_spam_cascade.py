"""Тесты каскада фильтрации спама L0 -> L1 -> L2 и сохранения рекламы с is_ad=True."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

from PIL import Image, ImageDraw

from app.spam_filter import (
    MediaPhashIndex,
    MentionGraph14d,
    SpamFilterCascade,
    compute_image_phash,
    normalize_homoglyphs,
)


def _make_synthetic_banner_bytes(*, size: tuple[int, int] = (256, 256)) -> bytes:
    img = Image.new("RGB", size, color=(20, 40, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 230, 90], fill=(250, 210, 40))
    draw.ellipse([70, 110, 190, 230], fill=(230, 60, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_l0_homoglyph_erid_ord_utm_and_short_length() -> None:
    cascade = SpamFilterCascade()

    # 1. Кириллическая 'е' в 'erid' (обфускация омоглифами)
    obfuscated = "Новые курсы по анализу данных с практикой. еrid: LjN8K9pZ2"
    assert "erid:" in normalize_homoglyphs(obfuscated)
    v_erid = cascade.evaluate(obfuscated)
    assert v_erid.stage_decided == "L0"
    assert v_erid.is_ad is True
    assert v_erid.should_persist is True
    assert "l0_erid_token" in v_erid.reasons

    # 2. Токен ОРД + UTM + реферальный бот
    v_ord = cascade.evaluate("Отличный сервис мониторинга серверов, токен ОРД: VK123456")
    assert v_ord.is_ad is True
    assert v_ord.should_persist is True
    assert "l0_ord_ad_marker" in v_ord.reasons

    v_utm = cascade.evaluate("Смотрите подробности https://promo.example.ru/?utm_source=tg_ch")
    assert v_utm.is_ad is True
    assert "l0_utm_tracking" in v_utm.reasons

    v_ref = cascade.evaluate("Получи бонус через бота t.me/super_bonus_bot?start=ref_991")
    assert v_ref.is_ad is True
    assert "l0_referral_bot" in v_ref.reasons

    # 3. Проверка длины (< 15 символов без информативного медиа)
    v_short = cascade.evaluate("Го в чат!")
    assert v_short.stage_decided == "L0"
    assert v_short.is_noise is True
    assert v_short.should_persist is False
    assert "l0_short_text_lt_15" in v_short.reasons


def test_l0_image_phash_detects_recompressed_ad_banner() -> None:
    raw_png = _make_synthetic_banner_bytes(size=(256, 256))
    known_hash = compute_image_phash(raw_png)

    # Пережимаем и ресайзим рекламный креатив в JPEG меньшего размера
    recompressed_img = Image.open(io.BytesIO(raw_png)).resize((180, 180))
    jpg_buf = io.BytesIO()
    recompressed_img.save(jpg_buf, format="JPEG", quality=75)
    incoming_hash = compute_image_phash(jpg_buf.getvalue())

    phash_idx = MediaPhashIndex([known_hash])
    cascade = SpamFilterCascade(phash_index=phash_idx)

    # Текст поста сам по себе не содержит слова erid/UTM — палится только по pHash картинки
    verdict = cascade.evaluate(
        "Полезная подборка сервисов для команд разработки и аналитики за март.",
        media_phashes=[incoming_hash],
    )
    assert verdict.stage_decided == "L0"
    assert verdict.is_ad is True
    assert verdict.spam_class == "native_ad"
    assert verdict.should_persist is True
    assert "l0_phash_ad_creative" in verdict.reasons


def test_l1_l2_last20_cataphora_and_14day_reciprocal_pr_graph() -> None:
    graph = MentionGraph14d(window_days=14, reciprocity_hours=48)
    t0 = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    # Канал partner_daily упоминал tech_digest за 10 часов до текущего поста
    graph.record_mention("partner_daily", "tech_digest", t0 - timedelta(hours=10))

    cascade = SpamFilterCascade(mention_graph=graph)

    # Взаимный пиар: ссылка в последних 20% текста + взаимное упоминание за 48ч + CTA
    mutual_pr_text = (
        "Сегодня разобрали важные архитектурные паттерны для высоконагруженных брокеров сообщений "
        "и практики надежного шардирования без простоя. "
        "Рекомендуем подписаться на коллег @partner_daily и переходить прямо по ссылке https://t.me/partner_daily"
    )
    verdict_pr = cascade.evaluate(
        mutual_pr_text,
        source_channel="@tech_digest",
        published_at=t0,
    )
    assert verdict_pr.is_ad is True
    assert verdict_pr.spam_class == "mutual_pr"
    assert verdict_pr.should_persist is True
    assert verdict_pr.l1_features is not None
    assert verdict_pr.l1_features.is_reciprocal_48h is True
    assert verdict_pr.l1_features.link_in_last_20_percent is True

    # Катафора в первом абзаце + ссылка в хвосте + коммерческий оффер (эскалация на L2)
    native_ad_text = (
        "Мало кто знает, но один инструмент сэкономит вам 10 часов в неделю.\n\n"
        "Мы протестировали новый курс и тариф со скидкой 40% для команды разработки, "
        "смотрите подробности и регистрируйтесь бесплатно по ссылке https://t.me/academy_offer"
    )
    verdict_ad = cascade.evaluate(
        native_ad_text,
        source_channel="@other_tech_blog",
        published_at=t0,
    )
    assert verdict_ad.stage_decided == "L2"
    assert verdict_ad.is_ad is True
    assert verdict_ad.should_persist is True
    assert verdict_ad.l1_features is not None
    assert verdict_ad.l1_features.has_first_para_cataphora is True
    assert verdict_ad.l1_features.link_in_last_20_percent is True

    # Чистый органический пост с техническим упоминанием в середине текста
    organic_text = (
        "Вышел релиз FastAPI 0.162 с поддержкой нового протокола маршрутизации t.me/python_news "
        "и улучшенной производительностью валидации моделей Pydantic при высоких нагрузках."
    )
    verdict_org = cascade.evaluate(
        organic_text,
        source_channel="@python_weekly",
        published_at=t0,
    )
    assert verdict_org.is_ad is False
    assert verdict_org.spam_class == "organic"


def test_simhash_datasketch_imagehash_and_transformers_l2_integration() -> None:
    """Проверяет интеграцию simhash, datasketch MinHashLSH, imagehash и transformers в каскаде."""
    import imagehash
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from app.spam_filter import (
        L2SpamClassifier,
        MinHashLshCreativeIndex,
        SimhashBlocklistIndex,
        TransformersSequenceClassifierBackend,
        compute_text_simhash,
    )

    # 1. imagehash.phash эквивалентность
    png_bytes = _make_synthetic_banner_bytes()
    img = Image.open(io.BytesIO(png_bytes))
    assert compute_image_phash(png_bytes) == int(str(imagehash.phash(img)), 16)

    # 2. simhash.SimhashIndex блоклист
    tmpl = "Только сегодня скидка на интенсив по системному дизайну переходи быстрее"
    sh = compute_text_simhash(tmpl)
    sh_idx = SimhashBlocklistIndex([sh])
    lsh_idx = MinHashLshCreativeIndex(threshold=0.5, num_perm=128)
    lsh_idx.add_text(tmpl, key="ad_tmpl_1")

    cascade_l0 = SpamFilterCascade(simhash_index=sh_idx, minhash_index=lsh_idx)
    verdict_dup = cascade_l0.evaluate(
        "Только сегодня скидка на интенсив по системному дизайну переходи быстрее!"
    )
    assert verdict_dup.stage_decided == "L0"
    assert verdict_dup.is_ad is True
    assert "l0_simhash_duplicate_ad" in verdict_dup.reasons
    assert "l0_minhash_duplicate_ad" in verdict_dup.reasons

    # 3. TransformersSequenceClassifierBackend в L2
    tok = Tokenizer(WordLevel(vocab={"[UNK]": 0, "[PAD]": 1, "курс": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")

    def _dummy_cls_model(input_ids, attention_mask, **kwargs):
        _ = attention_mask, kwargs
        bsz = input_ids.shape[0]
        # Классы: organic, native_ad, mutual_pr, clickbait, crypto_scam -> native_ad выигрывает
        logits = np.zeros((bsz, 5), dtype=np.float32)
        logits[:, 1] = 5.0
        return logits

    l2_backend = TransformersSequenceClassifierBackend(
        "deepvk/GeRaCl-USER2-base",
        tokenizer=hf_tok,
        model=_dummy_cls_model,
    )
    cascade_l2 = SpamFilterCascade(l2_classifier=L2SpamClassifier(backend=l2_backend))
    t0 = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    res_l2 = cascade_l2.evaluate(
        "Обзор архитектурных паттернов и новый материал уже доступен в закрытой группе @partner_hub",
        source_channel="@tech_ch",
        published_at=t0,
    )
    assert res_l2.stage_decided == "L2"
    assert res_l2.spam_class == "native_ad"
    assert res_l2.is_ad is True
    assert res_l2.should_persist is True
