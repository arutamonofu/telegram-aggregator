-- ============================================================================
-- PostgreSQL 16 DDL: Схема core_public (Глобальный дедуплицированный индекс)
-- Инвариант 152-ФЗ: здесь ФИЗИЧЕСКИ запрещено хранение ПДн читателей,
-- email, cookie, IP-адресов и пользовательских историй поиска.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS core_public;

-- 1. Перечисления (ENUMs)
CREATE TYPE core_public.channel_tier_enum AS ENUM ('TIER_1', 'TIER_2', 'TIER_3');

CREATE TYPE core_public.session_state_enum AS ENUM (
    'WARMING_UP',
    'STANDBY_READY',
    'ACTIVE_STREAMING',
    'ACTIVE_POLLING',
    'DEGRADED_QUARANTINE',
    'FROZEN_APPEAL',
    'TERMINATED_DEAD'
);

CREATE TYPE core_public.spam_class_enum AS ENUM (
    'organic',
    'native_ad',
    'mutual_pr',
    'clickbait',
    'crypto_scam'
);

CREATE TYPE core_public.stance_enum AS ENUM ('PRO', 'CON', 'MIXED', 'NEUTRAL');

CREATE TYPE core_public.media_kind_enum AS ENUM ('photo', 'document', 'video', 'voice', 'sticker');

-- 2. Таблица публичных каналов (разделение канальных курсоров от сессионных)
CREATE TABLE core_public.channels (
    channel_id              BIGINT PRIMARY KEY,                     -- MTProto channel_id (без префикса -100)
    username                VARCHAR(64) UNIQUE,                     -- Публичный @username без '@'
    title                   TEXT NOT NULL,
    tier                    core_public.channel_tier_enum NOT NULL DEFAULT 'TIER_2',
    last_msg_id             INTEGER NOT NULL DEFAULT 0,             -- Глобальный водяной знак (Watermark)
    max_edit_date_checked   TIMESTAMPTZ,                            -- Максимальный проверенный edit_date
    linked_chat_id          BIGINT,                                 -- Кэш ID связанной дискуссионной супергруппы
    subscribers_count       INTEGER NOT NULL DEFAULT 1,             -- N_c: число подписчиков внутри нашей платформы
    tg_participants_count   INTEGER,                                -- Размер аудитории в Telegram
    lead_prior              REAL NOT NULL DEFAULT 0.5               -- Байесовский приоритет первоисточника (за 90 дней)
        CHECK (lead_prior >= 0.0 AND lead_prior <= 1.0),
    poll_interval_sec       INTEGER NOT NULL DEFAULT 900,           -- Интервал поллинга (Dev: 900 с)
    next_poll_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pinned_msg_ids          INTEGER[] NOT NULL DEFAULT '{}',        -- Закрепленные посты (рескан с затуханием)
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_channels_poll_queue
    ON core_public.channels (next_poll_at ASC)
    WHERE is_active = TRUE;

CREATE INDEX idx_channels_tier_nc
    ON core_public.channels (tier, subscribers_count DESC)
    WHERE is_active = TRUE;

-- 3. Таблица MTProto-сессий и кэш per-account access_hash
CREATE TABLE core_public.mtproto_sessions (
    account_id              SERIAL PRIMARY KEY,
    phone_hash              VARCHAR(64) UNIQUE NOT NULL,            -- SHA-256 от номера технического аккаунта
    is_premium              BOOLEAN NOT NULL DEFAULT FALSE,
    fsm_state               core_public.session_state_enum NOT NULL DEFAULT 'ACTIVE_POLLING',
    risk_score              SMALLINT NOT NULL DEFAULT 0 CHECK (risk_score BETWEEN 0 AND 100),
    proxy_ipv4              INET,                                   -- Привязанный статический ISP IPv4 (SOCKS5)
    assigned_channels_count SMALLINT NOT NULL DEFAULT 0,
    flood_wait_until        TIMESTAMPTZ,                            -- Персистентный таймер карантина (при FLOOD_WAIT > 60 с)
    bbolt_file_path         TEXT NOT NULL,                          -- Путь к изолированному bbolt файлу сессии
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE core_public.channel_access_hashes (
    account_id              INTEGER NOT NULL REFERENCES core_public.mtproto_sessions(account_id) ON DELETE CASCADE,
    peer_id                 BIGINT NOT NULL,                        -- channel_id или linked_chat_id
    peer_type               VARCHAR(16) NOT NULL CHECK (peer_type IN ('channel', 'discussion')),
    access_hash             BIGINT NOT NULL,                        -- Привязан строго к auth_key аккаунта
    resolved_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, peer_id)
);

-- 4. Таблица событийных кластеров (скользящее окно дедупликации 48-72 ч)
CREATE TABLE core_public.event_clusters (
    cluster_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    window_start            TIMESTAMPTZ NOT NULL,
    window_end              TIMESTAMPTZ NOT NULL,
    medoid_post_id          BIGINT,                                 -- Текущий пост-медоид M_j для сверки Qwen3-Reranker
    origin_post_id          BIGINT,                                 -- Пост-первоисточник с максимальным OriginScore
    origin_channel_id       BIGINT REFERENCES core_public.channels(channel_id) ON DELETE SET NULL,
    posts_count             INTEGER NOT NULL DEFAULT 1,
    channel_ids             BIGINT[] NOT NULL DEFAULT '{}',         -- Уникальные каналы в кластере (для бейджа "Еще в N")
    centroid_embedding      REAL[]                                  -- 512d центроид mu_j
        CHECK (centroid_embedding IS NULL OR array_length(centroid_embedding, 1) = 512),
    extracted_entities      JSONB NOT NULL DEFAULT '{"ner": [], "numbers": [], "tickers": []}',
    canonical_event_card    JSONB,                                  -- Атомарная Map-выжимка, neutral_title, CQE-спаны
    stance_matrix           JSONB,                                  -- Предрассчитанная матрица ЗА vs ПРОТИВ (при наличии спора)
    last_hdbscan_checked_at TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_event_clusters_window
    ON core_public.event_clusters (window_end DESC, posts_count DESC);

CREATE INDEX idx_event_clusters_channel_ids_gin
    ON core_public.event_clusters USING GIN (channel_ids);

-- 5. Таблица постов (каноническое хранилище + метаданные классификации)
CREATE TABLE core_public.posts (
    post_id                 BIGSERIAL PRIMARY KEY,
    channel_id              BIGINT NOT NULL REFERENCES core_public.channels(channel_id) ON DELETE CASCADE,
    msg_id                  INTEGER NOT NULL,
    grouped_id              BIGINT,                                 -- ID медиа-альбома (если != NULL)
    album_msg_ids           INTEGER[],                              -- Все msg_id альбома (2..10 элементов)
    published_at            TIMESTAMPTZ NOT NULL,                   -- Оригинальная дата поста в Telegram
    edit_date               TIMESTAMPTZ,
    effective_published_at  TIMESTAMPTZ NOT NULL,                   -- Может сдвигаться на edit_date при >30% правке
    edit_version            INTEGER NOT NULL DEFAULT 1,
    text_raw                TEXT NOT NULL DEFAULT '',               -- Исходный текст (rtrim), база для UTF-16 AST
    text_nfkc               TEXT NOT NULL DEFAULT '',               -- NFKC-нормализованный текст для индексации и Смита-Ватермана
    content_sha256          BYTEA NOT NULL,                         -- Контроль реального изменения при edit_date
    entities_json           JSONB NOT NULL DEFAULT '[]',            -- Массив Telegram MessageEntity (UTF-16)
    has_media               BOOLEAN NOT NULL DEFAULT FALSE,
    media_manifest_json     JSONB NOT NULL DEFAULT '[]',            -- [{file_id, media_kind, mime_type, width, height, caption}]
    code_blocks_text        TEXT,                                   -- Выделенный текст messageEntityPre для dense_code
    ocr_text                TEXT,                                   -- Распознанный текст медиа для dense_ocr
    views_count             INTEGER NOT NULL DEFAULT 0,
    forwards_count          INTEGER NOT NULL DEFAULT 0,
    replies_count           INTEGER NOT NULL DEFAULT 0,             -- Актуальный replies.replies из Telegram
    replies_max_id          INTEGER NOT NULL DEFAULT 0,
    replies_synced_count    INTEGER NOT NULL DEFAULT 0,             -- Число уже выгруженных комментариев
    is_ad                   BOOLEAN NOT NULL DEFAULT FALSE,
    spam_class              core_public.spam_class_enum NOT NULL DEFAULT 'organic',
    spam_probability        REAL NOT NULL DEFAULT 0.0,
    neutral_headline        TEXT,                                   -- Антикликбейт-заголовок (шаг Global Map)
    atomic_summary          TEXT,                                   -- 2-предложенная выжимка
    cqe_spans_json          JSONB NOT NULL DEFAULT '[]',            -- Извлеченные дословные CQE-спаны
    cluster_id              UUID REFERENCES core_public.event_clusters(cluster_id) ON DELETE SET NULL,
    origin_score            REAL,
    stance                  core_public.stance_enum NOT NULL DEFAULT 'NEUTRAL',
    qdrant_point_ids        UUID[] NOT NULL DEFAULT '{}',           -- 1 точка (<=2000 симв.) или 2-3 чанка
    is_deleted              BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_posts_channel_msg UNIQUE (channel_id, msg_id)
);

CREATE INDEX idx_posts_channel_published
    ON core_public.posts (channel_id, published_at DESC)
    WHERE is_deleted = FALSE;

CREATE INDEX idx_posts_feed_organic
    ON core_public.posts (published_at DESC, channel_id)
    WHERE is_deleted = FALSE AND is_ad = FALSE;

CREATE INDEX idx_posts_cluster
    ON core_public.posts (cluster_id, origin_score DESC NULLS LAST)
    WHERE cluster_id IS NOT NULL AND is_deleted = FALSE;

-- Таблица идемпотентной дедупликации приемника по композитному ключу (channel_id, msg_id, edit_date)
-- edit_date_norm = COALESCE(edit_date, '1970-01-01T00:00:00Z'::timestamptz)
CREATE TABLE core_public.ingest_idempotency (
    channel_id              BIGINT NOT NULL,
    msg_id                  INTEGER NOT NULL,
    edit_date_norm          TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01T00:00:00Z'::timestamptz,
    processed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (channel_id, msg_id, edit_date_norm)
);

-- Внешние ссылки от event_clusters к posts (отложенные для избежания циклической зависимости при вставке)
ALTER TABLE core_public.event_clusters
    ADD CONSTRAINT fk_clusters_medoid_post
        FOREIGN KEY (medoid_post_id) REFERENCES core_public.posts(post_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT fk_clusters_origin_post
        FOREIGN KEY (origin_post_id) REFERENCES core_public.posts(post_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;

-- 6. Таблица маппинга смещений cp_to_utf16 и клауз для Glass-box RAG
CREATE TABLE core_public.post_utf16_offsets (
    post_id                 BIGINT PRIMARY KEY REFERENCES core_public.posts(post_id) ON DELETE CASCADE,
    cp_length               INTEGER NOT NULL,                       -- Длина строки в Unicode Codepoints (Python len)
    utf16_length            INTEGER NOT NULL,                       -- Длина строки в 16-битных единицах UTF-16 (JS .length)
    has_non_bmp             BOOLEAN NOT NULL DEFAULT FALSE,         -- Оптимизация: если FALSE, cp_idx == utf16_offset
    cp_to_utf16             INTEGER[] NOT NULL,                     -- Массив длины cp_length + 1: cp_to_utf16[cp_idx] = utf16
    non_bmp_anchor_points   JSONB NOT NULL DEFAULT '[]',            -- [{"cp_idx": 0, "cum_shift": 1}]
    sentences_json          JSONB NOT NULL DEFAULT '[]',            -- [{"sentence_id": 1, "tag": "P42_S1", "start_cp": 0, "end_cp": 44, "start_utf16": 0, "end_utf16": 45, "text": "..."}]
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. Таблица комментариев и бакет-кэша выжимок комментариев
CREATE TABLE core_public.post_comments (
    comment_id              BIGSERIAL PRIMARY KEY,
    post_id                 BIGINT NOT NULL REFERENCES core_public.posts(post_id) ON DELETE CASCADE,
    channel_id              BIGINT NOT NULL,
    post_msg_id             INTEGER NOT NULL,
    linked_chat_id          BIGINT NOT NULL,
    comment_msg_id          INTEGER NOT NULL,
    reply_to_msg_id         INTEGER,                                -- Родитель в дереве обсуждения
    author_peer_hash        VARCHAR(32) NOT NULL,                   -- Хеш автора для подсчета UniqueAuthors(Subtree)
    published_at            TIMESTAMPTZ NOT NULL,
    edit_date               TIMESTAMPTZ,
    text_raw                TEXT NOT NULL,
    entities_json           JSONB NOT NULL DEFAULT '[]',
    out_degree              SMALLINT NOT NULL DEFAULT 0,
    unique_authors_subtree  SMALLINT NOT NULL DEFAULT 0,
    ping_pong_depth         SMALLINT NOT NULL DEFAULT 0,
    graph_score             REAL NOT NULL DEFAULT 0.0,
    is_bug_report           BOOLEAN NOT NULL DEFAULT FALSE,         -- Флаг спасения через RuModernBERT-small
    semantic_class          VARCHAR(32),                            -- bug_report | constructive_counterargument | personal_case | noise
    mmr_duplicate_count     SMALLINT NOT NULL DEFAULT 1,
    is_deleted              BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_post_comments_msg UNIQUE (linked_chat_id, comment_msg_id)
);

CREATE INDEX idx_post_comments_post_tree
    ON core_public.post_comments (post_id, comment_msg_id ASC)
    WHERE is_deleted = FALSE;

CREATE TABLE core_public.comment_summaries (
    post_id                 BIGINT NOT NULL REFERENCES core_public.posts(post_id) ON DELETE CASCADE,
    bucket_threshold        SMALLINT NOT NULL CHECK (bucket_threshold IN (20, 50, 150, 500)),
    replies_count_at_calc   INTEGER NOT NULL,
    summary_text            TEXT NOT NULL,
    structured_points_json  JSONB NOT NULL,                         -- Ключевые баги, контраргументы, кейсы со ссылками [#c102]
    included_comment_ids    INTEGER[] NOT NULL,                     -- Отобранные после 88% сжатия comment_msg_id
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (post_id, bucket_threshold)
);

-- 8. File Reference Table и File Source Table (автообновление истекших file_reference)
CREATE TABLE core_public.file_references (
    file_id                 VARCHAR(128) PRIMARY KEY,               -- Канонический ID: "{media_kind}:{dc_id}:{tg_id}:{thumb_size}"
    tg_id                   BIGINT NOT NULL,                        -- InputPhotoFileLocation.ID / InputDocumentFileLocation.ID
    access_hash             BIGINT NOT NULL,
    dc_id                   SMALLINT NOT NULL,
    media_kind              core_public.media_kind_enum NOT NULL,
    thumb_size              VARCHAR(16) NOT NULL DEFAULT '',        -- Строковый PhotoSize.Type ("m", "x", "y" или "" для оригинала)
    file_reference          BYTEA NOT NULL,                         -- Текущий криптографический токен file_reference
    file_size               BIGINT NOT NULL DEFAULT 0,
    mime_type               VARCHAR(128),
    phash                   BIGINT,                                 -- Перцептивный хэш для дедупликации медиа (L0 + S3)
    local_cache_path        TEXT,                                   -- Dev: путь в локальном дисковом кэше (TTL 24ч)
    local_cache_expires_at  TIMESTAMPTZ,
    refreshed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_permanently_gone     BOOLEAN NOT NULL DEFAULT FALSE          -- TRUE, если пост удален во всех FileSource -> HTTP 410 Gone
);

CREATE INDEX idx_file_references_phash
    ON core_public.file_references (phash)
    WHERE phash IS NOT NULL;

CREATE TABLE core_public.file_sources (
    source_id               BIGSERIAL PRIMARY KEY,
    file_id                 VARCHAR(128) NOT NULL REFERENCES core_public.file_references(file_id) ON DELETE CASCADE,
    channel_id              BIGINT NOT NULL REFERENCES core_public.channels(channel_id) ON DELETE CASCADE,
    msg_id                  INTEGER NOT NULL,
    grouped_id              BIGINT,                                 -- Если файл в альбоме
    album_msg_ids           INTEGER[] NOT NULL DEFAULT '{}',        -- Все ID альбома для пакетного channels.getMessages
    is_source_deleted       BOOLEAN NOT NULL DEFAULT FALSE,
    last_refresh_attempt_at TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_file_sources_location UNIQUE (file_id, channel_id, msg_id)
);

CREATE INDEX idx_file_sources_lookup
    ON core_public.file_sources (file_id, is_source_deleted, created_at DESC);

-- Триггер автоматического обновления updated_at
CREATE OR REPLACE FUNCTION core_public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_channels_updated_at BEFORE UPDATE ON core_public.channels
    FOR EACH ROW EXECUTE FUNCTION core_public.set_updated_at();
CREATE TRIGGER trg_sessions_updated_at BEFORE UPDATE ON core_public.mtproto_sessions
    FOR EACH ROW EXECUTE FUNCTION core_public.set_updated_at();
CREATE TRIGGER trg_clusters_updated_at BEFORE UPDATE ON core_public.event_clusters
    FOR EACH ROW EXECUTE FUNCTION core_public.set_updated_at();
CREATE TRIGGER trg_posts_updated_at BEFORE UPDATE ON core_public.posts
    FOR EACH ROW EXECUTE FUNCTION core_public.set_updated_at();
CREATE TRIGGER trg_offsets_updated_at BEFORE UPDATE ON core_public.post_utf16_offsets
    FOR EACH ROW EXECUTE FUNCTION core_public.set_updated_at();
