-- ============================================================================
-- PostgreSQL 16 DDL: Схема tenant_rf (Профили, Zero-Trust подписки и квоты)
-- В Стадии 2 (MVP) эта схема физически выносится в кластер pg_users_rf в ЦОД РФ.
-- Ссылка на core_public логическая (без cross-schema FOREIGN KEY для чистоты разреза).
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS tenant_rf;

CREATE TYPE tenant_rf.plan_tier_enum AS ENUM ('TRIAL', 'BASIC', 'PROSUMER', 'B2B_WORKSPACE');
CREATE TYPE tenant_rf.finops_zone_enum AS ENUM ('GREEN', 'YELLOW', 'RED', 'GUARDRAIL');

-- 1. Пользователи (Zero-Trust: без номера телефона Telegram и без auth_key)
CREATE TABLE tenant_rf.users (
    user_id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email                   VARCHAR(255) UNIQUE NOT NULL,
    oauth_provider          VARCHAR(32),                            -- 'email' | 'google' | 'github'
    oauth_subject           VARCHAR(255),
    plan_tier               tenant_rf.plan_tier_enum NOT NULL DEFAULT 'TRIAL',
    trial_ends_at           TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Подписки пользователя (логическая связь с core_public.channels.channel_id)
CREATE TABLE tenant_rf.user_channel_subscriptions (
    user_id                 UUID NOT NULL REFERENCES tenant_rf.users(user_id) ON DELETE CASCADE,
    channel_id              BIGINT NOT NULL,                        -- Логический ID канала в core_public
    folder_tag              VARCHAR(64) NOT NULL DEFAULT 'default',
    added_via               VARCHAR(32) NOT NULL DEFAULT 'manual',  -- 'manual' | 'addlist' | 'preset'
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, channel_id)
);

CREATE INDEX idx_user_subs_user ON tenant_rf.user_channel_subscriptions (user_id, folder_tag);

-- 3. Витрина отраслевых пресетов (для старта в 1 клик)
CREATE TABLE tenant_rf.channel_presets (
    preset_id               VARCHAR(64) PRIMARY KEY,                -- 'it_dev', 'ai_ds', 'product_mgmt', 'macro', 'legal_tax'
    title                   VARCHAR(128) NOT NULL,
    description             TEXT NOT NULL,
    channel_ids             BIGINT[] NOT NULL,
    sort_order              SMALLINT NOT NULL DEFAULT 0,
    is_active               BOOLEAN NOT NULL DEFAULT TRUE
);

-- 4. Квоты триала и учет TechCOGS (FinOps)
CREATE TABLE tenant_rf.user_finops_quotas (
    user_id                 UUID NOT NULL REFERENCES tenant_rf.users(user_id) ON DELETE CASCADE,
    billing_period          CHAR(7) NOT NULL,                       -- 'YYYY-MM'
    rag_queries_used        INTEGER NOT NULL DEFAULT 0,
    rag_queries_limit       INTEGER NOT NULL DEFAULT 30,            -- 30 на весь триал / в мес. на BASIC; -1 для безлимита
    accumulated_tech_cogs_rub NUMERIC(10, 2) NOT NULL DEFAULT 0.00, -- Spend_u для маршрутизации Green/Yellow/Red
    current_zone            tenant_rf.finops_zone_enum NOT NULL DEFAULT 'GREEN',
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, billing_period)
);

-- Триггер автоматического обновления updated_at в схеме tenant_rf
CREATE OR REPLACE FUNCTION tenant_rf.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at BEFORE UPDATE ON tenant_rf.users
    FOR EACH ROW EXECUTE FUNCTION tenant_rf.set_updated_at();
CREATE TRIGGER trg_user_finops_quotas_updated_at BEFORE UPDATE ON tenant_rf.user_finops_quotas
    FOR EACH ROW EXECUTE FUNCTION tenant_rf.set_updated_at();
