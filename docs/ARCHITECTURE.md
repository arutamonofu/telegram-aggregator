# Техническая архитектура локального прототипа (Стадия 1: Dev)

**Версия документа:** 1.0 (Стадия 1: Dev $\rightarrow$ интерфейсная готовность к Стадии 2: MVP)  
**Базовая спецификация:** [docs/SPEC.md](file:///home/arutamonofu/projects/telegram-aggregator/docs/SPEC.md) (разделы 5, 7.1–7.5, 8.1)  
**Целевая нагрузка Стадии 1:** 50–300 публичных каналов Telegram, 1 000–5 000 постов/сут, 1–5 разработчиков на одной рабочей станции (8 vCPU, 32 ГБ RAM, опционально NVIDIA RTX 3060/4060+ 8–12 ГБ VRAM) или облачной ВМ.

---

## 1. Обзор архитектуры и чистые интерфейсы перехода Dev $\rightarrow$ MVP

Локальный прототип (Стадия 1: Dev) проектируется в одноконтурной среде `docker-compose.dev.yml`, но **с жестким разделением контрактов, схем БД и программных интерфейсов**, чтобы переход на Стадию 2 (MVP: двухконтурная топология РФ $\leftrightarrow$ ЕС по 152-ФЗ / ФЗ № 23-ФЗ, 60 MTProto-сессий в 3 ярусах, GPU-узел vLLM, гибридный S3 Edge-Proxy) выполнялся заменой реализаций за абстракциями без переписывания бизнес-логики, без переиндексации коллекции Qdrant и без изменения AST-рендерера на клиенте.

### 1.1. Топология потоков данных Стадии 1 (Dev)

```mermaid
flowchart TB
    TG["Telegram MTProto DC1-DC5 (Layer 225+)"]
    LLM["Внешние LLM API (GPT-6 Luna / Claude Sonnet 5.5 / DeepSeek V4.1 Flash)"]

    subgraph DEV["Локальная сеть docker-compose.dev.yml"]
        direction TB

        subgraph GO["services/ingestion-go (Go 1.24+, gotd/td v0.145.1+)"]
            G1["SessionPool (1-3 std аккаунта, bbolt)"]
            G2["Gap Engine (UpdateHook + AffectedHook + TooLong Patch)"]
            G3["Album Assembler (1500ms) + Comment Delta Sync"]
            G4["Media Range Streamer + FileRef Auto-Refresh + Disk Cache 24h"]
        end

        REDIS["Redis 7 (Streams: posts / comments / deletions + Semantic Cache)"]
        PG["PostgreSQL 16 (Схемы: core_public + tenant_rf)"]
        QDRANT["Qdrant v1.19+ (Коллекция public_channels_v1: 4x512d + Snowball BM25)"]

        subgraph PY["services/ml-pipeline-py (Python 3.12+, FastAPI)"]
            P1["Stream Consumer + UTF-16 Offset Mapper (cp_to_utf16)"]
            P2["L0 Regex/Homoglyph/pHash + L2 Cold-Start GeRaCl-USER2-base"]
            P3["Shared Backbone: Qwen3-Embedding-0.6B (512d MRL)"]
            P4["Single-Pass Clustering (72h) + OriginScore + Qwen3-Reranker-0.6B"]
            P5["Comment Graph Pruner (RuModernBERT-small ONNX INT8) + MMR"]
            P6["Glass-box RAG: Hybrid Retrieval -> Synthesis -> Smith-Waterman -> mDeBERTa-v3-xnli"]
        end

        subgraph WEB["apps/web-client (Next.js 15 / SvelteKit, TypeScript)"]
            W1["Toggle View: ИИ-Сводка vs Очищенная лента оригиналов"]
            W2["AST-парсер сущностей Layer 225+ + Shiki + Interval Split Pass"]
            W3["SSE-клиент + Media Player (HTTP 206 Partial Content)"]
        end
    end

    TG <-->|"MTProto TCP / SOCKS5"| G1
    G1 --> G2 --> G3 -->|"XADD (sync Entry ID ACK)"| REDIS
    G3 <-->|"last_msg_id, FileRef, FileSource"| PG
    G4 <-->|"upload.getFile 512KB + channels.getMessages"| TG
    G4 <-->|"FileRef / FileSource"| PG

    REDIS -->|"XREADGROUP cg:ml-pipeline"| P1
    P1 --> P2 --> P3 --> P4
    P3 <-->|"512d + BM25 + ACORN-1"| QDRANT
    P4 <-->|"posts, offsets, clusters, comments"| PG
    P5 <-->|"POST /internal/v1/comments/sync"| G3
    P6 <-->|"Structured JSON / Citations API"| LLM
    P6 <-->|"Hybrid Semantic Cache"| REDIS

    W3 <-->|"SSE /api/v1/rag/query & REST"| P6
    W3 -->|"GET /api/v1/media/{file_id} (Range 206)"| P6
    P6 -->|"Proxy HTTP 206"| G4
    W1 --> W2
```

### 1.2. Матрица чистых интерфейсов (Dev $\rightarrow$ MVP $\rightarrow$ Production)

| Абстракция (Интерфейс) | Реализация в Стадии 1 (Dev) | Реализация в Стадии 2 (MVP) | Инвариант контракта, сохраняемый при переходе |
| :--- | :--- | :--- | :--- |
| **`SessionScheduler` (Go)** | `StaticSingleTierScheduler`: 1–3 стандартных аккаунта, единый цикл стриминга/поллинга с интервалом 15 мин без FSM прогрева. | `TieredFSMOrchestrator`: 60 сессий (8 Prem + 52 Std), 3 яруса (Tier 1/2/3), 6 состояний FSM и 21-дневный прогрев. | Внутренние потребители читают только канонические события в Redis Streams (`IngestPostEventV1`, `IngestCommentsDeltaEventV1`). |
| **`MediaStorageProvider` (Go)** | `LocalDiskMediaStore`: прямой стриминг 512 КБ чанков из MTProto с кэшем на локальном диске (TTL 24 ч). | `HybridS3EdgeStore`: упреждающая загрузка превью 35 КБ в S3 Standard РФ + ленивый HTTP 206 стриминг оригиналов с TTL 30 дней. | Единый эндпоинт `GET /internal/v1/media/{file_id}` с поддержкой `Range`, таблицами `file_references` / `file_sources` и ответом `410 Gone`. |
| **`SpamFilterCascade` (Python)** | `ColdStartNLICascade`: L0 (Regex/омоглифы/pHash) + zero-shot NLI `deepvk/GeRaCl-USER2-base` (0.2B). | `TrainedEnsembleCascade`: L0 + L1 (`LightGBM + 2-layer MLP` поверх 512d + TF-IDF + 25 граф-признаков) + L2 (`deepvk/RuModernBERT-base` INT8). | Возвращает единый объект `SpamVerdict(spam_class, spam_probability, is_ad)`. Посты с `is_ad=True` всегда сохраняются в БД и Qdrant с флагом `is_ad`. |
| **`DeduplicationIndex` (Python)** | `UnifiedWindowClusterer`: Single-Pass Clustering в окне 72 ч поверх `public_channels_v1` с фильтром `published_at >= now - 72h`, без `HDBSCAN`. | `DualLevelClusterer`: 15-мин RAM Ring Buffer (brute-force cosine) + 6-часовые темпоральные шарды Qdrant (SQ8) + фоновый `HDBSCAN` каждые 6 ч. | Формат записи `event_clusters` и поле `cluster_id` в payload Qdrant неизменны. |
| **`StorageBoundary` (PostgreSQL)** | Логическое разделение в одном PostgreSQL 16 на две схемы: `core_public` и `tenant_rf` (без физического сетевого шлюза). | Физическое разделение на `core_public` (БД в ЦОД ЕС) и `pg_users_rf` (ЦОД РФ) с PII-Scrubber Gateway и туннелем AmneziaWG 3.1. | Внутри схемы `core_public` **физически запрещены** таблицы пользователей, `user_id`, cookie, IP-адреса и история поисковых запросов; связь идет только по `channel_id`. |
| **`PrivacyBoundaryMiddleware` (Python)** | `NoOpPassThroughMiddleware`: пропускает запросы и ответы напрямую внутри `docker-compose`. | `PiiScrubberClientMiddleware`: удаление `user_id`/IP/Cookie, маскирование ПДн по Приказу РКН № 140 (`[PER_1]`, `[PHONE_1]`) и деанонимизация SSE на шлюзе РФ. | Контракт FastAPI `/api/v1/rag/query` и формат SSE-фреймов не содержат персональных полей в полезной нагрузке моделей. |
| **`EmbeddingInferenceClient` (Python)** | `LocalEmbeddingAdapter`: ONNX Runtime / llama.cpp или локальный vLLM (одна модель `Qwen3-Embedding-0.6B` 512d MRL). | `RemoteVLLMEmbeddingClient`: выделенный GPU-узел L4/L40S с vLLM ($\ge$ v0.8.5, left-padding fix PR #648). | Строгая L2-нормализация 512d-векторов; посты при индексации, спам-фильтрации и кластеризации кодируются **строго без префиксов**, а поисковые запросы RAG — исключительно с англоязычной инструкцией `Instruct: Given a search query, retrieve relevant Telegram posts... Query: ...`. |

---

## 2. Структура монорепозитория и детальное устройство сервисов

### 2.1. Дерево каталогов монорепозитория

```text
telegram-aggregator/
├── docs/
│   ├── SPEC.md                               # Продуктово-техническая спецификация
│   └── ARCHITECTURE.md                       # Архитектура Стадии 1 (Dev) (данный документ)
├── deploy/
│   ├── docker-compose.dev.yml                # Локальный оркестратор Стадии 1
│   ├── .env.dev.example                      # Шаблон переменных окружения
│   ├── postgres/
│   │   └── migrations/
│   │       ├── 0001_init_core_public.sql     # DDL схемы core_public (каналы, посты, смещения, кластеры, FileRef)
│   │       └── 0002_init_tenant_rf.sql       # DDL схемы tenant_rf (пользователи, подписки, пресеты, FinOps квоты)
│   └── qdrant/
│       └── init_public_channels_v1.py        # Скрипт инициализации коллекции public_channels_v1 и индексов
├── packages/
│   └── contracts/                            # Единый источник правды для схем (JSON Schema + OpenAPI + TS)
│       ├── events/
│       │   ├── ingest_post_v1.schema.json
│       │   ├── ingest_comments_v1.schema.json
│       │   └── ingest_deletion_v1.schema.json
│       ├── rag/
│       │   └── glassbox_llm_output.schema.json
│       └── openapi/
│           └── ml_pipeline_v1.yaml
├── services/
│   ├── ingestion-go/                         # Go 1.24+ воркер MTProto сбора и стриминга медиа
│   │   ├── go.mod                            # github.com/gotd/td v0.145.1+, github.com/gotd/contrib, go.etcd.io/bbolt, github.com/redis/go-redis/v9, github.com/jackc/pgx/v5
│   │   ├── Dockerfile
│   │   ├── cmd/
│   │   │   └── ingestor/
│   │   │       └── main.go                   # Точка входа, DI, graceful shutdown
│   │   └── internal/
│   │       ├── config/                       # Конфигурация, лимиты 1-3 аккаунтов, SOCKS5 ISP IPv4
│   │       ├── domain/                       # Интерфейсы SessionScheduler, MediaStorageProvider, WatermarkStore
│   │       ├── mtproto/
│   │       │   ├── client_factory.go         # Сборка telegram.Client + UpdateHook + AffectedHook + floodwait
│   │       │   ├── pool.go                   # Реализация StaticSingleTierScheduler (Dev)
│   │       │   └── too_long_patch.go         # Middleware-патч updates.channelDifferenceTooLong (gotd/td#1711)
│   │       ├── updates/
│   │       │   ├── dispatcher.go             # Обработка updateNewChannelMessage, updateEditChannelMessage, delete
│   │       │   ├── watermark_storage.go      # Обертка updates.StateStorage: коммит pts строго после sync ACK от Redis/PostgreSQL
│   │       │   └── poller.go                 # Единый Dev-поллер messages.getHistory + батч edit_date (getMessages)
│   │       ├── album/
│   │       │   └── assembler.go              # Буферизация медиа-альбомов по (channel_id, grouped_id) в окне 1500 мс
│   │       ├── comments/
│   │       │   └── delta_sync.go             # Однократный GetDiscussionMessage + дельта-пагинация GetReplies (по 100)
│   │       ├── media/
│   │       │   ├── streamer.go               # Чтение upload.getFile чанками 512 КБ (part_size % 1024 == 0)
│   │       │   ├── fileref_refresher.go      # Перехват FILE_REFERENCE_EXPIRED -> FileSource -> ChannelsGetMessages
│   │       │   └── disk_cache.go             # Реализация LocalDiskMediaStore (TTL 24h)
│   │       ├── storage/
│   │       │   ├── bbolt/
│   │       │   │   └── store.go              # bbolt: session.Storage, updates.StateStorage, storage.PeerStorage
│   │       │   └── postgres/
│   │       │       └── repository.go         # Доступ к core_public.channels, channel_access_hashes, file_references, file_sources
│   │       ├── publisher/
│   │       │   └── redis_streams.go          # Публикация в stream:ingest:* с синхронным ожиданием ID записи
│   │       └── api/
│   │           └── http_server.go            # Внутренний HTTP сервер :8081 (media range, comments/sync, channels/register)
│   │
│   └── ml-pipeline-py/                       # Python 3.12+ FastAPI + ML-конвейер
│       ├── pyproject.toml                    # fastapi, qdrant-client>=1.17, psycopg[binary], redis, onnxruntime, transformers, openai, anthropic, httpx
│       ├── Dockerfile
│       └── app/
│           ├── main.py                       # Lifespan, запуск background consumers, FastAPI routers
│           ├── config.py                     # Пороги (tau_low=0.72, tau_high=0.88, p_entail=0.75, weights RRF)
│           ├── consumers/
│           │   ├── post_stream_worker.py     # XREADGROUP stream:ingest:posts -> ML-пайплайн -> Postgres/Qdrant -> XACK
│           │   └── comment_deletion_worker.py# Обработка stream:ingest:comments и stream:ingest:deletions
│           ├── preprocessing/
│           │   ├── normalizer.py             # NFKC-нормализация, выделение блоков кода (messageEntityPre), OCR-текста
│           │   ├── offsets.py                # Построение cp_to_utf16 / utf16_to_cp и сегментация [P{id}_S{sid}]
│           │   └── entities_ner.py           # Экстрактор NER, чисел, тикеров, номеров ФЗ, CLI-флагов (для Jaccard и CacheKey)
│           ├── embeddings/
│           │   ├── qwen3_embedder.py         # Qwen3-Embedding-0.6B -> усечение 512d MRL -> L2-нормализация
│           │   └── qwen3_reranker.py         # Qwen3-Reranker-0.6B (SequenceClassification)
│           ├── spam_filter/
│           │   ├── l0_deterministic.py       # Regex (erid, ОРД, UTM, рефералки), таблица омоглифов, pHash медиа
│           │   └── l2_geracl_nli.py          # Холодный старт zero-shot NLI (deepvk/GeRaCl-USER2-base)
│           ├── clustering/
│           │   ├── single_pass.py            # Скользящее окно 72ч, формула S(P_new, C_j), серая зона через Reranker
│           │   └── origin_score.py           # Расчет OriginScore с учетом t_effective (edit_date при >30% правки)
│           ├── comments/
│           │   ├── graph_pruner.py           # Детерминированная очистка + GraphScore(u) + Context-Ancestor Closure
│           │   ├── bug_rescue.py             # deepvk/RuModernBERT-small (ONNX INT8) + Regex стектрейсов
│           │   └── summarizer.py             # MMR-диверсификация + бакет-кэш (20/50/150/500) + вызов LLM
│           ├── retrieval/
│           │   ├── qdrant_repo.py            # Инициализация public_channels_v1, upsert Multi-Rep, query_points_groups
│           │   └── hybrid_search.py          # 5 prefetch-веток + Weighted RRF + буст w_bm25=1.6
│           ├── rag/
│           │   ├── query_planner.py          # Маршрутизация эшелонов (Fast/Mid/Frontier), переформулировка
│           │   ├── synthesizer.py            # Вызов LLM (Structured JSON / Anthropic Citations) + Stance MMR Quota
│           │   ├── smith_waterman.py         # Локальное выравнивание над обратимо нормализованными строками -> UTF-16
│           │   └── nli_gate.py               # Локальный гейт mDeBERTa-v3-xnli (ONNX, p_entail >= 0.75)
│           ├── cache/
│           │   └── semantic_cache.py         # Двухуровневый кэш (SHA256 CacheKey + косинус внутри бакета)
│           └── api/
│               ├── routes_rag.py             # POST /api/v1/rag/query (SSE), GET /api/v1/feed
│               ├── routes_posts.py           # POST /api/v1/posts/{id}/comments-summary
│               └── routes_sources.py         # POST /api/v1/sources/import, GET /api/v1/media/{file_id}
│
└── apps/
    └── web-client/                           # Next.js 15 / SvelteKit (App Router, TypeScript)
        ├── package.json                      # @telegraf/entity, shiki, lottie-web / tlottie
        ├── Dockerfile
        └── src/
            ├── lib/
            │   ├── ast/
            │   │   ├── types.ts              # TelegramAstNode, TextLeafNode, RagHighlightSpan
            │   │   ├── parser.ts             # Построение базового AST из UTF-16 entities (Layer 225+ / Bot API 10.4)
            │   │   ├── interval_split.ts     # Алгоритм Interval Split Pass для инъекции <mark data-cite="...">
            │   │   └── shiki_highlighter.ts  # Подсветка messageEntityPre через codeToHast
            │   ├── sse/
            │   │   └── rag_stream.ts         # Клиент EventSource / fetch-stream с обработкой keep-alive
            │   └── api/
            │       └── client.ts             # Типизированный клиент к FastAPI /api/v1/*
            └── components/
                ├── feed/
                │   ├── ToggleViewHeader.tsx  # Переключатель «ИИ-Сводка» <-> «Очищенная лента оригиналов»
                │   ├── PostCard.tsx          # Рендер AST, бейдж «Еще в N каналах», антикликбейт-шапка
                │   └── AlbumGrid.tsx         # Адаптивная сетка альбомов + HTTP 206 видео/лайтбокс
                ├── rag/
                │   ├── TwoTierResults.tsx    # Двухэтажная выдача (Синтез/Матрица ЗА vs ПРОТИВ + Стек оригиналов)
                │   └── CitationHoverCard.tsx # Всплывающая точная цитата + синхронная подсветка hover/focus
                └── comments/
                    └── CommentSummaryDrawer.tsx
```

---

### 2.2. Детальное устройство `services/ingestion-go`

Сервис написан на Go 1.24+ поверх `github.com/gotd/td` ($\ge$ `v0.145.1`, актуальная TL-схема **Layer 225+**). В Стадии 1 обслуживает 50–300 публичных каналов через 1–3 стандартных аккаунта Telegram.

#### 1. Инициализация `gotd/td`, `bbolt` и цепочка middleware
Вся локальная персистентность MTProto-транспорта изолирована в файлах `bbolt` (`go.etcd.io/bbolt`), а глобальные курсоры каналов (`last_msg_id`, `max_edit_date_checked`, `linked_chat_id`, `tier`) хранятся в PostgreSQL:
- **Update State Storage:** `github.com/gotd/contrib/bbolt` (`boltstor.NewStateStorage(boltDB)`) реализует интерфейс `updates.StateStorage` (`SetChannelPts`, `SetPts`, `SetQts`, `SetDateSeq`, `GetState`, `GetChannelPts`) и оборачивается в `WatermarkStateStorage`, который перехватывает вызовы фиксации `pts`/`qts`/`seq` и разрешает запись нового состояния в `bbolt` только после получения синхронного подтверждения (ACK) от Redis Streams (`client.XAdd(...).Result()`) и PostgreSQL (`tx.Commit(ctx)`).
- **Session Storage:** `telegram.SessionStorage` (`github.com/gotd/td/session.Storage`: `LoadSession` / `StoreSession`) создается через `boltstor.NewSessionStorage(boltDB, "mtproto_sessions", []byte(sessionKey))` (атомарное хранение `auth_key`, `dc_id`, `salt`, отпечатка устройства).
- **Peer Storage:** `boltstor.NewPeerStorage(boltDB)` реализует `storage.PeerStorage` (`github.com/gotd/contrib/storage`), сохраняя локальный кэш пиров для уменьшения повторных обращений к `api.ContactsResolveUsername`.
- **Update Manager & Middleware Stack (`telegram.Options`):**
  1. `floodwait.NewWaiter().WithMaxWait(60 * time.Second)` (из `github.com/gotd/contrib/middleware/floodwait`, реализует `telegram.Middleware` через `Handle(next tg.Invoker) telegram.InvokeFunc`): локально перехватывает `FLOOD_WAIT_X` при $X \le 60$ с, не блокируя остальные операции. При $X > 60$ с ошибка пробрасывается в координатор для записи `flood_wait_until` в PostgreSQL `core_public.mtproto_sessions`.
  2. `NewChannelTooLongPatchMiddleware`: кастомный `telegram.Middleware` (`telegram.MiddlewareFunc`), перехватывающий `*tg.UpdatesGetChannelDifferenceRequest` (см. ниже).
  3. `updhook.UpdateHook(gaps.Handle)` (из `github.com/gotd/td/telegram/updates/hook`): направляет входящие `tg.UpdatesClass` в gap engine `updates.Manager` (`updates.New(updates.Config{Handler: dispatcher, Storage: watermarkStorage, OnChannelTooLong: ...})`).
  4. `updhook.AffectedHook(gaps)`: перехватывает RPC-ответы типа `*tg.MessagesAffectedMessages` / `*tg.MessagesAffectedHistory`, предотвращая рассинхронизацию локального `pts`.

#### 2. Обязательный патч `updates.channelDifferenceTooLong` (обход `gotd/td#1711`)
При получении конструктора `*tg.UpdatesChannelDifferenceTooLong` (`#a4bcc6fe`) стандартный менеджер обновлений сбрасывает локальный `pts` через callback `updates.Config.OnChannelTooLong(channelID)`, теряя вложенные свежие сообщения из самого пакета. В `internal/mtproto/too_long_patch.go` реализуется `telegram.Middleware`, который инспектирует ответ RPC-вызова `*tg.UpdatesGetChannelDifferenceRequest` до его возврата в `telegram/updates`:

```go
// internal/mtproto/too_long_patch.go
package mtproto

import (
	"context"
	"fmt"

	"github.com/gotd/td/bin"
	"github.com/gotd/td/telegram"
	"github.com/gotd/td/tg"
)

type TooLongRecoveryHandler interface {
	RecoverChannelGap(ctx context.Context, channelID int64, diff *tg.UpdatesChannelDifferenceTooLong) error
}

// NewChannelTooLongPatchMiddleware перехватывает updates.getChannelDifference до сброса pts в gap engine.
func NewChannelTooLongPatchMiddleware(recovery TooLongRecoveryHandler) telegram.Middleware {
	return telegram.MiddlewareFunc(func(next tg.Invoker) telegram.InvokeFunc {
		return func(ctx context.Context, input bin.Encoder, output bin.Decoder) error {
			err := next.Invoke(ctx, input, output)
			if err != nil {
				return err
			}

			req, isGetChannelDiff := input.(*tg.UpdatesGetChannelDifferenceRequest)
			if !isGetChannelDiff {
				return nil
			}

			// output для UpdatesGetChannelDifferenceRequest — *tg.UpdatesChannelDifferenceBox
			box, ok := output.(*tg.UpdatesChannelDifferenceBox)
			if !ok {
				return nil
			}

			tooLong, isTooLong := box.ChannelDifference.(*tg.UpdatesChannelDifferenceTooLong)
			if !isTooLong {
				return nil
			}

			var channelID int64
			if ch, ok := req.Channel.(*tg.InputChannel); ok {
				channelID = ch.ChannelID
			}

			// 1. Принудительно извлекаем tooLong.Messages, tooLong.Chats, tooLong.Users и передаем в пайплайн.
			// 2. Извлекаем новый pts из if dlg, ok := tooLong.Dialog.(*tg.Dialog); ok { _ = dlg.Pts }.
			// 3. Догружаем окно [pg.last_msg_id + 1 .. min(tooLong.Messages.ID) - 1] через api.MessagesGetHistory.
			if recErr := recovery.RecoverChannelGap(ctx, channelID, tooLong); recErr != nil {
				return fmt.Errorf("channel %d too_long recovery failed: %w", channelID, recErr)
			}
			return nil
		}
	})
}
```

#### 3. Watermark Commit Invariant и сбор альбомов (`grouped_id`)
- **Watermark Commit Invariant:** Диспетчер обновлений (`internal/updates/dispatcher.go`), обертка `WatermarkStateStorage` (`internal/updates/watermark_storage.go`) и поллер (`internal/updates/poller.go`) **никогда не сдвигают** локальный `pts` (через `updates.StateStorage.SetChannelPts` в `bbolt`) и глобальный `core_public.channels.last_msg_id` (в PostgreSQL) до получения синхронного подтверждения (ACK) от брокера сообщений и хранилища о записи батча:
  1. Сначала `publisher/redis_streams.go` выполняет синхронный `redisClient.XAdd(ctx, &redis.XAddArgs{...}).Result()` и ожидает возврат присвоенного Stream Entry ID, а `postgres/repository.go` фиксирует обновление метаданных и курсора в транзакции (`tx.Commit(ctx)`).
  2. При ошибке записи в Redis Streams или PostgreSQL обработчик немедленно возвращает ненулевой `error` в `telegram/updates` (`updates.Manager`), не допуская сдвига `pts` и `last_msg_id`, что инициирует повторную вычитку неподтвержденного диапазона после восстановления соединения.
  3. **Идемпотентность на приемнике (`ml-pipeline-py`):** повторно доставленные события при ретраях или ребалансировке дедуплицируются строго по композитному ключу **`(channel_id, msg_id, edit_date)`** (где `edit_date = NULL` нормализуется к `'1970-01-01T00:00:00Z'`, таблица `core_public.ingest_idempotency` с `PRIMARY KEY (channel_id, msg_id, edit_date_norm)` + `ON CONFLICT DO NOTHING`).
- **Буферизация альбомов (`internal/album/assembler.go`):** Telegram отправляет медиа-альбомы (2–10 элементов) раздельными сообщениями `tg.Message` с общим `GroupedID != 0`. Ассемблер держит в RAM карту активных буферов по ключу `(channel_id, grouped_id)` с таймером сброса **1 500 мс** (сброс происходит раньше, если от того же канала приходит сообщение с другим `grouped_id` или `grouped_id == 0`). Подпись первого элемента становится основным текстом поста, подписи последующих сохраняются в `media_manifest_json[i].caption`.

#### 4. Дельта-выгрузка комментариев и автообновление `file_reference`
- **Комментарии без вступления в группы (`internal/comments/delta_sync.go`):**
  1. При обнаружении `msg.Replies.Comments == true` (если `channels.linked_chat_id IS NULL`) однократно вызывается `api.MessagesGetDiscussionMessage(ctx, &tg.MessagesGetDiscussionMessageRequest{Peer: inputPeer, MsgID: msg.ID})`, извлекается `chats[0]` и сохраняется `linked_chat_id` в `core_public.channels`.
  2. Сравнивается `msg.Replies.Replies` (из `api.MessagesGetHistory` или `tg.UpdateChannelMessageViews` / `tg.UpdateNewChannelMessage`) с `posts.replies_synced_count`.
  3. При росте счетчика запускается `api.MessagesGetReplies(ctx, &tg.MessagesGetRepliesRequest{Peer: ..., MsgID: ..., OffsetID: lastSyncedCommentID, Limit: 100})` — строго на дельту новых реплик.
- **Стриминг медиа и автообновление `file_reference` (`internal/media/`):**
  - Загрузка выполняется строго через `api.UploadGetFile(ctx, &tg.UploadGetFileRequest{Location: loc, Offset: offset, Limit: 524288})` (`tg.InputPhotoFileLocation` / `tg.InputDocumentFileLocation`) с `Limit = 524288` (512 КБ) и выравниванием `Offset % 524288 == 0` (инвариант MTProto: `part_size % 1024 == 0` и `524288 % part_size == 0`, предотвращающий `FILE_PART_SIZE_INVALID`).
  - Для фото и постеров в `core_public.file_references.thumb_size` сохраняется строго строковый идентификатор `Type` из `tg.PhotoSizeClass` (`"m"`, `"x"`, `"y"`).
  - При ошибке `tgerr.Is(err, "FILE_REFERENCE_EXPIRED")` (либо проверке через `rpcErr, ok := tgerr.As(err); ok && strings.HasPrefix(rpcErr.Type, "FILE_REFERENCE_")`) вызывается `FileRefRefresher.Refresh(ctx, fileID)`: запрашивается `core_public.file_sources` (сортировка `is_source_deleted = FALSE, created_at DESC`), выполняется `api.ChannelsGetMessages(ctx, &tg.ChannelsGetMessagesRequest{Channel: inputChannel, ID: inputMsgIDs})` (передаются все `album_msg_ids`, если файл является частью альбома `grouped_id`), новые байты `file_reference` атомарно записываются в `core_public.file_references` и запрос `api.UploadGetFile` прозрачно повторяется.

---

### 2.3. Детальное устройство `services/ml-pipeline-py`

Единый FastAPI-сервис на Python 3.12+, объединяющий фоновые воркеры Redis Streams (`asyncio` task group) и HTTP/SSE API для веб-клиента.

1. **Предобработка и карта смещений `cp_to_utf16` (`app/preprocessing/offsets.py`):**
   - Исходный `text_raw` приводится к NFKC (`unicodedata.normalize("NFKC", text_raw)`), при этом вычисляется таблица отображения `cp_to_utf16` длиной `len(text_nfkc) + 1`:
     ```python
     def build_cp_to_utf16(text: str) -> tuple[list[int], bool]:
         cp_to_utf16 = [0] * (len(text) + 1)
         utf16_pos = 0
         has_non_bmp = False
         for idx, ch in enumerate(text):
             cp_to_utf16[idx] = utf16_pos
             if ord(ch) > 0xFFFF:
                 utf16_pos += 2
                 has_non_bmp = True
             else:
                 utf16_pos += 1
         cp_to_utf16[len(text)] = utf16_pos
         return cp_to_utf16, has_non_bmp
     ```
   - Замечание по инварианту подсветки в DOM: чтобы координаты UTF-16 в RAG совпадали с `text_raw`, который рендерит фронтенд, смещения `cp_to_utf16` строятся именно поверх канонической отображаемой строки (с предварительным `rstrip` согласно спецификации Telegram entities) и сохраняются в `core_public.post_utf16_offsets`. Текст сегментируется на пронумерованные предложения `[P{post_id}_S{sent_id}]` с фиксацией `start_cp`, `end_cp`, `start_utf16`, `end_utf16`.

2. **Shared Embedding Backbone (`app/embeddings/qwen3_embedder.py`):**
   - Модель: **`Qwen/Qwen3-Embedding-0.6B`** (596M, Apache 2.0, контекст 32 768 токенов).
   - Инференс в Dev: локальный `vLLM` ($\ge$ v0.8.5 с `--runner pooling` / OpenAI-совместимым эндпоинтом `/v1/embeddings` и left-padding fix PR #648) при наличии десктопной GPU либо `onnxruntime.InferenceSession` (`ONNX Runtime`) / `llama.cpp` (Q8_0/FP16) + `transformers.AutoTokenizer` на CPU/GPU.
   - **MRL-усечение:** берется скрытое состояние последнего токена (last-token pooling), срез `vec[:512]` и L2-нормализация (`vec / np.linalg.norm(vec)`).
   - **Строгое разделение режимов кодирования (Симметрия префиксов, раздел 7.2.1 SPEC):**
     - **Режим индексации и внутренних ML-задач (`encode_documents` — строго без префиксов):** посты (`dense_body`), код (`dense_code`), OCR (`dense_ocr`), выжимки комментариев (`dense_comments`), признаки для фильтра рекламы (L1/L2) и векторы кандидатов/центроидов потоковой кластеризации ($e(P_{\text{new}})$, $\mu_j$) кодируются **строго без инструктивных префиксов** (`Instruct:` / `Query:` запрещены), что сохраняет математическую симметрию векторного пространства и исключает потерю 1–5% полноты. Для чанков лонгридов ($2000\text{–}4096$ симв.), а также для `dense_ocr` и `dense_comments` в начало текста добавляется только мета-шапка: `[{channel_title} | {neutral_headline} | {published_date}]\n{chunk_text}`.
     - **Режим поискового RAG-запроса (`encode_query` — с англоязычной инструкцией):** инструктивный префикс добавляется **исключительно** к пользовательскому поисковому запросу:
       `Instruct: Given a search query, retrieve relevant Telegram posts that answer the query\nQuery: {user_query}`.

3. **Каскад фильтрации спама и рекламы в режиме холодного старта (Стадия 1):**
   - **Ступень L0 (`app/spam_filter/l0_deterministic.py`, $<0.05$ мс/пост):**
     - Табличная нормализация русско-латинских омоглифов (`а/a`, `е/e`, `о/o`, `р/p`, `с/c`, `х/x`, `к/k`);
     - Regex-правила на токены ОРД (`erid=`, `erid:`, `Pb3XmBt`, `2Vtzq`), UTM-метки (`utm_source=tg`, `utm_campaign=`), ссылки на реферальных ботов;
     - Отсев пустых/сверхкоротких постов ($<15$ символов без значимого медиа);
     - Сверка перцептивных хэшей `pHash` изображений со словарем рекламных креативов.
   - **Ступень L2 холодного старта (`app/spam_filter/l2_geracl_nli.py`):**
     - Вместо ансамбля L1 (`LightGBM + MLP`), требующего накопления размеченной выборки, используется zero-shot NLI классификатор **`deepvk/GeRaCl-USER2-base`** (0.2B, через `transformers.AutoModelForSequenceClassification` / `onnxruntime.InferenceSession`) с гипотезами для 5 классов (`organic`, `native_ad`, `mutual_pr`, `clickbait`, `crypto_scam`) + эвристика позиционирования ссылок (присутствие `t.me/` или `@username` в последних 20% текста).
     - Посты с вердиктом `native_ad` / `mutual_pr` не удаляются, а сохраняются с `is_ad = true` в PostgreSQL и Qdrant.

4. **Потоковая дедупликация (Single-Pass Incremental Clustering) и `OriginScore`:**
   - В скользящем окне $W = 72$ часа по коллекции `public_channels_v1` (в Стадии 1 — через фильтр `published_at >= now() - 72h` без 6-часовых шардов и без фонового `HDBSCAN`) для каждого не-рекламного поста $P_{\text{new}}$ запрашивается Top-20 кандидатов по `dense_body` (вектор без префикса) + `bm25`.
   - Вычисляется композитный скор относительно каждого кластера-кандидата $C_j$ с медоидом $M_j$ и центроидом $\mu_j$:
     $$S(P_{\text{new}}, C_j) = w_1 \cos(e(P_{\text{new}}), \mu_j) + w_2 \text{BM25}_{\text{norm}}(P_{\text{new}}, M_j) + w_3 J_{\text{ent}}(E(P_{\text{new}}), E(M_j)) - \lambda \Delta t$$
     где базовые параметры калибровки: $w_1 = 0.55$, $w_2 = 0.20$, $w_3 = 0.25$, $\lambda = 0.003\ \text{ч}^{-1}$, $\tau_{\text{low}} = 0.72$, $\tau_{\text{high}} = 0.88$.
   - Если $S \in [0.72, 0.88)$ (серая зона), пара $(P_{\text{new}}, M_j)$ передается в кросс-энкодер **`Qwen/Qwen3-Reranker-0.6B`**, запускаемый в `vLLM` в режиме `Qwen3ForSequenceClassification` (`--runner pooling --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'` либо предварительно сконвертированный через `examples/pooling/score/convert_model_to_seq_cls.py`) через эндпоинт `/score` (или `vllm.LLM.score()`). При вероятности $p_{\text{same}} \ge 0.85$ пост присоединяется к $C_j$.
   - **Атрибуция первоисточника:** для каждого поста кластера пересчитывается `OriginScore` с эффективным временем публикации $t_i^{\text{effective}} = \text{edit\_date}$ при изменении $>30\%$ символов относительно первой версии (защита от грабберов-заглушек).

5. **Сжатие веток комментариев (Lazy On-Demand):**
   - Запускается только по запросу `POST /api/v1/posts/{post_id}/comments-summary`, кэшируется в `core_public.comment_summaries` по бакетам порогов `(20, 50, 150, 500)`.
   - Воронка: (1) детерминированный отсев стикеров/GIF и реплик $<15$ симв. без цифр/вопроса $\rightarrow$ (2) расчет `GraphScore(u)` по дереву `reply_to` $\rightarrow$ (3) спасение баг-репортов и конструктивных возражений через **`deepvk/RuModernBERT-small`** (34.6M, `onnxruntime.InferenceSession` INT8) + regex стектрейсов/кодов ошибок $\rightarrow$ (4) MMR-диверсификация с инжекцией `[Подтверждено в N аналогичных комментариях]` $\rightarrow$ (5) `Context-Ancestor Closure` (возврат усеченных до 200 символов предков `parent(u)` и форматирование `[#c102 -> ответ на #c98]`).

6. **Glass-box RAG: Пословная атрибуция, Смит — Ватерман и NLI-гейт:**
   - **Гибридный поиск:** 5 prefetch-веток в `QdrantClient.query_points_groups(group_by="post_id", group_size=1, limit=30)` $\rightarrow$ Top-30 постов $\rightarrow$ реранкинг `Qwen/Qwen3-Reranker-0.6B` (`/score`) $\rightarrow$ отбор Top-8..12 постов.
   - **Синтез:** прямой вызов внешнего LLM API: для Claude Sonnet 5.5 используется `anthropic.AsyncAnthropic.messages.create` с `citations={"enabled": True}` для документов-источников (возвращает `char_location` и `cited_text`); для GPT-6 Luna / GPT-6.1 Sol / локального vLLM используется `openai.AsyncOpenAI.chat.completions.create` со строгой JSON-схемой (`response_format` / vLLM `extra_body={"structured_outputs": {"json": ...}}` на базе `llguidance`), возвращающей список утверждений `claims[]`, привязанных к `post_id`, `sentence_id` и `exact_quote`.
   - **Выравнивание Смита — Ватермана (`app/rag/smith_waterman.py`):** над обратимо нормализованной строкой поста (унификация `ё -> е`, `«» -> ""`, `—/– -> -`, сжатие повторных пробелов) с матрицей весов `match=+2, mismatch=-1, gap_open=-2, gap_extend=-0.5`. При нормализованном сходстве $\ge 0.85$ восстанавливаются индексы `[start_cp, end_cp]` в исходном тексте, затем по таблице `cp_to_utf16` выполняется $O(1)$ трансляция в `utf16_range = [start_utf16, end_utf16]`.
   - **NLI-гейт (`app/rag/nli_gate.py`):** локальный энкодер **`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`** (дообученный на `PsiloQA`, `onnxruntime.InferenceSession`) проверяет каждую пару `(statement, verbatim_text)`. Проходят только утверждения с вероятностью класса `ENTAILMENT` $p_{\text{entail}} \ge 0.75$. При отсеве всех утверждений пользователю выводится: *«В выбранных каналах нет подтвержденных данных по запросу»*.

7. **Двухуровневый семантический кэш (`app/cache/semantic_cache.py`, `app/preprocessing/entities_ner.py`):**
   - **Защита от коллизий сущностей и межпользовательских утечек (разделы 7.3.2 и 7.8.5 SPEC):** кэширование по голому косинусному сходству вектора запроса запрещено (запросы с портами `8080` и `8081` дают $\cos > 0.96$). Для публичного контура (`public_channels_v1`) вычисляется детерминированный композитный ключ в Redis/Qdrant:
     $$\text{CacheKey} = \text{SHA256}\big(\text{Sorted}(\text{channel\_ids}) \parallel \text{TimeBucket} \parallel \text{ExactEntities}(\text{numbers}, \text{tickers}, \text{law\_ids}, \text{cli\_flags})\big)$$
     - `Sorted(channel_ids)`: отсортированный по возрастанию и дедуплицированный список `channel_id` подписки пользователя;
     - `TimeBucket`: нормализованное окно поиска (`time_window_days`) и временной бакет (TTL 1–3 часа для новостных запросов, 24–72 часа для ретроспективных обзоров);
     - `ExactEntities(numbers, tickers, law_ids, cli_flags)`: канонически отсортированное множество точных сущностей, извлеченных регулярными выражениями и NER до поиска (числа/порты вроде `8080`/`8081`, тикеры `SBER`/`VTBR`, номера законов `152-ФЗ`/`23-ФЗ`, CLI-флаги).
   - **Контракт изоляции приватного контура (готовность к Стадии 2/3):** для закрытого контура (`private_channels_v1`) обращение к публичному кэшу заблокировано на уровне шлюза; используется изолированный ключ с солью арендатора:
     $$\text{PrivateCacheKey} = \text{SHA256}\big(\text{tenant\_id} \parallel K_{\text{tenant\_salt}} \parallel \text{Sorted}(\text{channel\_aliases}) \parallel \text{TimeBucket} \parallel \text{ExactEntities}\big)$$
   - **Двухуровневая схема:**
     - *Уровень 1:* косинусное сравнение 512d-эмбеддингов поисковых запросов (`encode_query` с префиксом `Instruct:`) выполняется **исключительно внутри совпавшего хэш-бакета `CacheKey`** при пороге $\tau_{\text{cache}} \ge 0.95$ (в FinOps зонах $\tau_{\text{cache}} \in [0.90, 0.95]$), что гарантирует `False Cache Hit Rate = 0.0%`.
     - *Уровень 2:* при промахе Уровня 1 (уникальный набор каналов $S_u$) извлекаются предрассчитанные `canonical_event_card` из `core_public.event_clusters` по пересечению `cluster_id`.

---

### 2.4. Детальное устройство `apps/web-client`

Веб-клиент (Next.js 15 / SvelteKit, TypeScript) полностью изолирован от интерфейса Telegram (без личных чатов и счетчиков непрочитанного) и реализует два режима потребления (`Toggle View`: **«ИИ-Сводка»** и **«Очищенная лента оригиналов»**).

#### Конвейер AST-рендеринга и алгоритм `Interval Split Pass`
1. **Нормализация длины и парсинг сущностей (`src/lib/ast/parser.ts`):**
   - К `text_raw` применяется удаление замыкающих пробелов/переносов (как в спецификации Telegram) с сохранением `entities` (MTProto Layer 225+ / Bot API 9.5–10.4).
   - На базе `@telegraf/entity` (`toTree` / адаптированного `serialiseWith` с `escapers.HTML`) строится строго типизированное дерево `TelegramAstNode`:
     - `messageEntityPre`: не допускает вложенных сущностей; асинхронно передается в **Shiki** (`codeToHast` из пакета `shiki`) по `language` для построения дерева HAST без `dangerouslySetInnerHTML`.
     - `messageEntityBlockquote`: поддержка `collapsed: boolean` (интерактивное сворачивание).
     - `messageEntitySpoiler`: скрытие CSS-маской с раскрытием по клику.
     - `messageEntityFormattedDate`: локализованное форматирование `unix_time` по битовой маске `flags` (`relative`, `day_of_week`, `short_time`, `long_time`, `short_date`, `long_date`).
     - `messageEntityCustomEmoji`: обертка над `alt`-символом с асинхронной загрузкой анимации `document_id` в `tlottie` и фоллбэком на статический Twemoji.
     - `messageEntityTextUrl`: проверка домена по белому списку; внешние домены и Punycode перехватываются модальным окном подтверждения с декодированным URL.
2. **Алгоритм `Interval Split Pass` (`src/lib/ast/interval_split.ts`):**
   - Каждый `TextLeafNode` хранит глобальные UTF-16 координаты `[startUtf16, endUtf16)` в строке поста.
   - Перед рендерингом карточки поста из RAG-выдачи дерево AST проходит линейный проход, разрезающий `TextLeafNode` по границам всех `RagHighlightSpan` (`[qStart, qEnd)`):

```typescript
// apps/web-client/src/lib/ast/interval_split.ts
export interface RagHighlightSpan {
  claimId: string;
  startUtf16: number;
  endUtf16: number;
}

export type TelegramAstNode =
  | { type: 'text'; text: string; startUtf16: number; endUtf16: number; citeIds?: string[] }
  | { type: 'element'; tag: string; props?: Record<string, unknown>; startUtf16: number; endUtf16: number; children: TelegramAstNode[] }
  | { type: 'code_block'; language: string; code: string; hast: unknown; startUtf16: number; endUtf16: number };

/**
 * Разрезает текстовые листья AST по границам верифицированных UTF-16 спанов RAG,
 * сохраняя иерархию тегов (<strong>, <em>, <blockquote>) без смещения на эмодзи вне BMP.
 */
export function splitAstLeavesWithRagCitations(
  nodes: TelegramAstNode[],
  spans: RagHighlightSpan[],
): TelegramAstNode[] {
  if (spans.length === 0) return nodes;

  return nodes.flatMap((node): TelegramAstNode[] => {
    if (node.type === 'element') {
      return [{
        ...node,
        children: splitAstLeavesWithRagCitations(node.children, spans),
      }];
    }
    if (node.type === 'code_block') {
      // Для code_block подсветка цитат применяется через декораторы строк/диапазонов Shiki HAST
      return [node];
    }

    // Лист TextLeaf [leafStart, leafEnd)
    const leafStart = node.startUtf16;
    const leafEnd = node.endUtf16;
    const overlapping = spans.filter((s) => s.endUtf16 > leafStart && s.startUtf16 < leafEnd);
    if (overlapping.length === 0) return [node];

    // Собираем уникальные точки разреза внутри [leafStart, leafEnd]
    const cutPoints = new Set<number>([leafStart, leafEnd]);
    for (const span of overlapping) {
      cutPoints.add(Math.max(leafStart, span.startUtf16));
      cutPoints.add(Math.min(leafEnd, span.endUtf16));
    }
    const sortedCuts = Array.from(cutPoints).sort((a, b) => a - b);

    const result: TelegramAstNode[] = [];
    for (let i = 0; i < sortedCuts.length - 1; i++) {
      const segStart = sortedCuts[i];
      const segEnd = sortedCuts[i + 1];
      if (segStart >= segEnd) continue;

      // Относительные UTF-16 срезы внутри JS-строки листа (UTF-16 нативно в String.prototype.slice)
      const sliceText = node.text.slice(segStart - leafStart, segEnd - leafStart);
      const activeCiteIds = overlapping
        .filter((s) => s.startUtf16 <= segStart && s.endUtf16 >= segEnd)
        .map((s) => s.claimId);

      result.push({
        type: 'text',
        text: sliceText,
        startUtf16: segStart,
        endUtf16: segEnd,
        citeIds: activeCiteIds.length > 0 ? activeCiteIds : undefined,
      });
    }
    return result;
  });
}
```

---

## 3. Полная DDL-схема PostgreSQL 16

Ниже приведены полные SQL-миграции для создания схем `core_public` (глобальный дедуплицированный индекс и медиа-таблицы) и `tenant_rf` (данные пользователей, подписки и квоты). Разделение на две схемы внутри одного PostgreSQL 16 в Dev позволяет вынести `tenant_rf` в отдельный российский кластер `pg_users_rf` на Стадии 2 без изменения SQL-запросов сервисов.

### 3.1. Миграция `0001_init_core_public.sql`

```sql
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
```

### 3.2. Миграция `0002_init_tenant_rf.sql`

```sql
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
```

---

## 4. Схема коллекции `public_channels_v1` в Qdrant v1.19+

Единый дедуплицированный индекс реализован в рамках **одной коллекции `public_channels_v1`** по схеме **Multi-Representation**:
- **4 именованных плотных вектора 512d** (`Qwen/Qwen3-Embedding-0.6B` с усечением MRL до 512d и L2-нормализацией): `dense_body` (100% точек), `dense_code` ($\sim$10–12%), `dense_ocr` ($\sim$20–25%), `dense_comments` ($\sim$35–40%).
- **1 разреженный вектор `bm25`**: серверный расчет BM25 (`models.Document(text=..., model="Qdrant/bm25")`) со `SparseVectorParams(modifier=models.Modifier.IDF)`.
- **Квантизация (Стадия 1 Dev / Стадия 2 MVP):** **SQ8 (`INT8`)** с `always_ram=True` и исходными FP32-векторами `on_disk=True` на NVMe для обязательного `rescore=True` (на Стадии 3 при $>15\text{–}20$M постов переключается на 4-bit TurboQuant).
- **Условное дробление лонгридов:** посты $\le 2000$ символов хранятся одной точкой; посты $2000\text{–}4096$ символов дробятся на 2–3 смысловых чанка с контекстной шапкой `[Канал | Заголовок | Дата]` и связываются единым `post_id` (группировка через `query_points_groups(group_by="post_id")`).

### 4.1. Скрипт инициализации коллекции и индексов payload (`qdrant-client`)

> **Критический инвариант:** индексы payload (`channel_id`, `published_at`, `is_ad`, `post_id`, `cluster_id`, `text_searchable`) создаются **строго до загрузки векторов**, чтобы Qdrant построил ребра Filterable HNSW при первой индексации.

```python
# deploy/qdrant/init_public_channels_v1.py
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "public_channels_v1"
VECTOR_SIZE = 512


def init_public_channels_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return

    dense_params = models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE,
        on_disk=True,  # Исходные FP32 хранятся на NVMe для финального rescore=True
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=20000,  # КБ: переключение узких срезов (<0.5%) в SIMD full-scan
            on_disk=False,              # Граф HNSW закреплен в RAM
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,  # SQ8 в RAM
                quantile=0.99,
                always_ram=True,
            )
        ),
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense_body": dense_params,
            "dense_code": dense_params,
            "dense_ocr": dense_params,
            "dense_comments": dense_params,
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # Серверный расчет IDF по корпусу
                index=models.SparseIndexParams(on_disk=False),
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=20000,
            on_disk=False,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )

    # 1. Индекс фильтрации по списку подписок channel_id IN (30..500)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="channel_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            on_disk=False,  # Битовые маски зафиксированы в RAM
        ),
        wait=True,
    )

    # 2. Временной индекс окна поиска и скользящего окна дедупликации (72 ч)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="published_at",
        field_schema=models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME,
            is_principal=True,  # Оптимизация диапазонных сканов по времени
            on_disk=False,
        ),
        wait=True,
    )

    # 3. Булевый фильтр отсечения нативной рекламы (is_ad = false)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="is_ad",
        field_schema=models.BoolIndexParams(
            type=models.BoolIndexType.BOOL,
            on_disk=False,
        ),
        wait=True,
    )

    # 4. Целочисленный индекс для group_by="post_id"
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="post_id",
        field_schema=models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            lookup=True,
            range=False,
            on_disk=False,
        ),
        wait=True,
    )

    # 5. Ключевой индекс кластера событий (для фильтрации/обновления метаданных кластера)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="cluster_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            on_disk=False,
        ),
        wait=True,
    )

    # 6. Полнотекстовый индекс payload со стеммером Russian Snowball (для точных лексических фильтров)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text_searchable",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=40,
            lowercase=True,
            stemmer=models.SnowballParams(
                type=models.Snowball.SNOWBALL,
                language=models.SnowballLanguage.RUSSIAN,
            ),
            stopwords=models.Language.RUSSIAN,
            on_disk=False,
        ),
        wait=True,
    )
```

### 4.2. Структура точки (`PointStruct`) при записи в Qdrant

При записи поста разреженно заполняются только те именованные векторы, модальности которых присутствуют в посте (`dense_code`, `dense_ocr`, `dense_comments` опускаются, если соответствующих данных нет). Для серверного BM25 используется `models.Document` с моделью `"Qdrant/bm25"` и языком `"russian"`.

```python
# Пример записи точки (одиночный пост или чанк лонгрида)
point = models.PointStruct(
    id=str(point_uuid),  # UUIDv5(NAMESPACE_DNS, f"{channel_id}:{msg_id}:{chunk_idx}")
    vector={
        "dense_body": dense_body_512d,       # Обязательно (без префикса Instruct)
        # Опциональные модальности:
        **({"dense_code": dense_code_512d} if dense_code_512d else {}),
        **({"dense_ocr": dense_ocr_512d} if dense_ocr_512d else {}),
        **({"dense_comments": dense_comments_512d} if dense_comments_512d else {}),
        # Серверный BM25 (TF-вектор вычисляется на сервере, IDF применяется через Modifier.IDF)
        "bm25": models.Document(
            text=bm25_combined_text,         # Заголовок + тело + код + OCR
            model="Qdrant/bm25",
            options={"language": "russian", "avg_len": 256.0},
        ),
    },
    payload={
        "post_id": 42,                       # Ключ группировки group_by="post_id"
        "chunk_index": 0,                    # 0 для <=2000 симв., 0..2 для лонгридов
        "channel_id": "1092837465",          # Keyword (строковое представление BIGINT)
        "msg_id": 8421,
        "published_at": "2026-11-15T10:12:00Z",  # RFC 3339 для DatetimeRange
        "is_ad": False,
        "spam_class": "organic",
        "cluster_id": "c7b8d210-4f1a-4c32-9b1e-0192837465ab",
        "text_searchable": bm25_combined_text,
        "has_code": bool(dense_code_512d),
        "has_ocr": bool(dense_ocr_512d),
        "has_comments_summary": bool(dense_comments_512d),
    },
)
```

### 4.3. Эталонный гибридный запрос (`query_points_groups` + Weighted RRF + ACORN-1)

Вместо DBSF (разрушающего распределение на пустых модальностях и тяжелых хвостах BM25) применяется **Weighted RRF (`models.Rrf(k=60, weights=...)`)** с жесткими порогами `score_threshold` внутри каждого `Prefetch` и включенным 2-hop обходом **ACORN-1 (`AcornSearchParams`)**.

```python
# services/ml-pipeline-py/app/retrieval/qdrant_repo.py
from datetime import datetime
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "public_channels_v1"


def hybrid_search_posts(
    client: QdrantClient,
    query_text: str,
    query_dense_512d: list[float],
    subscribed_channel_ids: list[str],
    published_after: datetime,
    has_exact_entities: bool,
) -> models.GroupsResult:
    """
    Выполняет 5-веточный гибридный поиск с группировкой по post_id (Top-30 уникальных постов).
    При детекции тикеров, номеров ФЗ или кодов ошибок вес BM25 повышается с 0.9 до 1.6.
    """
    common_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="channel_id",
                match=models.MatchAny(any=subscribed_channel_ids),
            ),
            models.FieldCondition(
                key="published_at",
                range=models.DatetimeRange(gte=published_after),
            ),
            models.FieldCondition(
                key="is_ad",
                match=models.MatchValue(value=False),
            ),
        ]
    )

    acorn_search_params = models.SearchParams(
        hnsw_ef=128,
        exact=False,
        quantization=models.QuantizationSearchParams(
            ignore=False,
            rescore=True,       # Обязательное дооценивание по FP32 с NVMe
            oversampling=2.5,
        ),
        acorn=models.AcornSearchParams(
            enable=True,        # 2-hop обход для узких выборок 0.001%-0.5%
            max_selectivity=0.4,
        ),
    )

    w_bm25 = 1.6 if has_exact_entities else 0.9

    prefetches = [
        # 1. Основное тело поста (w = 1.0)
        models.Prefetch(
            query=query_dense_512d,
            using="dense_body",
            filter=common_filter,
            params=acorn_search_params,
            score_threshold=0.42,
            limit=50,
        ),
        # 2. Серверный Snowball BM25 (w = 0.9 или 1.6)
        models.Prefetch(
            query=models.Document(
                text=query_text,
                model="Qdrant/bm25",
                options={"language": "russian", "avg_len": 256.0},
            ),
            using="bm25",
            filter=common_filter,
            limit=50,
        ),
        # 3. Блоки кода (w = 0.6)
        models.Prefetch(
            query=query_dense_512d,
            using="dense_code",
            filter=common_filter,
            params=acorn_search_params,
            score_threshold=0.55,
            limit=25,
        ),
        # 4. Распознанный текст с изображений OCR (w = 0.5)
        models.Prefetch(
            query=query_dense_512d,
            using="dense_ocr",
            filter=common_filter,
            params=acorn_search_params,
            score_threshold=0.52,
            limit=25,
        ),
        # 5. Выжимки веток комментариев (w = 0.4)
        models.Prefetch(
            query=query_dense_512d,
            using="dense_comments",
            filter=common_filter,
            params=acorn_search_params,
            score_threshold=0.50,
            limit=25,
        ),
    ]

    return client.query_points_groups(
        collection_name=COLLECTION_NAME,
        prefetch=prefetches,
        query=models.RrfQuery(
            rrf=models.Rrf(
                k=60,
                weights=[1.0, w_bm25, 0.6, 0.5, 0.4],
            )
        ),
        group_by="post_id",
        group_size=1,
        limit=30,
        with_payload=True,
    )
```

---

## 5. Строгие JSON-схемы и контракты взаимодействия

### 5.1. Контракт 1: `services/ingestion-go` $\leftrightarrow$ `services/ml-pipeline-py` (Redis 7 Streams)

Обмен событиями выполняется через три персистентных потока Redis Streams (`MAXLEN ~ 100000` в Dev). Для соблюдения **Watermark Commit Invariant** воркер `ingestion-go` выполняет синхронный `XADD` и `PostgreSQL.COMMIT`, и обновляет локальный `pts` / глобальный `last_msg_id` строго после получения синхронного ACK (Stream Entry ID от Redis и подтверждения транзакции PostgreSQL); при ошибке записи обработчик возвращает отказ в `telegram/updates`, инициируя повторную вычитку неподтвержденного диапазона. Воркер `ml-pipeline-py` читает через Consumer Group `cg:ml-pipeline` (`XREADGROUP`), выполняет идемпотентную дедупликацию по композитному ключу **`(channel_id, msg_id, edit_date)`** (в таблице `core_public.ingest_idempotency`) и отправляет `XACK` только после фиксации в PostgreSQL и Qdrant.

#### 1. Поток `stream:ingest:posts` — JSON Schema (`IngestPostEventV1`)

Каждое сообщение потока имеет поле `payload` (UTF-8 JSON), соответствующее схеме Draft 2020-12:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://tg-agg.local/schemas/ingest_post_v1.schema.json",
  "title": "IngestPostEventV1",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "event_id",
    "event_type",
    "channel_id",
    "channel_title",
    "msg_id",
    "published_at",
    "edit_date",
    "text_raw",
    "entities",
    "replies",
    "media_items"
  ],
  "properties": {
    "schema_version": { "type": "string", "const": "1.0" },
    "event_id": { "type": "string", "format": "uuid" },
    "event_type": { "type": "string", "enum": ["post_new", "post_edit"] },
    "channel_id": { "type": "integer", "minimum": 1 },
    "channel_username": { "type": ["string", "null"] },
    "channel_title": { "type": "string", "minLength": 1 },
    "msg_id": { "type": "integer", "minimum": 1 },
    "grouped_id": { "type": ["integer", "null"] },
    "album_msg_ids": {
      "type": "array",
      "items": { "type": "integer", "minimum": 1 }
    },
    "published_at": { "type": "string", "format": "date-time" },
    "edit_date": { "type": ["string", "null"], "format": "date-time" },
    "text_raw": { "type": "string" },
    "show_caption_above_media": { "type": "boolean", "default": false },
    "views": { "type": "integer", "minimum": 0, "default": 0 },
    "forwards": { "type": "integer", "minimum": 0, "default": 0 },
    "entities": {
      "type": "array",
      "description": "Сущности Telegram MTProto Layer 225+; offset и length строго в UTF-16 code units.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["type", "offset", "length"],
        "properties": {
          "type": {
            "type": "string",
            "enum": [
              "bold", "italic", "underline", "strikethrough", "spoiler",
              "code", "pre", "blockquote", "text_url", "url", "mention",
              "hashtag", "cashtag", "custom_emoji", "formatted_date",
              "diff_insert", "diff_delete", "diff_replace"
            ]
          },
          "offset": { "type": "integer", "minimum": 0 },
          "length": { "type": "integer", "minimum": 1 },
          "language": { "type": ["string", "null"], "description": "Только для type='pre'" },
          "url": { "type": ["string", "null"], "description": "Только для type='text_url'" },
          "collapsed": { "type": ["boolean", "null"], "description": "Только для type='blockquote'" },
          "document_id": { "type": ["string", "null"], "description": "Строковый BIGINT для type='custom_emoji'" },
          "unix_time": { "type": ["integer", "null"], "description": "Для type='formatted_date'" },
          "date_flags": { "type": ["integer", "null"], "description": "Битовая маска formatted_date (Bot API 9.5)" },
          "old_text": { "type": ["string", "null"], "description": "Для DiffReplace" }
        }
      }
    },
    "replies": {
      "type": "object",
      "additionalProperties": false,
      "required": ["comments_enabled", "replies_count", "replies_max_id"],
      "properties": {
        "comments_enabled": { "type": "boolean" },
        "linked_chat_id": { "type": ["integer", "null"] },
        "replies_count": { "type": "integer", "minimum": 0 },
        "replies_max_id": { "type": "integer", "minimum": 0 }
      }
    },
    "media_items": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "file_id", "tg_id", "access_hash", "dc_id", "media_kind",
          "thumb_size", "file_reference_b64", "file_size"
        ],
        "properties": {
          "file_id": { "type": "string", "description": "{media_kind}:{dc_id}:{tg_id}:{thumb_size}" },
          "tg_id": { "type": "integer" },
          "access_hash": { "type": "integer" },
          "dc_id": { "type": "integer", "minimum": 1, "maximum": 5 },
          "media_kind": { "type": "string", "enum": ["photo", "document", "video", "voice", "sticker"] },
          "thumb_size": { "type": "string", "description": "Строковый type из PhotoSize (или '' для оригинала)" },
          "file_reference_b64": { "type": "string" },
          "mime_type": { "type": ["string", "null"] },
          "file_size": { "type": "integer", "minimum": 0 },
          "width": { "type": ["integer", "null"] },
          "height": { "type": ["integer", "null"] },
          "duration_sec": { "type": ["integer", "null"] },
          "caption": { "type": ["string", "null"], "description": "Индивидуальная подпись слота альбома" },
          "stripped_thumb_b64": { "type": ["string", "null"] }
        }
      }
    }
  }
}
```

#### 2. Поток `stream:ingest:comments` — JSON Schema (`IngestCommentsDeltaEventV1`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://tg-agg.local/schemas/ingest_comments_v1.schema.json",
  "title": "IngestCommentsDeltaEventV1",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "event_id", "channel_id", "post_msg_id", "linked_chat_id", "total_replies_count", "comments"],
  "properties": {
    "schema_version": { "type": "string", "const": "1.0" },
    "event_id": { "type": "string", "format": "uuid" },
    "channel_id": { "type": "integer" },
    "post_msg_id": { "type": "integer" },
    "linked_chat_id": { "type": "integer" },
    "total_replies_count": { "type": "integer", "minimum": 0 },
    "comments": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["comment_msg_id", "author_peer_hash", "published_at", "text_raw", "entities"],
        "properties": {
          "comment_msg_id": { "type": "integer" },
          "reply_to_msg_id": { "type": ["integer", "null"] },
          "author_peer_hash": { "type": "string", "minLength": 16, "maxLength": 64 },
          "published_at": { "type": "string", "format": "date-time" },
          "edit_date": { "type": ["string", "null"], "format": "date-time" },
          "text_raw": { "type": "string" },
          "entities": { "type": "array", "items": { "type": "object" } }
        }
      }
    }
  }
}
```

#### 3. Поток `stream:ingest:deletions` — JSON Schema (`IngestDeletionEventV1`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://tg-agg.local/schemas/ingest_deletion_v1.schema.json",
  "title": "IngestDeletionEventV1",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "event_id", "channel_id", "deleted_msg_ids", "occurred_at"],
  "properties": {
    "schema_version": { "type": "string", "const": "1.0" },
    "event_id": { "type": "string", "format": "uuid" },
    "channel_id": { "type": "integer" },
    "deleted_msg_ids": {
      "type": "array",
      "minItems": 1,
      "items": { "type": "integer", "minimum": 1 }
    },
    "occurred_at": { "type": "string", "format": "date-time" }
  }
}
```

### 5.2. Внутренний HTTP API `services/ingestion-go` (`:8081`) и цикл обновления `file_reference`

`ingestion-go` экспортирует три внутренних синхронных эндпоинта для `ml-pipeline-py`:

1. **`GET /internal/v1/media/{file_id}`**
   - Поддерживает заголовок `Range: bytes=<start>-<end>` и отвечает `206 Partial Content` (`Accept-Ranges: bytes`, `Content-Range`, `Content-Type`).
   - Внутри: проверка локального дискового кэша (TTL 24 ч) $\rightarrow$ при промахе вызов `upload.getFile` чанками по `524288` байт (512 КБ).
   - Если все источники в `core_public.file_sources` удалены (`is_permanently_gone = TRUE`), возвращает `410 Gone`.
2. **`POST /internal/v1/comments/sync`**
   - Тело запроса: `{"channel_id": 1092837465, "post_msg_id": 8421}`.
   - Инициирует немедленную дельта-синхронизацию `messages.getReplies` перед ленивой суммаризацией ветки и возвращает `{"status": "synced", "new_comments": 14, "total_replies": 142}`.
3. **`POST /internal/v1/channels/register`**
   - Тело запроса: `{"targets": ["@golang_news", "https://t.me/addlist/abc123"]}`.
   - Помещает запросы в темпированную очередь `contacts.resolveUsername` / `chatlists.checkChatlistInvite` с кэшированием `access_hash` в `core_public.channel_access_hashes` и возвращает `202 Accepted` со списком поставленных в очередь `channel_id` / `job_id`.

#### Диаграмма автообновления истекшего `file_reference`

```mermaid
sequenceDiagram
    autonumber
    participant Client as web-client / ml-pipeline-py
    participant Media as ingestion-go Media Streamer
    participant PG as PostgreSQL (file_references + file_sources)
    participant TG as Telegram MTProto DC

    Client->>Media: GET /internal/v1/media/{file_id} (Range: bytes=0-524287)
    Media->>PG: SELECT tg_id, access_hash, dc_id, thumb_size, file_reference FROM file_references
    PG-->>Media: file_reference (устаревший токен)
    Media->>TG: upload.getFile(location, offset=0, limit=524288)
    TG-->>Media: RPC Error 400 FILE_REFERENCE_EXPIRED
    Media->>PG: SELECT channel_id, msg_id, album_msg_ids FROM file_sources WHERE file_id=... AND NOT is_source_deleted
    PG-->>Media: Список источников [S1, S2...]
    loop Пока не найдется доступный источник
        Media->>TG: channels.getMessages(channel=S1.channel_id, id=[msg_id или album_msg_ids])
        alt Сообщение доступно
            TG-->>Media: Свежий tg.Message с новым file_reference
            Media->>PG: UPDATE file_references SET file_reference=new_bytes, refreshed_at=NOW()
            Media->>TG: upload.getFile(new_location, offset=0, limit=524288)
            TG-->>Media: upload.file (512 КБ чанк)
            Media-->>Client: HTTP 206 Partial Content (потоковый чанк)
        else Сообщение удалено в S1 (EmptyMessage)
            Media->>PG: UPDATE file_sources SET is_source_deleted=TRUE WHERE source_id=S1.id
        end
    end
    opt Все источники удалены
        Media->>PG: UPDATE file_references SET is_permanently_gone=TRUE
        Media-->>Client: HTTP 410 Gone (UI показывает превью и карточку поста)
    end
```

---

### 5.3. Контракт 2: `services/ml-pipeline-py` $\leftrightarrow$ Внешние LLM API (Glass-box RAG Schema)

При маршрутизации в `GPT-6 Luna` / `GPT-6.1 Sol` / `DeepSeek V4.1 Flash` (или локальный vLLM + `llguidance`) применяется строгая JSON Schema ограниченного декодирования, запрещающая генерацию тезисов без ссылки на конкретный `[P{post_id}_S{sent_id}]` и дословный спан:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://tg-agg.local/schemas/glassbox_llm_output.schema.json",
  "title": "RagSynthesisOutput",
  "type": "object",
  "additionalProperties": false,
  "required": ["mode", "claims", "stance_comparison_matrix"],
  "properties": {
    "mode": { "type": "string", "enum": ["qa", "digest", "stance_matrix"] },
    "claims": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["claim_id", "topic_group", "statement", "citations"],
        "properties": {
          "claim_id": { "type": "string", "pattern": "^c[0-9]+$" },
          "topic_group": { "type": "string" },
          "statement": { "type": "string", "minLength": 5 },
          "citations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": ["post_id", "sentence_id", "exact_quote"],
              "properties": {
                "post_id": { "type": "integer", "minimum": 1 },
                "sentence_id": { "type": "integer", "minimum": 1 },
                "exact_quote": { "type": "string", "minLength": 3 }
              }
            }
          }
        }
      }
    },
    "stance_comparison_matrix": {
      "type": "array",
      "description": "Заполняется при mode='stance_matrix' из стратифицированного буфера (<=8 PRO, <=8 CON).",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["aspect", "pro_points", "con_points"],
        "properties": {
          "aspect": { "type": "string" },
          "pro_points": {
            "type": "array",
            "items": { "$ref": "#/$defs/StanceArgument" }
          },
          "con_points": {
            "type": "array",
            "items": { "$ref": "#/$defs/StanceArgument" }
          }
        }
      }
    }
  },
  "$defs": {
    "StanceArgument": {
      "type": "object",
      "additionalProperties": false,
      "required": ["claim_id", "statement", "citations"],
      "properties": {
        "claim_id": { "type": "string" },
        "statement": { "type": "string" },
        "citations": {
          "type": "array",
          "minItems": 1,
          "maxItems": 3,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["post_id", "sentence_id", "exact_quote"],
            "properties": {
              "post_id": { "type": "integer" },
              "sentence_id": { "type": "integer" },
              "exact_quote": { "type": "string" }
            }
          }
        }
      }
    }
  }
}
```

---

### 5.4. Контракт 3: `services/ml-pipeline-py` $\leftrightarrow$ `apps/web-client` (OpenAPI 3.1.0 и протокол SSE)

#### 1. Спецификация OpenAPI 3.1.0 (`packages/contracts/openapi/ml_pipeline_v1.yaml`)

```yaml
openapi: 3.1.0
info:
  title: Telegram AI Aggregator ML & API Gateway (Stage 1 Dev)
  version: 1.0.0
servers:
  - url: http://localhost:8000
paths:
  /api/v1/rag/query:
    post:
      summary: Гибридный RAG-поиск и потоковый синтез Glass-box с NLI-верификацией (SSE)
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RagQueryRequest'
      responses:
        '200':
          description: Поток SSE (Content-Type text/event-stream; charset=utf-8)
          headers:
            Cache-Control:
              schema: { type: string, example: "no-cache, no-transform" }
            X-Accel-Buffering:
              schema: { type: string, example: "no" }
          content:
            text/event-stream:
              schema:
                type: string
  /api/v1/feed:
    get:
      summary: Двухрежимная лента (Toggle View: ИИ-Сводка vs Очищенная лента оригиналов)
      parameters:
        - name: mode
          in: query
          required: true
          schema: { type: string, enum: [digest, clean_originals] }
        - name: channel_ids
          in: query
          required: true
          schema: { type: string, description: "Список channel_id через запятую" }
        - name: time_window_hours
          in: query
          schema: { type: integer, default: 24, minimum: 1, maximum: 168 }
        - name: cursor
          in: query
          schema: { type: string, nullable: true }
        - name: limit
          in: query
          schema: { type: integer, default: 30, maximum: 100 }
      responses:
        '200':
          description: Лента событийных кластеров или очищенных оригинальных постов
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FeedResponse'
  /api/v1/posts/{post_id}/comments-summary:
    post:
      summary: Ленивая (On-Demand) ИИ-выжимка ветки комментариев с бакет-кэшированием
      parameters:
        - name: post_id
          in: path
          required: true
          schema: { type: integer }
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                force_refresh: { type: boolean, default: false }
      responses:
        '200':
          description: Структурированная выжимка комментариев (баги, контраргументы, кейсы)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CommentSummaryResponse'
  /api/v1/sources/import:
    post:
      summary: Импорт публичных каналов (@username, t.me/addlist/...) или пресетов
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [user_id, targets]
              properties:
                user_id: { type: string, format: uuid }
                targets: { type: array, items: { type: string }, minItems: 1 }
                preset_id: { type: string, nullable: true }
      responses:
        '202':
          description: Источники приняты на резолвинг в ingestion-go
  /api/v1/media/{file_id}:
    get:
      summary: Проксирование медиа из ingestion-go (превью и HTTP 206 Range-потоки)
      parameters:
        - name: file_id
          in: path
          required: true
          schema: { type: string }
        - name: Range
          in: header
          schema: { type: string, example: "bytes=0-524287" }
      responses:
        '206':
          description: Частичное содержимое медиафайла
        '410':
          description: Оригинальный пост удален во всех источниках

components:
  schemas:
    RagQueryRequest:
      type: object
      additionalProperties: false
      required: [user_id, query, channel_ids]
      properties:
        user_id: { type: string, format: uuid }
        query: { type: string, minLength: 2, maxLength: 1000 }
        channel_ids:
          type: array
          items: { type: integer }
          minItems: 1
          maxItems: 500
        mode: { type: string, enum: [qa, stance_matrix], default: qa }
        time_window_days: { type: integer, default: 90, minimum: 1, maximum: 365 }
        cluster_id: { type: string, format: uuid, nullable: true }

    FeedResponse:
      type: object
      required: [mode, items, next_cursor]
      properties:
        mode: { type: string, enum: [digest, clean_originals] }
        next_cursor: { type: string, nullable: true }
        items:
          type: array
          items:
            oneOf:
              - $ref: '#/components/schemas/DigestClusterCard'
              - $ref: '#/components/schemas/CleanPostItem'

    DigestClusterCard:
      type: object
      required: [cluster_id, neutral_title, atomic_summary, origin_post, also_in_channels_count, verified_citations]
      properties:
        cluster_id: { type: string, format: uuid }
        topic_rubric: { type: string }
        neutral_title: { type: string }
        atomic_summary: { type: string }
        also_in_channels_count: { type: integer }
        origin_post: { $ref: '#/components/schemas/CleanPostItem' }
        verified_citations:
          type: array
          items: { $ref: '#/components/schemas/VerifiedCitation' }

    CleanPostItem:
      type: object
      required: [post_id, channel_id, channel_title, msg_id, published_at, neutral_headline, text_raw, entities, media_manifest, replies_count, deeplink]
      properties:
        post_id: { type: integer }
        channel_id: { type: integer }
        channel_username: { type: string, nullable: true }
        channel_title: { type: string }
        msg_id: { type: integer }
        published_at: { type: string, format: date-time }
        neutral_headline: { type: string, nullable: true }
        text_raw: { type: string }
        entities: { type: array, items: { type: object } }
        media_manifest: { type: array, items: { type: object } }
        replies_count: { type: integer }
        cluster_id: { type: string, format: uuid, nullable: true }
        also_in_channels_count: { type: integer, default: 0 }
        deeplink: { type: string, format: uri }

    VerifiedCitation:
      type: object
      required: [post_id, channel_id, channel_title, msg_id, published_at, deeplink, utf16_range, verbatim_text, nli_entailment_score, sw_alignment_score]
      properties:
        post_id: { type: integer }
        channel_id: { type: integer }
        channel_title: { type: string }
        msg_id: { type: integer }
        published_at: { type: string, format: date-time }
        deeplink: { type: string, format: uri }
        utf16_range:
          type: array
          minItems: 2
          maxItems: 2
          items: { type: integer, minimum: 0 }
        verbatim_text: { type: string }
        nli_entailment_score: { type: number, minimum: 0.75, maximum: 1.0 }
        sw_alignment_score: { type: number, minimum: 0.85, maximum: 1.0 }

    CommentSummaryResponse:
      type: object
      required: [post_id, bucket_threshold, total_comments, compressed_comments_count, summary_text, highlights]
      properties:
        post_id: { type: integer }
        bucket_threshold: { type: integer, enum: [20, 50, 150, 500] }
        total_comments: { type: integer }
        compressed_comments_count: { type: integer }
        summary_text: { type: string }
        highlights:
          type: array
          items:
            type: object
            required: [category, text, comment_refs]
            properties:
              category: { type: string, enum: [bug_report, counterargument, personal_case] }
              text: { type: string }
              comment_refs: { type: array, items: { type: string } }
```

#### 2. Протокол SSE-потока `POST /api/v1/rag/query`

Сервер выставляет заголовки `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no`. Во время длительных операций (поиск, реранкинг, NLI-проверка) каждые **15 секунд** отправляется пустой комментарий `: keep-alive\n\n`.

Типы передаваемых событий (`event:` + `data:`):

1. **`event: pipeline_stage`** — прогресс этапов конвейера для индикатора в UI:
   ```json
   {"stage": "hybrid_retrieval", "status": "completed", "latency_ms": 48, "candidates_count": 30}
   ```
   Допустимые значения `stage`: `"cache_check"`, `"hybrid_retrieval"`, `"reranking"`, `"llm_synthesis"`, `"nli_verification"`.

2. **`event: verified_claim`** — отправляется **строго после** прохождения Смита — Ватермана ($\ge 0.85$) и NLI-гейта `mDeBERTa-v3-xnli` ($p_{\text{entail}} \ge 0.75$):
   ```json
   {
     "claim_id": "c1",
     "topic_group": "Транспортный слой",
     "statement": "На слое 225 в gotd/td ответ updates.channelDifferenceTooLong может сбрасывать pts без обработки вложенных сообщений.",
     "citations": [
       {
         "post_id": 42,
         "channel_id": 1092837465,
         "channel_title": "Go Backend Digest",
         "msg_id": 8421,
         "published_at": "2026-11-14T09:15:00Z",
         "deeplink": "https://t.me/gobackenddigest/8421",
         "utf16_range": [15, 45],
         "verbatim_text": "потеряно 14% пакетов из-за BGP",
         "nli_entailment_score": 0.94,
         "sw_alignment_score": 0.968
       }
     ]
   }
   ```

3. **`event: stance_row`** — строка сопоставительной таблицы «Аргументы ЗА vs Аргументы ПРОТИВ» (каждый аргумент также прошел двойную NLI-верификацию):
   ```json
   {
     "aspect": "Потребление памяти на 200+ сессий",
     "pro_points": [ { "claim_id": "c2", "statement": "...", "citations": [ /* VerifiedCitation */ ] } ],
     "con_points": [ { "claim_id": "c3", "statement": "...", "citations": [ /* VerifiedCitation */ ] } ]
   }
   ```

4. **`event: source_posts`** — нижний этаж выдачи (стек оригинальных постов с предсобранными интервалами подсветки для `Interval Split Pass` в `apps/web-client`):
   ```json
   {
     "posts": [
       {
         "post": { /* CleanPostItem: post_id, text_raw, entities, media_manifest... */ },
         "rag_highlights": [
           { "claimId": "c1", "startUtf16": 15, "endUtf16": 45 }
         ]
       }
     ]
   }
   ```

5. **`event: done`** — финальное событие завершения стрима:
   ```json
   {
     "cache_hit": false,
     "model_tier": "fast",
     "verified_claims_count": 4,
     "rejected_claims_count": 1,
     "total_latency_ms": 1840,
     "quota_remaining": 29
   }
   ```

---

## 6. Локальная среда (`docker-compose.dev.yml`) и Quality Gates перехода на MVP

### 6.1. Эталонная конфигурация `deploy/docker-compose.dev.yml`

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: tg_agg_dev
      POSTGRES_USER: tg_agg
      POSTGRES_PASSWORD: dev_password_change_me
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./postgres/migrations:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tg_agg -d tg_agg_dev"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7.4-alpine
    command: ["redis-server", "--appendonly", "yes", "--maxmemory", "1gb", "--maxmemory-policy", "noeviction"]
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  qdrant:
    image: qdrant/qdrant:v1.19.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

  ingestion-go:
    build:
      context: ../services/ingestion-go
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
    environment:
      DATABASE_URL: postgres://tg_agg:dev_password_change_me@postgres:5432/tg_agg_dev?sslmode=disable
      REDIS_ADDR: redis:6379
      BBOLT_DIR: /var/lib/ingestor/bbolt
      MEDIA_CACHE_DIR: /var/lib/ingestor/media_cache
      MEDIA_CACHE_TTL_HOURS: "24"
      ALBUM_BUFFER_MS: "1500"
      HTTP_LISTEN_ADDR: ":8081"
      TG_APP_ID: ${TG_APP_ID}
      TG_APP_HASH: ${TG_APP_HASH}
      SOCKS5_PROXY_URL: ${SOCKS5_PROXY_URL:-}
    ports:
      - "8081:8081"
    volumes:
      - bbolt_data:/var/lib/ingestor/bbolt
      - media_cache_24h:/var/lib/ingestor/media_cache

  ml-pipeline-py:
    build:
      context: ../services/ml-pipeline-py
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
      qdrant: { condition: service_started }
      ingestion-go: { condition: service_started }
    environment:
      DATABASE_URL: postgresql://tg_agg:dev_password_change_me@postgres:5432/tg_agg_dev
      REDIS_URL: redis://redis:6379/0
      QDRANT_URL: http://qdrant:6333
      QDRANT_GRPC_URL: qdrant:6334
      INGESTION_INTERNAL_URL: http://ingestion-go:8081
      EMBEDDING_MODEL_ID: Qwen/Qwen3-Embedding-0.6B
      EMBEDDING_DIM_MRL: "512"
      RERANKER_MODEL_ID: Qwen/Qwen3-Reranker-0.6B
      COLDSTART_SPAM_MODEL_ID: deepvk/GeRaCl-USER2-base
      COMMENT_RESCUE_MODEL_ID: deepvk/RuModernBERT-small
      NLI_GATE_MODEL_ID: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
      CLUSTERING_TAU_LOW: "0.72"
      CLUSTERING_TAU_HIGH: "0.88"
      RERANK_MERGE_THRESHOLD: "0.85"
      SW_SIMILARITY_THRESHOLD: "0.85"
      NLI_ENTAILMENT_THRESHOLD: "0.75"
      LLM_API_KEY: ${LLM_API_KEY}
      LLM_BASE_URL: ${LLM_BASE_URL}
    ports:
      - "8000:8000"

  web-client:
    build:
      context: ../apps/web-client
    depends_on:
      ml-pipeline-py: { condition: service_started }
    environment:
      NEXT_PUBLIC_API_BASE_URL: http://localhost:8000
      INTERNAL_API_BASE_URL: http://ml-pipeline-py:8000
    ports:
      - "3000:3000"

volumes:
  pg_data:
  redis_data:
  qdrant_storage:
  bbolt_data:
  media_cache_24h:
```

### 6.2. Критерии перехода на Стадию 2: MVP (Quality Gates)

1. **Надежность транспортного ядра (`services/ingestion-go`):**
   - Непрерывный сбор **300 публичных каналов** в течение **72 часов** с нулевым числом необработанных `FLOOD_WAIT` и без рассинхронизации локального `pts` в `bbolt` (включая искусственно спровоцированные переполнения `updates.channelDifferenceTooLong`);
   - 100% успешное автовосстановление истекших `file_reference` через `file_references` / `file_sources` при чтении ретроспективных постов и альбомов.
2. **Точность ML-конвейера (`services/ml-pipeline-py`, на отложенной выборке 500 постов и Gold-Set):**
   - Холодный старт фильтрации спама L0 + `GeRaCl-USER2-base`: ложные срабатывания на органических ИТ-постах $\text{FPR} < 1.5\%$ (на L0 $< 0.5\%$);
   - Потоковая дедупликация в 72-часовом окне: $BCubed\ F_1 \ge 0.88$, парная точность склейки на пороге $\tau_{\text{high}} = 0.88$ — $PairwisePrecision \ge 0.99$, атрибуция первоисточника `OriginScore` $Precision@1 \ge 0.85$;
   - Гибридный поиск `public_channels_v1`: $Recall@30 \ge 0.95$ относительно `exact: true` на узких фильтрах `channel_id IN (...)` ($0.1\text{–}0.5\%$ базы);
   - Верификация Glass-box RAG: **100%** тезисов на выходе SSE имеют статус `ENTAILMENT` ($p_{\text{entail}} \ge 0.75$) в `mDeBERTa-v3-xnli`, $0\%$ ложных попаданий в семантический кэш на состязательных парах (8080 vs 8081).
3. **Точность рендеринга (`apps/web-client`):**
   - **0 символов сдвига** при наложении `<mark data-cite="...">` на постах с суррогатными парами эмодзи вне BMP (`> U+FFFF`), вложенным форматированием (`bold` + `italic` + `text_url`) и блоками кода `Shiki` (`messageEntityPre`).
