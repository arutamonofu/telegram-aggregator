import { DATE_FLAG_DAY_OF_WEEK, DATE_FLAG_LONG_DATE, DATE_FLAG_SHORT_TIME } from '@/lib/ast/formatted_date';
import type {
  CleanOriginalsResponse,
  CleanPostItem,
  CommentInputItem,
  CommentSummaryResponse,
  DigestFeedResponse,
  IndustryPreset,
  RagSourcePostItem,
  StanceMatrixRow,
  VerifiedClaim,
} from '@/lib/types/feed';

function makePreviewSvgDataUrl(
  bgStart: string,
  bgEnd: string,
  title: string,
  subtitle: string,
  metric: string,
): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="${bgStart}"/>
        <stop offset="100%" stop-color="${bgEnd}"/>
      </linearGradient>
    </defs>
    <rect width="640" height="420" rx="20" fill="url(#g)"/>
    <rect x="28" y="28" width="584" height="364" rx="14" fill="rgba(255,255,255,0.78)" stroke="rgba(15,23,42,0.08)"/>
    <text x="52" y="88" font-family="system-ui, sans-serif" font-size="16" font-weight="600" fill="#475569">${subtitle}</text>
    <text x="52" y="136" font-family="system-ui, sans-serif" font-size="28" font-weight="700" fill="#0f172a">${title}</text>
    <text x="52" y="252" font-family="ui-monospace, monospace" font-size="56" font-weight="700" fill="#0284c7">${metric}</text>
    <rect x="52" y="302" width="536" height="48" rx="10" fill="#f1f5f9"/>
    <text x="68" y="331" font-family="ui-monospace, monospace" font-size="14" fill="#334155">Edge Preview 35KB (S3 Standard) · HTTP 206 Ready</text>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const INDUSTRY_PRESETS: IndustryPreset[] = [
  {
    id: 'it_dev',
    title: 'IT & Dev',
    description: 'Высоконагруженные системы, Go/Rust, Kubernetes, отказы сетей BGP и постмортемы',
    badgeColor: 'bg-sky-50 text-sky-800 ring-sky-200',
    channels: [
      { channelId: '-1001001', handle: '@infra_postmortem', title: 'Infra Postmortem', topic: 'IT & Dev' },
      { channelId: '-1001002', handle: '@highload_notes', title: 'Highload Engineering', topic: 'IT & Dev' },
    ],
  },
  {
    id: 'ai_ds',
    title: 'AI & Data Science',
    description: 'vLLM, квантование, гибридный RAG, эмбеддинги, бенчмарки open-weight LLM',
    badgeColor: 'bg-violet-50 text-violet-800 ring-violet-200',
    channels: [
      { channelId: '-1002001', handle: '@llm_systems_ru', title: 'LLM Systems & RAG', topic: 'AI & Data Science' },
      { channelId: '-1002002', handle: '@ml_inference_lab', title: 'ML Inference Lab', topic: 'AI & Data Science' },
    ],
  },
  {
    id: 'product_mgmt',
    title: 'Product Management',
    description: 'Юнит-экономика SaaS, FinOps, метрики удержания, UX без информационного шума',
    badgeColor: 'bg-emerald-50 text-emerald-800 ring-emerald-200',
    channels: [
      { channelId: '-1003001', handle: '@product_economics', title: 'Product & FinOps', topic: 'Product Management' },
    ],
  },
  {
    id: 'macro',
    title: 'Макроэкономика и Финансы',
    description: 'Ставка ЦБ, долговой рынок, капекс на ЦОД и серверные кластеры, отчеты эмитентов',
    badgeColor: 'bg-amber-50 text-amber-800 ring-amber-200',
    channels: [
      { channelId: '-1004001', handle: '@macro_pulse_ru', title: 'Macro Pulse Desk', topic: 'Макроэкономика' },
    ],
  },
  {
    id: 'legal_tax',
    title: 'Право и Налоги РФ',
    description: '152-ФЗ (редакция 2025–2026), Приказ РКН № 140, налоги ИТ-компаний и комплаенс',
    badgeColor: 'bg-slate-100 text-slate-800 ring-slate-200',
    channels: [
      { channelId: '-1005001', handle: '@digital_law_152', title: 'Digital Law 152-ФЗ', topic: 'Право РФ' },
    ],
  },
];

export const SEED_CLEAN_POSTS: CleanPostItem[] = [
  {
    id: 'post-bgp-42',
    channelId: '-1001001',
    channelTitle: 'Infra Postmortem',
    channelHandle: '@infra_postmortem',
    channelAvatarInitials: 'IP',
    postId: 42,
    publishedAt: '2026-11-15T08:15:00Z',
    editDate: '2026-11-15T08:17:00Z',
    sourcePlatform: 'TG',
    clusterId: 'cluster-bgp-2026',
    neutralHeadline: 'Авария BGP в европейском сегменте привела к потере 14% трафика ЦОД',
    originalHeadline: '🔥 СРОЧНО!! ЕВРОПА ЛЕГЛА — BGP снёс 14% трафика ЦОД, инженеры в шоке!!',
    text: '🔥 Сбой BGP: потеряно 14% трафика ЦОД',
    entities: [{ type: 'messageEntityBold', offset: 3, length: 22 }],
    media: [],
    originScore: 0.94,
    duplicateCount: 6,
    duplicateSources: [
      {
        channelId: '-1001009',
        channelTitle: 'NetOps Daily',
        channelHandle: '@netops_daily',
        postId: '1892',
        publishedAt: '2026-11-15T08:19:00Z',
        editDate: '2026-11-15T08:21:00Z',
        originScore: 0.42,
        noveltyPct: 24,
        rewriteNote: 'Добавлены графики потерь по узлу Frankfurt-2',
      },
      {
        channelId: '-1001012',
        channelTitle: 'DC Monitor',
        channelHandle: '@dc_monitor',
        postId: '551',
        publishedAt: '2026-11-15T08:22:00Z',
        originScore: 0.35,
        noveltyPct: 6,
        rewriteNote: 'Пересказ без новых фактов',
      },
      {
        channelId: '-1001099',
        channelTitle: 'FastNews Grabber',
        channelHandle: '@fast_grabber_bot',
        postId: '9910',
        publishedAt: '2026-11-15T08:16:00Z',
        editDate: '2026-11-15T08:25:00Z',
        originScore: 0.08,
        isGrabberStubFiltered: true,
        noveltyPct: 0,
        rewriteNote: 'Пустой пост-заглушка отсечен по edit_date (+9 мин)',
      },
    ],
    commentCount: 38,
    rubric: 'IT & Dev',
  },
  {
    id: 'post-k8s-88',
    channelId: '-1002001',
    channelTitle: 'LLM Systems & RAG',
    channelHandle: '@llm_systems_ru',
    channelAvatarInitials: 'LS',
    postId: 88,
    publishedAt: '2026-11-15T08:40:00Z',
    sourcePlatform: 'TG',
    clusterId: 'cluster-qdrant-2026',
    neutralHeadline: 'Технический разбор переключения Qdrant ACORN-1 и обновление конфига MTProto-прокси',
    originalHeadline: '🚀 Убийца полного скана?! Как мы переключили Qdrant ACORN-1 и словили инцидент',
    text:
      '🔥 Оптимизация маршрутизации: окно 2026-11-16 (новая схема).\n\n' +
      'const rr = { thresholdKb: 80000, rescore: true };\n\n' +
      'Подробный отчет в журнале: портал инцидента или https://t.me/core_updates.\n' +
      'Детали миграции: при включении ACL-фильтра селективности 0.4% полнота Recall@30 стабилизируется без полного сканирования.\n' +
      'Внутренний тикет: INC-2026-8841.',
    entities: [
      { type: 'custom_emoji', offset: 0, length: 2, document_id: '5368324170671202286' },
      {
        type: 'messageEntityFormattedDate',
        offset: 36,
        length: 10,
        unix_time: 1794830400,
        date_flags: DATE_FLAG_DAY_OF_WEEK | DATE_FLAG_SHORT_TIME | DATE_FLAG_LONG_DATE,
      },
      { type: 'diff_replace', offset: 48, length: 11, old_text: 'старый пул' },
      {
        type: 'messageEntityPre',
        offset: 63,
        length: 47,
        language: 'typescript',
      },
      {
        type: 'messageEntityTextUrl',
        offset: 139,
        length: 16,
        url: 'https://xn--e1afmkfd.xn--p1ai/status',
      },
      { type: 'expandable_blockquote', offset: 189, length: 121, collapsed: true },
      { type: 'spoiler', offset: 330, length: 13 },
    ],
    media: [],
    originScore: 0.91,
    duplicateCount: 3,
    commentCount: 64,
    rubric: 'AI & Data Science',
  },
  {
    id: 'post-album-104',
    channelId: '-1001002',
    channelTitle: 'Highload Engineering',
    channelHandle: '@highload_notes',
    channelAvatarInitials: 'HE',
    postId: 104,
    publishedAt: '2026-11-15T07:50:00Z',
    sourcePlatform: 'MAX',
    clusterId: 'cluster-bgp-2026',
    neutralHeadline: 'Телеметрия медиа-прокси и буферизации альбомов grouped_id (4 снимка дашборда)',
    originalHeadline: '⚡ Минус 42% нагрузки на RPC! Секрет буферизации альбомов в проде',
    text: 'После включения буферизации альбомов (окно 1500 мс) и чанкования Range 206 по 512 КБ нагрузка RPC сократилась на 42%. Отдельно проверен сценарий HTTP 410 Gone на четвертом вложении.',
    entities: [{ type: 'bold', offset: 54, length: 29 }],
    groupedId: 'grp_9941',
    showCaptionAboveMedia: true,
    media: [
      {
        mediaId: 'm-1',
        fileId: 'file_bgp_latency',
        mediaType: 'photo',
        mimeType: 'image/svg+xml',
        width: 640,
        height: 420,
        thumbnailDataUrl: makePreviewSvgDataUrl('#e0f2fe', '#f8fafc', 'HTTP 206 Range Chunks', 'Media Edge-Proxy · 512 KB align', '-42% RPC'),
        streamUrl: '/api/v1/media/file_bgp_latency',
        caption: 'График выравнивания 512 КБ чанков',
      },
      {
        mediaId: 'm-2',
        fileId: 'file_s3_cache',
        mediaType: 'photo',
        mimeType: 'image/svg+xml',
        width: 640,
        height: 420,
        thumbnailDataUrl: makePreviewSvgDataUrl('#ecfdf5', '#f8fafc', 'Pre-fetch Thumbnails', 'S3 Standard РФ · TTL 30d', '35 KB avg'),
        streamUrl: '/api/v1/media/file_s3_cache',
        caption: 'Кэш легких превью в Selectel/Yandex',
      },
      {
        mediaId: 'm-3',
        fileId: 'file_album_buffer',
        mediaType: 'photo',
        mimeType: 'image/svg+xml',
        width: 640,
        height: 420,
        thumbnailDataUrl: makePreviewSvgDataUrl('#fef3c7', '#f8fafc', 'grouped_id Aggregator', 'Window 1500ms · Zero splits', '99.4%'),
        streamUrl: '/api/v1/media/file_album_buffer',
      },
      {
        mediaId: 'm-4',
        fileId: 'file_gone_archive',
        mediaType: 'photo',
        mimeType: 'image/svg+xml',
        width: 640,
        height: 420,
        thumbnailDataUrl: makePreviewSvgDataUrl('#ffe4e6', '#f8fafc', 'Archived Source Fallback', 'Orig deleted · Cached S3 preview', '410 Gone'),
        streamUrl: '/api/v1/media/file_gone_archive',
        caption: 'Демонстрация отката при удалении оригинала (HTTP 410 Gone)',
        fileGone: true,
      },
    ],
    originScore: 0.89,
    duplicateCount: 2,
    commentCount: 19,
    rubric: 'IT & Dev',
  },
  {
    id: 'post-macro-201',
    channelId: '-1002002',
    channelTitle: 'ML Inference Lab',
    channelHandle: '@ml_inference_lab',
    channelAvatarInitials: 'MI',
    postId: 201,
    publishedAt: '2026-11-15T06:30:00Z',
    sourcePlatform: 'RSS',
    clusterId: 'cluster-finops-gpu',
    neutralHeadline: 'Сравнение стоимости облачного Batch API и собственных GPU H100 при 5 000 пользователей',
    originalHeadline: '💸 Не покупайте H100! Почему Batch API разносит своё железо в пух и прах',
    text: 'При нагрузке 2 500–5 000 подписчиков облачный GPT-6 Luna Batch выигрывает у аренды кластера 1× H100 + L40S ($3 198/мес) до достижения устойчивого порога 6 800 пользователей.',
    entities: [{ type: 'bold', offset: 110, length: 12 }],
    media: [],
    originScore: 0.92,
    duplicateCount: 4,
    commentCount: 27,
    rubric: 'AI & Data Science',
  },
];

export const SEED_DIGEST_RESPONSE: DigestFeedResponse = {
  mode: 'digest',
  user_id: 'demo_user',
  finops_zone: 'GREEN',
  reduce_model: 'DeepSeek V4.1 Flash / Shared Map',
  total_candidate_clusters: 42,
  selected_cards_count: 3,
  rubrics: ['Инфраструктура и Сети', 'AI и Инференс', 'Product & FinOps'],
  scheduleWindow: '09:00',
  nextEditionTime: '18:00',
  pendingNextSlotCount: 14,
  fromDemoFallback: true,
  cards: [
    {
      clusterId: 'cluster-bgp-2026',
      rubric: 'Инфраструктура и Сети',
      neutralTitle: 'Потеря 14% трафика ЦОД из-за транзитного сбоя BGP',
      importanceScore: 0.95,
      primarySource: {
        channelId: '-1001001',
        channelTitle: 'Infra Postmortem',
        channelHandle: '@infra_postmortem',
        originScore: 0.94,
        postId: '42',
        publishedAt: '2026-11-15T08:15:00Z',
      },
      otherChannelsCount: 6,
      otherChannels: SEED_CLEAN_POSTS[0].duplicateSources ?? [],
      summaryBullets: [
        {
          text: 'Подтверждено снижение транзитного трафика до дата-центров на 14% во время перестройки BGP-маршрутов.',
          citations: [
            {
              claimId: 'c1',
              channelId: '-1001001',
              channelTitle: 'Infra Postmortem',
              channelHandle: '@infra_postmortem',
              postId: 'post-bgp-42',
              quoteText: 'потеряно 14% трафика',
              publishedAt: '2026-11-15T08:15:00Z',
              originScore: 0.94,
              startUtf16: 13,
              endUtf16: 33,
            },
          ],
        },
        {
          text: 'В комментариях инженеры зафиксировали деградацию прерывистых TCP-соединений и рекомендовали fallback через AmneziaWG / NaiveProxy.',
          citations: [
            {
              claimId: 'c2',
              channelId: '-1001002',
              channelTitle: 'Highload Engineering',
              channelHandle: '@highload_notes',
              postId: 'post-album-104',
              quoteText: 'нагрузка RPC сократилась на 42%',
              publishedAt: '2026-11-15T07:50:00Z',
              originScore: 0.89,
              startUtf16: 97,
              endUtf16: 126,
            },
          ],
        },
      ],
    },
    {
      clusterId: 'cluster-qdrant-2026',
      rubric: 'AI и Инференс',
      neutralTitle: 'Стабилизация полноты Recall@30 на узких фильтрах ACL в Qdrant ACORN-1',
      importanceScore: 0.88,
      primarySource: {
        channelId: '-1002001',
        channelTitle: 'LLM Systems & RAG',
        channelHandle: '@llm_systems_ru',
        originScore: 0.91,
        postId: '88',
        publishedAt: '2026-11-15T08:40:00Z',
      },
      otherChannelsCount: 3,
      otherChannels: [],
      summaryBullets: [
        {
          text: 'При селективности фильтрации 0.4% сохраняется стабильный Recall@30 при пороге full_scan_threshold 80 000 KB.',
          citations: [
            {
              claimId: 'c3',
              channelId: '-1002001',
              channelTitle: 'LLM Systems & RAG',
              channelHandle: '@llm_systems_ru',
              postId: 'post-k8s-88',
              quoteText: 'полнота Recall@30 стабилизируется без полного сканирования',
              publishedAt: '2026-11-15T08:40:00Z',
              originScore: 0.91,
              startUtf16: 239,
              endUtf16: 299,
            },
          ],
        },
      ],
    },
    {
      clusterId: 'cluster-finops-gpu',
      rubric: 'Product & FinOps',
      neutralTitle: 'Порог окупаемости перехода с Batch API на собственные GPU (6 800 пользователей)',
      importanceScore: 0.84,
      primarySource: {
        channelId: '-1002002',
        channelTitle: 'ML Inference Lab',
        channelHandle: '@ml_inference_lab',
        originScore: 0.92,
        postId: '201',
        publishedAt: '2026-11-15T06:30:00Z',
      },
      otherChannelsCount: 4,
      otherChannels: [],
      summaryBullets: [
        {
          text: 'До масштаба 6 800 платящих подписчиков оптимизированный Batch API экономичнее аренды собственного H100-кластера ($3 198/мес).',
          citations: [
            {
              claimId: 'c4',
              channelId: '-1002002',
              channelTitle: 'ML Inference Lab',
              channelHandle: '@ml_inference_lab',
              postId: 'post-macro-201',
              quoteText: '$3 198/мес) до достижения устойчивого порога 6 800 пользователей',
              publishedAt: '2026-11-15T06:30:00Z',
              originScore: 0.92,
              startUtf16: 110,
              endUtf16: 171,
            },
          ],
        },
      ],
    },
  ],
};

export const SEED_CLEAN_RESPONSE: CleanOriginalsResponse = {
  mode: 'clean_originals',
  user_id: 'demo_user',
  total: SEED_CLEAN_POSTS.length,
  items: SEED_CLEAN_POSTS,
  fromDemoFallback: true,
};

export const SEED_COMMENTS_BY_POST: Record<string, CommentInputItem[]> = {
  'post-bgp-42': [
    {
      comment_id: 'cm-1',
      author_alias: 'net_arch',
      text: 'Подтверждаю: на узле Frankfurt-2 AS-path ушел в петлю ровно на 14 минут, потом стабилизировалось.',
      likes: 12,
      replies_count: 3,
      is_bug_report_hint: false,
    },
    {
      comment_id: 'cm-2',
      parent_id: 'cm-1',
      author_alias: 'dev_ops_42',
      text: 'Баг-репорт: клиентский SSE-стрим не отправлял keep-alive на этапе реранка, из-за чего Nginx закрывал соединение ровно на 60-й секунде!',
      likes: 1,
      replies_count: 0,
      is_bug_report_hint: true,
    },
    {
      comment_id: 'cm-3',
      author_alias: 'sre_lead',
      text: 'Плюс один, без keep-alive раз в 15с stateful NAT тоже режет длинные сессии.',
      likes: 8,
      replies_count: 1,
      is_bug_report_hint: false,
    },
  ],
};

export function buildDemoCommentSummary(postId: string): CommentSummaryResponse {
  const comments = SEED_COMMENTS_BY_POST[postId] ?? [];
  return {
    post_id: postId,
    bucket: comments.length > 0 ? '300_plus' : 'light',
    cache_hit: false,
    kept_after_pruning: comments.length || 2,
    digest_text:
      comments.length > 0
        ? 'Пользователи подтвердили кратковременную петлю AS-path во Франкфурте и выделили критический баг-репорт в глубине ветки: без SSE `: keep-alive` каждые 15 секунд Nginx/NAT разрывает длинные запросы.'
        : 'Ключевые технические замечания: требуется контроль UTF-16 смещений и валидация HTTP Range 206 при ретраях.',
    selected: [
      {
        comment_id: 'cm-2',
        parent_comment_id: 'cm-1',
        parent_author_alias: 'net_arch',
        parent_preview: 'Подтверждаю: на узле Frankfurt-2 AS-path ушел в петлю ровно на 14 минут...',
        author_alias: 'dev_ops_42',
        likes: 18,
        confirmed_count: 4,
        text: 'Баг-репорт: клиентский SSE-стрим не отправлял keep-alive на этапе реранка, из-за чего Nginx закрывал соединение ровно на 60-й секунде!',
        category: 'bug_report',
        importance_score: 0.96,
        is_saved_rare_bug: true,
      },
      {
        comment_id: 'cm-3',
        author_alias: 'sre_lead',
        likes: 11,
        text: 'Контраргумент: локальный stateful NAT режет длинные сессии уже на 45-й секунде, одного таймера 60 с в Nginx недостаточно.',
        category: 'counter_argument',
        importance_score: 0.89,
      },
      {
        comment_id: 'cm-1',
        author_alias: 'net_arch',
        likes: 12,
        text: 'Кейс продакшена: на узле Frankfurt-2 AS-path ушел в петлю на 14 минут — переключение на резервный шлюз AmneziaWG удержало SLA.',
        category: 'user_case',
        importance_score: 0.84,
      },
    ],
    fromDemoFallback: true,
  };
}

export const SEED_DEMO_RAG_CLAIMS: VerifiedClaim[] = [
  {
    claim_id: 'c1',
    topic_group: 'Инцидент маршрутизации',
    statement:
      'Во время сбоя BGP в дата-центрах зафиксирована потеря 14% входящего и транзитного трафика.',
    citations: [
      {
        claim_id: 'c1',
        post_id: 'post-bgp-42',
        channel_id: '-1001001',
        channel_title: 'Infra Postmortem',
        channel_handle: '@infra_postmortem',
        quote: 'потеряно 14% трафика',
        start_utf16: 13,
        end_utf16: 33,
        published_at: '2026-11-15T08:15:00Z',
        origin_score: 0.94,
      },
    ],
  },
  {
    claim_id: 'c2',
    topic_group: 'Медиа и пропускная способность',
    statement:
      'Пакетная буферизация альбомов (1500 мс) и ленивое Range-чтение по 512 КБ снизили количество RPC на 42%.',
    citations: [
      {
        claim_id: 'c2',
        post_id: 'post-album-104',
        channel_id: '-1001002',
        channel_title: 'Highload Engineering',
        channel_handle: '@highload_notes',
        quote: 'нагрузка RPC сократилась на 42%',
        start_utf16: 97,
        end_utf16: 126,
        published_at: '2026-11-15T07:50:00Z',
        origin_score: 0.89,
      },
    ],
  },
];

export const SEED_DEMO_STANCE_ROWS: StanceMatrixRow[] = [
  {
    aspect: 'Переход на собственные GPU (H100 + L40S) vs облачные Batch API',
    nuances: [
      'Требование 152-ФЗ РФ для закрытых каналов смещает порог окупаемости вниз до 1 000–1 500 пользователей.',
      'При сезонном трафике Batch API (-50%) сохраняет гибкость без простоев GPU.',
    ],
    pro_points: [
      {
        claim_id: 'pro-gpu',
        statement:
          'При достижении устойчивого масштаба 6 800+ подписчиков (и в контуре 152-ФЗ РФ) собственные GPU обеспечивают полную независимость и экономию OpEx.',
        stance: 'PRO',
        confirmed_channels_count: 3,
        citations: [
          {
            claim_id: 'pro-gpu',
            post_id: 'post-macro-201',
            channel_id: '-1002002',
            channel_title: 'ML Inference Lab',
            channel_handle: '@ml_inference_lab',
            quote: 'устойчивого порога 6 800 пользователей',
            start_utf16: 135,
            end_utf16: 171,
            published_at: '2026-11-15T06:30:00Z',
            origin_score: 0.92,
          },
        ],
      },
    ],
    con_points: [
      {
        claim_id: 'con-api',
        statement:
          'На масштабе 2 500–5 000 пользователей аренда кластера 1× H100 + L40S ($3 198/мес) дороже оптимизированного Batch API.',
        stance: 'CON',
        confirmed_channels_count: 2,
        citations: [
          {
            claim_id: 'con-api',
            post_id: 'post-macro-201',
            channel_id: '-1002002',
            channel_title: 'ML Inference Lab',
            channel_handle: '@ml_inference_lab',
            quote: 'облачный GPT-6 Luna Batch выигрывает у аренды кластера 1× H100 + L40S ($3 198/мес)',
            start_utf16: 40,
            end_utf16: 122,
            published_at: '2026-11-15T06:30:00Z',
            origin_score: 0.92,
          },
        ],
      },
    ],
  },
];

export const SEED_DEMO_RAG_SOURCE_POSTS: RagSourcePostItem[] = [
  {
    post: SEED_CLEAN_POSTS[0],
    rag_highlights: [{ claimId: 'c1', startUtf16: 13, endUtf16: 33 }],
  },
  {
    post: SEED_CLEAN_POSTS[2],
    rag_highlights: [{ claimId: 'c2', startUtf16: 97, endUtf16: 126 }],
  },
  {
    post: SEED_CLEAN_POSTS[3],
    rag_highlights: [
      { claimId: 'pro-gpu', startUtf16: 135, endUtf16: 171 },
      { claimId: 'con-api', startUtf16: 40, endUtf16: 122 },
    ],
  },
];
