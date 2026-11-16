import type {
  CleanPostItem,
  DirectAnswerStatus,
  RagDoneMetadata,
  RagMode,
  RagSourcePostItem,
  StanceMatrixRow,
  VerifiedCitation,
  VerifiedClaim,
  VerifiedDirectAnswer,
} from '@/lib/types/feed';

export interface SseFrame {
  event: string;
  data: string;
  id?: string;
}

export interface RagQueryRequest {
  user_id: string;
  query: string;
  mode: RagMode;
  channel_ids: string[];
  time_window_days?: number;
  cluster_id?: string;
  plan_tier?: string;
  deep_analysis?: boolean;
  preloaded_posts?: Array<Record<string, unknown>>;
}

export interface RagStreamCallbacks {
  onStage?: (payload: { stage: string; message?: string }) => void;
  onDirectAnswer?: (answer: VerifiedDirectAnswer) => void;
  onClaim?: (claim: VerifiedClaim) => void;
  onStanceRow?: (row: StanceMatrixRow) => void;
  onSourcePosts?: (posts: RagSourcePostItem[]) => void;
  onKeepAlive?: (timestampMs: number) => void;
  onDone?: (meta: RagDoneMetadata) => void;
  onError?: (errorMessage: string) => void;
}

function normalizeCitation(raw: Record<string, unknown>): VerifiedCitation {
  const range = Array.isArray(raw.utf16_range) ? raw.utf16_range : undefined;
  const rawUser = raw.channel_username ? String(raw.channel_username).replace(/^@/, '') : '';
  const channelHandle = raw.channel_handle
    ? String(raw.channel_handle)
    : rawUser
      ? `@${rawUser}`
      : raw.channel_title
        ? String(raw.channel_title)
        : undefined;

  return {
    claim_id: raw.claim_id ? String(raw.claim_id) : undefined,
    post_id: String(raw.post_id ?? ''),
    channel_id: String(raw.channel_id ?? ''),
    channel_title: raw.channel_title ? String(raw.channel_title) : undefined,
    channel_handle: channelHandle,
    quote: String(raw.quote ?? raw.verbatim_text ?? ''),
    start_utf16: Number(raw.start_utf16 ?? range?.[0] ?? 0),
    end_utf16: Number(raw.end_utf16 ?? range?.[1] ?? 0),
    published_at: raw.published_at ? String(raw.published_at) : undefined,
    origin_score: typeof raw.origin_score === 'number' ? raw.origin_score : undefined,
  };
}

function normalizeClaim(raw: Record<string, unknown>): VerifiedClaim {
  const rawCites = Array.isArray(raw.citations) ? raw.citations : [];
  return {
    claim_id: String(raw.claim_id ?? ''),
    topic_group: String(raw.topic_group ?? 'Факты'),
    statement: String(raw.statement ?? ''),
    verification_status:
      (raw.verification_status as VerifiedClaim['verification_status']) ?? 'entailed',
    citations: rawCites.map((c) => normalizeCitation((c ?? {}) as Record<string, unknown>)),
  };
}

export function normalizeDirectAnswer(raw: Record<string, unknown>): VerifiedDirectAnswer {
  const rawCites = Array.isArray(raw.citations) ? raw.citations : [];
  const rawStatus = String(raw.status ?? 'direct').toLowerCase();
  const status: DirectAnswerStatus =
    rawStatus === 'indirect' || rawStatus === 'not_found' ? rawStatus : 'direct';
  return {
    status,
    text: String(raw.text ?? ''),
    citations: rawCites.map((c) => normalizeCitation((c ?? {}) as Record<string, unknown>)),
  };
}

function normalizeSourcePostItem(raw: Record<string, unknown>): RagSourcePostItem {
  const p = (raw.post ?? {}) as Record<string, unknown>;
  const rawUser = p.channel_username ? String(p.channel_username).replace(/^@/, '') : '';
  const channelHandle = p.channelHandle
    ? String(p.channelHandle)
    : rawUser
      ? `@${rawUser}`
      : String(p.channel_title ?? p.channel_id ?? '');
  const channelTitle = String(p.channelTitle ?? p.channel_title ?? (rawUser || channelHandle));
  const rawText = String(p.text ?? p.text_raw ?? '');
  const rawHeadline = p.neutralHeadline ?? p.neutral_headline;
  const fallbackHeadline = rawText.split('\n')[0]?.slice(0, 110) || 'Публикация';

  const normalizedPost: CleanPostItem = {
    id: String(p.id ?? p.post_id ?? ''),
    channelId: String(p.channelId ?? p.channel_id ?? ''),
    channelTitle,
    channelHandle,
    postId: (p.postId ?? p.post_id ?? p.msg_id ?? '') as string | number,
    publishedAt: String(p.publishedAt ?? p.published_at ?? new Date().toISOString()),
    neutralHeadline: rawHeadline ? String(rawHeadline) : fallbackHeadline,
    originalHeadline: p.originalHeadline ? String(p.originalHeadline) : undefined,
    sourcePlatform: (p.sourcePlatform as CleanPostItem['sourcePlatform']) ?? 'TG',
    clusterId: p.clusterId ? String(p.clusterId) : p.cluster_id ? String(p.cluster_id) : undefined,
    text: rawText,
    entities: Array.isArray(p.entities) ? (p.entities as CleanPostItem['entities']) : [],
    media: Array.isArray(p.media) ? (p.media as CleanPostItem['media']) : [],
    commentCount: Number(p.commentCount ?? p.replies_count ?? 0),
    isPrivateChannel: Boolean(p.isPrivateChannel),
  };

  const highlights = Array.isArray(raw.rag_highlights)
    ? (raw.rag_highlights as RagSourcePostItem['rag_highlights'])
    : [];

  return {
    post: normalizedPost,
    rag_highlights: highlights,
  };
}

/**
 * Incremental SSE frame decoder supporting arbitrary chunk splits across TCP/HTTP boundaries
 * and `: keep-alive` heartbeat comments (SPEC 7.5.3).
 */
export class IncrementalSseParser {
  private buffer = '';

  pushChunk(
    chunkText: string,
    onFrame: (frame: SseFrame) => void,
    onComment?: (comment: string) => void,
  ): void {
    this.buffer += chunkText.replace(/\r\n/g, '\n');

    let separatorIdx = this.buffer.indexOf('\n\n');
    while (separatorIdx !== -1) {
      const rawBlock = this.buffer.slice(0, separatorIdx);
      this.buffer = this.buffer.slice(separatorIdx + 2);
      this.processBlock(rawBlock, onFrame, onComment);
      separatorIdx = this.buffer.indexOf('\n\n');
    }
  }

  flush(onFrame: (frame: SseFrame) => void, onComment?: (comment: string) => void): void {
    if (this.buffer.trim().length > 0) {
      this.processBlock(this.buffer, onFrame, onComment);
      this.buffer = '';
    }
  }

  private processBlock(
    rawBlock: string,
    onFrame: (frame: SseFrame) => void,
    onComment?: (comment: string) => void,
  ): void {
    const lines = rawBlock.split('\n');
    let eventName = 'message';
    let eventId: string | undefined;
    const dataLines: string[] = [];

    for (const line of lines) {
      if (!line) continue;
      if (line.startsWith(':')) {
        onComment?.(line.slice(1).trim());
        continue;
      }
      const colonIdx = line.indexOf(':');
      const field = colonIdx === -1 ? line : line.slice(0, colonIdx);
      let value = colonIdx === -1 ? '' : line.slice(colonIdx + 1);
      if (value.startsWith(' ')) {
        value = value.slice(1);
      }

      if (field === 'event') {
        eventName = value;
      } else if (field === 'data') {
        dataLines.push(value);
      } else if (field === 'id') {
        eventId = value;
      }
    }

    if (dataLines.length > 0) {
      onFrame({
        event: eventName,
        data: dataLines.join('\n'),
        id: eventId,
      });
    }
  }
}

export function dispatchParsedSseFrame(
  frame: SseFrame,
  callbacks: RagStreamCallbacks,
): void {
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(frame.data) as Record<string, unknown>;
  } catch {
    if (frame.event === 'error') {
      callbacks.onError?.(frame.data);
    }
    return;
  }

  switch (frame.event) {
    case 'pipeline_stage':
      callbacks.onStage?.(payload as { stage: string; message?: string });
      break;
    case 'direct_answer':
      callbacks.onDirectAnswer?.(normalizeDirectAnswer(payload));
      break;
    case 'verified_claim':
      callbacks.onClaim?.(normalizeClaim(payload));
      break;
    case 'stance_row': {
      const rawPro = Array.isArray(payload.pro_points) ? payload.pro_points : [];
      const rawCon = Array.isArray(payload.con_points) ? payload.con_points : [];
      const normalizePt = (pt: Record<string, unknown>) => ({
        claim_id: String(pt.claim_id ?? ''),
        statement: String(pt.statement ?? ''),
        stance: (pt.stance as 'PRO' | 'CON' | 'MIXED') ?? 'PRO',
        confirmed_channels_count:
          typeof pt.confirmed_channels_count === 'number'
            ? pt.confirmed_channels_count
            : undefined,
        citations: Array.isArray(pt.citations)
          ? pt.citations.map((c) => normalizeCitation((c ?? {}) as Record<string, unknown>))
          : [],
      });
      callbacks.onStanceRow?.({
        aspect: String(payload.aspect ?? 'Общая оценка'),
        pro_points: rawPro.map((p) => normalizePt((p ?? {}) as Record<string, unknown>)),
        con_points: rawCon.map((p) => normalizePt((p ?? {}) as Record<string, unknown>)),
        nuances: Array.isArray(payload.nuances) ? payload.nuances.map(String) : undefined,
      });
      break;
    }
    case 'source_posts': {
      // Backend routes_rag.py sends `{"source_posts": [...]}`; SPEC 5.4 also documents `"posts"`
      const rawList =
        (payload.source_posts as Array<Record<string, unknown>> | undefined) ??
        (payload.posts as Array<Record<string, unknown>> | undefined) ??
        [];
      callbacks.onSourcePosts?.(rawList.map(normalizeSourcePostItem));
      break;
    }
    case 'done':
      callbacks.onDone?.(payload as unknown as RagDoneMetadata);
      break;
    case 'error':
      callbacks.onError?.(
        String(payload.message ?? payload.detail ?? 'Ошибка потокового поиска SSE'),
      );
      break;
    default:
      break;
  }
}

/**
 * Execute streaming hybrid RAG query over `POST /api/v1/rag/query`.
 * Uses Fetch ReadableStream + `IncrementalSseParser` to support POST body + `: keep-alive` frames.
 */
export async function streamRagQuery(
  request: RagQueryRequest,
  callbacks: RagStreamCallbacks,
  options?: {
    signal?: AbortSignal;
    endpointUrl?: string;
    fetchImpl?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  },
): Promise<{ usedFallback: boolean }> {
  const endpoint = options?.endpointUrl ?? '/api/v1/rag/query';
  const fetchFn = options?.fetchImpl ?? fetch;

  const response = await fetchFn(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(request),
    signal: options?.signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`RAG SSE HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  const parser = new IncrementalSseParser();

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const textChunk = decoder.decode(value, { stream: true });
    parser.pushChunk(
      textChunk,
      (frame) => dispatchParsedSseFrame(frame, callbacks),
      () => callbacks.onKeepAlive?.(Date.now()),
    );
  }

  const tail = decoder.decode();
  if (tail) {
    parser.pushChunk(
      tail,
      (frame) => dispatchParsedSseFrame(frame, callbacks),
      () => callbacks.onKeepAlive?.(Date.now()),
    );
  }
  parser.flush(
    (frame) => dispatchParsedSseFrame(frame, callbacks),
    () => callbacks.onKeepAlive?.(Date.now()),
  );

  return { usedFallback: false };
}
