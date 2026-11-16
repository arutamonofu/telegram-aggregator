import type { NextRequest } from 'next/server';
import {
  fetchLiveChannelsBatch,
  type LiveChannelInput,
} from '@/lib/live/live_fetcher';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function POST(request: NextRequest): Promise<Response> {
  let payload: { channels?: LiveChannelInput[]; forceRefresh?: boolean };
  try {
    payload = (await request.json()) as {
      channels?: LiveChannelInput[];
      forceRefresh?: boolean;
    };
  } catch {
    return Response.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const channels = Array.isArray(payload.channels)
    ? payload.channels.slice(0, 25)
    : [];

  const result = await fetchLiveChannelsBatch(channels, {
    forceRefresh: Boolean(payload.forceRefresh),
  });

  return Response.json(result);
}
