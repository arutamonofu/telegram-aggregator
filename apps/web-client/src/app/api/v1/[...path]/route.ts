import dns from 'node:dns';
import net from 'node:net';
import type { NextRequest } from 'next/server';

try {
  dns.setDefaultResultOrder?.('ipv4first');
  net.setDefaultAutoSelectFamily?.(false);
} catch {
  // Ignore if unsupported by the active JS runtime
}

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

function resolveBackendBaseUrl(): string {
  const raw = (
    process.env.INTERNAL_API_BASE_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    'http://127.0.0.1:8000'
  ).replace(/\/$/, '');
  return raw.replace(/^http:\/\/localhost(?=:|\/|$)/i, 'http://127.0.0.1');
}

async function proxyToBackend(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await context.params;
  const joinedPath = path.join('/');
  const incomingUrl = new URL(request.url);
  const targetUrl = `${resolveBackendBaseUrl()}/api/v1/${joinedPath}${incomingUrl.search}`;

  const headers = new Headers();
  const contentType = request.headers.get('content-type');
  if (contentType) headers.set('content-type', contentType);
  const accept = request.headers.get('accept');
  if (accept) headers.set('accept', accept);

  const method = request.method.toUpperCase();
  const hasBody = method !== 'GET' && method !== 'HEAD';
  const isStreamingPath =
    joinedPath.startsWith('rag/query') ||
    joinedPath.startsWith('media/') ||
    Boolean(accept?.includes('text/event-stream'));

  try {
    const upstream = await fetch(targetUrl, {
      method,
      headers,
      body: hasBody ? await request.arrayBuffer() : undefined,
      signal: AbortSignal.timeout(isStreamingPath ? 60000 : 12000),
      cache: 'no-store',
    });

    const responseHeaders = new Headers();
    const upstreamType = upstream.headers.get('content-type');
    if (upstreamType) responseHeaders.set('content-type', upstreamType);
    const cacheControl = upstream.headers.get('cache-control');
    if (cacheControl) responseHeaders.set('cache-control', cacheControl);

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Upstream connection failed';
    return Response.json(
      { error: 'Backend API unavailable', detail: message },
      { status: 502 },
    );
  }
}

export const GET = proxyToBackend;
export const POST = proxyToBackend;
export const PUT = proxyToBackend;
export const PATCH = proxyToBackend;
export const DELETE = proxyToBackend;
export const OPTIONS = proxyToBackend;
