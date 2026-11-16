import dns from 'node:dns';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';
import tls from 'node:tls';
import zlib from 'node:zlib';
import {
  extractTelegramChannelTitle,
  parseRssOrAtomXml,
  parseTelegramPublicHtml,
} from '@/lib/live/tme_public_parser';
import type { CleanPostItem, SourcePlatform } from '@/lib/types/feed';

// Prefer IPv4 DNS resolution and disable Happy Eyeballs multi-family probing on load.
// On Linux VPN/TUN setups where IPv6 is unrouted or blackholed, Happy Eyeballs in Node.js
// undici triggers spurious timeouts and connection resets.
try {
  dns.setDefaultResultOrder?.('ipv4first');
  net.setDefaultAutoSelectFamily?.(false);
} catch {
  // Ignore if unsupported by the active JS runtime
}

export interface LiveChannelInput {
  channelId: string;
  handle: string;
  title?: string;
  platform?: SourcePlatform;
}

export interface LiveChannelBatchOptions {
  maxConcurrency?: number;
  timeoutMs?: number;
  maxRetries?: number;
  retryDelayMs?: number;
  interRequestDelayMs?: number;
  forceRefresh?: boolean;
  bypassCache?: boolean;
  skipDirectTme?: boolean;
  proxyUrl?: string;
  fetchImpl?: typeof fetch;
}

export interface LiveChannelBatchResult {
  items: CleanPostItem[];
  errors: Record<string, string>;
  channelTitles: Record<string, string>;
  fetchedAt: string;
}

interface FetchTextResponse {
  ok: boolean;
  status: number;
  text: () => Promise<string>;
}

interface CachedChannelEntry {
  items: CleanPostItem[];
  resolvedTitle?: string;
  fetchedAtMs: number;
}

interface SingleChannelOutcome {
  items: CleanPostItem[];
  resolvedTitle?: string;
  error?: string;
}

const USERNAME_RE = /^[a-zA-Z][a-zA-Z0-9_]{3,30}$/;

const FRESH_TTL_MS = 60_000;
const MIN_BURST_INTERVAL_MS = 5_000;
const STALE_TTL_MS = 30 * 60_000;
const DEFAULT_TIMEOUT_MS = 12_000;
const DEFAULT_MAX_RETRIES = 1;
const MAX_RSS_REDIRECTS = 3;

const channelCache = new Map<string, CachedChannelEntry>();
const inFlightRequests = new Map<string, Promise<SingleChannelOutcome>>();

let sharedHttpsAgent: https.Agent | null = null;
let sharedHttpAgent: http.Agent | null = null;

function getIpv4Agents(): { httpsAgent: https.Agent; httpAgent: http.Agent } {
  if (!sharedHttpsAgent) {
    sharedHttpsAgent = new https.Agent({
      keepAlive: true,
      maxSockets: 2,
      maxFreeSockets: 2,
      family: 4,
      timeout: DEFAULT_TIMEOUT_MS,
    });
  }
  if (!sharedHttpAgent) {
    sharedHttpAgent = new http.Agent({
      keepAlive: true,
      maxSockets: 2,
      maxFreeSockets: 2,
      family: 4,
      timeout: DEFAULT_TIMEOUT_MS,
    });
  }
  return { httpsAgent: sharedHttpsAgent, httpAgent: sharedHttpAgent };
}

function clampConcurrency(value: number | undefined): number {
  const fallback = Number(process.env.LIVE_FETCH_CONCURRENCY);
  const raw =
    typeof value === 'number' && Number.isFinite(value)
      ? value
      : Number.isFinite(fallback) && fallback > 0
        ? fallback
        : 2;
  return Math.max(1, Math.min(4, Math.floor(raw)));
}

function sleep(ms: number): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function clearLiveFetcherCache(): void {
  channelCache.clear();
  inFlightRequests.clear();
}

export function isSafePublicHttpUrl(rawUrl: string): boolean {
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
      return false;
    }
    const host = parsed.hostname.toLowerCase();
    if (
      host === 'localhost' ||
      host.startsWith('127.') ||
      host.startsWith('10.') ||
      host.startsWith('192.168.') ||
      host.startsWith('169.254.') ||
      host === '0.0.0.0' ||
      host === '[::1]' ||
      host === '::1'
    ) {
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

function extractErrorDetails(err: unknown): {
  name: string;
  message: string;
  code: string;
} {
  if (!(err instanceof Error)) {
    return { name: '', message: String(err ?? ''), code: '' };
  }
  const cause = (err as Error & { cause?: unknown }).cause;
  const errCode =
    typeof (err as Error & { code?: unknown }).code === 'string'
      ? (err as Error & { code: string }).code
      : '';
  const causeCode =
    cause && typeof cause === 'object' && typeof (cause as { code?: unknown }).code === 'string'
      ? (cause as { code: string }).code
      : '';
  const causeMessage =
    cause instanceof Error
      ? cause.message
      : cause && typeof cause === 'object' && typeof (cause as { message?: unknown }).message === 'string'
        ? (cause as { message: string }).message
        : '';

  return {
    name: err.name ?? '',
    message: [err.message, causeMessage].filter(Boolean).join(' '),
    code: causeCode || errCode,
  };
}

export function formatLiveFetchError(err: unknown): string {
  const { name, message, code } = extractErrorDetails(err);
  const upperCode = code.toUpperCase();

  if (
    name === 'TimeoutError' ||
    name === 'AbortError' ||
    upperCode === 'ETIMEDOUT' ||
    upperCode === 'UND_ERR_CONNECT_TIMEOUT' ||
    upperCode === 'UND_ERR_HEADERS_TIMEOUT' ||
    upperCode === 'UND_ERR_BODY_TIMEOUT' ||
    /aborted due to timeout|timed out/i.test(message)
  ) {
    return 'Превышено время ожидания ответа от t.me (таймаут сети/VPN)';
  }

  if (upperCode === 'ENOTFOUND' || upperCode === 'EAI_AGAIN') {
    return `Ошибка DNS при обращении к t.me (проверьте DNS/VPN)${upperCode ? ` [${upperCode}]` : ''}`;
  }

  if (
    upperCode === 'ECONNRESET' ||
    upperCode === 'ECONNREFUSED' ||
    upperCode === 'ENETUNREACH' ||
    upperCode === 'EHOSTUNREACH' ||
    upperCode === 'EPIPE' ||
    upperCode === 'UND_ERR_SOCKET' ||
    /fetch failed|socket hang up|econnreset|enetunreach/i.test(message)
  ) {
    const suffix = upperCode ? ` (${upperCode})` : '';
    return `Соединение с t.me прервано (сбой сети или перегрузка VPN-туннеля)${suffix}`;
  }

  return message || 'Не удалось загрузить публичную ленту';
}

function isRetryableStatus(status: number): boolean {
  return status === 429 || status === 502 || status === 503 || status === 504;
}

function isRetryableError(err: unknown): boolean {
  const { name, message, code } = extractErrorDetails(err);
  const upperCode = code.toUpperCase();
  if (
    name === 'TimeoutError' ||
    name === 'AbortError' ||
    upperCode === 'ETIMEDOUT' ||
    upperCode === 'ECONNRESET' ||
    upperCode === 'EAI_AGAIN' ||
    upperCode === 'ENETUNREACH' ||
    upperCode === 'EHOSTUNREACH' ||
    upperCode === 'UND_ERR_SOCKET' ||
    upperCode === 'UND_ERR_CONNECT_TIMEOUT'
  ) {
    return true;
  }
  return /fetch failed|aborted due to timeout|timed out|socket hang up|econnreset/i.test(message);
}

function decodeCompressedBuffer(buf: Buffer, encodingHeader: string | undefined): string {
  const enc = (encodingHeader ?? '').toLowerCase();
  try {
    if (enc.includes('br')) {
      return zlib.brotliDecompressSync(buf).toString('utf8');
    }
    if (enc.includes('gzip')) {
      return zlib.gunzipSync(buf).toString('utf8');
    }
    if (enc.includes('deflate')) {
      return zlib.inflateSync(buf).toString('utf8');
    }
  } catch {
    // Fall back to raw UTF-8 if server sent unencoded payload with stale header
  }
  return buf.toString('utf8');
}

function ipv4NodeHttpFetch(
  targetUrl: string,
  headers: Record<string, string>,
  timeoutMs: number,
  redirectsRemaining = MAX_RSS_REDIRECTS,
): Promise<FetchTextResponse> {
  return new Promise((resolve, reject) => {
    let parsed: URL;
    try {
      parsed = new URL(targetUrl);
    } catch (err) {
      reject(err);
      return;
    }

    const isHttps = parsed.protocol === 'https:';
    const mod = isHttps ? https : http;
    const { httpsAgent, httpAgent } = getIpv4Agents();

    let settled = false;
    const req = mod.request(
      parsed,
      {
        method: 'GET',
        agent: isHttps ? httpsAgent : httpAgent,
        family: 4,
        headers: {
          ...headers,
          'Accept-Encoding': 'gzip, deflate, br',
          Connection: 'keep-alive',
        },
        timeout: timeoutMs,
      },
      (res) => {
        const status = res.statusCode ?? 0;
        const location = res.headers.location;
        if (
          location &&
          [301, 302, 303, 307, 308].includes(status) &&
          redirectsRemaining > 0
        ) {
          res.resume();
          try {
            const nextUrl = new URL(location, parsed).toString();
            if (!isSafePublicHttpUrl(nextUrl)) {
              settled = true;
              reject(new Error('Unsafe redirect URL'));
              return;
            }
            settled = true;
            ipv4NodeHttpFetch(nextUrl, headers, timeoutMs, redirectsRemaining - 1).then(
              resolve,
              reject,
            );
            return;
          } catch (err) {
            settled = true;
            reject(err);
            return;
          }
        }

        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer | string) => {
          chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
        });
        res.on('end', () => {
          if (settled) return;
          settled = true;
          const bodyText = decodeCompressedBuffer(
            Buffer.concat(chunks),
            Array.isArray(res.headers['content-encoding'])
              ? res.headers['content-encoding'][0]
              : res.headers['content-encoding'],
          );
          resolve({
            ok: status >= 200 && status < 300,
            status,
            text: async () => bodyText,
          });
        });
        res.on('error', (err) => {
          if (settled) return;
          settled = true;
          reject(err);
        });
      },
    );

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      const timeoutErr = new Error('The operation was aborted due to timeout');
      timeoutErr.name = 'TimeoutError';
      req.destroy(timeoutErr);
      reject(timeoutErr);
    }, timeoutMs);

    req.on('timeout', () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      const timeoutErr = new Error('The operation was aborted due to timeout');
      timeoutErr.name = 'TimeoutError';
      req.destroy(timeoutErr);
      reject(timeoutErr);
    });

    req.on('error', (err) => {
      clearTimeout(timer);
      if (settled) return;
      settled = true;
      reject(err);
    });

    req.on('close', () => {
      clearTimeout(timer);
    });

    req.end();
  });
}

function resolveLiveProxyUrl(explicitProxyUrl?: string): string | null {
  const raw = (
    explicitProxyUrl ??
    process.env.LIVE_FETCH_PROXY_URL ??
    process.env.SOCKS5_PROXY_URL ??
    process.env.HTTPS_PROXY ??
    process.env.https_proxy ??
    process.env.HTTP_PROXY ??
    process.env.http_proxy ??
    process.env.ALL_PROXY ??
    process.env.all_proxy ??
    ''
  ).trim();
  return raw || null;
}

function createBufferedSocketReader(socket: net.Socket | tls.TLSSocket) {
  let buffered: Buffer = Buffer.alloc(0);
  let ended = false;
  let streamErr: Error | null = null;
  let pendingResolve: (() => void) | null = null;
  let pendingReject: ((err: Error) => void) | null = null;

  const onData = (chunk: Buffer | string) => {
    const buf = typeof chunk === 'string' ? Buffer.from(chunk) : chunk;
    buffered = buffered.length === 0 ? buf : Buffer.concat([buffered, buf]);
    if (pendingResolve) {
      const wake = pendingResolve;
      pendingResolve = null;
      pendingReject = null;
      wake();
    }
  };

  const onError = (err: Error) => {
    streamErr = err;
    if (pendingReject) {
      const fail = pendingReject;
      pendingResolve = null;
      pendingReject = null;
      fail(err);
    }
  };

  const onClose = () => {
    ended = true;
    if (pendingReject) {
      const fail = pendingReject;
      pendingResolve = null;
      pendingReject = null;
      fail(streamErr ?? new Error('Proxy connection closed unexpectedly'));
    }
  };

  socket.on('data', onData);
  socket.on('error', onError);
  socket.on('close', onClose);
  socket.on('end', onClose);

  const waitForData = async (): Promise<void> => {
    if (streamErr) throw streamErr;
    if (ended) throw new Error('Proxy connection closed unexpectedly');
    await new Promise<void>((resolve, reject) => {
      pendingResolve = resolve;
      pendingReject = reject;
    });
  };

  return {
    async readExact(length: number): Promise<Buffer> {
      while (buffered.length < length) {
        await waitForData();
      }
      const out = buffered.subarray(0, length);
      buffered = buffered.subarray(length);
      return out;
    },
    async readUntil(delimiter: Buffer, maxBytes = 16384): Promise<Buffer> {
      while (true) {
        const idx = buffered.indexOf(delimiter);
        if (idx !== -1) {
          const endIdx = idx + delimiter.length;
          const out = buffered.subarray(0, endIdx);
          buffered = buffered.subarray(endIdx);
          return out;
        }
        if (buffered.length > maxBytes) {
          throw new Error('Proxy response headers exceeded limit');
        }
        await waitForData();
      }
    },
    detach(): void {
      socket.off('data', onData);
      socket.off('error', onError);
      socket.off('close', onClose);
      socket.off('end', onClose);
      if (buffered.length > 0) {
        socket.unshift(buffered);
        buffered = Buffer.alloc(0);
      }
    },
  };
}

async function openProxyTunnelSocket(
  proxyUrlStr: string,
  targetHost: string,
  targetPort: number,
  targetIsHttps: boolean,
  timeoutMs: number,
): Promise<net.Socket | tls.TLSSocket> {
  const parsedProxy = new URL(proxyUrlStr);
  const proto = parsedProxy.protocol.toLowerCase();
  if (
    proto !== 'socks5:' &&
    proto !== 'socks5h:' &&
    proto !== 'http:' &&
    proto !== 'https:'
  ) {
    throw new Error(`Unsupported proxy protocol: ${proto}`);
  }

  const rawHost = parsedProxy.hostname.replace(/^\[|\]$/g, '');
  const connectHost = rawHost.toLowerCase() === 'localhost' ? '127.0.0.1' : rawHost;
  const proxyPort =
    Number(parsedProxy.port) ||
    (proto.startsWith('socks5') ? 1080 : proto === 'https:' ? 443 : 80);
  const username = parsedProxy.username ? decodeURIComponent(parsedProxy.username) : '';
  const password = parsedProxy.password ? decodeURIComponent(parsedProxy.password) : '';
  const family = net.isIPv6(connectHost) ? 6 : 4;

  const baseSocket: net.Socket | tls.TLSSocket =
    proto === 'https:'
      ? tls.connect({
          host: connectHost,
          port: proxyPort,
          servername: rawHost,
          family,
        })
      : net.connect({
          host: connectHost,
          port: proxyPort,
          family,
        });

  let tunnelTimer: ReturnType<typeof setTimeout> | null = null;
  const withTimeout = <T>(promise: Promise<T>): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      tunnelTimer = setTimeout(() => {
        const timeoutErr = new Error('The operation was aborted due to timeout');
        timeoutErr.name = 'TimeoutError';
        baseSocket.destroy(timeoutErr);
        reject(timeoutErr);
      }, timeoutMs);
      promise.then(
        (val) => {
          if (tunnelTimer) clearTimeout(tunnelTimer);
          resolve(val);
        },
        (err) => {
          if (tunnelTimer) clearTimeout(tunnelTimer);
          baseSocket.destroy();
          reject(err);
        },
      );
    });

  return withTimeout(
    (async () => {
      await new Promise<void>((resolve, reject) => {
        const readyEvent = proto === 'https:' ? 'secureConnect' : 'connect';
        const onReady = () => {
          baseSocket.off('error', onErr);
          resolve();
        };
        const onErr = (err: Error) => {
          baseSocket.off(readyEvent, onReady);
          reject(err);
        };
        baseSocket.once(readyEvent, onReady);
        baseSocket.once('error', onErr);
      });

      const reader = createBufferedSocketReader(baseSocket);
      try {
        if (proto === 'socks5:' || proto === 'socks5h:') {
          const hasAuth = username.length > 0 || password.length > 0;
          baseSocket.write(
            Buffer.from(hasAuth ? [0x05, 0x02, 0x00, 0x02] : [0x05, 0x01, 0x00]),
          );
          const greetingResp = await reader.readExact(2);
          if (greetingResp[0] !== 0x05) {
            throw new Error(`Invalid SOCKS5 version: ${greetingResp[0]}`);
          }
          if (greetingResp[1] === 0x02) {
            if (!hasAuth) {
              throw new Error('SOCKS5 proxy requires authentication');
            }
            const userBuf = Buffer.from(username, 'utf8');
            const passBuf = Buffer.from(password, 'utf8');
            if (userBuf.length > 255 || passBuf.length > 255) {
              throw new Error('SOCKS5 credentials exceed 255 bytes');
            }
            baseSocket.write(
              Buffer.concat([
                Buffer.from([0x01, userBuf.length]),
                userBuf,
                Buffer.from([passBuf.length]),
                passBuf,
              ]),
            );
            const authResp = await reader.readExact(2);
            if (authResp[1] !== 0x00) {
              throw new Error(`SOCKS5 authentication failed (status=${authResp[1]})`);
            }
          } else if (greetingResp[1] !== 0x00) {
            throw new Error(`SOCKS5 unsupported auth method (${greetingResp[1]})`);
          }

          const hostBuf = Buffer.from(targetHost, 'utf8');
          if (hostBuf.length > 255) {
            throw new Error('Target hostname exceeds 255 bytes for SOCKS5');
          }
          const portBuf = Buffer.allocUnsafe(2);
          portBuf.writeUInt16BE(targetPort, 0);
          baseSocket.write(
            Buffer.concat([
              Buffer.from([0x05, 0x01, 0x00, 0x03, hostBuf.length]),
              hostBuf,
              portBuf,
            ]),
          );

          const connectResp = await reader.readExact(4);
          if (connectResp[0] !== 0x05) {
            throw new Error(`Invalid SOCKS5 connect response version: ${connectResp[0]}`);
          }
          if (connectResp[1] !== 0x00) {
            throw new Error(`SOCKS5 CONNECT failed (REP=${connectResp[1]})`);
          }
          const atyp = connectResp[3];
          if (atyp === 0x01) {
            await reader.readExact(6);
          } else if (atyp === 0x04) {
            await reader.readExact(18);
          } else if (atyp === 0x03) {
            const lenBuf = await reader.readExact(1);
            await reader.readExact(lenBuf[0] + 2);
          } else {
            throw new Error(`Unsupported SOCKS5 address type: ${atyp}`);
          }
        } else {
          const connectLines = [
            `CONNECT ${targetHost}:${targetPort} HTTP/1.1`,
            `Host: ${targetHost}:${targetPort}`,
            'Proxy-Connection: keep-alive',
          ];
          if (username || password) {
            const basic = Buffer.from(`${username}:${password}`, 'utf8').toString('base64');
            connectLines.push(`Proxy-Authorization: Basic ${basic}`);
          }
          baseSocket.write(`${connectLines.join('\r\n')}\r\n\r\n`);
          const headerBuf = await reader.readUntil(Buffer.from('\r\n\r\n'));
          const statusLine = headerBuf.toString('utf8').split('\r\n')[0] ?? '';
          if (!/^HTTP\/1\.[01]\s+200\b/i.test(statusLine)) {
            throw new Error(`Proxy CONNECT failed: ${statusLine || 'invalid response'}`);
          }
        }
      } finally {
        reader.detach();
      }

      if (!targetIsHttps) {
        return baseSocket;
      }

      const tlsSocket = tls.connect({
        socket: baseSocket,
        servername: targetHost,
      });
      await new Promise<void>((resolve, reject) => {
        const onSecure = () => {
          tlsSocket.off('error', onErr);
          resolve();
        };
        const onErr = (err: Error) => {
          tlsSocket.off('secureConnect', onSecure);
          reject(err);
        };
        tlsSocket.once('secureConnect', onSecure);
        tlsSocket.once('error', onErr);
      });
      return tlsSocket;
    })(),
  );
}

async function proxiedNodeHttpFetch(
  targetUrl: string,
  proxyUrlStr: string,
  headers: Record<string, string>,
  timeoutMs: number,
  redirectsRemaining = MAX_RSS_REDIRECTS,
): Promise<FetchTextResponse> {
  const parsed = new URL(targetUrl);
  const isHttps = parsed.protocol === 'https:';
  const targetHost = parsed.hostname.replace(/^\[|\]$/g, '');
  const targetPort = Number(parsed.port) || (isHttps ? 443 : 80);

  const tunneledSocket = await openProxyTunnelSocket(
    proxyUrlStr,
    targetHost,
    targetPort,
    isHttps,
    timeoutMs,
  );

  return new Promise((resolve, reject) => {
    let settled = false;
    const oneShotAgent = new http.Agent({ keepAlive: false });
    oneShotAgent.createConnection = () => tunneledSocket;

    const req = http.request(
      {
        host: targetHost,
        port: targetPort,
        path: `${parsed.pathname}${parsed.search}`,
        method: 'GET',
        agent: oneShotAgent,
        headers: {
          ...headers,
          Host: parsed.host,
          'Accept-Encoding': 'gzip, deflate, br',
          Connection: 'close',
        },
        timeout: timeoutMs,
      },
      (res) => {
        const status = res.statusCode ?? 0;
        const location = res.headers.location;
        if (
          location &&
          [301, 302, 303, 307, 308].includes(status) &&
          redirectsRemaining > 0
        ) {
          res.resume();
          tunneledSocket.destroy();
          try {
            const nextUrl = new URL(location, parsed).toString();
            if (!isSafePublicHttpUrl(nextUrl)) {
              settled = true;
              reject(new Error('Unsafe redirect URL'));
              return;
            }
            settled = true;
            proxiedNodeHttpFetch(
              nextUrl,
              proxyUrlStr,
              headers,
              timeoutMs,
              redirectsRemaining - 1,
            ).then(resolve, reject);
            return;
          } catch (err) {
            settled = true;
            reject(err);
            return;
          }
        }

        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer | string) => {
          chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
        });
        res.on('end', () => {
          if (settled) return;
          settled = true;
          const bodyText = decodeCompressedBuffer(
            Buffer.concat(chunks),
            Array.isArray(res.headers['content-encoding'])
              ? res.headers['content-encoding'][0]
              : res.headers['content-encoding'],
          );
          resolve({
            ok: status >= 200 && status < 300,
            status,
            text: async () => bodyText,
          });
        });
        res.on('error', (err) => {
          if (settled) return;
          settled = true;
          tunneledSocket.destroy();
          reject(err);
        });
      },
    );

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      const timeoutErr = new Error('The operation was aborted due to timeout');
      timeoutErr.name = 'TimeoutError';
      req.destroy(timeoutErr);
      tunneledSocket.destroy(timeoutErr);
      reject(timeoutErr);
    }, timeoutMs);

    req.on('timeout', () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      const timeoutErr = new Error('The operation was aborted due to timeout');
      timeoutErr.name = 'TimeoutError';
      req.destroy(timeoutErr);
      tunneledSocket.destroy(timeoutErr);
      reject(timeoutErr);
    });

    req.on('error', (err) => {
      clearTimeout(timer);
      if (settled) return;
      settled = true;
      tunneledSocket.destroy();
      reject(err);
    });

    req.on('close', () => {
      clearTimeout(timer);
    });

    req.end();
  });
}

async function fetchWithRetry(
  url: string,
  headers: Record<string, string>,
  options: {
    timeoutMs: number;
    maxRetries: number;
    retryDelayMs: number;
    proxyUrl?: string;
    fetchImpl?: typeof fetch;
  },
): Promise<FetchTextResponse> {
  let lastError: unknown;
  const activeProxyUrl = resolveLiveProxyUrl(options.proxyUrl);

  for (let attempt = 0; attempt <= options.maxRetries; attempt += 1) {
    if (attempt > 0) {
      await sleep(options.retryDelayMs * attempt);
    }

    try {
      let resp: FetchTextResponse;
      if (options.fetchImpl) {
        resp = await options.fetchImpl(url, {
          headers: {
            ...headers,
            Connection: 'keep-alive',
          },
          signal: AbortSignal.timeout(options.timeoutMs),
          cache: 'no-store',
        });
      } else if (activeProxyUrl) {
        resp = await proxiedNodeHttpFetch(
          url,
          activeProxyUrl,
          headers,
          options.timeoutMs,
        );
      } else if (attempt === 0) {
        resp = await globalThis.fetch(url, {
          headers: {
            ...headers,
            Connection: 'keep-alive',
          },
          signal: AbortSignal.timeout(options.timeoutMs),
          cache: 'no-store',
        });
      } else {
        // On retry in production, switch to dedicated IPv4 keep-alive agent to bypass
        // any undici/Happy Eyeballs IPv6 VPN socket issue.
        resp = await ipv4NodeHttpFetch(url, headers, options.timeoutMs);
      }

      if (!resp.ok && isRetryableStatus(resp.status) && attempt < options.maxRetries) {
        lastError = new Error(`HTTP ${resp.status}`);
        continue;
      }
      return resp;
    } catch (err) {
      lastError = err;
      if (attempt >= options.maxRetries || !isRetryableError(err)) {
        throw err;
      }
    }
  }

  throw lastError instanceof Error ? lastError : new Error('fetch failed');
}

async function tryTelegramMirrorAndRssFallbacks(
  descriptor: {
    username: string;
    channelId: string;
    handle: string;
    title?: string;
  },
  options: {
    timeoutMs: number;
    proxyUrl?: string;
    fetchImpl?: typeof fetch;
  },
): Promise<SingleChannelOutcome | null> {
  const encodedUser = encodeURIComponent(descriptor.username);
  const mirrorTimeout = Math.min(options.timeoutMs, 10_000);
  const activeProxyUrl = resolveLiveProxyUrl(options.proxyUrl);

  const requestText = async (
    targetUrl: string,
    headers: Record<string, string>,
  ): Promise<FetchTextResponse> => {
    if (options.fetchImpl) {
      return options.fetchImpl(targetUrl, {
        headers,
        signal: AbortSignal.timeout(mirrorTimeout),
        cache: 'no-store',
      });
    }
    if (activeProxyUrl) {
      try {
        return await proxiedNodeHttpFetch(
          targetUrl,
          activeProxyUrl,
          headers,
          mirrorTimeout,
        );
      } catch {
        // If local proxy tunnel is unreachable for bridges, fall back to direct IPv4
      }
    }
    return ipv4NodeHttpFetch(targetUrl, headers, mirrorTimeout);
  };

  // 1. Reader/mirror HTML proxy (r.jina.ai)
  try {
    const jinaResp = await requestText(
      `https://r.jina.ai/https://t.me/s/${encodedUser}`,
      {
        Accept: 'text/html, text/plain',
        'X-Return-Format': 'html',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
      },
    );
    if (jinaResp.ok) {
      const html = await jinaResp.text();
      if (html.includes('tgme_widget_message')) {
        const resolvedTitle = extractTelegramChannelTitle(html) ?? undefined;
        const parsed = parseTelegramPublicHtml(html, {
          channelId: descriptor.channelId,
          handle: `@${descriptor.username}`,
          title: resolvedTitle ?? descriptor.title,
        });
        if (parsed.length > 0) {
          return { items: parsed, resolvedTitle };
        }
      }
    }
  } catch {
    // Continue to RSS bridges
  }

  // 2. Public Telegram RSS bridges
  const rssBridgeUrls = [
    `https://tg.i-c-a.su/rss/${encodedUser}`,
    `https://rsshub.app/telegram/channel/${encodedUser}`,
  ];

  for (const bridgeUrl of rssBridgeUrls) {
    try {
      const rssResp = await requestText(bridgeUrl, {
        Accept: 'application/rss+xml, application/atom+xml, application/xml, text/xml',
        'User-Agent': 'CalmFeedReader/1.0',
      });
      if (!rssResp.ok) continue;
      const xml = await rssResp.text();
      const parsedRss = parseRssOrAtomXml(xml, {
        channelId: descriptor.channelId,
        handle: `@${descriptor.username}`,
        title: descriptor.title,
      });
      if (parsedRss.length === 0) continue;

      const normalizedTitle = parsedRss[0]?.channelTitle;
      const normalized: CleanPostItem[] = parsedRss.map((item) => ({
        ...item,
        id: `tg-${descriptor.username}-${item.postId}`,
        channelHandle: `@${descriptor.username}`,
        sourcePlatform: 'TG',
        clusterId: `cluster-${descriptor.username}-${item.postId}`,
        rubric: 'Подписки',
      }));
      return {
        items: normalized,
        resolvedTitle: normalizedTitle,
      };
    } catch {
      // Try next bridge
    }
  }

  return null;
}

function remapCachedItems(
  items: CleanPostItem[],
  ch: LiveChannelInput,
  resolvedTitle?: string,
): CleanPostItem[] {
  return items.map((item) => ({
    ...item,
    channelId: ch.channelId,
    channelTitle: resolvedTitle || item.channelTitle || ch.title || ch.handle,
  }));
}

async function fetchSingleChannelUncached(
  descriptor: {
    sourceKey: string;
    kind: 'rss' | 'tg';
    url: string;
    handle: string;
    username?: string;
    channelId: string;
    title?: string;
  },
  options: {
    timeoutMs: number;
    maxRetries: number;
    retryDelayMs: number;
    skipDirectTme?: boolean;
    proxyUrl?: string;
    fetchImpl?: typeof fetch;
  },
): Promise<SingleChannelOutcome> {
  if (descriptor.kind === 'rss') {
    try {
      const resp = await fetchWithRetry(
        descriptor.url,
        {
          Accept: 'application/rss+xml, application/atom+xml, application/xml, text/xml',
          'User-Agent': 'CalmFeedReader/1.0',
        },
        options,
      );
      if (!resp.ok) {
        return { items: [], error: `HTTP ${resp.status}` };
      }
      const xml = await resp.text();
      const parsed = parseRssOrAtomXml(xml, {
        channelId: descriptor.channelId,
        handle: descriptor.handle,
        title: descriptor.title,
      });
      return {
        items: parsed,
        resolvedTitle: parsed[0]?.channelTitle,
      };
    } catch (err) {
      return {
        items: [],
        error: formatLiveFetchError(err),
      };
    }
  }

  const username = descriptor.username ?? descriptor.handle.replace(/^@/, '');
  const skipDirectTme =
    options.skipDirectTme ??
    /^(1|true|yes)$/i.test(process.env.LIVE_FETCH_SKIP_DIRECT_TME ?? '');

  if (skipDirectTme) {
    const fallback = await tryTelegramMirrorAndRssFallbacks(
      {
        username,
        channelId: descriptor.channelId,
        handle: descriptor.handle,
        title: descriptor.title,
      },
      {
        timeoutMs: options.timeoutMs,
        proxyUrl: options.proxyUrl,
        fetchImpl: options.fetchImpl,
      },
    );
    return (
      fallback ?? {
        items: [],
        error: 'Не удалось загрузить канал через зеркала/RSS-бриджи',
      }
    );
  }

  try {
    const resp = await fetchWithRetry(
      descriptor.url,
      {
        Accept: 'text/html,application/xhtml+xml',
        'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
      },
      options,
    );
    if (!resp.ok) {
      return {
        items: [],
        error: `Telegram web preview HTTP ${resp.status}`,
      };
    }
    const html = await resp.text();
    const resolvedTitle = extractTelegramChannelTitle(html) ?? undefined;
    const parsed = parseTelegramPublicHtml(html, {
      channelId: descriptor.channelId,
      handle: `@${username}`,
      title: resolvedTitle ?? descriptor.title,
    });
    if (parsed.length === 0) {
      return {
        items: [],
        resolvedTitle,
        error: 'Публичная веб-лента канала пуста или скрыта (закрытый канал)',
      };
    }
    return {
      items: parsed,
      resolvedTitle,
    };
  } catch (err) {
    if (!options.fetchImpl && isRetryableError(err)) {
      const fallback = await tryTelegramMirrorAndRssFallbacks(
        {
          username,
          channelId: descriptor.channelId,
          handle: descriptor.handle,
          title: descriptor.title,
        },
        {
          timeoutMs: options.timeoutMs,
          proxyUrl: options.proxyUrl,
        },
      );
      if (fallback) {
        return fallback;
      }
    }
    return {
      items: [],
      error: formatLiveFetchError(err),
    };
  }
}

/**
 * Fetches public Telegram (`t.me/s/<username>`) and RSS channels with:
 * - Bounded concurrency pool (default 2) + slight pacing to avoid bursting VPN/TUN sockets
 * - IPv4-first DNS + IPv4 keep-alive agent retry fallback
 * - Singleflight in-flight deduplication and TTL + stale-while-error caching
 */
export async function fetchLiveChannelsBatch(
  channels: LiveChannelInput[],
  options: LiveChannelBatchOptions = {},
): Promise<LiveChannelBatchResult> {
  const maxConcurrency = clampConcurrency(options.maxConcurrency);
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  const retryDelayMs = options.retryDelayMs ?? (options.fetchImpl ? 10 : 350);
  const interRequestDelayMs = options.interRequestDelayMs ?? (options.fetchImpl ? 0 : 100);
  const forceRefresh = Boolean(options.forceRefresh);
  const bypassCache = Boolean(options.bypassCache);

  const allItems: CleanPostItem[] = [];
  const errors: Record<string, string> = {};
  const channelTitles: Record<string, string> = {};

  interface NetworkTask {
    ch: LiveChannelInput;
    sourceKey: string;
    kind: 'rss' | 'tg';
    url: string;
    handle: string;
    username?: string;
    staleEntry?: CachedChannelEntry;
  }

  const networkTasks: NetworkTask[] = [];
  const now = Date.now();

  for (const ch of channels) {
    const handle = (ch.handle ?? '').trim();
    if (!handle) continue;

    const isRssUrl = ch.platform === 'RSS' || /^https?:\/\//i.test(handle);
    let sourceKey: string;
    let url: string;
    let kind: 'rss' | 'tg';
    let username: string | undefined;

    if (isRssUrl) {
      if (!isSafePublicHttpUrl(handle)) {
        errors[handle] = 'Недопустимый URL RSS-ленты';
        continue;
      }
      kind = 'rss';
      url = handle;
      sourceKey = `rss:${handle}`;
    } else {
      username = handle.replace(/^@/, '');
      if (!USERNAME_RE.test(username)) {
        errors[handle] = 'Некорректное имя канала Telegram';
        continue;
      }
      kind = 'tg';
      url = `https://t.me/s/${encodeURIComponent(username)}`;
      sourceKey = `tg:${username.toLowerCase()}`;
    }

    const cached = bypassCache ? undefined : channelCache.get(sourceKey);
    const ageMs = cached ? now - cached.fetchedAtMs : Number.POSITIVE_INFINITY;
    const isWithinBurstWindow = cached && ageMs < MIN_BURST_INTERVAL_MS;
    const isFresh = cached && ageMs < FRESH_TTL_MS;

    if (cached && (isWithinBurstWindow || (!forceRefresh && isFresh))) {
      if (cached.resolvedTitle) {
        channelTitles[ch.channelId] = cached.resolvedTitle;
      }
      allItems.push(...remapCachedItems(cached.items, ch, cached.resolvedTitle));
      continue;
    }

    const staleEntry = cached && ageMs < STALE_TTL_MS ? cached : undefined;
    networkTasks.push({
      ch,
      sourceKey,
      kind,
      url,
      handle,
      username,
      staleEntry,
    });
  }

  let cursor = 0;
  let workerNetworkIndex = 0;

  async function runWorker(): Promise<void> {
    while (true) {
      const idx = cursor;
      cursor += 1;
      if (idx >= networkTasks.length) {
        return;
      }

      const task = networkTasks[idx];
      let flight = inFlightRequests.get(task.sourceKey);
      let isOwner = false;

      if (!flight) {
        const launchSeq = workerNetworkIndex;
        workerNetworkIndex += 1;
        isOwner = true;

        flight = (async () => {
          if (launchSeq > 0 && interRequestDelayMs > 0) {
            await sleep(Math.min(interRequestDelayMs, 250));
          }
          const result = await fetchSingleChannelUncached(
            {
              sourceKey: task.sourceKey,
              kind: task.kind,
              url: task.url,
              handle: task.handle,
              username: task.username,
              channelId: task.ch.channelId,
              title: task.ch.title,
            },
            {
              timeoutMs,
              maxRetries,
              retryDelayMs,
              skipDirectTme: options.skipDirectTme,
              proxyUrl: options.proxyUrl,
              fetchImpl: options.fetchImpl,
            },
          );
          if (!result.error && result.items.length > 0 && !bypassCache) {
            channelCache.set(task.sourceKey, {
              items: result.items,
              resolvedTitle: result.resolvedTitle,
              fetchedAtMs: Date.now(),
            });
          }
          return result;
        })().finally(() => {
          inFlightRequests.delete(task.sourceKey);
        });

        inFlightRequests.set(task.sourceKey, flight);
      }

      const outcome = await flight;

      if (!outcome.error) {
        if (outcome.resolvedTitle) {
          channelTitles[task.ch.channelId] = outcome.resolvedTitle;
        }
        allItems.push(
          ...remapCachedItems(outcome.items, task.ch, outcome.resolvedTitle),
        );
      } else if (task.staleEntry && task.staleEntry.items.length > 0) {
        if (task.staleEntry.resolvedTitle) {
          channelTitles[task.ch.channelId] = task.staleEntry.resolvedTitle;
        }
        allItems.push(
          ...remapCachedItems(
            task.staleEntry.items,
            task.ch,
            task.staleEntry.resolvedTitle,
          ),
        );
      } else {
        errors[task.handle] = outcome.error;
        if (isOwner && outcome.resolvedTitle) {
          channelTitles[task.ch.channelId] = outcome.resolvedTitle;
        }
      }
    }
  }

  const workerCount = Math.min(maxConcurrency, networkTasks.length);
  if (workerCount > 0) {
    await Promise.all(Array.from({ length: workerCount }, () => runWorker()));
  }

  allItems.sort((a, b) => (a.publishedAt < b.publishedAt ? 1 : -1));

  return {
    items: allItems,
    errors,
    channelTitles,
    fetchedAt: new Date().toISOString(),
  };
}
