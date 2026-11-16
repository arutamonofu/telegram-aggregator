/**
 * URL safety verification, Punycode -> Unicode decoding (RFC 3492),
 * IDN homograph warnings, and allowlist enforcement for Telegram links.
 */

const PUNYCODE_BASE = 36;
const PUNYCODE_TMIN = 1;
const PUNYCODE_TMAX = 26;
const PUNYCODE_SKEW = 38;
const PUNYCODE_DAMP = 700;
const PUNYCODE_INITIAL_BIAS = 72;
const PUNYCODE_INITIAL_N = 128;
const PUNYCODE_DELIMITER = '-';

export const DEFAULT_ALLOWED_LINK_DOMAINS: readonly string[] = [
  't.me',
  'telegram.me',
  'telegram.org',
  'core.telegram.org',
  'telegra.ph',
  'fragment.com',
];

export interface InspectedLink {
  rawUrl: string;
  normalizedUrl: string;
  protocol: string;
  asciiHost: string;
  decodedHost: string;
  decodedFullUrl: string;
  isAllowedProtocol: boolean;
  isAllowlistedDomain: boolean;
  requiresInterception: boolean;
  isPunycode: boolean;
  hasMixedScripts: boolean;
  warningReason?: string;
}

function adaptBias(delta: number, numPoints: number, firstTime: boolean): number {
  let d = firstTime ? Math.floor(delta / PUNYCODE_DAMP) : delta >> 1;
  d += Math.floor(d / numPoints);
  let k = 0;
  const limit = Math.floor(((PUNYCODE_BASE - PUNYCODE_TMIN) * PUNYCODE_TMAX) / 2);
  while (d > limit) {
    d = Math.floor(d / (PUNYCODE_BASE - PUNYCODE_TMIN));
    k += PUNYCODE_BASE;
  }
  return k + Math.floor(((PUNYCODE_BASE - PUNYCODE_TMIN + 1) * d) / (d + PUNYCODE_SKEW));
}

function decodePunycodeDigit(code: number): number {
  if (code >= 48 && code <= 57) return code - 48 + 26; // 0-9 -> 26..35
  if (code >= 65 && code <= 90) return code - 65; // A-Z -> 0..25
  if (code >= 97 && code <= 122) return code - 97; // a-z -> 0..25
  return PUNYCODE_BASE;
}

/**
 * Decode a single Punycode label (RFC 3492). Supports labels with or without `xn--`.
 */
export function decodePunycodeLabel(label: string): string {
  const lower = label.toLowerCase();
  if (!lower.startsWith('xn--')) {
    return label;
  }
  const input = lower.slice(4);
  if (input.length === 0) {
    return label;
  }

  const output: number[] = [];
  const basicEnd = input.lastIndexOf(PUNYCODE_DELIMITER);
  let ic = 0;

  if (basicEnd > 0) {
    for (let j = 0; j < basicEnd; j += 1) {
      const cp = input.charCodeAt(j);
      if (cp >= 0x80) {
        return label;
      }
      output.push(cp);
    }
    ic = basicEnd + 1;
  } else if (basicEnd === 0) {
    ic = 1;
  }

  let n = PUNYCODE_INITIAL_N;
  let i = 0;
  let bias = PUNYCODE_INITIAL_BIAS;

  while (ic < input.length) {
    const oldi = i;
    let w = 1;
    let k = PUNYCODE_BASE;

    while (true) {
      if (ic >= input.length) {
        return label;
      }
      const digit = decodePunycodeDigit(input.charCodeAt(ic));
      ic += 1;
      if (digit >= PUNYCODE_BASE) {
        return label;
      }
      i += digit * w;
      const t =
        k <= bias
          ? PUNYCODE_TMIN
          : k >= bias + PUNYCODE_TMAX
            ? PUNYCODE_TMAX
            : k - bias;
      if (digit < t) {
        break;
      }
      w *= PUNYCODE_BASE - t;
      k += PUNYCODE_BASE;
    }

    const outLenPlusOne = output.length + 1;
    bias = adaptBias(i - oldi, outLenPlusOne, oldi === 0);
    n += Math.floor(i / outLenPlusOne);
    i %= outLenPlusOne;
    output.splice(i, 0, n);
    i += 1;
  }

  try {
    return String.fromCodePoint(...output);
  } catch {
    return label;
  }
}

/**
 * Decode a hostname containing Punycode labels (`xn--...`) into Unicode.
 */
export function decodeDomainPunycode(hostname: string): string {
  if (!hostname) return '';
  return hostname
    .split('.')
    .map((label) => decodePunycodeLabel(label))
    .join('.');
}

function detectMixedLatinAndCyrillic(host: string): boolean {
  const hasLatin = /[a-z]/i.test(host);
  const hasCyrillic = /[\u0400-\u04FF]/.test(host);
  return hasLatin && hasCyrillic;
}

function isHostAllowlisted(asciiHost: string, allowlist: readonly string[]): boolean {
  const host = asciiHost.toLowerCase().replace(/\.$/, '');
  if (!host) return false;
  return allowlist.some((allowed) => {
    const cleanAllowed = allowed.toLowerCase();
    return host === cleanAllowed || host.endsWith(`.${cleanAllowed}`);
  });
}

/**
 * Inspect an arbitrary link (including Telegram bare URLs without scheme or Punycode IDN hosts).
 */
export function inspectLinkSafety(
  rawUrl: string,
  allowlist: readonly string[] = DEFAULT_ALLOWED_LINK_DOMAINS,
): InspectedLink {
  const trimmed = rawUrl.trim();
  const candidate =
    /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed) || trimmed.startsWith('//')
      ? trimmed.startsWith('//')
        ? `https:${trimmed}`
        : trimmed
      : `https://${trimmed}`;

  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    return {
      rawUrl,
      normalizedUrl: rawUrl,
      protocol: '',
      asciiHost: '',
      decodedHost: rawUrl,
      decodedFullUrl: rawUrl,
      isAllowedProtocol: false,
      isAllowlistedDomain: false,
      requiresInterception: true,
      isPunycode: false,
      hasMixedScripts: false,
      warningReason: 'Некорректный формат URL',
    };
  }

  const protocol = parsed.protocol.toLowerCase();
  const isAllowedProtocol =
    protocol === 'https:' || protocol === 'http:' || protocol === 'mailto:' || protocol === 'tel:' || protocol === 'tg:';

  if (!isAllowedProtocol) {
    return {
      rawUrl,
      normalizedUrl: parsed.href,
      protocol,
      asciiHost: parsed.hostname,
      decodedHost: parsed.hostname,
      decodedFullUrl: parsed.href,
      isAllowedProtocol: false,
      isAllowlistedDomain: false,
      requiresInterception: true,
      isPunycode: false,
      hasMixedScripts: false,
      warningReason: `Заблокирована небезопасная схема URL: ${protocol}`,
    };
  }

  if (protocol === 'mailto:' || protocol === 'tel:' || protocol === 'tg:') {
    return {
      rawUrl,
      normalizedUrl: parsed.href,
      protocol,
      asciiHost: '',
      decodedHost: '',
      decodedFullUrl: parsed.href,
      isAllowedProtocol: true,
      isAllowlistedDomain: protocol === 'tg:',
      requiresInterception: protocol !== 'tg:',
      isPunycode: false,
      hasMixedScripts: false,
    };
  }

  const asciiHost = parsed.hostname.toLowerCase();
  const decodedHost = decodeDomainPunycode(asciiHost);
  const isPunycode = asciiHost !== decodedHost || asciiHost.includes('xn--');
  const hasMixedScripts = detectMixedLatinAndCyrillic(decodedHost);
  const isAllowlistedDomain = isHostAllowlisted(asciiHost, allowlist);

  let decodedPathAndQuery = '';
  try {
    decodedPathAndQuery = `${decodeURI(parsed.pathname)}${parsed.search}${parsed.hash}`;
  } catch {
    decodedPathAndQuery = `${parsed.pathname}${parsed.search}${parsed.hash}`;
  }
  const portSuffix = parsed.port ? `:${parsed.port}` : '';
  const decodedFullUrl = `${parsed.protocol}//${decodedHost}${portSuffix}${decodedPathAndQuery}`;

  let warningReason: string | undefined;
  if (hasMixedScripts) {
    warningReason = 'Обнаружено смешение алфавитов (IDN homograph — возможный фишинг)';
  } else if (isPunycode) {
    warningReason = `Интернационализированный домен (Punycode): ${asciiHost} → ${decodedHost}`;
  } else if (!isAllowlistedDomain) {
    warningReason = 'Внешний домен не входит в локальный белый список';
  }

  return {
    rawUrl,
    normalizedUrl: parsed.href,
    protocol,
    asciiHost,
    decodedHost,
    decodedFullUrl,
    isAllowedProtocol: true,
    isAllowlistedDomain,
    requiresInterception: !isAllowlistedDomain || isPunycode || hasMixedScripts,
    isPunycode,
    hasMixedScripts,
    warningReason,
  };
}
