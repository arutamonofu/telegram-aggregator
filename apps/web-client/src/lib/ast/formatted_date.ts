/**
 * Formatter and validator for Telegram `messageEntityFormattedDate` / `date_time`
 * (MTProto Layer 225+, Bot API 9.5+).
 *
 * Limits per SPEC 7.5.5:
 * - No nested entities allowed inside formatted_date
 * - UTF-16 length <= 31 when date_flags != 0 (or format string set), <= 127 otherwise
 * - Flag bitmask:
 *   bit 0 (1):  relative (`r`)
 *   bit 1 (2):  day_of_week (`w`)
 *   bit 2 (4):  short_time (`t`)
 *   bit 3 (8):  long_time (`T`)
 *   bit 4 (16): short_date (`d`)
 *   bit 5 (32): long_date (`D`)
 */

export const DATE_FLAG_RELATIVE = 1 << 0;
export const DATE_FLAG_DAY_OF_WEEK = 1 << 1;
export const DATE_FLAG_SHORT_TIME = 1 << 2;
export const DATE_FLAG_LONG_TIME = 1 << 3;
export const DATE_FLAG_SHORT_DATE = 1 << 4;
export const DATE_FLAG_LONG_DATE = 1 << 5;

export const MAX_FORMATTED_DATE_WITH_FLAGS_UTF16 = 31;
export const MAX_FORMATTED_DATE_NO_FLAGS_UTF16 = 127;

export interface DecodedDateFlags {
  relative: boolean;
  dayOfWeek: boolean;
  shortTime: boolean;
  longTime: boolean;
  shortDate: boolean;
  longDate: boolean;
  hasAnyFlag: boolean;
}

export function decodeDateFlags(bitmask?: number, formatString?: string): DecodedDateFlags {
  const mask = typeof bitmask === 'number' && Number.isFinite(bitmask) ? bitmask : 0;
  const fmt = formatString ?? '';

  const relative = (mask & DATE_FLAG_RELATIVE) !== 0 || fmt.includes('r');
  const dayOfWeek = (mask & DATE_FLAG_DAY_OF_WEEK) !== 0 || fmt.includes('w');
  const shortTime = (mask & DATE_FLAG_SHORT_TIME) !== 0 || fmt.includes('t');
  const longTime = (mask & DATE_FLAG_LONG_TIME) !== 0 || fmt.includes('T');
  const shortDate = (mask & DATE_FLAG_SHORT_DATE) !== 0 || fmt.includes('d');
  const longDate = (mask & DATE_FLAG_LONG_DATE) !== 0 || fmt.includes('D');

  return {
    relative,
    dayOfWeek,
    shortTime,
    longTime,
    shortDate,
    longDate,
    hasAnyFlag: relative || dayOfWeek || shortTime || longTime || shortDate || longDate,
  };
}

export function isFormattedDateLengthValid(
  utf16Length: number,
  dateFlags?: number,
  dateTimeFormat?: string,
): boolean {
  const flags = decodeDateFlags(dateFlags, dateTimeFormat);
  const maxLen = flags.hasAnyFlag
    ? MAX_FORMATTED_DATE_WITH_FLAGS_UTF16
    : MAX_FORMATTED_DATE_NO_FLAGS_UTF16;
  return utf16Length > 0 && utf16Length <= maxLen;
}

function formatRelativeDate(date: Date, nowMs: number, locale: string): string {
  const diffSeconds = Math.round((date.getTime() - nowMs) / 1000);
  const absSeconds = Math.abs(diffSeconds);
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });

  if (absSeconds < 60) {
    return rtf.format(diffSeconds, 'second');
  }
  const diffMinutes = Math.round(diffSeconds / 60);
  if (Math.abs(diffMinutes) < 60) {
    return rtf.format(diffMinutes, 'minute');
  }
  const diffHours = Math.round(diffSeconds / 3600);
  if (Math.abs(diffHours) < 24) {
    return rtf.format(diffHours, 'hour');
  }
  const diffDays = Math.round(diffSeconds / 86400);
  if (Math.abs(diffDays) < 30) {
    return rtf.format(diffDays, 'day');
  }
  const diffMonths = Math.round(diffDays / 30);
  if (Math.abs(diffMonths) < 12) {
    return rtf.format(diffMonths, 'month');
  }
  const diffYears = Math.round(diffDays / 365);
  return rtf.format(diffYears, 'year');
}

/**
 * Formats a Telegram `messageEntityFormattedDate` respecting the exact flag order
 * specified in SPEC 7.5.5:
 * `relative`, or combination of `day_of_week`, `short/long_time`, `short/long_date`.
 */
export function formatTelegramDynamicDate(params: {
  unixTime?: number;
  dateFlags?: number;
  dateTimeFormat?: string;
  fallbackText: string;
  locale?: string;
  timeZone?: string;
  nowMs?: number;
}): {
  displayText: string;
  isoString: string;
  fullLocalString: string;
} {
  const {
    unixTime,
    dateFlags,
    dateTimeFormat,
    fallbackText,
    locale = 'ru-RU',
    timeZone,
    nowMs = Date.now(),
  } = params;

  if (typeof unixTime !== 'number' || !Number.isFinite(unixTime) || unixTime <= 0) {
    return {
      displayText: fallbackText,
      isoString: '',
      fullLocalString: fallbackText,
    };
  }

  const date = new Date(unixTime * 1000);
  const isoString = date.toISOString();
  const fullLocalString = new Intl.DateTimeFormat(locale, {
    dateStyle: 'long',
    timeStyle: 'medium',
    ...(timeZone ? { timeZone } : {}),
  }).format(date);

  const flags = decodeDateFlags(dateFlags, dateTimeFormat);

  if (!flags.hasAnyFlag) {
    return {
      displayText: fallbackText || fullLocalString,
      isoString,
      fullLocalString,
    };
  }

  if (flags.relative) {
    return {
      displayText: formatRelativeDate(date, nowMs, locale),
      isoString,
      fullLocalString,
    };
  }

  const parts: string[] = [];

  if (flags.dayOfWeek) {
    parts.push(
      new Intl.DateTimeFormat(locale, {
        weekday: 'short',
        ...(timeZone ? { timeZone } : {}),
      }).format(date),
    );
  }

  if (flags.longTime) {
    parts.push(
      new Intl.DateTimeFormat(locale, {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        ...(timeZone ? { timeZone } : {}),
      }).format(date),
    );
  } else if (flags.shortTime) {
    parts.push(
      new Intl.DateTimeFormat(locale, {
        hour: '2-digit',
        minute: '2-digit',
        ...(timeZone ? { timeZone } : {}),
      }).format(date),
    );
  }

  if (flags.longDate) {
    parts.push(
      new Intl.DateTimeFormat(locale, {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
        ...(timeZone ? { timeZone } : {}),
      }).format(date),
    );
  } else if (flags.shortDate) {
    parts.push(
      new Intl.DateTimeFormat(locale, {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        ...(timeZone ? { timeZone } : {}),
      }).format(date),
    );
  }

  return {
    displayText: parts.length > 0 ? parts.join(' · ') : fullLocalString,
    isoString,
    fullLocalString,
  };
}
