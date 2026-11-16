import { afterEach, describe, expect, it } from 'bun:test';
import fs from 'node:fs';
import path from 'node:path';
import { escapers, serialiseWith, serialisers } from '@telegraf/entity';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { common, createLowlight } from 'lowlight';
import React from 'react';
import { codeToHast } from 'shiki';
import { AstRenderer } from '@/components/ast/AstRenderer';
import {
  DATE_FLAG_DAY_OF_WEEK,
  DATE_FLAG_LONG_DATE,
  DATE_FLAG_RELATIVE,
  DATE_FLAG_SHORT_TIME,
  isFormattedDateLengthValid,
} from '@/lib/ast/formatted_date';
import { splitAstLeavesWithRagCitations } from '@/lib/ast/interval_split';
import {
  computeTwemojiCodepoints,
  parseTelegramPostAstAsync,
  parseTelegramPostAstSync,
  rtrimUtf16,
  serialiseBotApiMessageWithTelegraf,
} from '@/lib/ast/parser';
import {
  highlightCodeWithLowlightSync,
  highlightCodeWithShikiAsync,
} from '@/lib/ast/shiki_highlighter';
import { decodeDomainPunycode, inspectLinkSafety } from '@/lib/ast/url_safety';

afterEach(() => {
  cleanup();
});

function walkSourceFiles(dir: string): string[] {
  const results: string[] = [];
  if (!fs.existsSync(dir)) return results;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === '__tests__' || entry.name === 'node_modules') continue;
      results.push(...walkSourceFiles(full));
    } else if (entry.name.endsWith('.ts') || entry.name.endsWith('.tsx')) {
      results.push(full);
    }
  }
  return results;
}

describe('External library API verification (@telegraf/entity, shiki, lowlight)', () => {
  it('verifies @telegraf/entity serialiseWith and escapers.HTML signatures', () => {
    const serialise = serialiseWith(serialisers.HTML, escapers.HTML);
    expect(typeof serialise).toBe('function');
    expect(escapers.HTML('<script>&"\'')).toBe('&lt;script&gt;&amp;"\'');
    expect(
      serialiseBotApiMessageWithTelegraf('<raw>', []),
    ).toBe('&lt;raw&gt;');
    expect(
      serialiseBotApiMessageWithTelegraf('Hello world', [
        { type: 'bold', offset: 0, length: 5 },
      ]),
    ).toBe('<b>Hello</b> world');
  });

  it('verifies shiki codeToHast and lowlight createLowlight(common) HAST output', async () => {
    const shikiHast = await codeToHast('const x = 1;', {
      lang: 'typescript',
      theme: 'light-plus',
    });
    expect(shikiHast.type).toBe('root');
    expect(shikiHast.children.length).toBeGreaterThan(0);

    const low = createLowlight(common);
    expect(low.registered('typescript')).toBe(true);
    const lowHast = low.highlight('typescript', 'const x = 1;');
    expect(lowHast.type).toBe('root');
    expect(lowHast.children.length).toBeGreaterThan(0);
  });
});

describe('SPEC 7.5.5 Reference Test: Interval Split Pass on surrogate emoji UTF-16 post', () => {
  it('splits TextLeaf nodes at [13, 33] across bold [3, 25) producing exactly two <mark data-cite="c1"> without emoji shift', () => {
    const text = '🔥 Сбой BGP: потеряно 14% трафика ЦОД';
    expect(text.length).toBe(37);
    expect(text.slice(0, 3)).toBe('🔥 ');
    expect(text.slice(3, 25)).toBe('Сбой BGP: потеряно 14%');
    expect(text.slice(13, 33)).toBe('потеряно 14% трафика');
    expect(text.slice(25, 37)).toBe(' трафика ЦОД');

    const primaryAst = parseTelegramPostAstSync(text, [
      { type: 'messageEntityBold', offset: 3, length: 22 },
    ]);

    expect(primaryAst.trimmedText).toBe(text);
    expect(primaryAst.nodes.length).toBe(3);

    const [firstLeaf, boldNode, tailLeaf] = primaryAst.nodes;
    expect(firstLeaf.kind).toBe('text');
    if (firstLeaf.kind === 'text') {
      expect(firstLeaf.startUtf16).toBe(0);
      expect(firstLeaf.endUtf16).toBe(3);
      expect(firstLeaf.text).toBe('🔥 ');
    }

    expect(boldNode.kind).toBe('element');
    if (boldNode.kind === 'element') {
      expect(boldNode.tag).toBe('bold');
      expect(boldNode.children.length).toBe(1);
      const innerLeaf = boldNode.children[0];
      expect(innerLeaf.kind).toBe('text');
      if (innerLeaf.kind === 'text') {
        expect(innerLeaf.startUtf16).toBe(3);
        expect(innerLeaf.endUtf16).toBe(25);
        expect(innerLeaf.text).toBe('Сбой BGP: потеряно 14%');
      }
    }

    expect(tailLeaf.kind).toBe('text');
    if (tailLeaf.kind === 'text') {
      expect(tailLeaf.startUtf16).toBe(25);
      expect(tailLeaf.endUtf16).toBe(37);
      expect(tailLeaf.text).toBe(' трафика ЦОД');
    }

    const splitNodes = splitAstLeavesWithRagCitations(primaryAst.nodes, [
      { claim_id: 'c1', utf16_range: [13, 33] },
    ]);

    const { container } = render(React.createElement(AstRenderer, { nodes: splitNodes }));
    expect(container.textContent).toBe(text);

    const strongEls = container.querySelectorAll('strong');
    expect(strongEls.length).toBe(1);
    expect(strongEls[0].textContent).toBe('Сбой BGP: потеряно 14%');

    const markEls = container.querySelectorAll('mark[data-cite="c1"]');
    expect(markEls.length).toBe(2);

    // Segment #1: inside <strong>
    expect(markEls[0].textContent).toBe('потеряно 14%');
    expect(markEls[0].closest('strong')).toBe(strongEls[0]);

    // Segment #2: outside <strong>
    expect(markEls[1].textContent).toBe(' трафика');
    expect(markEls[1].closest('strong')).toBeNull();

    // Unified hover synchronization across both split segments
    expect(markEls[0].getAttribute('data-cite-active')).toBe('false');
    expect(markEls[1].getAttribute('data-cite-active')).toBe('false');

    fireEvent.mouseEnter(markEls[0]);
    expect(markEls[0].getAttribute('data-cite-active')).toBe('true');
    expect(markEls[1].getAttribute('data-cite-active')).toBe('true');

    fireEvent.mouseLeave(markEls[0]);
    expect(markEls[0].getAttribute('data-cite-active')).toBe('false');
    expect(markEls[1].getAttribute('data-cite-active')).toBe('false');

    // Unified focus synchronization via keyboard navigation
    fireEvent.focus(markEls[1]);
    expect(markEls[0].getAttribute('data-cite-active')).toBe('true');
    expect(markEls[1].getAttribute('data-cite-active')).toBe('true');

    fireEvent.blur(markEls[1]);
    expect(markEls[0].getAttribute('data-cite-active')).toBe('false');
    expect(markEls[1].getAttribute('data-cite-active')).toBe('false');
  });
});

describe('2026 Entities, Code Highlighting, Link Interception & DOM Safety', () => {
  it('applies UTF-16 rtrim and renders code blocks via Shiki and lowlight without innerHTML', async () => {
    const raw = 'Пример кода:\nconst ans = 42;   \n\n';
    const { trimmedText, trimmedTrailingCount } = rtrimUtf16(raw);
    expect(trimmedTrailingCount).toBe(5);
    expect(trimmedText).toBe('Пример кода:\nconst ans = 42;');

    const offset = 'Пример кода:\n'.length;
    const len = 'const ans = 42;'.length;

    const syncAst = parseTelegramPostAstSync(raw, [
      { type: 'pre', offset, length: len, language: 'ts' },
      // Nested entity inside `pre` must be rejected cleanly
      { type: 'bold', offset, length: 5 },
    ]);
    expect(syncAst.warnings.some((w) => w.includes('Nested entities'))).toBe(true);
    const syncBlock = syncAst.nodes.find((n) => n.kind === 'code_block');
    expect(syncBlock?.kind).toBe('code_block');
    if (syncBlock?.kind === 'code_block') {
      expect(syncBlock.highlighterEngine).toBe('lowlight');
    }

    const asyncAst = await parseTelegramPostAstAsync(raw, [
      { type: 'pre', offset, length: len, language: 'ts' },
    ]);
    const shikiBlock = asyncAst.nodes.find((n) => n.kind === 'code_block');
    expect(shikiBlock?.kind).toBe('code_block');
    if (shikiBlock?.kind === 'code_block') {
      expect(shikiBlock.highlighterEngine).toBe('shiki');
    }

    const low = highlightCodeWithLowlightSync('line1\nline2', 'python');
    expect(low.engine).toBe('lowlight');

    const shiki = await highlightCodeWithShikiAsync('const y = 2;', 'ts');
    expect(shiki.engine).toBe('shiki');

    const { container } = render(React.createElement(AstRenderer, { nodes: asyncAst.nodes }));
    const blockEl = container.querySelector('[data-testid="ast-code-block"]');
    expect(blockEl?.getAttribute('data-highlighter')).toBe('shiki');
    expect(container.querySelector('[data-line="1"]')).not.toBeNull();
  });

  it('supports collapsed expandable blockquote and clickable spoiler', () => {
    const text = 'Цитата с длинным объяснением и спойлер внутри.';
    const ast = parseTelegramPostAstSync(text, [
      { type: 'expandable_blockquote', offset: 0, length: 29 },
      { type: 'spoiler', offset: 32, length: 7 },
    ]);

    const { container } = render(React.createElement(AstRenderer, { nodes: ast.nodes }));
    const quote = container.querySelector('blockquote');
    expect(quote?.getAttribute('data-collapsed')).toBe('true');

    const expandBtn = screen.getByRole('button', { name: 'Развернуть цитату' });
    fireEvent.click(expandBtn);
    expect(quote?.getAttribute('data-collapsed')).toBe('false');

    const spoiler = screen.getByRole('button', { name: 'Показать скрытый текст' });
    expect(spoiler.getAttribute('data-revealed')).toBe('false');
    fireEvent.click(spoiler);
    expect(spoiler.getAttribute('data-revealed')).toBe('true');
  });

  it('validates and renders messageEntityFormattedDate with bitmask flags and timestamp copy menu', () => {
    expect(isFormattedDateLengthValid(31, DATE_FLAG_RELATIVE)).toBe(true);
    expect(isFormattedDateLengthValid(32, DATE_FLAG_RELATIVE)).toBe(false);
    expect(isFormattedDateLengthValid(127, 0)).toBe(true);
    expect(isFormattedDateLengthValid(128, 0)).toBe(false);

    const text = 'Событие состоится 2026-11-16 в 12:00 UTC';
    const ast = parseTelegramPostAstSync(text, [
      {
        type: 'messageEntityFormattedDate',
        offset: 18,
        length: 10,
        unix_time: 1794830400,
        date_flags: DATE_FLAG_DAY_OF_WEEK | DATE_FLAG_SHORT_TIME | DATE_FLAG_LONG_DATE,
      },
      // Inner bold inside formatted_date should be ignored
      { type: 'bold', offset: 18, length: 4 },
    ]);
    expect(ast.warnings.some((w) => w.includes('formatted_date'))).toBe(true);

    const { container } = render(React.createElement(AstRenderer, { nodes: ast.nodes }));
    const timeEl = container.querySelector('time');
    expect(timeEl).not.toBeNull();
    expect(timeEl?.getAttribute('data-unix-time')).toBe('1794830400');

    if (timeEl) {
      fireEvent.click(timeEl);
    }
    expect(screen.getByText(/Копировать Unix: 1794830400/)).toBeTruthy();
  });

  it('renders custom_emoji with document_id + Twemoji fallback, plus diff_replace and rich widget', () => {
    const codepoints = computeTwemojiCodepoints('🔥');
    expect(codepoints).toBe('1f525');

    const text = '🔥 Новая версия [Widget]';
    const ast = parseTelegramPostAstSync(text, [
      { type: 'custom_emoji', offset: 0, length: 2, document_id: '5368324170671202286' },
      { type: 'diff_replace', offset: 3, length: 12, old_text: 'Старая версия' },
      {
        type: 'widget',
        offset: 16,
        length: 8,
        widget_payload: { kind: 'metric_badge', title: 'P99', subtitle: '42 ms' },
      },
    ]);

    const { container } = render(React.createElement(AstRenderer, { nodes: ast.nodes }));
    const customEmojiEl = container.querySelector('[data-custom-emoji-id="5368324170671202286"]');
    expect(customEmojiEl).not.toBeNull();
    expect(customEmojiEl?.getAttribute('data-twemoji-url')).toContain('1f525.svg');
    expect(container.querySelector('[data-diff="replace"]')?.textContent).toContain('Старая версия');
    expect(container.querySelector('[data-widget-kind="metric_badge"]')?.textContent).toContain('42 ms');
  });

  it('decodes Punycode -> Unicode and intercepts external links via modal while allowing t.me', () => {
    expect(decodeDomainPunycode('xn--e1afmkfd.xn--p1ai')).toBe('пример.рф');

    const phish = inspectLinkSafety('https://xn--80ak6aa92e.com/login');
    expect(phish.isPunycode).toBe(true);
    expect(phish.decodedHost).toBe('\u0430\u0440\u0440\u04cf\u0435.com');
    expect(phish.hasMixedScripts).toBe(true);
    expect(phish.requiresInterception).toBe(true);

    const mixedLabel = inspectLinkSafety('https://xn--pple-43d.com/verify');
    expect(mixedLabel.decodedHost).toBe('\u0430pple.com');
    expect(mixedLabel.hasMixedScripts).toBe(true);

    const safeTg = inspectLinkSafety('https://t.me/durov/123');
    expect(safeTg.isAllowlistedDomain).toBe(true);
    expect(safeTg.requiresInterception).toBe(false);

    const text = 'Проверьте внешний сайт и канал';
    const ast = parseTelegramPostAstSync(text, [
      {
        type: 'messageEntityTextUrl',
        offset: 10,
        length: 12,
        url: 'https://xn--80ak6aa92e.com/login',
      },
    ]);

    render(React.createElement(AstRenderer, { nodes: ast.nodes }));
    const link = screen.getByRole('link', { name: 'внешний сайт' });
    fireEvent.click(link);

    const modal = screen.getByTestId('external-link-modal');
    expect(modal).toBeTruthy();
    expect(screen.getByTestId('decoded-unicode-host').textContent).toBe('\u0430\u0440\u0440\u04cf\u0435.com');
    expect(screen.getByTestId('ascii-punycode-host').textContent).toBe('xn--80ak6aa92e.com');
    expect(screen.getByTestId('link-warning-banner').textContent).toContain('смешение алфавитов');
  });

  it('guarantees zero dangerouslySetInnerHTML across all non-test web-client src files', () => {
    const srcRoot = path.resolve(import.meta.dir, '../../..');
    const files = walkSourceFiles(srcRoot);
    expect(files.length).toBeGreaterThan(0);

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      expect(content.includes('dangerouslySetInnerHTML')).toBe(false);
    }
  });

  it('renders SPEC 9.2 gotd Layer 225 post with emoji, Go code block, and exact UTF-16 <mark> citation without DOM shift', () => {
    const prose =
      '🔥 Разбор бага библиотеки gotd на слое 225 протокола MTProto.\n' +
      'При чтении длинной истории канала в gotd на 225 слое возникает ошибка ' +
      'updates.channelDifferenceTooLong, если локальный pts устарел и нужен откат на messages.getHistory.\n' +
      'Пример обработчика на языке программирования Go:\n';
    const goCode =
      'case *tg.UpdatesChannelDifferenceTooLong:\n' +
      '    return client.MessagesGetHistory(ctx, 225)';
    const text = prose + goCode;

    // Exact UTF-16 range produced by Python Smith-Waterman aligner in calibration_qa.py ([62, 230])
    const citeRange: [number, number] = [62, 230];
    const expectedSentence =
      'При чтении длинной истории канала в gotd на 225 слое возникает ошибка ' +
      'updates.channelDifferenceTooLong, если локальный pts устарел и нужен откат на messages.getHistory.';
    expect(text.slice(citeRange[0], citeRange[1])).toBe(expectedSentence);

    const ast = parseTelegramPostAstSync(text, [
      {
        type: 'pre',
        offset: prose.length,
        length: goCode.length,
        language: 'go',
      },
    ]);

    const splitNodes = splitAstLeavesWithRagCitations(ast.nodes, [
      { claim_id: 'c1', utf16_range: citeRange },
    ]);

    const { container } = render(React.createElement(AstRenderer, { nodes: splitNodes }));
    const mark = container.querySelector('mark[data-cite="c1"]');
    expect(mark).not.toBeNull();
    expect(mark?.textContent).toBe(expectedSentence);

    const preBlock = container.querySelector('pre code') ?? container.querySelector('pre');
    expect(preBlock).not.toBeNull();
    expect(preBlock?.textContent).toContain('UpdatesChannelDifferenceTooLong');
    expect(preBlock?.textContent).toContain('MessagesGetHistory(ctx, 225)');

    expect(container.textContent?.startsWith(prose)).toBe(true);
    expect(container.querySelectorAll('mark[data-cite="c1"]').length).toBe(1);
  });
});
