import type {
  Element as HastElement,
  ElementContent,
  Properties as HastProperties,
  Root as HastRoot,
  Text as HastText,
} from 'hast';
import { common as lowlightCommonGrammars, createLowlight } from 'lowlight';
import {
  type BundledLanguage,
  type BundledTheme,
  bundledLanguages,
  bundledLanguagesAlias,
  bundledThemes,
  codeToHast,
} from 'shiki';

const lowlightInstance = createLowlight(lowlightCommonGrammars);

const SHIKI_LANGUAGE_ALIASES: Record<string, string> = {
  py: 'python',
  js: 'javascript',
  ts: 'typescript',
  sh: 'bash',
  shell: 'bash',
  yml: 'yaml',
  md: 'markdown',
  'c++': 'cpp',
  'c#': 'csharp',
  cs: 'csharp',
  golang: 'go',
  rs: 'rust',
  txt: 'text',
  plaintext: 'text',
  plain: 'text',
};

const LOWLIGHT_LANGUAGE_ALIASES: Record<string, string> = {
  py: 'python',
  js: 'javascript',
  ts: 'typescript',
  sh: 'bash',
  zsh: 'bash',
  yml: 'yaml',
  'c++': 'cpp',
  'c#': 'csharp',
  cs: 'csharp',
  golang: 'go',
  rs: 'rust',
  html: 'xml',
  svg: 'xml',
  text: 'plaintext',
  txt: 'plaintext',
  plain: 'plaintext',
};

export interface HighlightedCodeResult {
  hast: HastRoot;
  engine: 'shiki' | 'lowlight' | 'plain';
  resolvedLanguage: string;
}

export function resolveShikiLanguage(language?: string): BundledLanguage | 'text' {
  const clean = (language ?? '').trim().toLowerCase();
  if (!clean || clean === 'text' || clean === 'plaintext' || clean === 'plain' || clean === 'txt') {
    return 'text';
  }
  const mapped = SHIKI_LANGUAGE_ALIASES[clean] ?? clean;
  if (mapped in bundledLanguages || mapped in bundledLanguagesAlias) {
    return mapped as BundledLanguage;
  }
  return 'text';
}

export function resolveShikiTheme(preferred: string = 'light-plus'): BundledTheme {
  if (preferred in bundledThemes) {
    return preferred as BundledTheme;
  }
  if ('github-light' in bundledThemes) {
    return 'github-light' as BundledTheme;
  }
  return Object.keys(bundledThemes)[0] as BundledTheme;
}

function resolveLowlightLanguage(language?: string): string | null {
  const clean = (language ?? '').trim().toLowerCase();
  if (!clean) {
    return null;
  }
  const mapped = LOWLIGHT_LANGUAGE_ALIASES[clean] ?? clean;
  return lowlightInstance.registered(mapped) ? mapped : null;
}

function makeTextNode(value: string): HastText {
  return { type: 'text', value };
}

function makeElement(
  tagName: string,
  properties: HastProperties,
  children: ElementContent[],
): HastElement {
  return {
    type: 'element',
    tagName,
    properties,
    children,
  };
}

/**
 * Splits flat lowlight inline HAST nodes by `\n` without HTML regexes or innerHTML,
 * producing per-line `<span class="line" data-line="N">` wrappers matching Shiki's structure.
 */
function groupHastInlineNodesIntoLines(inlineNodes: ElementContent[]): HastElement[] {
  const lines: ElementContent[][] = [[]];

  const appendTextSplittingNewlines = (
    textValue: string,
    wrapSpanProperties?: HastProperties,
  ) => {
    const parts = textValue.split('\n');
    for (let i = 0; i < parts.length; i += 1) {
      if (i > 0) {
        lines.push([]);
      }
      const chunk = parts[i];
      if (chunk.length > 0) {
        const textNode = makeTextNode(chunk);
        lines[lines.length - 1].push(
          wrapSpanProperties
            ? makeElement('span', { ...wrapSpanProperties }, [textNode])
            : textNode,
        );
      }
    }
  };

  const walk = (
    node: ElementContent,
    inheritedSpanProps?: HastProperties,
  ) => {
    if (node.type === 'text') {
      appendTextSplittingNewlines(node.value, inheritedSpanProps);
      return;
    }
    if (node.type === 'element') {
      const nextProps: HastProperties | undefined =
        node.tagName === 'span'
          ? { ...(node.properties ?? {}) }
          : inheritedSpanProps;
      for (const child of node.children) {
        walk(child, nextProps);
      }
    }
  };

  for (const node of inlineNodes) {
    walk(node);
  }

  return lines.map((lineChildren, idx) =>
    makeElement(
      'span',
      {
        className: ['line'],
        'data-line': idx + 1,
      },
      lineChildren.length > 0 ? lineChildren : [makeTextNode(' ')],
    ),
  );
}

/**
 * Synchronous HAST highlighter powered by `lowlight` (offline PWA fallback and instant initial SSR AST).
 */
export function highlightCodeWithLowlightSync(
  code: string,
  language?: string,
): HighlightedCodeResult {
  const normalizedCode = code.replace(/\r\n?/g, '\n');
  const resolvedLang = resolveLowlightLanguage(language);

  try {
    const rawRoot: HastRoot = resolvedLang
      ? lowlightInstance.highlight(resolvedLang, normalizedCode)
      : normalizedCode.trim().length > 0
        ? lowlightInstance.highlightAuto(normalizedCode)
        : { type: 'root', children: [] };

    const inlineChildren = rawRoot.children.filter(
      (child): child is ElementContent => child.type === 'text' || child.type === 'element',
    );
    const lineElements = groupHastInlineNodesIntoLines(inlineChildren);

    const hast: HastRoot = {
      type: 'root',
      children: [
        makeElement(
          'pre',
          {
            className: ['shiki-fallback', 'lowlight-pre'],
            tabIndex: 0,
            'data-language': resolvedLang ?? (language || 'plaintext'),
          },
          [
            makeElement(
              'code',
              {
                className: ['code-block-code'],
              },
              lineElements,
            ),
          ],
        ),
      ],
    };

    return {
      hast,
      engine: 'lowlight',
      resolvedLanguage: resolvedLang ?? (language || 'plaintext'),
    };
  } catch {
    return buildPlainCodeHast(normalizedCode, language);
  }
}

export function buildPlainCodeHast(code: string, language?: string): HighlightedCodeResult {
  const normalized = code.replace(/\r\n?/g, '\n');
  const lineElements = normalized.split('\n').map((line, idx) =>
    makeElement(
      'span',
      {
        className: ['line'],
        'data-line': idx + 1,
      },
      [makeTextNode(line.length > 0 ? line : ' ')],
    ),
  );

  return {
    hast: {
      type: 'root',
      children: [
        makeElement(
          'pre',
          {
            className: ['plain-code-pre'],
            tabIndex: 0,
            'data-language': language || 'plaintext',
          },
          [makeElement('code', { className: ['code-block-code'] }, lineElements)],
        ),
      ],
    },
    engine: 'plain',
    resolvedLanguage: language || 'plaintext',
  };
}

/**
 * Primary async HAST highlighter powered by Shiki (`codeToHast`) with VS Code theme and line numbers,
 * gracefully falling back to `lowlight` in offline/restricted environments or unknown grammars.
 */
export async function highlightCodeWithShikiAsync(
  code: string,
  language?: string,
  preferredTheme: string = 'light-plus',
): Promise<HighlightedCodeResult> {
  const normalizedCode = code.replace(/\r\n?/g, '\n');
  const resolvedLang = resolveShikiLanguage(language);
  const resolvedTheme = resolveShikiTheme(preferredTheme);

  try {
    const hast = await codeToHast(normalizedCode, {
      lang: resolvedLang,
      theme: resolvedTheme,
      transformers: [
        {
          name: 'telegram-aggregator-line-numbers',
          pre(node) {
            node.properties['data-language'] = language || resolvedLang;
          },
          line(node, line) {
            node.properties['data-line'] = line;
          },
        },
      ],
    });
    return {
      hast,
      engine: 'shiki',
      resolvedLanguage: resolvedLang,
    };
  } catch {
    return highlightCodeWithLowlightSync(normalizedCode, language);
  }
}

/**
 * Parse inline CSS string from a HAST element `style` property into a plain React style map.
 */
export function parseHastInlineStyle(styleValue: unknown): Record<string, string> | undefined {
  if (typeof styleValue !== 'string' || !styleValue.trim()) {
    return undefined;
  }
  const styleObj: Record<string, string> = {};
  const declarations = styleValue.split(';');
  for (const decl of declarations) {
    const colonIdx = decl.indexOf(':');
    if (colonIdx <= 0) continue;
    const rawProp = decl.slice(0, colonIdx).trim();
    const rawVal = decl.slice(colonIdx + 1).trim();
    if (!rawProp || !rawVal) continue;
    if (rawProp.startsWith('--')) {
      styleObj[rawProp] = rawVal;
      continue;
    }
    const camelProp = rawProp.replace(/-([a-z])/g, (_, ch: string) => ch.toUpperCase());
    styleObj[camelProp] = rawVal;
  }
  return Object.keys(styleObj).length > 0 ? styleObj : undefined;
}
