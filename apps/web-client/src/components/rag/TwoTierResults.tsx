'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { SEED_CLEAN_POSTS } from '../../lib/api/seed_data';
import { parseTelegramPostAstSync } from '../../lib/ast/parser';
import { evaluateRagEntitlements } from '../../lib/entitlements';
import { formatRuDateTime } from '../../lib/format_date';
import { runLocalFts5Search } from '../../lib/local-vault/fts5_and_crypto';
import { streamRagQuery } from '../../lib/sse/rag_stream';
import type {
  CleanPostItem,
  DensityMode,
  RagDoneMetadata,
  RagSourcePostItem,
  SearchEngineMode,
  StanceMatrixRow,
  VerifiedCitation,
  VerifiedClaim,
  VerifiedDirectAnswer,
  ZeroTrustUserSession,
} from '../../lib/types/feed';
import { AstRenderer } from '../ast/AstRenderer';
import { IconArrowUp, IconSearch, IconSparkles, IconTrash } from '../ui/Icons';
import { CitationHoverCard } from './CitationHoverCard';

export interface ExternalSearchRequest {
  id: number;
  text: string;
  mode?: SearchEngineMode;
}

export interface TwoTierResultsProps {
  session: ZeroTrustUserSession;
  density: DensityMode;
  availablePosts?: CleanPostItem[];
  externalQuery?: ExternalSearchRequest | null;
  onExternalQueryConsumed?: () => void;
  onConsumeRagQuota?: () => void;
  onAcceptPrivacyConsent?: () => void;
  layoutVariant?: 'full' | 'pane';
  focusSignal?: number;
}

type SsePhase = 'idle' | 'searching' | 'drafting' | 'verifying' | 'done' | 'refusal';
type MobileMatrixTab = 'pro' | 'con' | 'nuances';

const TIME_WINDOWS: readonly { days: number; label: string }[] = [
  { days: 1, label: 'За 24 часа' },
  { days: 7, label: 'За 7 дней' },
  { days: 30, label: 'За 30 дней' },
];

const FALLBACK_EXAMPLE_QUERIES: readonly string[] = [
  'вакансии',
  'итоги недели',
  'обновление',
];

function deriveExampleQueries(posts: CleanPostItem[]): string[] {
  const candidates: string[] = [];
  const seen = new Set<string>();
  for (const post of posts.slice(0, 8)) {
    const words = post.text
      .replace(/https?:\/\/\S+/g, ' ')
      .split(/[\s,.;:!?()[\]{}«»"']+/)
      .map((w) => w.trim())
      .filter((w) => w.length >= 4 && w.length <= 18 && !w.startsWith('@') && !w.startsWith('#'));
    for (const word of words) {
      const lower = word.toLowerCase();
      if (!seen.has(lower)) {
        seen.add(lower);
        candidates.push(word);
        break;
      }
    }
    if (candidates.length >= 3) break;
  }
  return candidates.length > 0 ? candidates : [...FALLBACK_EXAMPLE_QUERIES];
}

export function TwoTierResults({
  session,
  density,
  availablePosts = SEED_CLEAN_POSTS,
  externalQuery = null,
  onExternalQueryConsumed,
  onConsumeRagQuota,
  onAcceptPrivacyConsent,
  layoutVariant = 'full',
  focusSignal,
}: TwoTierResultsProps) {
  const entitlements = useMemo(() => evaluateRagEntitlements(session), [session]);

  const [query, setQuery] = useState<string>('');
  const [engineMode, setEngineMode] = useState<SearchEngineMode>(
    entitlements.canRunCloudRag ? 'rag_synthesis' : 'local_fts5'
  );
  const [timeWindowDays, setTimeWindowDays] = useState<number>(7);

  const [phase, setPhase] = useState<SsePhase>('idle');
  const [sources, setSources] = useState<RagSourcePostItem[]>([]);
  const [directAnswer, setDirectAnswer] = useState<VerifiedDirectAnswer | null>(null);
  const [claims, setClaims] = useState<VerifiedClaim[]>([]);
  const [stanceRows, setStanceRows] = useState<StanceMatrixRow[]>([]);
  const [doneMeta, setDoneMeta] = useState<RagDoneMetadata | null>(null);
  const [activeSourceId, setActiveSourceId] = useState<string>('');
  const [sourcesCollapsed, setSourcesCollapsed] = useState<boolean>(false);

  const exampleQueries = useMemo(
    () => deriveExampleQueries(availablePosts),
    [availablePosts],
  );
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [fallbackNote, setFallbackNote] = useState<string | null>(null);
  const [needsConsentPrompt, setNeedsConsentPrompt] = useState<boolean>(false);
  const [mobileMatrixTab, setMobileMatrixTab] = useState<MobileMatrixTab>('pro');
  const [ftsElapsedMs, setFtsElapsedMs] = useState<number | null>(null);
  const [isMobile, setIsMobile] = useState<boolean>(false);

  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const sourceRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    if (focusSignal && focusSignal > 0) {
      searchInputRef.current?.focus();
    }
  }, [focusSignal]);

  useEffect(() => {
    const syncViewport = () => {
      setIsMobile(window.innerWidth < 768);
    };
    syncViewport();
    window.addEventListener('resize', syncViewport);
    return () => window.removeEventListener('resize', syncViewport);
  }, []);

  const { sourcesByKey, sourceIndexByKey } = useMemo(() => {
    const byKey: Record<string, RagSourcePostItem> = {};
    const indexByKey: Record<string, number> = {};
    sources.forEach((src, idx) => {
      const displayIdx = idx + 1;
      byKey[src.post.id] = src;
      indexByKey[src.post.id] = displayIdx;
      const pidKey = String(src.post.postId ?? '').trim();
      if (pidKey && byKey[pidKey] === undefined) {
        byKey[pidKey] = src;
        indexByKey[pidKey] = displayIdx;
      }
    });
    return { sourcesByKey: byKey, sourceIndexByKey: indexByKey };
  }, [sources]);

  const jumpToSource = (postId: string) => {
    const targetId = sourcesByKey[postId]?.post.id ?? postId;
    if (sourcesCollapsed) {
      setSourcesCollapsed(false);
    }
    setActiveSourceId(targetId);
    const scrollTarget = () => {
      const el = sourceRefs.current[targetId];
      if (el && typeof el.scrollIntoView === 'function') {
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    };
    if (sourcesCollapsed) {
      setTimeout(scrollTarget, 0);
    } else {
      scrollTarget();
    }
  };

  const dedupeCitationsByPost = (citations: VerifiedCitation[]): VerifiedCitation[] => {
    const byPost = new Map<string, VerifiedCitation>();
    for (const cit of citations) {
      const key = String(cit.post_id);
      const existing = byPost.get(key);
      if (!existing) {
        byPost.set(key, cit);
      } else if (cit.quote && !existing.quote.includes(cit.quote)) {
        byPost.set(key, {
          ...existing,
          quote: existing.quote ? `${existing.quote} … ${cit.quote}` : cit.quote,
        });
      }
    }
    return Array.from(byPost.values());
  };

  useEffect(() => {
    const handleDigitJump = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }
      if (/^[1-9]$/.test(event.key)) {
        const idx = Number(event.key) - 1;
        const src = sources[idx];
        if (src) {
          event.preventDefault();
          jumpToSource(src.post.id);
        }
      }
    };
    window.addEventListener('keydown', handleDigitJump);
    return () => window.removeEventListener('keydown', handleDigitJump);
  }, [sources]);

  const runSearch = async (
    overrideMode?: SearchEngineMode,
    overrideDays?: number,
    overrideQuery?: string,
    bypassConsentCheck = false,
  ) => {
    const effectiveQuery = (overrideQuery ?? query).trim();
    if (!effectiveQuery) {
      searchInputRef.current?.focus();
      setErrorMsg('Введите тему или ключевое слово для поиска.');
      return;
    }

    const effectiveDays = overrideDays ?? timeWindowDays;
    const requestedMode = overrideMode ?? engineMode;
    const effectiveMode: SearchEngineMode =
      requestedMode === 'rag_synthesis' && !entitlements.canRunCloudRag
        ? 'local_fts5'
        : requestedMode;

    if (effectiveMode !== engineMode) {
      setEngineMode(effectiveMode);
    }
    if (effectiveDays !== timeWindowDays) {
      setTimeWindowDays(effectiveDays);
    }
    if (overrideQuery !== undefined) {
      setQuery(overrideQuery);
    }

    setErrorMsg(null);
    setFallbackNote(null);
    setNeedsConsentPrompt(false);

    if (effectiveMode === 'local_fts5') {
      const localRes = runLocalFts5Search(effectiveQuery, availablePosts);
      setSources(localRes.matches);
      setDirectAnswer(null);
      setClaims([]);
      setStanceRows([]);
      setDoneMeta(null);
      setFtsElapsedMs(localRes.elapsedMs);
      setPhase(localRes.matches.length > 0 ? 'done' : 'refusal');
      if (localRes.matches[0]) {
        setActiveSourceId(localRes.matches[0].post.id);
      }
      return;
    }

    if (!bypassConsentCheck && !session.crossBorderConsentAccepted && !session.localInferenceOnlyRf) {
      setNeedsConsentPrompt(true);
      return;
    }

    setSourcesCollapsed(false);
    setPhase('searching');
    setFtsElapsedMs(null);
    setSources([]);
    setDirectAnswer(null);
    setClaims([]);
    setStanceRows([]);
    setDoneMeta(null);

    const isComparative = /\b(vs|или|против|сравни|плюсы|минусы)\b/i.test(effectiveQuery);
    const ragMode = isComparative ? 'stance_matrix' : 'qa';

    const executeSsePass = async (
      preloadedPostsPayload?: Array<Record<string, unknown>>,
      byNumericId?: Map<string, CleanPostItem>,
    ) => {
      const collectedClaims: VerifiedClaim[] = [];
      const collectedStanceRows: StanceMatrixRow[] = [];
      let collectedDirectAnswer: VerifiedDirectAnswer | null = null;
      let collectedSources: RagSourcePostItem[] = [];
      let collectedMeta: RagDoneMetadata | null = null;
      let streamError: string | null = null;

      await streamRagQuery(
        {
          user_id: session.userId,
          query: effectiveQuery,
          mode: ragMode,
          channel_ids: session.subscribedChannels.map((c) => c.channelId),
          time_window_days: effectiveDays,
          plan_tier: session.planTier,
          ...(preloadedPostsPayload ? { preloaded_posts: preloadedPostsPayload } : {}),
        },
        {
          onStage: (stage) => {
            setPhase(stage.stage.includes('verify') ? 'verifying' : 'drafting');
          },
          onDirectAnswer: (answer) => {
            collectedDirectAnswer = answer;
            setDirectAnswer(answer);
          },
          onSourcePosts: (items) => {
            const enriched = items.map((item) => {
              const original = byNumericId?.get(String(item.post.postId));
              if (!original) return item;
              return {
                ...item,
                post: {
                  ...original,
                  text: item.post.text || original.text,
                },
              };
            });
            collectedSources = enriched;
            if (enriched.length > 0) {
              setSources(enriched);
              setActiveSourceId(enriched[0].post.id);
            }
          },
          onClaim: (claim) => {
            setPhase('verifying');
            collectedClaims.push(claim);
            setClaims((prev) => [...prev, claim]);
          },
          onStanceRow: (row) => {
            collectedStanceRows.push(row);
            setStanceRows((prev) => [...prev, row]);
          },
          onDone: (meta) => {
            collectedMeta = meta;
            if (!collectedDirectAnswer && meta.direct_answer) {
              collectedDirectAnswer = meta.direct_answer;
              setDirectAnswer(meta.direct_answer);
            }
            setDoneMeta(meta);
          },
          onError: (msg) => {
            streamError = msg;
          },
        }
      );

      return {
        directAnswer: collectedDirectAnswer as VerifiedDirectAnswer | null,
        claims: collectedClaims,
        stanceRows: collectedStanceRows,
        sources: collectedSources,
        meta: collectedMeta as RagDoneMetadata | null,
        streamError,
      };
    };

    try {
      let passRes = await executeSsePass();
      let hasResults =
        passRes.claims.length > 0 ||
        passRes.stanceRows.length > 0 ||
        passRes.sources.length > 0;

      // If Qdrant returned no verified data (e.g., live-synced channels not in Qdrant),
      // retry seamlessly against the client's loaded posts via `preloaded_posts`.
      if (!hasResults && !passRes.streamError && availablePosts.length > 0) {
        const cutoffMs = Date.now() - effectiveDays * 86_400_000;
        const windowPosts = availablePosts.filter((p) => {
          const ts = Date.parse(p.publishedAt);
          return Number.isNaN(ts) || ts >= cutoffMs;
        });
        const candidatePool = windowPosts.length > 0 ? windowPosts : availablePosts;
        const localRanked = runLocalFts5Search(effectiveQuery, candidatePool);
        const rankedPosts = localRanked.matches.map((m) => m.post);
        const combinedPosts = (
          rankedPosts.length > 0 ? rankedPosts : candidatePool
        ).slice(0, 24);

        const byNumericId = new Map<string, CleanPostItem>();
        const usedNumericIds = new Set<number>();
        const preloadedPayload = combinedPosts.map((p, idx) => {
          const rawNum = Number(p.postId);
          const msgNum = Number.isFinite(rawNum) && rawNum > 0 ? Math.floor(rawNum) : 100000 + idx;
          let uniqueId = msgNum;
          if (usedNumericIds.has(uniqueId)) {
            uniqueId = 1_000_000 + idx;
            while (usedNumericIds.has(uniqueId)) {
              uniqueId += 1;
            }
          }
          usedNumericIds.add(uniqueId);
          byNumericId.set(String(uniqueId), p);
          return {
            post_id: uniqueId,
            channel_id: p.channelId,
            channel_username: p.channelHandle.replace(/^@/, ''),
            channel_title: p.channelTitle,
            msg_id: msgNum,
            published_at: p.publishedAt,
            text: p.text,
            neutral_headline: p.neutralHeadline,
          };
        });

        setDirectAnswer(null);
        setClaims([]);
        setStanceRows([]);
        setSources([]);
        passRes = await executeSsePass(preloadedPayload, byNumericId);
        hasResults =
          passRes.claims.length > 0 ||
          passRes.stanceRows.length > 0 ||
          passRes.sources.length > 0;
      }

      if (passRes.streamError && !hasResults) {
        setErrorMsg(passRes.streamError);
        setPhase('refusal');
        return;
      }

      if (hasResults) {
        setPhase('done');
        if (passRes.claims.length > 0 || passRes.stanceRows.length > 0) {
          onConsumeRagQuota?.();
        }
        return;
      }

      // Final graceful fallback: show matching posts by text if any exist locally
      const localFallback = runLocalFts5Search(effectiveQuery, availablePosts);
      if (localFallback.matches.length > 0) {
        setSources(localFallback.matches);
        setActiveSourceId(localFallback.matches[0].post.id);
        setFallbackNote(
          'Точный ИИ-ответ не прошёл проверку цитат, поэтому показаны подходящие посты по тексту.',
        );
        setPhase('done');
      } else {
        setPhase('refusal');
      }
    } catch {
      const localRes = runLocalFts5Search(effectiveQuery, availablePosts);
      if (localRes.matches.length > 0) {
        setSources(localRes.matches);
        setActiveSourceId(localRes.matches[0].post.id);
        setFallbackNote(
          'Сервер ИИ-ответов сейчас недоступен — показаны подходящие посты из локального поиска.',
        );
        setPhase('done');
      } else {
        setPhase('refusal');
      }
    }
  };

  const flattenedCounts = useMemo(() => {
    let pro = 0;
    let con = 0;
    let nuances = 0;
    for (const row of stanceRows) {
      pro += row.pro_points.length;
      con += row.con_points.length;
      nuances += row.nuances?.length ?? 0;
    }
    return { pro, con, nuances };
  }, [stanceRows]);

  const isStreaming = phase === 'searching' || phase === 'drafting' || phase === 'verifying';
  const lastExternalIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (!externalQuery || externalQuery.id === lastExternalIdRef.current) {
      return;
    }
    lastExternalIdRef.current = externalQuery.id;
    onExternalQueryConsumed?.();
    void runSearch(externalQuery.mode ?? 'rag_synthesis', undefined, externalQuery.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalQuery]);

  const renderDirectAnswerCard = (answer: VerifiedDirectAnswer, compactFont = false) => {
    const uniqueCites = dedupeCitationsByPost(answer.citations);
    const isIndirect = answer.status === 'indirect';
    return (
      <div
        className={`direct-answer-card ${isIndirect ? 'direct-answer-indirect' : 'direct-answer-direct'}`}
        data-testid="direct-answer-card"
      >
        <div className="row-between" style={{ gap: '0.4rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
          <span className={`badge ${isIndirect ? 'badge-unsupported' : 'badge-entailed'}`}>
            {isIndirect ? 'Прямого подтверждения в каналах нет' : 'Ответ по источникам'}
          </span>
          {doneMeta?.verified_claims_count !== undefined ? (
            <span className="typo-meta tabular-nums" style={{ fontSize: '11.5px' }}>
              Фактов: {doneMeta.verified_claims_count}
            </span>
          ) : null}
        </div>
        <p className="typo-body" style={compactFont ? { fontSize: '14.5px' } : undefined}>
          {answer.text}
        </p>
        {uniqueCites.length > 0 ? (
          <div className="row" style={{ gap: '0.35rem', flexWrap: 'wrap', marginTop: '0.35rem' }}>
            {uniqueCites.map((cit, idx) => (
              <CitationHoverCard
                key={`direct-cit-${cit.post_id}-${idx}`}
                index={sourceIndexByKey[cit.post_id] ?? idx + 1}
                citation={{
                  postId: cit.post_id,
                  channelHandle: cit.channel_handle ?? cit.channel_id,
                  quotePreview: cit.quote,
                  utf16Start: cit.start_utf16,
                  utf16End: cit.end_utf16,
                }}
                source={sourcesByKey[cit.post_id]}
                verificationStatus="entailed"
                onOpenSource={jumpToSource}
              />
            ))}
          </div>
        ) : null}
      </div>
    );
  };

  const renderClaimCard = (claim: VerifiedClaim, compactFont = false) => {
    const uniqueCites = dedupeCitationsByPost(claim.citations);
    return (
      <div key={claim.claim_id} className="claim-item">
        <p className="typo-body" style={compactFont ? { fontSize: '14.5px' } : undefined}>
          {claim.statement}
        </p>
        <div
          className="row"
          style={{ gap: '0.35rem', flexWrap: 'wrap', marginTop: '0.3rem' }}
        >
          {uniqueCites.map((cit, idx) => (
            <CitationHoverCard
              key={`${claim.claim_id}-cit-${cit.post_id}-${idx}`}
              index={sourceIndexByKey[cit.post_id] ?? idx + 1}
              citation={{
                postId: cit.post_id,
                channelHandle: cit.channel_handle ?? cit.channel_id,
                quotePreview: cit.quote,
                utf16Start: cit.start_utf16,
                utf16End: cit.end_utf16,
              }}
              source={sourcesByKey[cit.post_id]}
              verificationStatus={claim.verification_status ?? 'entailed'}
              onOpenSource={jumpToSource}
            />
          ))}
        </div>
      </div>
    );
  };

  const normalizeCompareText = (text: string) =>
    text.replace(/\[[^\]]+\]/g, '').replace(/[\s.,!?;:]+/g, ' ').trim().toLowerCase();

  const shouldShowClaimsList =
    claims.length > 0 &&
    (!directAnswer ||
      claims.length > 1 ||
      directAnswer.status === 'indirect' ||
      normalizeCompareText(claims[0].statement) !== normalizeCompareText(directAnswer.text));

  if (layoutVariant === 'pane') {
    const resetPane = () => {
      setPhase('idle');
      setDirectAnswer(null);
      setClaims([]);
      setSources([]);
      setStanceRows([]);
      setDoneMeta(null);
      setErrorMsg(null);
      setFallbackNote(null);
      setFtsElapsedMs(null);
      setSourcesCollapsed(false);
      setQuery('');
    };

    const renderSourceList = (items: RagSourcePostItem[]) =>
      items.map((src, index) => {
        const parsed = parseTelegramPostAstSync(src.post.text, src.post.entities);
        return (
          <article
            key={src.post.id}
            ref={(el) => {
              sourceRefs.current[src.post.id] = el;
            }}
            className={`rag-source-card ${
              activeSourceId === src.post.id ? 'rag-source-active' : ''
            }`}
            style={{ padding: '12px 14px' }}
          >
            <div
              className="row-between"
              style={{ gap: '0.35rem', marginBottom: '0.3rem', flexWrap: 'wrap' }}
            >
              <div className="row" style={{ gap: '0.35rem', minWidth: 0 }}>
                <kbd className="kbd-hint">{index + 1}</kbd>
                <strong
                  className="typo-meta truncate"
                  style={{ color: 'var(--text-primary)' }}
                >
                  {src.post.channelTitle}
                </strong>
                <span className="typo-meta truncate">{src.post.channelHandle}</span>
              </div>
              <span className="typo-meta tabular-nums" style={{ fontSize: '11px' }}>
                {formatRuDateTime(src.post.publishedAt)}
              </span>
            </div>

            <AstRenderer
              nodes={parsed.nodes}
              richContainers={parsed.richContainers}
              density="compact"
              cssHighlightSpans={src.rag_highlights}
            />
          </article>
        );
      });

    return (
      <section
        aria-label="Поиск и анализ по источникам"
        data-testid="search-pane"
        className="search-pane"
      >
        <div className="search-pane-header">
          <div className="row-between" style={{ gap: '0.4rem' }}>
            <div className="segmented-tabs" role="group" aria-label="Режим поиска">
              <button
                type="button"
                className={engineMode === 'rag_synthesis' ? 'active' : ''}
                disabled={!entitlements.canRunCloudRag}
                onClick={() => setEngineMode('rag_synthesis')}
                data-tooltip="Ответ с проверкой по цитатам · остаток запросов на месяц"
              >
                <span className="row" style={{ gap: '0.35rem', alignItems: 'center' }}>
                  <span>ИИ-ответ</span>
                  <span
                    className="tabular-nums"
                    data-testid="rag-quota-badge"
                    style={{
                      fontSize: '10.5px',
                      fontWeight: 600,
                      padding: '1px 6px',
                      borderRadius: '9999px',
                      backgroundColor:
                        engineMode === 'rag_synthesis'
                          ? 'var(--accent-primary-bg)'
                          : 'var(--bg-surface)',
                      color:
                        engineMode === 'rag_synthesis'
                          ? 'var(--accent-primary)'
                          : 'var(--text-secondary)',
                    }}
                  >
                    {entitlements.quotaRemaining}/{entitlements.quotaTotal}
                  </span>
                </span>
              </button>
              <button
                type="button"
                className={engineMode === 'local_fts5' ? 'active' : ''}
                onClick={() => setEngineMode('local_fts5')}
                data-tooltip="Быстрый поиск по тексту (бесплатно)"
              >
                По словам
              </button>
            </div>

            <div className="row" style={{ gap: '0.25rem' }}>
              {phase !== 'idle' ? (
                <button
                  type="button"
                  className="icon-btn"
                  onClick={resetPane}
                  data-testid="clear-search-pane-btn"
                  data-tooltip="Сбросить"
                  aria-label="Сбросить"
                >
                  <IconTrash size={14} />
                </button>
              ) : null}
            </div>
          </div>

          {engineMode === 'local_fts5' ? (
            <form
              className="row"
              style={{ gap: '0.4rem' }}
              onSubmit={(e) => {
                e.preventDefault();
                void runSearch('local_fts5');
              }}
            >
              <input
                ref={searchInputRef}
                type="search"
                className="text-input"
                style={{ flex: 1, minWidth: 0, padding: '6px 10px', fontSize: '13px' }}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Слово или фраза…"
                aria-label="Поисковый запрос"
              />
              <button
                type="submit"
                className="icon-btn-primary"
                aria-label="Найти"
                data-tooltip="Найти"
              >
                <IconSearch size={15} />
              </button>
            </form>
          ) : null}

          {entitlements.isQuotaWarning ? (
            <div className="quota-banner quota-banner-warn" role="status" style={{ marginTop: 0 }}>
              <span>Осталось {entitlements.quotaRemaining} из {entitlements.quotaTotal}.</span>
            </div>
          ) : null}

          {entitlements.isQuotaExhausted ? (
            <div
              className="quota-banner quota-banner-exhausted"
              role="status"
              style={{ marginTop: 0 }}
            >
              <span>Лимит исчерпан — доступен поиск «По словам».</span>
            </div>
          ) : null}

          {needsConsentPrompt ? (
            <div className="quota-banner quota-banner-warn" role="status" style={{ marginTop: 0 }}>
              <div className="stack" style={{ gap: '0.4rem' }}>
                <span style={{ fontSize: '12px' }}>
                  Для облачной модели требуется согласие (152-ФЗ).
                </span>
                <div className="row" style={{ gap: '0.4rem' }}>
                  <button
                    type="button"
                    className="btn-primary"
                    style={{ padding: '4px 10px', fontSize: '12px' }}
                    onClick={() => {
                      onAcceptPrivacyConsent?.();
                      void runSearch('rag_synthesis', undefined, undefined, true);
                    }}
                  >
                    Разрешить
                  </button>
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ padding: '4px 10px', fontSize: '12px' }}
                    onClick={() => void runSearch('local_fts5')}
                  >
                    По словам
                  </button>
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div className="search-pane-body">
          {phase === 'idle' ? (
            <div className="search-pane-empty">
              <p className="typo-meta" style={{ lineHeight: 1.5 }}>
                {engineMode === 'rag_synthesis'
                  ? 'Задайте вопрос — ответ соберётся по цитатам из ваших каналов.'
                  : 'Мгновенный поиск по сохранённым постам без расхода лимита.'}
              </p>
              <div className="row" style={{ gap: '0.35rem', flexWrap: 'wrap' }}>
                {exampleQueries.map((example) => (
                  <button
                    key={example}
                    type="button"
                    className="chip-btn"
                    onClick={() => void runSearch(engineMode, undefined, example)}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {isStreaming ? (
                <div className="sse-stepper" aria-label="Этапы подготовки ответа">
                  <span
                    className={phase === 'searching' ? 'step-active row' : 'step-done'}
                    style={{ gap: '0.3rem' }}
                  >
                    {phase === 'searching' ? (
                      <span className="loading-spinner" aria-hidden="true" />
                    ) : null}
                    <span>1. Поиск</span>
                  </span>
                  <span
                    className={
                      phase === 'drafting'
                        ? 'step-active row'
                        : phase === 'verifying'
                          ? 'step-done'
                          : ''
                    }
                    style={{ gap: '0.3rem' }}
                  >
                    {phase === 'drafting' ? (
                      <span className="loading-spinner" aria-hidden="true" />
                    ) : null}
                    <span>2. Ответ</span>
                  </span>
                  <span
                    className={phase === 'verifying' ? 'step-active row' : ''}
                    style={{ gap: '0.3rem' }}
                  >
                    {phase === 'verifying' ? (
                      <span className="loading-spinner" aria-hidden="true" />
                    ) : null}
                    <span>3. Цитаты</span>
                  </span>
                </div>
              ) : null}

              {phase === 'refusal' ? (
                <div className="refusal-stub-card" role="status" style={{ padding: '12px 14px' }}>
                  <div className="font-semibold" style={{ fontSize: '13.5px' }}>
                    {engineMode === 'rag_synthesis'
                      ? 'В каналах нет точных данных по запросу'
                      : 'Совпадений не найдено'}
                  </div>
                  <p className="typo-meta" style={{ marginTop: '0.2rem' }}>
                    {errorMsg ??
                      (engineMode === 'rag_synthesis'
                        ? 'Уточните запрос, расширьте период или поищите по словам.'
                        : 'Попробуйте другое слово.')}
                  </p>
                  {engineMode === 'rag_synthesis' ? (
                    <div className="row" style={{ gap: '0.4rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                      <button
                        type="button"
                        className="btn-secondary"
                        style={{ padding: '4px 10px', fontSize: '12px' }}
                        onClick={() => void runSearch('rag_synthesis', 30)}
                      >
                        За 30 дней
                      </button>
                      <button
                        type="button"
                        className="btn-primary"
                        style={{ padding: '4px 10px', fontSize: '12px' }}
                        onClick={() => void runSearch('local_fts5')}
                      >
                        По словам
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : (
                <>
                  {fallbackNote ? (
                    <p className="typo-meta" role="status">
                      {fallbackNote}
                    </p>
                  ) : null}

                  {engineMode === 'rag_synthesis' && directAnswer ? (
                    renderDirectAnswerCard(directAnswer, true)
                  ) : null}

                  {engineMode === 'rag_synthesis' && shouldShowClaimsList ? (
                    <div className="stack" style={{ gap: '0.5rem' }}>
                      <div className="row-between">
                        <span className="typo-meta font-semibold" style={{ fontSize: '11.5px' }}>
                          {directAnswer ? 'Подтверждающие факты из постов' : 'Факты из источников'}
                        </span>
                        {!directAnswer && doneMeta?.verified_claims_count !== undefined ? (
                          <span className="typo-meta tabular-nums" style={{ fontSize: '11.5px' }}>
                            Подтверждено фактов: {doneMeta.verified_claims_count}
                          </span>
                        ) : null}
                      </div>
                      {claims.map((claim) => renderClaimCard(claim, true))}
                    </div>
                  ) : null}

                  {engineMode === 'rag_synthesis' && stanceRows.length > 0 ? (
                    <div className="stack" style={{ gap: '0.5rem' }}>
                      {stanceRows.map((row) => (
                        <div key={row.aspect} className="stance-row-card">
                          <strong className="typo-meta" style={{ color: 'var(--text-primary)' }}>
                            {row.aspect}
                          </strong>
                          <div className="stance-two-col" style={{ marginTop: '0.35rem' }}>
                            <div className="stance-cell stance-pro">
                              <span className="badge badge-entailed">ЗА</span>
                              {row.pro_points.map((pt, i) => (
                                <p key={i} className="typo-meta" style={{ color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                                  {pt.statement}
                                </p>
                              ))}
                            </div>
                            <div className="stance-cell stance-con">
                              <span className="badge badge-unsupported">ПРОТИВ</span>
                              {row.con_points.map((pt, i) => (
                                <p key={i} className="typo-meta" style={{ color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                                  {pt.statement}
                                </p>
                              ))}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : null}

                  {engineMode === 'local_fts5' ? (
                    <div className="typo-meta tabular-nums" style={{ fontSize: '12px' }}>
                      Найдено совпадений: {sources.length}
                      {ftsElapsedMs !== null ? ` • ${ftsElapsedMs} мс` : ''}
                    </div>
                  ) : sources.length > 0 ? (
                    <div className="row-between">
                      <button
                        type="button"
                        className="btn-link"
                        style={{ fontSize: '12px' }}
                        onClick={() => setSourcesCollapsed((prev) => !prev)}
                      >
                        {sourcesCollapsed
                          ? `Показать источники (${sources.length})`
                          : `Скрыть источники (${sources.length})`}
                      </button>
                      {!sourcesCollapsed ? (
                        <span className="typo-meta" style={{ fontSize: '11px' }}>
                          1–9
                        </span>
                      ) : null}
                    </div>
                  ) : null}

                  {sources.length > 0 && (engineMode === 'local_fts5' || !sourcesCollapsed) ? (
                    <div className="stack" style={{ gap: '0.55rem' }}>
                      {renderSourceList(sources)}
                    </div>
                  ) : null}
                </>
              )}
            </>
          )}
        </div>

        {engineMode === 'rag_synthesis' ? (
          <div className="search-pane-footer">
            {errorMsg && phase === 'idle' ? (
              <p
                className="typo-meta"
                style={{ color: 'var(--verify-unsupported-accent)', marginBottom: '0.35rem', fontSize: '11.5px' }}
              >
                {errorMsg}
              </p>
            ) : null}
            <form
              className="ai-chat-island"
              onSubmit={(e) => {
                e.preventDefault();
                void runSearch('rag_synthesis');
              }}
              aria-label="Запрос по источникам"
            >
              <span className="ai-chat-sparkle" aria-hidden="true">
                <IconSparkles size={15} />
              </span>
              <input
                ref={searchInputRef}
                type="search"
                className="ai-chat-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Спросить по источникам…"
                aria-label="Поисковый запрос"
              />
              <button
                type="submit"
                className="ai-chat-send-btn"
                disabled={isStreaming || !query.trim()}
                aria-label="Отправить запрос"
                data-tooltip="Спросить"
              >
                <IconArrowUp size={16} />
              </button>
            </form>
            <div className="search-pane-footer-meta">
              <span className="typo-meta" style={{ fontSize: '11.5px' }}>
                Период поиска:
              </span>
              <div className="search-window-pills" role="group" aria-label="Период поиска">
                {TIME_WINDOWS.map((w) => (
                  <button
                    key={w.days}
                    type="button"
                    className={
                      timeWindowDays === w.days
                        ? 'search-window-pill active'
                        : 'search-window-pill'
                    }
                    onClick={() => setTimeWindowDays(w.days)}
                    aria-pressed={timeWindowDays === w.days}
                  >
                    {w.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : null}
      </section>
    );
  }

  return (
    <section aria-label="Поиск и анализ по источникам" className="stack">
      <div className="rag-control-card">
        <div className="row-between" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
          <div className="segmented-tabs" role="group" aria-label="Режим поиска">
            <button
              type="button"
              className={engineMode === 'rag_synthesis' ? 'active' : ''}
              disabled={!entitlements.canRunCloudRag}
              onClick={() => setEngineMode('rag_synthesis')}
              title={
                entitlements.canRunCloudRag
                  ? 'Соберёт ответ и проверит факты по цитатам из ваших каналов'
                  : 'Месячный лимит ИИ-ответов исчерпан — доступен бесплатный поиск по тексту'
              }
            >
              <span className="row" style={{ gap: '0.35rem', alignItems: 'center' }}>
                <span>ИИ-ответ по источникам</span>
                <span
                  className="tabular-nums"
                  style={{
                    fontSize: '10.5px',
                    fontWeight: 600,
                    padding: '1px 6px',
                    borderRadius: '9999px',
                    backgroundColor:
                      engineMode === 'rag_synthesis'
                        ? 'var(--accent-primary-bg)'
                        : 'var(--bg-surface)',
                    color:
                      engineMode === 'rag_synthesis'
                        ? 'var(--accent-primary)'
                        : 'var(--text-secondary)',
                  }}
                >
                  {entitlements.quotaRemaining}/{entitlements.quotaTotal}
                </span>
              </span>
            </button>
            <button
              type="button"
              className={engineMode === 'local_fts5' ? 'active' : ''}
              onClick={() => setEngineMode('local_fts5')}
              title="Быстрый поиск по тексту загруженных постов без расхода лимита"
            >
              Поиск по тексту (бесплатно)
            </button>
          </div>

          <div className="row" style={{ gap: '0.4rem', flexWrap: 'wrap' }}>
            {entitlements.statusChipLabel ? (
              <span className="badge badge-neutral tabular-nums">
                {entitlements.statusChipLabel}
              </span>
            ) : null}
            {entitlements.showFrontierEscalation ? (
              <span className="badge badge-pending tabular-nums">
                Глубокий анализ: {entitlements.frontierQuotaRemaining}/3
              </span>
            ) : null}
          </div>
        </div>

        {entitlements.isQuotaWarning ? (
          <div className="quota-banner quota-banner-warn" role="status">
            <span>
              Осталось {entitlements.quotaRemaining} из {entitlements.quotaTotal} ИИ-ответов.
              Текстовый поиск по архиву всегда бесплатен.
            </span>
          </div>
        ) : null}

        {entitlements.isQuotaExhausted ? (
          <div className="quota-banner quota-banner-exhausted" role="status">
            <span>
              Лимит ИИ-ответов исчерпан (0 / {entitlements.quotaTotal}). Включён бесплатный поиск
              по сохранённым постам.
            </span>
          </div>
        ) : null}

        <form
          className="row"
          style={{ gap: '0.5rem', marginTop: '0.65rem', flexWrap: 'wrap' }}
          onSubmit={(e) => {
            e.preventDefault();
            void runSearch();
          }}
        >
          <input
            ref={searchInputRef}
            type="search"
            className="text-input"
            style={{ flex: 1, minWidth: '220px' }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Спросите или найдите по каналам (например: H100 vs Batch API)…"
            aria-label="Поисковый запрос"
          />
          <select
            className="select-input"
            aria-label="Период поиска"
            value={timeWindowDays}
            onChange={(e) => setTimeWindowDays(Number(e.target.value))}
          >
            {TIME_WINDOWS.map((w) => (
              <option key={w.days} value={w.days}>
                {w.label}
              </option>
            ))}
          </select>
          <button type="submit" className="btn-primary" disabled={isStreaming}>
            {isStreaming ? (
              <>
                <span className="loading-spinner" aria-hidden="true" />
                <span>Ищем…</span>
              </>
            ) : engineMode === 'rag_synthesis' ? (
              'Найти ответ'
            ) : (
              'Найти посты'
            )}
          </button>
        </form>

        <div className="row" style={{ gap: '0.4rem', marginTop: '0.55rem', flexWrap: 'wrap' }}>
          <span className="typo-meta" style={{ fontSize: '12px' }}>
            Например:
          </span>
          {exampleQueries.map((example) => (
            <button
              key={example}
              type="button"
              className="chip-btn"
              onClick={() => void runSearch(undefined, undefined, example)}
            >
              {example}
            </button>
          ))}
        </div>

        {needsConsentPrompt ? (
          <div className="quota-banner quota-banner-warn" role="status">
            <div className="row-between" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
              <span>
                Для ИИ-ответа запрос передаётся в облачную модель (152-ФЗ). Вы можете разрешить
                обработку или переключиться на поиск по тексту.
              </span>
              <div className="row" style={{ gap: '0.4rem' }}>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => {
                    onAcceptPrivacyConsent?.();
                    void runSearch('rag_synthesis', undefined, undefined, true);
                  }}
                >
                  Разрешить и найти
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => void runSearch('local_fts5')}
                >
                  Искать по тексту
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {errorMsg ? (
          <p
            className="typo-meta"
            style={{ color: 'var(--verify-unsupported-accent)', marginTop: '0.45rem' }}
          >
            {errorMsg}
          </p>
        ) : null}
      </div>

      {isStreaming ? (
        <div className="sse-stepper" aria-label="Этапы подготовки ответа">
          <span className={phase === 'searching' ? 'step-active row' : 'step-done'} style={{ gap: '0.35rem' }}>
            {phase === 'searching' ? <span className="loading-spinner" aria-hidden="true" /> : null}
            <span>1. Ищем посты</span>
          </span>
          <span
            className={
              phase === 'drafting'
                ? 'step-active row'
                : phase === 'verifying'
                  ? 'step-done'
                  : ''
            }
            style={{ gap: '0.35rem' }}
          >
            {phase === 'drafting' ? <span className="loading-spinner" aria-hidden="true" /> : null}
            <span>2. Составляем ответ</span>
          </span>
          <span className={phase === 'verifying' ? 'step-active row' : ''} style={{ gap: '0.35rem' }}>
            {phase === 'verifying' ? <span className="loading-spinner" aria-hidden="true" /> : null}
            <span>3. Проверяем цитаты</span>
          </span>
        </div>
      ) : null}

      {phase === 'idle' ? (
        <div className="rag-panel">
          <h2 className="typo-h3">Поиск по вашим каналам</h2>
          <p className="typo-body" style={{ marginTop: '0.25rem', color: 'var(--text-secondary)' }}>
            Введите тему, вопрос или ключевое слово. В режиме ИИ-ответа система соберёт сводный
            ответ со ссылками на цитаты, а при текстовом поиске сразу покажет подходящие посты.
          </p>
        </div>
      ) : phase === 'refusal' ? (
        <div className="refusal-stub-card" role="status">
          <h2 className="typo-h3">В ваших каналах нет точных данных по запросу</h2>
          <p className="typo-body" style={{ marginTop: '0.25rem' }}>
            Мы не стали додумывать ответ без подтверждения в постах. Попробуйте уточнить формулировку,
            расширить период или выполнить поиск по точному совпадению.
          </p>
          <div className="row" style={{ gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.65rem' }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => searchInputRef.current?.focus()}
            >
              Уточнить запрос
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => void runSearch('rag_synthesis', 30)}
            >
              Искать за 30 дней
            </button>
            <button
              type="button"
              className="btn-primary"
              onClick={() => void runSearch('local_fts5')}
            >
              Искать по тексту
            </button>
          </div>
        </div>
      ) : (
        <div className="rag-two-tier-desktop">
          <div className="stack">
            {engineMode === 'rag_synthesis' ? (
              <>
                {isStreaming || claims.length > 0 || fallbackNote || stanceRows.length === 0 ? (
                  <div className="rag-panel">
                    <div className="row-between" style={{ marginBottom: '0.45rem', flexWrap: 'wrap', gap: '0.4rem' }}>
                      <h2 className="typo-h3">Ответ с опорой на посты</h2>
                      {doneMeta?.verified_claims_count !== undefined ? (
                        <span className="typo-meta tabular-nums">
                          Подтверждено фактов: {doneMeta.verified_claims_count}
                        </span>
                      ) : null}
                    </div>

                    {isStreaming && claims.length === 0 ? (
                      <p className="typo-meta row" style={{ gap: '0.4rem' }} role="status">
                        <span className="loading-spinner" aria-hidden="true" />
                        <span>
                          {phase === 'searching'
                            ? 'Ищем релевантные посты в подключённых каналах…'
                            : phase === 'drafting'
                              ? 'Формируем ответ по найденным постам…'
                              : 'Проверяем цитаты и факты…'}
                        </span>
                      </p>
                    ) : null}

                    {fallbackNote ? (
                      <p className="typo-meta" role="status">
                        {fallbackNote}
                      </p>
                    ) : null}

                    {directAnswer ? (
                      <div style={{ marginBottom: shouldShowClaimsList ? '0.7rem' : 0 }}>
                        {renderDirectAnswerCard(directAnswer, false)}
                      </div>
                    ) : null}

                    {shouldShowClaimsList ? (
                      <div className="stack" style={{ gap: '0.6rem' }}>
                        {directAnswer ? (
                          <div className="typo-meta font-semibold" style={{ fontSize: '12px' }}>
                            Подтверждающие факты из постов
                          </div>
                        ) : null}
                        {claims.map((claim) => renderClaimCard(claim, false))}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {stanceRows.length > 0 ? (
                  <div className="rag-panel">
                    <h2 className="typo-h3" style={{ marginBottom: '0.2rem' }}>
                      Сравнение подходов по теме
                    </h2>
                    <p className="typo-meta" style={{ marginBottom: '0.55rem' }}>
                      Аргументы «за», ограничения и условия из постов по вашему сравнительному запросу.
                    </p>

                    {isMobile ? (
                      <>
                        <div className="segmented-tabs" style={{ marginBottom: '0.5rem' }}>
                          <button
                            type="button"
                            className={mobileMatrixTab === 'pro' ? 'active' : ''}
                            onClick={() => setMobileMatrixTab('pro')}
                          >
                            ЗА ({flattenedCounts.pro})
                          </button>
                          <button
                            type="button"
                            className={mobileMatrixTab === 'con' ? 'active' : ''}
                            onClick={() => setMobileMatrixTab('con')}
                          >
                            ПРОТИВ ({flattenedCounts.con})
                          </button>
                          <button
                            type="button"
                            className={mobileMatrixTab === 'nuances' ? 'active' : ''}
                            onClick={() => setMobileMatrixTab('nuances')}
                          >
                            Нюансы ({flattenedCounts.nuances})
                          </button>
                        </div>

                        {stanceRows.map((row) => (
                          <div key={row.aspect} className="stack" style={{ gap: '0.35rem', marginBottom: '0.6rem' }}>
                            <strong className="typo-meta">{row.aspect}</strong>
                            {mobileMatrixTab === 'pro'
                              ? row.pro_points.map((pt, i) => (
                                  <div key={i} className="stance-cell stance-pro">
                                    <p className="typo-meta" style={{ color: 'var(--text-primary)' }}>
                                      {pt.statement}
                                    </p>
                                  </div>
                                ))
                              : mobileMatrixTab === 'con'
                              ? row.con_points.map((pt, i) => (
                                  <div key={i} className="stance-cell stance-con">
                                    <p className="typo-meta" style={{ color: 'var(--text-primary)' }}>
                                      {pt.statement}
                                    </p>
                                  </div>
                                ))
                              : (row.nuances ?? []).map((nText, i) => (
                                  <div key={i} className="stance-cell">
                                    <p className="typo-meta" style={{ color: 'var(--text-primary)' }}>
                                      {nText}
                                    </p>
                                  </div>
                                ))}
                          </div>
                        ))}
                      </>
                    ) : (
                      <div className="stack" style={{ gap: '0.65rem' }}>
                        {stanceRows.map((row) => (
                          <div key={row.aspect} className="stance-row-card">
                            <div className="row-between" style={{ marginBottom: '0.4rem' }}>
                              <strong className="typo-meta" style={{ color: 'var(--text-primary)' }}>
                                {row.aspect}
                              </strong>
                            </div>

                            <div className="stance-two-col">
                              <div className="stance-cell stance-pro">
                                <span className="badge badge-entailed">ЗА</span>
                                {row.pro_points.map((pt, i) => (
                                  <p key={i} className="typo-meta" style={{ color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                                    {pt.statement}
                                  </p>
                                ))}
                              </div>
                              <div className="stance-cell stance-con">
                                <span className="badge badge-unsupported">ПРОТИВ</span>
                                {row.con_points.map((pt, i) => (
                                  <p key={i} className="typo-meta" style={{ color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                                    {pt.statement}
                                  </p>
                                ))}
                              </div>
                            </div>

                            {row.nuances && row.nuances.length > 0 ? (
                              <div className="stance-nuances">
                                <strong className="typo-meta">Условия: </strong>
                                <span className="typo-meta">{row.nuances.join(' • ')}</span>
                              </div>
                            ) : null}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : null}
              </>
            ) : (
              <div className="rag-panel">
                <h2 className="typo-h3">Поиск по сохранённым постам</h2>
                <p className="typo-meta tabular-nums" style={{ marginTop: '0.25rem' }}>
                  Найдено совпадений: {sources.length}
                  {ftsElapsedMs !== null ? ` • ${ftsElapsedMs} мс` : ''}
                </p>
              </div>
            )}
          </div>

          <div className="stack">
            <div className="row-between">
              <h2 className="typo-h3">Источники ({sources.length})</h2>
              {!isMobile && sources.length > 0 ? (
                <span className="typo-meta">Клавиши 1–9: перейти к посту</span>
              ) : null}
            </div>

            {isStreaming && sources.length === 0 ? (
              <div className="rag-source-card" role="status">
                <p className="typo-meta">Подбираем посты-источники…</p>
              </div>
            ) : null}

            {sources.map((src, index) => {
              const parsed = parseTelegramPostAstSync(
                src.post.text,
                src.post.entities
              );

              return (
                <article
                  key={src.post.id}
                  ref={(el) => {
                    sourceRefs.current[src.post.id] = el;
                  }}
                  className={`rag-source-card ${
                    activeSourceId === src.post.id ? 'rag-source-active' : ''
                  }`}
                >
                  <div className="row-between" style={{ gap: '0.4rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
                    <div className="row" style={{ gap: '0.4rem' }}>
                      <kbd className="kbd-hint">{index + 1}</kbd>
                      <strong className="typo-meta" style={{ color: 'var(--text-primary)' }}>
                        {src.post.channelTitle}
                      </strong>
                      <span className="typo-meta">{src.post.channelHandle}</span>
                    </div>
                    <span className="typo-meta tabular-nums">{formatRuDateTime(src.post.publishedAt)}</span>
                  </div>

                  <AstRenderer
                    nodes={parsed.nodes}
                    richContainers={parsed.richContainers}
                    density={density}
                    cssHighlightSpans={src.rag_highlights}
                  />
                </article>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
