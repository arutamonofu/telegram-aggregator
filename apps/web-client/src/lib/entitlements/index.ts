import type {
  CleanPostItem,
  FinopsZone,
  PlanTier,
  RbacRole,
  ZeroTrustUserSession,
} from '@/lib/types/feed';

export interface RagEntitlementState {
  planTier: PlanTier;
  finopsZone: FinopsZone;
  quotaTotal: number;
  quotaUsed: number;
  quotaRemaining: number;
  usageRatio: number;
  isUnifiedPoolTier: boolean;
  isQuotaWarning: boolean;
  isQuotaExhausted: boolean;
  canRunCloudRag: boolean;
  canRunLocalFts5: boolean;
  showFrontierEscalation: boolean;
  frontierQuotaRemaining: number;
  isRedZoneFastOverview: boolean;
  isGuardrailThrottled: boolean;
  statusChipLabel: string | null;
  upgradeCtaRequired: boolean;
}

/**
 * Evaluates unified FinOps quota & graceful degradation state (SPEC 7.8.6 & 10.3.4).
 * Trial & Basic share a unified 30 RAG-query pool; Local FTS5 search remains free even at 30/30.
 */
export function evaluateRagEntitlements(session: ZeroTrustUserSession): RagEntitlementState {
  const isUnifiedPoolTier = session.planTier === 'TRIAL' || session.planTier === 'BASIC';
  const quotaTotal = Math.max(1, session.ragQuotaTotal || 30);
  const quotaUsed = Math.max(0, session.ragQuotaUsed || 0);
  const quotaRemaining = Math.max(0, quotaTotal - quotaUsed);
  const usageRatio = quotaUsed / quotaTotal;

  const isQuotaWarning = isUnifiedPoolTier && usageRatio >= 0.8 && quotaRemaining > 0;
  const isQuotaExhausted = isUnifiedPoolTier && quotaRemaining === 0;

  const isYellow = session.planTier === 'PROSUMER' && session.finopsZone === 'YELLOW';
  const isRed = session.finopsZone === 'RED';
  const isGuardrail = session.finopsZone === 'GUARDRAIL';

  let statusChipLabel: string | null = null;
  if (isRed) {
    statusChipLabel = 'Ускоренный обзор • окно 14 дней';
  } else if (isGuardrail) {
    statusChipLabel = 'Защита от перегрузки • троттлинг 3–5 с';
  }

  return {
    planTier: session.planTier,
    finopsZone: session.finopsZone,
    quotaTotal,
    quotaUsed,
    quotaRemaining,
    usageRatio,
    isUnifiedPoolTier,
    isQuotaWarning,
    isQuotaExhausted,
    canRunCloudRag: !isQuotaExhausted,
    canRunLocalFts5: true,
    showFrontierEscalation: isYellow && session.frontierQuotaRemaining > 0,
    frontierQuotaRemaining: session.frontierQuotaRemaining,
    isRedZoneFastOverview: isRed,
    isGuardrailThrottled: isGuardrail,
    statusChipLabel,
    upgradeCtaRequired: isQuotaExhausted,
  };
}

export function canSharePostExternally(
  post: Pick<CleanPostItem, 'isPrivateChannel' | 'workspaceName'>,
  session: Pick<ZeroTrustUserSession, 'rbacRole' | 'workspaceScope'>,
): { allowed: boolean; reason?: string } {
  if (post.isPrivateChannel) {
    return {
      allowed: false,
      reason: `Внешний шаринг в личные заметки заблокирован политикой закрытого пространства${
        post.workspaceName ? ` «${post.workspaceName}»` : ''
      }.`,
    };
  }
  if (session.rbacRole === 'VIEWER') {
    return {
      allowed: false,
      reason: 'Роль «Наблюдатель» разрешает только чтение без внешнего экспорта.',
    };
  }
  return { allowed: true };
}

export function formatRbacRoleLabel(role: RbacRole): string {
  switch (role) {
    case 'ADMIN':
      return 'Администратор';
    case 'ANALYST':
      return 'Аналитик';
    case 'VIEWER':
      return 'Наблюдатель';
  }
}

export interface B2BExportTargetPayload {
  target: 'slack' | 'notion' | 'telegram' | 'webhook';
  workspaceName: string;
  exportedAt: string;
  generationSessionId: string;
  rbacRole: RbacRole;
  verifiedSources: Array<{
    postId: string;
    channelHandle: string;
    publishedAt: string;
    quote: string;
  }>;
}

export function buildB2BExportBundle(params: {
  target: 'slack' | 'notion' | 'telegram' | 'webhook';
  workspaceName: string;
  generationSessionId: string;
  rbacRole: RbacRole;
  verifiedSources: B2BExportTargetPayload['verifiedSources'];
}): B2BExportTargetPayload {
  return {
    target: params.target,
    workspaceName: params.workspaceName,
    exportedAt: new Date().toISOString(),
    generationSessionId: params.generationSessionId,
    rbacRole: params.rbacRole,
    verifiedSources: params.verifiedSources,
  };
}
