import type { RagHighlightInput, RawTelegramEntity } from '@/lib/ast/types';

export type ViewMode = 'digest' | 'clean_originals' | 'search';
export type RagMode = 'qa' | 'digest' | 'stance_matrix';
export type PlanTier = 'TRIAL' | 'BASIC' | 'PROSUMER' | 'B2B_WORKSPACE';
export type FinopsZone = 'GREEN' | 'YELLOW' | 'RED' | 'GUARDRAIL';

export type WorkspaceScope = 'personal' | 'team_b2b' | 'hybrid';
export type DensityMode = 'comfortable' | 'compact';
export type ThemeMode = 'light' | 'dark' | 'sepia';
export type SourcePlatform = 'TG' | 'MAX' | 'RSS';
export type RbacRole = 'ADMIN' | 'ANALYST' | 'VIEWER';
export type AgentConnectionStatus = 'online' | 'offline_cache' | 'session_expired';
export type SearchEngineMode = 'rag_synthesis' | 'local_fts5';
export type VerificationStatus = 'entailed' | 'pending' | 'unsupported';

export interface MediaAttachmentItem {
  mediaId: string;
  fileId: string;
  mediaType: 'photo' | 'video' | 'document';
  mimeType: string;
  width?: number;
  height?: number;
  durationSec?: number;
  sizeBytes?: number;
  /** Lightweight 35 KB prefetched thumbnail (still renders even on HTTP 410 Gone). */
  thumbnailDataUrl: string;
  /** Lazy Range-streaming proxy URL (HTTP 206). */
  streamUrl: string;
  caption?: string;
  captionEntities?: RawTelegramEntity[];
  fileGone?: boolean;
}

export interface DuplicateChannelRef {
  channelId: string;
  channelTitle: string;
  channelHandle: string;
  postId: string;
  publishedAt: string;
  editDate?: string;
  originScore?: number;
  /** Marks empty grabber stub posts filtered out by edit_date analysis (SPEC 7.2.3 / 10.3.1). */
  isGrabberStubFiltered?: boolean;
  /** Information novelty score compared to primary origin (0..100%). */
  noveltyPct?: number;
  rewriteNote?: string;
}

export interface CleanPostItem {
  id: string;
  channelId: string;
  channelTitle: string;
  channelHandle: string;
  channelAvatarInitials?: string;
  postId: number | string;
  publishedAt: string;
  editDate?: string;
  neutralHeadline: string;
  /** Original author clickbait/opening line for 1-click/hover reversible AI-modification (SPEC 10.1, 10.3.1). */
  originalHeadline?: string;
  sourcePlatform?: SourcePlatform;
  clusterId?: string;
  text: string;
  entities: RawTelegramEntity[];
  groupedId?: string;
  showCaptionAboveMedia?: boolean;
  media: MediaAttachmentItem[];
  originScore?: number;
  duplicateCount?: number;
  duplicateSources?: DuplicateChannelRef[];
  commentCount?: number;
  rubric?: string;
  isPrivateChannel?: boolean;
  workspaceName?: string;
}

export interface DigestCardCitation {
  claimId: string;
  channelId: string;
  channelTitle: string;
  channelHandle: string;
  postId: string;
  quoteText: string;
  publishedAt: string;
  originScore: number;
  startUtf16: number;
  endUtf16: number;
}

export interface DigestCard {
  clusterId: string;
  rubric: string;
  neutralTitle: string;
  summaryBullets: Array<{
    text: string;
    citations: DigestCardCitation[];
  }>;
  primarySource: {
    channelId: string;
    channelTitle: string;
    channelHandle: string;
    originScore: number;
    postId: string;
    publishedAt: string;
  };
  otherChannelsCount: number;
  otherChannels: DuplicateChannelRef[];
  importanceScore: number;
  estimatedReadSec?: number;
  groupingMode?: 'semantic_cluster' | 'channel_latest';
}

export interface DigestFeedResponse {
  mode: 'digest';
  user_id: string;
  finops_zone: FinopsZone;
  reduce_model: string;
  total_candidate_clusters: number;
  selected_cards_count: number;
  rubrics: string[];
  cards: DigestCard[];
  scheduleWindow?: '09:00' | '18:00' | 'manual';
  nextEditionTime?: string;
  pendingNextSlotCount?: number;
  fromDemoFallback?: boolean;
}

export interface CleanOriginalsResponse {
  mode: 'clean_originals';
  user_id: string;
  total: number;
  items: CleanPostItem[];
  fromDemoFallback?: boolean;
}

export interface VerifiedCitation {
  claim_id?: string;
  post_id: string;
  channel_id: string;
  channel_title?: string;
  channel_handle?: string;
  quote: string;
  start_utf16: number;
  end_utf16: number;
  published_at?: string;
  origin_score?: number;
}

export interface VerifiedClaim {
  claim_id: string;
  topic_group: string;
  statement: string;
  verification_status?: VerificationStatus;
  citations: VerifiedCitation[];
}

export type DirectAnswerStatus = 'direct' | 'indirect' | 'not_found';

export interface VerifiedDirectAnswer {
  status: DirectAnswerStatus;
  text: string;
  citations: VerifiedCitation[];
}

export interface VerifiedStancePoint {
  claim_id: string;
  statement: string;
  stance: 'PRO' | 'CON' | 'MIXED';
  confirmed_channels_count?: number;
  citations: VerifiedCitation[];
}

export interface StanceMatrixRow {
  aspect: string;
  pro_points: VerifiedStancePoint[];
  con_points: VerifiedStancePoint[];
  nuances?: string[];
}

export interface RagSourcePostItem {
  post: CleanPostItem;
  rag_highlights: RagHighlightInput[];
}

export interface RagDoneMetadata {
  status: string;
  mode: RagMode;
  answer_status?: DirectAnswerStatus;
  direct_answer?: VerifiedDirectAnswer | null;
  cache_hit?: boolean;
  verified_claims_count?: number;
  rejected_claims_count?: number;
  rows_count?: number;
  fallback_message?: string;
  total_latency_ms?: number;
  session_id?: string;
}

export interface CommentInputItem {
  comment_id: string;
  parent_id?: string | null;
  author_alias: string;
  text: string;
  likes: number;
  replies_count: number;
  published_at?: string;
  is_bug_report_hint?: boolean;
}

export interface SelectedCommentItem {
  comment_id: string;
  parent_comment_id?: string | null;
  parent_author_alias?: string;
  parent_preview?: string;
  author_alias: string;
  text: string;
  likes?: number;
  confirmed_count?: number;
  category: 'bug_report' | 'counter_argument' | 'user_case' | 'expertise' | 'other';
  importance_score: number;
  is_saved_rare_bug?: boolean;
}

export interface CommentSummaryResponse {
  post_id: string;
  bucket: string;
  cache_hit: boolean;
  kept_after_pruning: number;
  digest_text: string;
  selected: SelectedCommentItem[];
  fromDemoFallback?: boolean;
}

export interface IndustryPreset {
  id: 'it_dev' | 'ai_ds' | 'product_mgmt' | 'macro' | 'legal_tax';
  title: string;
  description: string;
  badgeColor: string;
  channels: Array<{
    channelId: string;
    handle: string;
    title: string;
    topic: string;
  }>;
}

export interface SubscribedChannel {
  channelId: string;
  handle: string;
  title: string;
  sourceType: 'public_preset' | 'imported_link' | 'imported_addlist' | 'private_agent';
  platform?: SourcePlatform;
  folder?: string;
  topic?: string;
}

export interface UserFolder {
  id: string;
  name: string;
  channelIds: string[];
}

export interface ZeroTrustUserSession {
  userId: string;
  email: string;
  authMethod: 'email_otp' | 'passkey' | 'oauth_google' | 'oauth_github' | 'guest_trial';
  planTier: PlanTier;
  finopsZone: FinopsZone;
  workspaceScope: WorkspaceScope;
  rbacRole: RbacRole;
  ragQuotaTotal: number;
  ragQuotaUsed: number;
  frontierQuotaRemaining: number;
  localInferenceOnlyRf: boolean;
  crossBorderConsentAccepted: boolean;
  subscribedChannels: SubscribedChannel[];
  appliedPresetIds: string[];
  desktopAgentConnected: boolean;
  agentStatus: AgentConnectionStatus;
  desktopAgentEndpoint: string;
  readClusterIds: string[];
}
