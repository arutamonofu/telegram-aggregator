"""Потоковая дедупликация/кластеризация (72ч, Single-Pass) и расчет OriginScore (раздел 7.2.3)."""

from app.clustering.origin_score import (
    ClusterOriginResult,
    OriginCandidate,
    compute_edit_change_ratio,
    compute_origin_score,
    recompute_cluster_origin,
    resolve_effective_published_at,
)
from app.clustering.ring_buffer import HotRingBuffer, RingBufferEntry
from app.clustering.single_pass import (
    ClusterAssignmentResult,
    ClusterMember,
    EventClusterState,
    SimilarityBreakdown,
    SinglePassClusterer,
    compute_bm25_norm,
    compute_composite_similarity,
)

__all__ = [
    "ClusterAssignmentResult",
    "ClusterMember",
    "ClusterOriginResult",
    "EventClusterState",
    "HotRingBuffer",
    "OriginCandidate",
    "RingBufferEntry",
    "SimilarityBreakdown",
    "SinglePassClusterer",
    "compute_bm25_norm",
    "compute_composite_similarity",
    "compute_edit_change_ratio",
    "compute_origin_score",
    "recompute_cluster_origin",
    "resolve_effective_published_at",
]
