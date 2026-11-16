"""Трехступенчатая он-деманд воронка предочистки и графового сжатия комментариев (7.2.4)."""

from app.comments.bug_rescue import (
    BugReportRescuer,
    RescueVerdict,
    TransformersBugRescueBackend,
)
from app.comments.graph_pruner import (
    RawComment,
    ScoredComment,
    clean_comment_text,
    filter_and_score_comment_tree,
    is_noise_or_spam_comment,
)
from app.comments.summarizer import (
    CommentBucketCache,
    CommentFunnelResult,
    CommentFunnelSummarizer,
    SelectedCommentItem,
    resolve_comment_bucket,
)

__all__ = [
    "BugReportRescuer",
    "CommentBucketCache",
    "CommentFunnelResult",
    "CommentFunnelSummarizer",
    "RawComment",
    "RescueVerdict",
    "ScoredComment",
    "SelectedCommentItem",
    "TransformersBugRescueBackend",
    "clean_comment_text",
    "filter_and_score_comment_tree",
    "is_noise_or_spam_comment",
    "resolve_comment_bucket",
]
