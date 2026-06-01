from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from songrec.db.repositories import FingerprintMatchRow
from songrec.fingerprint import Fingerprint


@dataclass(frozen=True)
class MatchDecision:
    min_score: int = 20
    min_confidence: float = 0.005
    min_margin: float = 1.5


@dataclass
class MatchResult:
    track_id: int
    title: str
    score: int
    confidence: float
    offset_ms: int
    second_score: int
    margin: float
    is_confident: bool
    reject_reason: str | None = None


def match(
    query_fingerprints: list[Fingerprint],
    db_rows: list[FingerprintMatchRow],
    decision: MatchDecision | None = None,
) -> MatchResult | None:
    decision = decision or MatchDecision()
    query_by_hash: dict[str, list[Fingerprint]] = {}

    for fingerprint in query_fingerprints:
        query_by_hash.setdefault(fingerprint.hash, []).append(fingerprint)

    votes: Counter[tuple[int, str, int]] = Counter()

    for row in db_rows:
        for query_fingerprint in query_by_hash.get(row.hash, []):
            delta = row.track_offset_ms - query_fingerprint.offset_ms
            bucket_ms = round(delta / 100) * 100
            votes[(row.track_id, row.title, bucket_ms)] += 1

    if not votes:
        return None

    most_common = votes.most_common(2)
    (track_id, title, offset_ms), score = most_common[0]
    second_score = most_common[1][1] if len(most_common) > 1 else 0

    confidence = score / max(len(query_fingerprints), 1)
    margin = score / max(second_score, 1)

    reject_reasons: list[str] = []

    if score < decision.min_score:
        reject_reasons.append(
            f"score {score} < min_score {decision.min_score}"
        )

    if confidence < decision.min_confidence:
        reject_reasons.append(
            f"confidence {confidence:.4f} < min_confidence {decision.min_confidence:.4f}"
        )

    if margin < decision.min_margin:
        reject_reasons.append(
            f"margin {margin:.2f} < min_margin {decision.min_margin:.2f}"
        )

    return MatchResult(
        track_id=track_id,
        title=title,
        score=score,
        confidence=confidence,
        offset_ms=offset_ms,
        second_score=second_score,
        margin=margin,
        is_confident=not reject_reasons,
        reject_reason="; ".join(reject_reasons) if reject_reasons else None,
    )
