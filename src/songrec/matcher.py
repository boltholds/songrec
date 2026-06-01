from collections import Counter
from dataclasses import dataclass

from songrec.db.repositories import FingerprintMatchRow
from songrec.fingerprint import Fingerprint


@dataclass
class MatchResult:
    track_id: int
    title: str
    score: int
    confidence: float
    offset_ms: int


def match(
    query_fingerprints: list[Fingerprint],
    db_rows: list[FingerprintMatchRow],
) -> MatchResult | None:
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

    (track_id, title, offset_ms), score = votes.most_common(1)[0]

    confidence = score / max(len(query_fingerprints), 1)

    return MatchResult(
        track_id=track_id,
        title=title,
        score=score,
        confidence=confidence,
        offset_ms=offset_ms,
    )