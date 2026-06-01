from __future__ import annotations

from dataclasses import replace

import librosa
import numpy as np

from songrec.db.repositories import FingerprintRepository
from songrec.fingerprint import find_peaks, fingerprint_audio, fingerprint_peaks
from songrec.matcher import MatchDecision, MatchResult, match

DEFAULT_SPEED_FACTORS: tuple[float, ...] = (0.90, 0.95, 1.0, 1.05, 1.10)


def parse_speed_factors(value: str | None) -> list[float]:
    if not value:
        return list(DEFAULT_SPEED_FACTORS)

    factors: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        factor = float(item)
        if factor <= 0:
            raise ValueError("Speed factors must be positive")
        factors.append(factor)

    return factors or list(DEFAULT_SPEED_FACTORS)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-9:
        return audio.astype(np.float32)
    return (audio / peak * 0.95).astype(np.float32)


def stretch_query(audio: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0 or audio.size == 0:
        return audio.astype(np.float32)

    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=factor)
    return normalize_audio(stretched)


def result_rank_key(result: MatchResult) -> tuple[int, int, float, float]:
    """Sort key for choosing the best candidate across candidate factors."""
    return (
        1 if result.is_confident else 0,
        result.score,
        result.margin,
        result.confidence,
    )


def recognize_audio_fast(
    audio: np.ndarray,
    fingerprint_repo: FingerprintRepository,
    decision: MatchDecision | None = None,
) -> tuple[MatchResult | None, int]:
    decision = decision or MatchDecision()
    query_fingerprints = fingerprint_audio(audio)
    rows = fingerprint_repo.find_matches(query_fingerprints)
    result = match(
        query_fingerprints=query_fingerprints,
        db_rows=rows,
        decision=decision,
    )

    if result is not None:
        result = replace(result, speed_factor=1.0, mode="fast")

    return result, len(query_fingerprints)


def recognize_audio_multi_speed(
    audio: np.ndarray,
    fingerprint_repo: FingerprintRepository,
    decision: MatchDecision | None = None,
    speed_factors: list[float] | tuple[float, ...] | None = None,
) -> tuple[MatchResult | None, int]:
    """
    Legacy robustness mode: physically time-stretch query audio for each factor.

    Kept for comparison/benchmarking. Prefer scale_aware for MVP-3.2.
    """
    decision = decision or MatchDecision()
    speed_factors = list(speed_factors or DEFAULT_SPEED_FACTORS)

    best_result: MatchResult | None = None
    total_fingerprints = 0

    for factor in speed_factors:
        candidate_audio = stretch_query(audio, factor)
        query_fingerprints = fingerprint_audio(candidate_audio)
        total_fingerprints += len(query_fingerprints)
        rows = fingerprint_repo.find_matches(query_fingerprints)
        result = match(
            query_fingerprints=query_fingerprints,
            db_rows=rows,
            decision=decision,
        )

        if result is None:
            continue

        result = replace(result, speed_factor=float(factor), mode="multi_speed")

        if best_result is None or result_rank_key(result) > result_rank_key(best_result):
            best_result = result

    return best_result, total_fingerprints


def recognize_audio_scale_aware(
    audio: np.ndarray,
    fingerprint_repo: FingerprintRepository,
    decision: MatchDecision | None = None,
    speed_factors: list[float] | tuple[float, ...] | None = None,
) -> tuple[MatchResult | None, int]:
    """
    Speed-aware matching without resampling the query audio.

    The audio is transformed to a peak constellation once. For each candidate
    scale we normalize landmark delta_time and offset onto the original-track
    time axis, then reuse the existing hash index and offset-voting matcher.

    For a query sped up by 1.10x, use scale ~= 1.10.
    For a query slowed down by 0.95x, use scale ~= 0.95.
    """
    decision = decision or MatchDecision()
    speed_factors = list(speed_factors or DEFAULT_SPEED_FACTORS)

    peaks = find_peaks(audio)
    best_result: MatchResult | None = None
    total_fingerprints = 0

    for factor in speed_factors:
        query_fingerprints = fingerprint_peaks(peaks, time_scale=float(factor))
        total_fingerprints += len(query_fingerprints)
        rows = fingerprint_repo.find_matches(query_fingerprints)
        result = match(
            query_fingerprints=query_fingerprints,
            db_rows=rows,
            decision=decision,
        )

        if result is None:
            continue

        result = replace(result, speed_factor=float(factor), mode="scale_aware")

        if best_result is None or result_rank_key(result) > result_rank_key(best_result):
            best_result = result

    return best_result, total_fingerprints


def recognize_audio(
    audio: np.ndarray,
    fingerprint_repo: FingerprintRepository,
    decision: MatchDecision | None = None,
    mode: str = "fast",
    speed_factors: list[float] | tuple[float, ...] | None = None,
) -> tuple[MatchResult | None, int]:
    if mode == "fast":
        return recognize_audio_fast(audio, fingerprint_repo, decision=decision)

    if mode == "multi_speed":
        return recognize_audio_multi_speed(
            audio,
            fingerprint_repo,
            decision=decision,
            speed_factors=speed_factors,
        )

    if mode == "scale_aware":
        return recognize_audio_scale_aware(
            audio,
            fingerprint_repo,
            decision=decision,
            speed_factors=speed_factors,
        )

    raise ValueError(f"Unknown recognition mode: {mode}")
