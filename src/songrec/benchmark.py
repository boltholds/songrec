from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import librosa
import numpy as np
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session

from songrec.audio import load_audio
from songrec.db.repositories import FingerprintRepository, TrackRepository
from songrec.indexer import iter_audio_files
from songrec.matcher import MatchDecision, MatchResult
from songrec.recognition.speed import DEFAULT_SPEED_FACTORS, recognize_audio, parse_speed_factors

console = Console()


@dataclass(frozen=True)
class BenchmarkCase:
    track_path: Path
    title: str
    duration_sec: int
    start_sec: float
    mode: str
    expected_track_id: int | None


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    result: MatchResult | None
    is_correct: bool
    is_false_positive: bool
    latency_ms: float
    query_fingerprints_count: int


def slice_audio(
    audio: np.ndarray,
    sample_rate: int,
    start_sec: float,
    duration_sec: int,
) -> np.ndarray:
    start_sample = int(start_sec * sample_rate)
    end_sample = start_sample + int(duration_sec * sample_rate)
    return audio[start_sample:end_sample]


def choose_start_positions(
    audio_duration_sec: float,
    fragment_duration_sec: int,
    samples_per_track: int,
) -> list[float]:
    max_start = audio_duration_sec - fragment_duration_sec

    if max_start <= 0:
        return []

    if samples_per_track <= 1:
        return [max_start / 2]

    safe_start = max(0.0, max_start * 0.10)
    safe_end = max_start * 0.90

    if safe_end <= safe_start:
        return [max_start / 2]

    return np.linspace(safe_start, safe_end, samples_per_track).tolist()


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-9:
        return audio.astype(np.float32)
    return (audio / peak * 0.95).astype(np.float32)


def add_white_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    if audio.size == 0:
        return audio

    signal_power = float(np.mean(audio**2))
    if signal_power <= 1e-12:
        return audio

    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(42).normal(0.0, np.sqrt(noise_power), audio.shape)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def change_volume(audio: np.ndarray, gain: float = 0.35) -> np.ndarray:
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def speed_change(audio: np.ndarray, rate: float = 1.05) -> np.ndarray:
    if audio.size == 0:
        return audio
    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
    return normalize_audio(stretched)


def apply_mode(fragment: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clean":
        return fragment
    if mode == "noise":
        return add_white_noise(fragment, snr_db=10.0)
    if mode == "volume":
        return change_volume(fragment, gain=0.35)
    if mode == "speed-0.95":
        return speed_change(fragment, rate=0.95)
    if mode == "speed-1.05":
        return speed_change(fragment, rate=1.05)
    if mode == "speed-1.10":
        return speed_change(fragment, rate=1.10)
    if mode == "negative":
        # Synthetic out-of-library audio. It should be rejected by confidence thresholds.
        rng = np.random.default_rng(123)
        return rng.normal(0.0, 0.1, fragment.shape).astype(np.float32)

    raise ValueError(f"Unknown benchmark mode: {mode}")


def recognize_audio_fragment(
    fragment: np.ndarray,
    fingerprint_repo: FingerprintRepository,
    decision: MatchDecision,
    recognition_mode: str = "fast",
    speed_factors: list[float] | None = None,
) -> tuple[MatchResult | None, int]:
    return recognize_audio(
        audio=fragment,
        fingerprint_repo=fingerprint_repo,
        decision=decision,
        mode=recognition_mode,
        speed_factors=speed_factors or list(DEFAULT_SPEED_FACTORS),
    )


def parse_modes(modes: str) -> list[str]:
    parsed = [mode.strip() for mode in modes.split(",") if mode.strip()]
    return parsed or ["clean"]


def run_benchmark(
    music_dir: Path,
    session: Session,
    durations_sec: list[int],
    samples_per_track: int,
    modes: list[str] | None = None,
    sample_rate: int = 11_025,
    decision: MatchDecision | None = None,
    recognition_mode: str = "fast",
    speed_factors: list[float] | None = None,
) -> list[BenchmarkResult]:
    files = list(iter_audio_files(music_dir))
    modes = modes or ["clean"]
    decision = decision or MatchDecision()

    if not files:
        console.print("[yellow]No audio files found.[/yellow]")
        return []

    track_repo = TrackRepository(session)
    fingerprint_repo = FingerprintRepository(session)

    results: list[BenchmarkResult] = []

    for track_path in files:
        title = track_path.stem
        track = track_repo.get_by_path(track_path)

        if track is None:
            console.print(
                f"[yellow]Skip not indexed:[/yellow] {track_path}. "
                "Run `songrec index` first."
            )
            continue

        console.print(f"[cyan]Benchmarking:[/cyan] {track_path}")

        audio = load_audio(track_path, sample_rate=sample_rate)
        audio_duration_sec = len(audio) / sample_rate

        for duration_sec in durations_sec:
            starts = choose_start_positions(
                audio_duration_sec=audio_duration_sec,
                fragment_duration_sec=duration_sec,
                samples_per_track=samples_per_track,
            )

            for start_sec in starts:
                original_fragment = slice_audio(
                    audio=audio,
                    sample_rate=sample_rate,
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                )

                for mode in modes:
                    expected_track_id = None if mode == "negative" else track.id
                    case = BenchmarkCase(
                        track_path=track_path,
                        title=title,
                        duration_sec=duration_sec,
                        start_sec=start_sec,
                        mode=mode,
                        expected_track_id=expected_track_id,
                    )

                    fragment = apply_mode(original_fragment, mode)

                    started_at = perf_counter()
                    result, query_fingerprints_count = recognize_audio_fragment(
                        fragment=fragment,
                        fingerprint_repo=fingerprint_repo,
                        decision=decision,
                        recognition_mode=recognition_mode,
                        speed_factors=speed_factors,
                    )
                    latency_ms = (perf_counter() - started_at) * 1000

                    confident_result = (
                        result if result is not None and result.is_confident else None
                    )

                    if expected_track_id is None:
                        is_correct = confident_result is None
                        is_false_positive = confident_result is not None
                    else:
                        is_correct = (
                            confident_result is not None
                            and confident_result.track_id == expected_track_id
                        )
                        is_false_positive = False

                    results.append(
                        BenchmarkResult(
                            case=case,
                            result=result,
                            is_correct=is_correct,
                            is_false_positive=is_false_positive,
                            latency_ms=latency_ms,
                            query_fingerprints_count=query_fingerprints_count,
                        )
                    )

    return results


def _avg(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def print_benchmark_report(results: list[BenchmarkResult]) -> None:
    if not results:
        console.print("[yellow]No benchmark results.[/yellow]")
        return

    total = len(results)
    correct = sum(result.is_correct for result in results)
    accuracy = correct / total
    avg_latency_ms = _avg([result.latency_ms for result in results])
    matched_results = [result.result for result in results if result.result is not None]
    avg_score = _avg([float(result.score) for result in matched_results])
    avg_margin = _avg([float(result.margin) for result in matched_results])
    false_positives = sum(result.is_false_positive for result in results)

    summary_table = Table(title="Benchmark summary")
    summary_table.add_column("Metric")
    summary_table.add_column("Value")

    summary_table.add_row("Total cases", str(total))
    summary_table.add_row("Correct", str(correct))
    summary_table.add_row("Top-1 accuracy", f"{accuracy:.2%}")
    summary_table.add_row("False positives", str(false_positives))
    summary_table.add_row("Avg latency", f"{avg_latency_ms:.2f} ms")
    summary_table.add_row("Avg score", f"{avg_score:.2f}")
    summary_table.add_row("Avg margin", f"{avg_margin:.2f}")

    recognition_modes = sorted({
        result.result.mode
        for result in results
        if result.result is not None
    })
    if recognition_modes:
        summary_table.add_row("Recognition mode", ", ".join(recognition_modes))

    console.print(summary_table)

    by_mode_table = Table(title="Accuracy by benchmark mode")
    by_mode_table.add_column("Mode")
    by_mode_table.add_column("Cases")
    by_mode_table.add_column("Correct")
    by_mode_table.add_column("Accuracy")
    by_mode_table.add_column("False positives")
    by_mode_table.add_column("Avg latency")

    modes = sorted({result.case.mode for result in results})

    for mode in modes:
        bucket = [result for result in results if result.case.mode == mode]
        bucket_total = len(bucket)
        bucket_correct = sum(result.is_correct for result in bucket)
        bucket_accuracy = bucket_correct / bucket_total
        bucket_latency = _avg([result.latency_ms for result in bucket])
        bucket_false_positives = sum(result.is_false_positive for result in bucket)

        by_mode_table.add_row(
            mode,
            str(bucket_total),
            str(bucket_correct),
            f"{bucket_accuracy:.2%}",
            str(bucket_false_positives),
            f"{bucket_latency:.2f} ms",
        )

    console.print(by_mode_table)

    by_duration_table = Table(title="Accuracy by fragment duration")
    by_duration_table.add_column("Duration")
    by_duration_table.add_column("Cases")
    by_duration_table.add_column("Correct")
    by_duration_table.add_column("Accuracy")
    by_duration_table.add_column("Avg latency")

    durations = sorted({result.case.duration_sec for result in results})

    for duration_sec in durations:
        bucket = [result for result in results if result.case.duration_sec == duration_sec]
        bucket_total = len(bucket)
        bucket_correct = sum(result.is_correct for result in bucket)
        bucket_accuracy = bucket_correct / bucket_total
        bucket_latency = _avg([result.latency_ms for result in bucket])

        by_duration_table.add_row(
            f"{duration_sec}s",
            str(bucket_total),
            str(bucket_correct),
            f"{bucket_accuracy:.2%}",
            f"{bucket_latency:.2f} ms",
        )

    console.print(by_duration_table)

    failures = [result for result in results if not result.is_correct]

    if not failures:
        console.print("[green]All benchmark cases passed.[/green]")
        return

    failure_table = Table(title="Failed cases")
    failure_table.add_column("Mode")
    failure_table.add_column("Expected")
    failure_table.add_column("Got")
    failure_table.add_column("Duration")
    failure_table.add_column("Start")
    failure_table.add_column("Score")
    failure_table.add_column("Margin")
    failure_table.add_column("Factor")
    failure_table.add_column("Reason")

    for failure in failures[:20]:
        got = failure.result.title if failure.result is not None else "No match"
        score = str(failure.result.score) if failure.result is not None else "-"
        margin = f"{failure.result.margin:.2f}" if failure.result is not None else "-"
        factor = f"{failure.result.speed_factor:.2f}" if failure.result is not None else "-"
        reason = (
            failure.result.reject_reason
            if failure.result is not None and failure.result.reject_reason
            else "-"
        )

        failure_table.add_row(
            failure.case.mode,
            failure.case.title,
            got,
            f"{failure.case.duration_sec}s",
            f"{failure.case.start_sec:.2f}s",
            score,
            margin,
            factor,
            reason,
        )

    console.print(failure_table)

    if len(failures) > 20:
        console.print(f"[yellow]Showing 20 of {len(failures)} failed cases.[/yellow]")
