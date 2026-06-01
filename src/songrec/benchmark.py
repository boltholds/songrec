from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session

from songrec.audio import load_audio
from songrec.db.repositories import FingerprintRepository, TrackRepository
from songrec.fingerprint import fingerprint_audio
from songrec.indexer import SUPPORTED_EXTENSIONS, iter_audio_files
from songrec.matcher import MatchResult, match

console = Console()


@dataclass(frozen=True)
class BenchmarkCase:
    track_path: Path
    title: str
    duration_sec: int
    start_sec: float


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    result: MatchResult | None
    is_correct: bool
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

    # Не режем совсем с края: там часто тишина, интро или fade-in.
    safe_start = max(0.0, max_start * 0.10)
    safe_end = max_start * 0.90

    if safe_end <= safe_start:
        return [max_start / 2]

    return np.linspace(safe_start, safe_end, samples_per_track).tolist()


def recognize_audio_fragment(
    fragment: np.ndarray,
    fingerprint_repo: FingerprintRepository,
) -> tuple[MatchResult | None, int]:
    query_fingerprints = fingerprint_audio(fragment)
    rows = fingerprint_repo.find_matches(query_fingerprints)
    result = match(query_fingerprints=query_fingerprints, db_rows=rows)
    return result, len(query_fingerprints)


def run_benchmark(
    music_dir: Path,
    session: Session,
    durations_sec: list[int],
    samples_per_track: int,
    sample_rate: int = 11_025,
) -> list[BenchmarkResult]:
    files = list(iter_audio_files(music_dir))

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
                case = BenchmarkCase(
                    track_path=track_path,
                    title=title,
                    duration_sec=duration_sec,
                    start_sec=start_sec,
                )

                fragment = slice_audio(
                    audio=audio,
                    sample_rate=sample_rate,
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                )

                started_at = perf_counter()
                result, query_fingerprints_count = recognize_audio_fragment(
                    fragment=fragment,
                    fingerprint_repo=fingerprint_repo,
                )
                latency_ms = (perf_counter() - started_at) * 1000

                is_correct = result is not None and result.track_id == track.id

                results.append(
                    BenchmarkResult(
                        case=case,
                        result=result,
                        is_correct=is_correct,
                        latency_ms=latency_ms,
                        query_fingerprints_count=query_fingerprints_count,
                    )
                )

    return results


def print_benchmark_report(results: list[BenchmarkResult]) -> None:
    if not results:
        console.print("[yellow]No benchmark results.[/yellow]")
        return

    total = len(results)
    correct = sum(result.is_correct for result in results)
    accuracy = correct / total
    avg_latency_ms = sum(result.latency_ms for result in results) / total
    avg_score = (
        sum(result.result.score for result in results if result.result is not None)
        / max(sum(result.result is not None for result in results), 1)
    )

    summary_table = Table(title="Benchmark summary")
    summary_table.add_column("Metric")
    summary_table.add_column("Value")

    summary_table.add_row("Total cases", str(total))
    summary_table.add_row("Correct", str(correct))
    summary_table.add_row("Top-1 accuracy", f"{accuracy:.2%}")
    summary_table.add_row("Avg latency", f"{avg_latency_ms:.2f} ms")
    summary_table.add_row("Avg score", f"{avg_score:.2f}")

    console.print(summary_table)

    by_duration_table = Table(title="Accuracy by fragment duration")
    by_duration_table.add_column("Duration")
    by_duration_table.add_column("Cases")
    by_duration_table.add_column("Correct")
    by_duration_table.add_column("Accuracy")
    by_duration_table.add_column("Avg latency")

    durations = sorted({result.case.duration_sec for result in results})

    for duration_sec in durations:
        bucket = [
            result for result in results if result.case.duration_sec == duration_sec
        ]
        bucket_total = len(bucket)
        bucket_correct = sum(result.is_correct for result in bucket)
        bucket_accuracy = bucket_correct / bucket_total
        bucket_latency = sum(result.latency_ms for result in bucket) / bucket_total

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
    failure_table.add_column("Expected")
    failure_table.add_column("Got")
    failure_table.add_column("Duration")
    failure_table.add_column("Start")
    failure_table.add_column("Score")

    for failure in failures[:20]:
        got = failure.result.title if failure.result is not None else "No match"
        score = str(failure.result.score) if failure.result is not None else "-"

        failure_table.add_row(
            failure.case.title,
            got,
            f"{failure.case.duration_sec}s",
            f"{failure.case.start_sec:.2f}s",
            score,
        )

    console.print(failure_table)

    if len(failures) > 20:
        console.print(f"[yellow]Showing 20 of {len(failures)} failed cases.[/yellow]")
