from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from songrec.benchmark import parse_modes, print_benchmark_report, run_benchmark
from songrec.recognition.speed import parse_speed_factors
from songrec.matcher import MatchDecision
from songrec.db.session import create_session_factory, drop_database
from songrec.graphs.recognize_graph import build_recognition_graph
from songrec.indexer import index_directory
from songrec.separation import DemucsSeparator, SeparationError

app = typer.Typer()
console = Console()


@app.command()
def index(
    music_dir: Path,
    db_path: Path = Path("songrec.sqlite3"),
) -> None:
    """
    Index all audio files in a directory.
    """
    if not music_dir.exists():
        raise typer.BadParameter(f"Directory does not exist: {music_dir}")

    session_factory = create_session_factory(db_path)

    with session_factory() as session:
        index_directory(music_dir, session)


@app.command("reset-db")
def reset_db(
    db_path: Path = Path("songrec.sqlite3"),
) -> None:
    """
    Delete local fingerprint database.
    """
    drop_database(db_path)
    console.print(f"[green]Deleted database:[/green] {db_path}")


@app.command()
def benchmark(
    music_dir: Path,
    db_path: Path = Path("songrec.sqlite3"),
    durations: str = "3,5,10,15",
    samples_per_track: int = 3,
    modes: str = "clean",
    min_score: int = 20,
    min_confidence: float = 0.005,
    min_margin: float = 1.5,
    recognition_mode: str = typer.Option("fast", "--recognition-mode", help="fast, multi_speed, or scale_aware"),
    speed_factors: str = typer.Option("0.90,0.95,1.0,1.05,1.10", help="Comma-separated factors for multi_speed"),
    threshold_sweep: bool = typer.Option(False, "--threshold-sweep", help="Show approximate threshold calibration table"),
) -> None:
    """
    Cut fragments from indexed tracks and measure recognition accuracy.

    Modes: clean,noise,volume,speed-0.95,speed-1.05,speed-1.10,negative
    Recognition modes: fast,multi_speed,scale_aware
    """
    if not music_dir.exists():
        raise typer.BadParameter(f"Directory does not exist: {music_dir}")

    if not db_path.exists():
        raise typer.BadParameter(f"Database does not exist: {db_path}")

    durations_sec = [
        int(value.strip())
        for value in durations.split(",")
        if value.strip()
    ]

    if not durations_sec:
        raise typer.BadParameter("Durations list is empty")

    session_factory = create_session_factory(db_path)

    with session_factory() as session:
        results = run_benchmark(
            music_dir=music_dir,
            session=session,
            durations_sec=durations_sec,
            samples_per_track=samples_per_track,
            modes=parse_modes(modes),
            decision=MatchDecision(
                min_score=min_score,
                min_confidence=min_confidence,
                min_margin=min_margin,
            ),
            recognition_mode=recognition_mode,
            speed_factors=parse_speed_factors(speed_factors),
        )

    print_benchmark_report(results, show_threshold_sweep=threshold_sweep)


@app.command()
def recognize(
    audio_path: Path,
    db_path: Path = Path("songrec.sqlite3"),
    mode: str = typer.Option("fast", "--mode", help="fast, multi_speed, or scale_aware"),
    speed_factors: str = typer.Option("0.90,0.95,1.0,1.05,1.10", help="Comma-separated factors for multi_speed"),
) -> None:
    """
    Recognize a song from an audio fragment.
    """
    graph = build_recognition_graph()

    state = graph.invoke(
        {
            "audio_path": audio_path,
            "db_path": db_path,
            "mode": mode,
            "speed_factors": parse_speed_factors(speed_factors),
        }
    )

    if state.get("error"):
        console.print(f"[red]{state['error']}[/red]")
        return

    result = state.get("result")

    if result is None:
        console.print("[red]No match found.[/red]")
        return

    if not result.is_confident:
        console.print("[yellow]Low-confidence match. Treat as no confident match.[/yellow]")
        if result.reject_reason:
            console.print(f"[yellow]Reason:[/yellow] {result.reject_reason}")

    table = Table(title="Recognition result")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Track", result.title)
    table.add_row("Track ID", str(result.track_id))
    table.add_row("Score", str(result.score))
    table.add_row("Confidence", f"{result.confidence:.4f}")
    table.add_row("Second score", str(result.second_score))
    table.add_row("Margin", f"{result.margin:.2f}x")
    table.add_row("Confident", str(result.is_confident))
    table.add_row("Recognition mode", result.mode)
    table.add_row("Speed factor", f"{result.speed_factor:.2f}")
    table.add_row("Offset", f"{result.offset_ms / 1000:.2f} sec")

    console.print(table)


@app.command()
def separate(
    audio_path: Path,
    output_dir: Path = Path("songrec_storage/stems"),
    model_name: str = typer.Option("htdemucs", "--model"),
    device: str | None = typer.Option(None, "--device", help="Demucs device, for example cuda or cpu"),
    jobs: int | None = typer.Option(None, "--jobs", help="Demucs worker count"),
) -> None:
    """
    Split an audio file into vocals.wav and no_vocals.wav with Demucs.
    """
    if not audio_path.exists():
        raise typer.BadParameter(f"Audio file does not exist: {audio_path}")

    separator = DemucsSeparator(
        output_dir=output_dir,
        model_name=model_name,
        device=device,
        jobs=jobs,
    )

    try:
        result = separator.separate(audio_path)
    except SeparationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Separation result")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Input", str(result.input_path))
    table.add_row("Model", result.model_name)
    table.add_row("Vocals", str(result.vocals_path))
    table.add_row("Instrumental", str(result.instrumental_path))
    table.add_row("Output dir", str(result.output_dir))
    console.print(table)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """
    Run SongRec FastAPI service.
    """
    import uvicorn

    uvicorn.run(
        "songrec.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()