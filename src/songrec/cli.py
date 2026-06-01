from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from songrec.db.session import create_session_factory, drop_database
from songrec.graphs.recognize_graph import build_recognition_graph
from songrec.indexer import index_directory



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


@app.command()
def recognize(
    audio_path: Path,
    db_path: Path = Path("songrec.sqlite3"),
) -> None:
    """
    Recognize a song from an audio fragment.
    """
    graph = build_recognition_graph()

    state = graph.invoke(
        {
            "audio_path": audio_path,
            "db_path": db_path,
        }
    )

    if state.get("error"):
        console.print(f"[red]{state['error']}[/red]")
        return

    result = state.get("result")

    if result is None:
        console.print("[red]No match found.[/red]")
        return

    table = Table(title="Recognition result")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Track", result.title)
    table.add_row("Track ID", str(result.track_id))
    table.add_row("Score", str(result.score))
    table.add_row("Confidence", f"{result.confidence:.4f}")
    table.add_row("Offset", f"{result.offset_ms / 1000:.2f} sec")

    console.print(table)

@app.command("reset-db")
def reset_db(
    db_path: Path = Path("songrec.sqlite3"),
) -> None:
    """
    Delete local fingerprint database.
    """
    drop_database(db_path)
    console.print(f"[green]Deleted database:[/green] {db_path}")
    

if __name__ == "__main__":
    app()