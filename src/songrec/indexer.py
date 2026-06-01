from pathlib import Path

from rich.console import Console
from sqlalchemy.orm import Session

from songrec.audio import load_audio
from songrec.db.repositories import FingerprintRepository, TrackRepository
from songrec.fingerprint import fingerprint_audio

console = Console()

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def iter_audio_files(directory: Path):
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def index_directory(directory: Path, session: Session) -> None:
    files = list(iter_audio_files(directory))

    if not files:
        console.print("[yellow]No audio files found.[/yellow]")
        return

    track_repo = TrackRepository(session)
    fingerprint_repo = FingerprintRepository(session)

    for path in files:
        console.print(f"[cyan]Indexing:[/cyan] {path}")

        audio = load_audio(path)
        fingerprints = fingerprint_audio(audio)

        track_id = track_repo.add_track(
            path=path,
            title=path.stem,
        )
        fingerprint_repo.delete_by_track_id(track_id)
        
        fingerprint_repo.add_fingerprints(
            track_id=track_id,
            fingerprints=fingerprints,
        )

        session.commit()

        console.print(
            f"[green]OK[/green] {path.name}: {len(fingerprints)} fingerprints"
        )