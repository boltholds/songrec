from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from songrec.api.dependencies import ApiSettings, get_session, get_settings
from songrec.api.schemas import (
    TrackDeleteResponse,
    TrackDetailResponse,
    TrackListResponse,
    TrackRead,
    TrackUploadResponse,
)
from songrec.audio import load_audio
from songrec.db.repositories import FingerprintRepository, TrackRepository
from songrec.fingerprint import fingerprint_audio

router = APIRouter(prefix="/tracks", tags=["tracks"])

SUPPORTED_UPLOAD_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "track"
    name = re.sub(r"[^A-Za-zА-Яа-я0-9._()\- ]+", "_", name)
    return name[:180]


def ensure_supported_audio(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio extension {suffix!r}. Supported: {supported}",
        )


async def save_upload(file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    destination.write_bytes(content)


@router.get("", response_model=TrackListResponse)
def list_tracks(session: Session = Depends(get_session)) -> TrackListResponse:
    repo = TrackRepository(session)
    tracks = repo.list_tracks()

    return TrackListResponse(
        total=len(tracks),
        tracks=[
            TrackRead(id=track.id, title=track.title, path=track.path)
            for track in tracks
        ],
    )


@router.post("", response_model=TrackUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_track(
    file: UploadFile = File(...),
    title: str | None = None,
    session: Session = Depends(get_session),
    settings: ApiSettings = Depends(get_settings),
) -> TrackUploadResponse:
    filename = safe_filename(file.filename or "track")
    ensure_supported_audio(filename)

    destination = settings.tracks_dir / filename
    await save_upload(file, destination)

    try:
        audio = load_audio(destination)
        fingerprints = fingerprint_audio(audio)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not process audio file: {exc}",
        ) from exc

    track_repo = TrackRepository(session)
    fingerprint_repo = FingerprintRepository(session)

    track_id = track_repo.add_track(
        path=destination,
        title=title or Path(filename).stem,
    )
    fingerprint_repo.delete_by_track_id(track_id)
    fingerprint_repo.add_fingerprints(track_id=track_id, fingerprints=fingerprints)
    session.commit()

    return TrackUploadResponse(
        id=track_id,
        title=title or Path(filename).stem,
        path=str(destination),
        fingerprints_count=len(fingerprints),
    )


@router.get("/{track_id}", response_model=TrackDetailResponse)
def get_track(
    track_id: int,
    session: Session = Depends(get_session),
) -> TrackDetailResponse:
    track_repo = TrackRepository(session)
    fingerprint_repo = FingerprintRepository(session)

    track = track_repo.get_track(track_id)
    if track is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track {track_id} not found",
        )

    return TrackDetailResponse(
        id=track.id,
        title=track.title,
        path=track.path,
        fingerprints_count=fingerprint_repo.count_by_track_id(track.id),
    )


@router.delete("/{track_id}", response_model=TrackDeleteResponse)
def delete_track(
    track_id: int,
    session: Session = Depends(get_session),
) -> TrackDeleteResponse:
    track_repo = TrackRepository(session)
    fingerprint_repo = FingerprintRepository(session)

    track = track_repo.get_track(track_id)
    if track is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track {track_id} not found",
        )

    title = track.title
    path_value = track.path
    fingerprints_deleted = fingerprint_repo.count_by_track_id(track.id)

    fingerprint_repo.delete_by_track_id(track.id)
    track_repo.delete_track(track)
    session.commit()

    file_path = Path(path_value)
    file_existed = file_path.exists()
    if file_existed:
        file_path.unlink(missing_ok=True)

    return TrackDeleteResponse(
        deleted=True,
        id=track_id,
        title=title,
        path=path_value,
        fingerprints_deleted=fingerprints_deleted,
        file_deleted=file_existed,
    )
