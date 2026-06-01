from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from songrec.api.schemas import HealthResponse, StatsResponse
from songrec.db.repositories import FingerprintRepository, TrackRepository
from songrec.api.dependencies import get_session

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/stats", response_model=StatsResponse)
def stats(session: Session = Depends(get_session)) -> StatsResponse:
    track_repo = TrackRepository(session)
    fingerprint_repo = FingerprintRepository(session)

    return StatsResponse(
        tracks_count=track_repo.count_tracks(),
        fingerprints_count=fingerprint_repo.count_fingerprints(),
    )
