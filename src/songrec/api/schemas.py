from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "songrec"


class TrackRead(BaseModel):
    id: int
    title: str
    path: str


class TrackDetailResponse(BaseModel):
    id: int
    title: str
    path: str
    fingerprints_count: int


class TrackDeleteResponse(BaseModel):
    deleted: bool
    id: int
    title: str
    path: str
    fingerprints_deleted: int
    file_deleted: bool


class TrackListResponse(BaseModel):
    tracks: list[TrackRead]
    total: int


class TrackUploadResponse(BaseModel):
    id: int
    title: str
    path: str
    fingerprints_count: int


class StatsResponse(BaseModel):
    tracks_count: int
    fingerprints_count: int


class RecognitionResultResponse(BaseModel):
    track_id: int
    title: str
    score: int
    second_score: int
    margin: float
    confidence: float
    offset_sec: float
    is_confident: bool
    reject_reason: str | None = None
    speed_factor: float = 1.0
    mode: str = "fast"


class RecognizeResponse(BaseModel):
    status: str = Field(description="matched, rejected, or not_found")
    result: RecognitionResultResponse | None
    latency_ms: float


class SeparationResponse(BaseModel):
    input_filename: str
    model_name: str
    mode: str
    vocals_path: str
    instrumental_path: str
    output_dir: str
    latency_ms: float
