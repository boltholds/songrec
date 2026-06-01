from __future__ import annotations

from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from songrec.api.dependencies import ApiSettings, get_settings
from songrec.api.routes.tracks import ensure_supported_audio, safe_filename, save_upload
from songrec.api.schemas import RecognitionResultResponse, RecognizeResponse
from songrec.graphs.recognize_graph import build_recognition_graph
from songrec.recognition.speed import parse_speed_factors

router = APIRouter(tags=["recognition"])


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(...),
    mode: str = Query("fast", pattern="^(fast|multi_speed|scale_aware)$"),
    speed_factors: str = Query("0.90,0.95,1.0,1.05,1.10"),
    settings: ApiSettings = Depends(get_settings),
) -> RecognizeResponse:
    filename = safe_filename(file.filename or "query.wav")
    ensure_supported_audio(filename)

    destination = settings.queries_dir / f"{uuid4().hex}_{filename}"
    await save_upload(file, destination)

    started_at = perf_counter()

    try:
        graph = build_recognition_graph()
        state = graph.invoke(
            {
                "audio_path": destination,
                "db_path": settings.db_path,
                "mode": mode,
                "speed_factors": parse_speed_factors(speed_factors),
            }
        )
    finally:
        destination.unlink(missing_ok=True)

    latency_ms = (perf_counter() - started_at) * 1000

    if state.get("error"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=state["error"],
        )

    result = state.get("result")

    if result is None:
        return RecognizeResponse(
            status="not_found",
            result=None,
            latency_ms=latency_ms,
        )

    response_result = RecognitionResultResponse(
        track_id=result.track_id,
        title=result.title,
        score=result.score,
        second_score=result.second_score,
        margin=result.margin,
        confidence=result.confidence,
        offset_sec=result.offset_ms / 1000,
        is_confident=result.is_confident,
        reject_reason=result.reject_reason,
        speed_factor=result.speed_factor,
        mode=result.mode,
    )

    return RecognizeResponse(
        status="matched" if result.is_confident else "rejected",
        result=response_result,
        latency_ms=latency_ms,
    )
