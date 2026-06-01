from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from songrec.api.dependencies import ApiSettings, get_settings
from songrec.api.routes.tracks import ensure_supported_audio, safe_filename, save_upload
from songrec.api.schemas import SeparationResponse
from songrec.separation import DemucsSeparator, SeparationError

router = APIRouter(tags=["separation"])


@router.post("/separate", response_model=SeparationResponse)
async def separate_audio(
    file: UploadFile = File(...),
    model_name: str = Query("htdemucs"),
    device: str | None = Query(None, description="Demucs device, for example cuda or cpu"),
    jobs: int | None = Query(None, ge=1, le=16),
    settings: ApiSettings = Depends(get_settings),
) -> SeparationResponse:
    filename = safe_filename(file.filename or "query.wav")
    ensure_supported_audio(filename)

    upload_path = settings.queries_dir / f"{uuid4().hex}_{filename}"
    await save_upload(file, upload_path)

    request_output_dir = settings.stems_dir / upload_path.stem
    separator = DemucsSeparator(
        output_dir=request_output_dir,
        model_name=model_name,
        device=device,
        jobs=jobs,
    )

    started_at = perf_counter()
    try:
        result = separator.separate(upload_path)
    except SeparationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        upload_path.unlink(missing_ok=True)

    latency_ms = (perf_counter() - started_at) * 1000

    return SeparationResponse(
        input_filename=filename,
        model_name=result.model_name,
        mode=result.mode,
        vocals_path=str(result.vocals_path),
        instrumental_path=str(result.instrumental_path),
        output_dir=str(result.output_dir),
        latency_ms=latency_ms,
    )
