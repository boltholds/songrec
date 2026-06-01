from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from songrec.api.dependencies import ApiSettings, get_settings
from songrec.api.routes.tracks import ensure_supported_audio, safe_filename, save_upload
from songrec.api.schemas import LyricsTranscriptionResponse, TranscriptionSegmentResponse
from songrec.lyrics import AsrError, FasterWhisperAsr
from songrec.lyrics.pipeline import LyricsRecognitionPipeline
from songrec.separation import DemucsSeparator, SeparationError

router = APIRouter(prefix="/lyrics", tags=["lyrics"])


@router.post("/transcribe", response_model=LyricsTranscriptionResponse)
async def transcribe_lyrics(
    file: UploadFile = File(...),
    model_name: str = Query("small", description="faster-whisper model name"),
    language: str | None = Query(None, description="Optional language code, for example en or ru"),
    device: str = Query("auto", description="faster-whisper device: auto, cpu, cuda"),
    compute_type: str = Query("default", description="faster-whisper compute type"),
    beam_size: int = Query(5, ge=1, le=10),
    separate_vocals: bool = Query(False, description="Run Demucs first and transcribe vocals.wav"),
    demucs_model: str = Query("htdemucs"),
    demucs_device: str | None = Query(None, description="Demucs device, for example cuda or cpu"),
    settings: ApiSettings = Depends(get_settings),
) -> LyricsTranscriptionResponse:
    filename = safe_filename(file.filename or "query.wav")
    ensure_supported_audio(filename)

    upload_path = settings.queries_dir / f"{uuid4().hex}_{filename}"
    await save_upload(file, upload_path)

    separator = None
    if separate_vocals:
        separator = DemucsSeparator(
            output_dir=settings.stems_dir / upload_path.stem,
            model_name=demucs_model,
            device=demucs_device,
        )

    pipeline = LyricsRecognitionPipeline(
        asr=FasterWhisperAsr(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
        ),
        separator=separator,
    )

    try:
        result = pipeline.transcribe(
            upload_path,
            language=language,
            separate_vocals=separate_vocals,
        )
    except (AsrError, SeparationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        upload_path.unlink(missing_ok=True)

    transcription = result.transcription
    return LyricsTranscriptionResponse(
        input_filename=filename,
        model_name=transcription.model_name,
        language=transcription.language,
        separated=result.separated,
        asr_audio_path=str(result.asr_audio_path),
        vocals_path=str(result.vocals_path) if result.vocals_path else None,
        instrumental_path=str(result.instrumental_path) if result.instrumental_path else None,
        text=transcription.text,
        normalized_text=transcription.normalized_text,
        segments=[
            TranscriptionSegmentResponse(
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                text=segment.text,
            )
            for segment in transcription.segments
        ],
        latency_ms=result.latency_ms,
    )
