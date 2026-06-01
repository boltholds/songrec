from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from songrec.lyrics.normalizer import normalize_lyrics_text


class AsrError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    audio_path: Path
    model_name: str
    language: str | None
    text: str
    normalized_text: str
    segments: list[TranscriptionSegment]
    latency_ms: float


class FasterWhisperAsr:
    """
    Thin optional wrapper around faster-whisper.

    faster-whisper is deliberately not imported at module import time. This keeps
    the rest of SongRec usable on machines where the ASR stack is not installed.
    """

    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AsrError(
                "faster-whisper is not installed. Install it with: "
                "poetry run python -m pip install faster-whisper"
            ) from exc

        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:  # noqa: BLE001 - preserve original backend error text
            raise AsrError(f"Failed to load faster-whisper model '{self.model_name}': {exc}") from exc

        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise AsrError(f"Audio file does not exist: {audio_path}")

        model = self._load_model()
        started_at = perf_counter()

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=self.beam_size,
                vad_filter=vad_filter,
            )
            segments = [
                TranscriptionSegment(
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    text=segment.text.strip(),
                )
                for segment in segments_iter
            ]
        except Exception as exc:  # noqa: BLE001
            raise AsrError(f"ASR transcription failed: {exc}") from exc

        text = " ".join(segment.text for segment in segments).strip()
        latency_ms = (perf_counter() - started_at) * 1000
        detected_language = getattr(info, "language", None) or language

        return TranscriptionResult(
            audio_path=audio_path,
            model_name=self.model_name,
            language=detected_language,
            text=text,
            normalized_text=normalize_lyrics_text(text),
            segments=segments,
            latency_ms=latency_ms,
        )
