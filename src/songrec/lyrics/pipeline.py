from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from songrec.lyrics.asr import FasterWhisperAsr, TranscriptionResult
from songrec.separation import DemucsSeparator


@dataclass(frozen=True)
class LyricsRecognitionResult:
    source_audio_path: Path
    asr_audio_path: Path
    separated: bool
    vocals_path: Path | None
    instrumental_path: Path | None
    transcription: TranscriptionResult
    latency_ms: float


class LyricsRecognitionPipeline:
    def __init__(
        self,
        asr: FasterWhisperAsr,
        separator: DemucsSeparator | None = None,
    ) -> None:
        self.asr = asr
        self.separator = separator

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        separate_vocals: bool = False,
    ) -> LyricsRecognitionResult:
        started_at = perf_counter()
        asr_audio_path = audio_path
        vocals_path: Path | None = None
        instrumental_path: Path | None = None

        if separate_vocals:
            if self.separator is None:
                raise ValueError("separate_vocals=True requires a separator")

            separation = self.separator.separate(audio_path)
            vocals_path = separation.vocals_path
            instrumental_path = separation.instrumental_path
            asr_audio_path = vocals_path

        transcription = self.asr.transcribe(asr_audio_path, language=language)

        return LyricsRecognitionResult(
            source_audio_path=audio_path,
            asr_audio_path=asr_audio_path,
            separated=separate_vocals,
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
            transcription=transcription,
            latency_ms=(perf_counter() - started_at) * 1000,
        )
