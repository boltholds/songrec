from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeparationResult:
    input_path: Path
    vocals_path: Path
    instrumental_path: Path
    output_dir: Path
    model_name: str
    mode: str = "demucs_two_stems"


class SeparationError(RuntimeError):
    pass


class SourceSeparator:
    def separate(self, audio_path: Path) -> SeparationResult:
        raise NotImplementedError
