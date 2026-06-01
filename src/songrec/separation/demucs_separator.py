from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from songrec.separation.base import SeparationError, SeparationResult, SourceSeparator


class DemucsSeparator(SourceSeparator):
    """Demucs wrapper for vocals / no_vocals source separation.

    Demucs is intentionally used through ``python -m demucs`` so the core
    SongRec package can still work without importing heavy torch dependencies.
    """

    def __init__(
        self,
        output_dir: Path,
        model_name: str = "htdemucs",
        python_executable: str | None = None,
        device: str | None = None,
        jobs: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.model_name = model_name
        self.python_executable = python_executable or sys.executable
        self.device = device
        self.jobs = jobs

    def separate(self, audio_path: Path) -> SeparationResult:
        if not audio_path.exists():
            raise SeparationError(f"Audio file does not exist: {audio_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            self.python_executable,
            "-m",
            "demucs",
            "-n",
            self.model_name,
            "--two-stems",
            "vocals",
            "-o",
            str(self.output_dir),
        ]

        if self.device:
            command.extend(["--device", self.device])

        if self.jobs is not None:
            command.extend(["-j", str(self.jobs)])

        command.append(str(audio_path))

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise SeparationError(
                "Python executable for Demucs was not found. "
                "Check python_executable or run inside the Poetry environment."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or str(exc)

            if "No module named demucs" in details:
                details = (
                    "Demucs is not installed in this environment. "
                    "Install it with: poetry run python -m pip install demucs"
                )

            raise SeparationError(f"Demucs separation failed: {details}") from exc

        stem_dir = self.output_dir / self.model_name / audio_path.stem
        vocals_path = stem_dir / "vocals.wav"
        instrumental_path = stem_dir / "no_vocals.wav"

        if not vocals_path.exists() or not instrumental_path.exists():
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            raise SeparationError(
                "Demucs finished, but expected stems were not found. "
                f"Expected: {vocals_path} and {instrumental_path}. "
                f"stdout={stdout!r} stderr={stderr!r}"
            )

        return SeparationResult(
            input_path=audio_path,
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
            output_dir=stem_dir,
            model_name=self.model_name,
        )
