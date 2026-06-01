from pathlib import Path

import librosa
import numpy as np


def load_audio(path: Path, sample_rate: int = 11_025) -> np.ndarray:
    audio, _ = librosa.load(
        path,
        sr=sample_rate,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    return audio.astype(np.float32)