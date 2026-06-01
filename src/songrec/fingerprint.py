from dataclasses import dataclass
from hashlib import sha1

import librosa
import numpy as np
from scipy.ndimage import maximum_filter


@dataclass(frozen=True)
class Fingerprint:
    hash: str
    offset_ms: int


def find_peaks(
    audio: np.ndarray,
    sample_rate: int = 11_025,
    n_fft: int = 2048,
    hop_length: int = 512,
    neighborhood_size: int = 20,
    amp_min_db: float = -45.0,
) -> list[tuple[int, int]]:
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    spectrogram = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max,
    )

    local_max = maximum_filter(
        spectrogram,
        size=neighborhood_size,
    ) == spectrogram

    detected_peaks = local_max & (spectrogram > amp_min_db)

    freq_bins, time_bins = np.where(detected_peaks)

    peaks = list(zip(time_bins.tolist(), freq_bins.tolist()))
    peaks.sort(key=lambda item: item[0])

    return peaks


def generate_hashes(
    peaks: list[tuple[int, int]],
    sample_rate: int = 11_025,
    hop_length: int = 512,
    fan_value: int = 15,
    min_delta_frames: int = 1,
    max_delta_frames: int = 80,
    time_scale: float = 1.0,
) -> list[Fingerprint]:
    """
    Generate Shazam-like landmark hashes.

    time_scale makes matching speed-aware without physically stretching audio.
    If a query was sped up by 1.10x, its peak distances are compressed;
    hashing with time_scale=1.10 maps query deltas and offsets back toward
    the original track time axis.
    """
    fingerprints: list[Fingerprint] = []

    for index, anchor in enumerate(peaks):
        anchor_time, anchor_freq = anchor
        targets = peaks[index + 1 : index + 1 + fan_value]

        for target_time, target_freq in targets:
            raw_delta_time = target_time - anchor_time
            scaled_delta_time = int(round(raw_delta_time * time_scale))

            if scaled_delta_time < min_delta_frames:
                continue

            if scaled_delta_time > max_delta_frames:
                continue

            raw_hash = f"{anchor_freq}|{target_freq}|{scaled_delta_time}".encode()
            fingerprint_hash = sha1(raw_hash).hexdigest()[:20]

            offset_sec = librosa.frames_to_time(
                anchor_time,
                sr=sample_rate,
                hop_length=hop_length,
            )

            fingerprints.append(
                Fingerprint(
                    hash=fingerprint_hash,
                    offset_ms=int(offset_sec * 1000 * time_scale),
                )
            )

    return fingerprints


def fingerprint_audio(
    audio: np.ndarray,
    time_scale: float = 1.0,
) -> list[Fingerprint]:
    peaks = find_peaks(audio)
    return generate_hashes(peaks, time_scale=time_scale)


def fingerprint_peaks(
    peaks: list[tuple[int, int]],
    time_scale: float = 1.0,
) -> list[Fingerprint]:
    return generate_hashes(peaks, time_scale=time_scale)
