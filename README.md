
## MVP-1.1: robustness benchmark

The benchmark command supports several modes:

```bash
poetry run songrec benchmark music --modes clean
poetry run songrec benchmark music --modes clean,noise,volume,negative
poetry run songrec benchmark music --modes clean,speed-0.95,speed-1.05,speed-1.10
```

Available modes:

- `clean` — original in-library fragments
- `noise` — white-noise augmented fragments
- `volume` — quieter fragments
- `speed-0.95` — slowed fragment
- `speed-1.05` — slightly sped-up fragment
- `speed-1.10` — sped-up fragment
- `negative` — synthetic out-of-library noise, expected to be rejected

The matcher now reports:

- `score`
- `second_score`
- `margin`
- `confidence`
- `is_confident`
- `reject_reason`

Thresholds can be tuned from CLI:

```bash
poetry run songrec benchmark music \
  --modes clean,noise,volume,negative \
  --min-score 20 \
  --min-confidence 0.005 \
  --min-margin 1.5
```

## MVP-2: FastAPI service

Install/update dependencies after unpacking this archive:

```bash
poetry install
```

Run the API:

```bash
poetry run songrec serve --reload
```

Alternative entrypoint:

```bash
poetry run songrec-api --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Environment variables:

```bash
SONGREC_DB_PATH=songrec.sqlite3
SONGREC_STORAGE_DIR=songrec_storage
```

Endpoints:

```text
GET  /health
GET  /stats
GET  /tracks
POST /tracks
POST /recognize
```

Upload a track into the fingerprint database:

```bash
curl -X POST "http://127.0.0.1:8000/tracks" \
  -F "file=@music/Daughter - Youth.mp3"
```

Recognize an audio fragment:

```bash
curl -X POST "http://127.0.0.1:8000/recognize" \
  -F "file=@queries/youth_10s.mp3"
```

Example recognition response:

```json
{
  "status": "matched",
  "result": {
    "track_id": 1,
    "title": "Daughter - Youth",
    "score": 842,
    "second_score": 140,
    "margin": 6.01,
    "confidence": 0.043,
    "offset_sec": 61.2,
    "is_confident": true,
    "reject_reason": null
  },
  "latency_ms": 104.8
}
```

### Track CRUD endpoints

MVP-2.1 adds track detail and delete endpoints:

```bash
curl "http://127.0.0.1:8000/tracks/6"
```

```bash
curl -X DELETE "http://127.0.0.1:8000/tracks/24"
```

`DELETE /tracks/{track_id}` removes the track row, its fingerprints, and the uploaded audio file from `songrec_storage/tracks` when the file exists.

## MVP-3: sped-up / slowed recognition

MVP-3 adds a `multi_speed` recognition mode. The default `fast` mode still runs one normal fingerprint pass. The new mode tries several time-stretched versions of the query audio and chooses the best candidate.

CLI recognition:

```bash
poetry run songrec recognize queries/youth_10s_sped.mp3 --mode multi_speed
```

Custom speed factors:

```bash
poetry run songrec recognize queries/youth_10s_sped.mp3 \
  --mode multi_speed \
  --speed-factors 0.90,0.95,1.0,1.05,1.10
```

API recognition:

```bash
curl -X POST "http://127.0.0.1:8000/recognize?mode=multi_speed" \
  -F "file=@queries/youth_10s_sped.mp3"
```

API with custom factors:

```bash
curl -X POST "http://127.0.0.1:8000/recognize?mode=multi_speed&speed_factors=0.90,0.95,1.0,1.05,1.10" \
  -F "file=@queries/youth_10s_sped.mp3"
```

Benchmark fast mode:

```bash
poetry run songrec benchmark music \
  --modes clean,speed-0.95,speed-1.05,speed-1.10 \
  --recognition-mode fast
```

Benchmark multi-speed mode:

```bash
poetry run songrec benchmark music \
  --modes clean,speed-0.95,speed-1.05,speed-1.10 \
  --recognition-mode multi_speed
```

The recognition response now includes:

```json
{
  "speed_factor": 0.95,
  "mode": "multi_speed"
}
```

`speed_factor` is the transformation applied to the query before fingerprinting. For example, a query that was sped up by `1.05x` is often recovered by applying a factor close to `0.95`.

## MVP-3.1: benchmark reporting and threshold calibration

MVP-3.1 separates recognition quality from confidence-gate policy.

The benchmark now reports:

- `Policy accuracy` — correct final behavior after confidence thresholds
- `Raw top-1 accuracy` — whether the best candidate was the expected track before rejection
- `Correct rejected` — expected track was found but rejected by thresholds
- `Wrong accepted` — dangerous wrong confident answer
- `Wrong rejected` — wrong candidate found but safely rejected
- `False positives` — confident match for negative/out-of-library cases

Run the extended benchmark:

```bash
poetry run songrec benchmark music \
  --modes clean,speed-0.95,speed-1.05,speed-1.10 \
  --recognition-mode multi_speed
```

Show an approximate threshold calibration table:

```bash
poetry run songrec benchmark music \
  --modes clean,speed-0.95,speed-1.05,speed-1.10,negative \
  --recognition-mode multi_speed \
  --threshold-sweep
```

Try a softer policy after checking that false positives remain zero:

```bash
poetry run songrec benchmark music \
  --modes clean,speed-0.95,speed-1.05,speed-1.10,negative \
  --recognition-mode multi_speed \
  --min-score 15 \
  --min-confidence 0.003 \
  --min-margin 1.20
```

The threshold sweep is calculated from already selected best matches. For `multi_speed` it is an approximation, but it is useful for choosing a safer starting policy.

## MVP-3.2: scale-aware recognition

MVP-3.2 adds `scale_aware` recognition mode. Unlike `multi_speed`, it does not physically time-stretch the query audio for every factor. It detects the peak constellation once, then hashes query landmarks with scaled `delta_time` and scaled offsets:

```text
track_time ≈ query_time * scale + offset
```

This keeps the original fingerprint index compatible, but makes sped-up / slowed matching much cheaper than full audio resampling.

CLI:

```bash
poetry run songrec recognize queries/youth_10s_sped.mp3 --mode scale_aware
```

API:

```bash
curl -X POST "http://127.0.0.1:8000/recognize?mode=scale_aware" \
  -F "file=@queries/youth_10s_sped.mp3"
```

Benchmark:

```bash
poetry run songrec benchmark music \
  --modes clean,speed-0.95,speed-1.05,speed-1.10,negative \
  --recognition-mode scale_aware \
  --threshold-sweep
```

Compare against the legacy resampling mode:

```bash
poetry run songrec benchmark music \
  --modes clean,speed-0.95,speed-1.05,speed-1.10,negative \
  --recognition-mode multi_speed
```

`multi_speed` is kept as a baseline. `scale_aware` is the preferred MVP-3.2 mode.

## MVP-4: source separation

MVP-4 adds a Demucs-based separation layer. It splits an input track into two stems:

```text
vocals.wav
no_vocals.wav
```

This is the foundation for the next stages: lyrics ASR from vocals and instrumental-only matching from `no_vocals`.

Demucs is a heavy optional dependency. The rest of SongRec still works without it. Install Demucs into the Poetry environment only when you want to use separation:

```bash
poetry run python -m pip install demucs
```

CLI separation:

```bash
poetry run songrec separate "music/Daughter - Youth.mp3"
```

With explicit device:

```bash
poetry run songrec separate "music/Daughter - Youth.mp3" --device cuda
```

API separation:

```bash
curl -X POST "http://127.0.0.1:8000/separate" \
  -F "file=@music/Daughter - Youth.mp3"
```

Windows `cmd` multiline form:

```cmd
curl -X POST "http://127.0.0.1:8000/separate" ^
  -F "file=@music/Daughter - Youth.mp3"
```

Example response:

```json
{
  "input_filename": "Daughter - Youth.mp3",
  "model_name": "htdemucs",
  "mode": "demucs_two_stems",
  "vocals_path": "songrec_storage/stems/.../vocals.wav",
  "instrumental_path": "songrec_storage/stems/.../no_vocals.wav",
  "output_dir": "songrec_storage/stems/...",
  "latency_ms": 12345.6
}
```

The separation layer is intentionally independent from fingerprinting. Current recognition modes remain unchanged:

```text
fast
multi_speed
scale_aware
```

Next planned step: use `vocals.wav` for Whisper/faster-whisper lyrics recognition and `no_vocals.wav` for instrumental fingerprints / chroma features.

## MVP-5: Lyrics ASR

MVP-5 adds an optional lyrics transcription layer. It can transcribe an audio fragment directly, or run Demucs first and transcribe the isolated `vocals.wav` stem.

The ASR backend is optional, so the rest of SongRec still works without it. Install it when you want lyrics recognition:

```bash
poetry run python -m pip install faster-whisper
```

For the full vocals pipeline, install Demucs too:

```bash
poetry run python -m pip install demucs
```

Recommended Windows setup for ML/audio dependencies is Python 3.11 or 3.12. Python 3.13 may cause binary compatibility issues with torch/torchaudio/Demucs.

### CLI

Transcribe a vocals file or any audio fragment:

```bash
poetry run songrec transcribe "songrec_storage/stems/htdemucs/Daughter - Youth/vocals.wav" --model small --language en
```

Run source separation first, then transcribe vocals:

```bash
poetry run songrec transcribe "music/Daughter - Youth.mp3" --separate-vocals --model small --language en
```

Use GPU if your faster-whisper installation supports it:

```bash
poetry run songrec transcribe "music/Daughter - Youth.mp3" --separate-vocals --device cuda --compute-type float16
```

### API

Transcribe uploaded audio directly:

```bash
curl -X POST "http://127.0.0.1:8000/lyrics/transcribe" ^
  -F "file=@queries/youth_10s.mp3"
```

Run Demucs first and transcribe vocals:

```bash
curl -X POST "http://127.0.0.1:8000/lyrics/transcribe?separate_vocals=true&language=en" ^
  -F "file=@music/Daughter - Youth.mp3"
```

Example response:

```json
{
  "input_filename": "youth_10s.mp3",
  "model_name": "small",
  "language": "en",
  "separated": false,
  "asr_audio_path": "songrec_storage\\queries\\..._youth_10s.mp3",
  "vocals_path": null,
  "instrumental_path": null,
  "text": "...",
  "normalized_text": "...",
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 4.2,
      "text": "..."
    }
  ],
  "latency_ms": 1234.5
}
```

### Architecture

```text
songrec/lyrics/
  asr.py          optional faster-whisper wrapper
  normalizer.py   text cleanup for future lyrics search
  pipeline.py     optional Demucs -> vocals -> ASR workflow
```

MVP-5 does not yet search a lyrics database. It prepares the ASR and normalization layer. The next step is lyrics indexing/search: store lyrics per track, split into lines/chunks, and match normalized ASR output against that index.
