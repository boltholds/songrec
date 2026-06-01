
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

MVP-3.2 completed:
- implemented scale-aware matching without audio time-stretch
- supports sped-up/slowed tracks through time-scale voting
- achieved 100% raw top-1 accuracy on clean/speed/negative benchmark
- achieved 99.49% calibrated policy accuracy
- kept wrong accepted = 0 and false positives = 0
- reduced avg latency from ~589 ms multi_speed to ~471 ms scale_aware