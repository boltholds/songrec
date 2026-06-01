
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
