MVP-1 status: completed.

Implemented:
- local music library indexing
- spectrogram peak fingerprinting
- SQLAlchemy-backed fingerprint storage
- LangGraph recognition pipeline
- offset-voting matcher
- CLI interface
- benchmark runner

Benchmark:
- 276 test cases
- 100% top-1 accuracy on clean fragments
- 94.19 ms average recognition latency
- stable recognition on 3s, 5s, 10s and 15s fragments