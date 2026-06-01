from __future__ import annotations

from fastapi import FastAPI

from songrec.api.dependencies import get_settings
from songrec.api.routes.health import router as health_router
from songrec.api.routes.lyrics import router as lyrics_router
from songrec.api.routes.recognize import router as recognize_router
from songrec.api.routes.separation import router as separation_router
from songrec.api.routes.tracks import router as tracks_router
from songrec.db.session import create_session_factory


def create_app() -> FastAPI:
    settings = get_settings()
    create_session_factory(settings.db_path)

    app = FastAPI(
        title="SongRec API",
        description="Hybrid music recognition service with fingerprinting, scale-aware matching and source separation and lyrics ASR.",
        version="0.5.0",
    )

    app.include_router(health_router)
    app.include_router(tracks_router)
    app.include_router(recognize_router)
    app.include_router(separation_router)
    app.include_router(lyrics_router)

    return app
