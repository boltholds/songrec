from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from sqlalchemy.orm import Session

from songrec.db.session import create_session_factory


@dataclass(frozen=True)
class ApiSettings:
    db_path: Path = Path(os.getenv("SONGREC_DB_PATH", "songrec.sqlite3"))
    storage_dir: Path = Path(os.getenv("SONGREC_STORAGE_DIR", "songrec_storage"))

    @property
    def tracks_dir(self) -> Path:
        return self.storage_dir / "tracks"

    @property
    def queries_dir(self) -> Path:
        return self.storage_dir / "queries"

    @property
    def stems_dir(self) -> Path:
        return self.storage_dir / "stems"


_settings = ApiSettings()


def get_settings() -> ApiSettings:
    _settings.tracks_dir.mkdir(parents=True, exist_ok=True)
    _settings.queries_dir.mkdir(parents=True, exist_ok=True)
    _settings.stems_dir.mkdir(parents=True, exist_ok=True)
    return _settings


def get_session() -> Generator[Session, None, None]:
    settings = get_settings()
    session_factory = create_session_factory(settings.db_path)

    with session_factory() as session:
        yield session
