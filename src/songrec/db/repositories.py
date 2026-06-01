from pathlib import Path
from typing import NamedTuple

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from songrec.db.models import FingerprintRecord, Track
from songrec.fingerprint import Fingerprint


class FingerprintMatchRow(NamedTuple):
    hash: str
    track_id: int
    track_offset_ms: int
    title: str


class TrackRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_track(self, path: Path, title: str) -> int:
        path_value = str(path)

        existing_track = self.session.scalar(
            select(Track).where(Track.path == path_value)
        )

        if existing_track is not None:
            return existing_track.id

        track = Track(
            path=path_value,
            title=title,
        )

        self.session.add(track)
        self.session.flush()

        return track.id

    def get_track(self, track_id: int) -> Track | None:
        return self.session.get(Track, track_id)

    def delete_track(self, track: Track) -> None:
        self.session.delete(track)

    def get_by_path(self, path: Path) -> Track | None:
        return self.session.scalar(
            select(Track).where(Track.path == str(path))
        )

    def list_tracks(self) -> list[Track]:
        return list(
            self.session.scalars(
                select(Track).order_by(Track.id.asc())
            ).all()
        )

    def count_tracks(self) -> int:
        return int(self.session.scalar(select(func.count(Track.id))) or 0)


class FingerprintRepository:
    def __init__(self, session: Session):
        self.session = session

    def delete_by_track_id(self, track_id: int) -> None:
        self.session.execute(
            delete(FingerprintRecord).where(FingerprintRecord.track_id == track_id)
        )

    def add_fingerprints(
        self,
        track_id: int,
        fingerprints: list[Fingerprint],
    ) -> None:
        records = [
            FingerprintRecord(
                hash=fingerprint.hash,
                track_id=track_id,
                offset_ms=fingerprint.offset_ms,
            )
            for fingerprint in fingerprints
        ]

        self.session.add_all(records)

    def count_fingerprints(self) -> int:
        return int(
            self.session.scalar(select(func.count(FingerprintRecord.id))) or 0
        )

    def count_by_track_id(self, track_id: int) -> int:
        return int(
            self.session.scalar(
                select(func.count(FingerprintRecord.id)).where(
                    FingerprintRecord.track_id == track_id
                )
            )
            or 0
        )

    def find_matches(
        self,
        query_fingerprints: list[Fingerprint],
    ) -> list[FingerprintMatchRow]:
        hashes = list({fingerprint.hash for fingerprint in query_fingerprints})

        if not hashes:
            return []

        statement = (
            select(
                FingerprintRecord.hash,
                FingerprintRecord.track_id,
                FingerprintRecord.offset_ms,
                Track.title,
            )
            .join(Track, Track.id == FingerprintRecord.track_id)
            .where(FingerprintRecord.hash.in_(hashes))
        )

        rows = self.session.execute(statement).all()

        return [
            FingerprintMatchRow(
                hash=row.hash,
                track_id=row.track_id,
                track_offset_ms=row.offset_ms,
                title=row.title,
            )
            for row in rows
        ]