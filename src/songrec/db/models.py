from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from songrec.db.base import Base


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)

    fingerprints: Mapped[list["FingerprintRecord"]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
    )


class FingerprintRecord(Base):
    __tablename__ = "fingerprints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    hash: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    track_id: Mapped[int] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False,
    )
    offset_ms: Mapped[int] = mapped_column(nullable=False)

    track: Mapped[Track] = relationship(back_populates="fingerprints")


Index("idx_fingerprints_hash_track", FingerprintRecord.hash, FingerprintRecord.track_id)