from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from songrec.db.base import Base


def make_database_url(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def create_session_factory(db_path: Path) -> sessionmaker[Session]:
    engine = create_engine(
        make_database_url(db_path),
        echo=False,
        future=True,
    )

    Base.metadata.create_all(engine)

    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    
def drop_database(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()