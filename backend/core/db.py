"""Database setup for the questionnaire app.

This module configures the SQLModel engine and provides a session generator.
It is intended to be used by service code / API routers to acquire a database
session.

Example (FastAPI):

    from fastapi import Depends, FastAPI
    from backend.db import get_session, init_db

    app = FastAPI()

    @app.on_event("startup")
    def on_startup():
        init_db()

    @app.get("/users")
    def list_users(session: Session = Depends(get_session)):
        return session.exec(select(User)).all()
"""

from typing import Generator

from sqlmodel import Session, SQLModel, create_engine


DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def init_db() -> None:
    """Create database tables.

    This should be invoked once at startup (or as part of a migration script).
    """

    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Get a new SQLModel session (use as a dependency in FastAPI)."""

    with Session(engine) as session:
        yield session
