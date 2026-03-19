"""API-level tests for the questionnaire backend.

These tests exercise the FastAPI app routes through TestClient while using an
in-memory SQLite database so they are isolated and deterministic.
"""

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.pool import StaticPool


from backend.main import app
from backend.db import get_session
from backend.models import create_prompt_template


# Use an in-memory SQLite instance for all API tests.
ENGINE=create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SQLModel.metadata.create_all(ENGINE)


def _get_test_session():
    with Session(ENGINE) as session:
        yield session


app.dependency_overrides[get_session] = _get_test_session
client = TestClient(app)


def test_get_prompts_empty():
    response = client.get("/prompts/")
    assert response.status_code == 200
    assert response.json() == []


def test_get_prompts_returns_data():
    # Insert a prompt using the same engine used by the API dependency.
    with Session(ENGINE) as session:
        create_prompt_template(
            session=session,
            category="Algorithm",
            round=1,
            variant="AAVE",
            text="Translate this to AAVE",
        )

    response = client.get("/prompts/")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["category"] == "Algorithm"
    assert data[0]["round"] == 1
    assert data[0]["variant"] == "AAVE"
    assert data[0]["text"] == "Translate this to AAVE"
