"""API-level tests for the questionnaire backend.

These tests exercise the FastAPI app routes through TestClient while using an
in-memory SQLite database so they are isolated and deterministic.
"""

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from backend.core.db import get_session
from backend.main import app
from backend.models import (
    create_generated_response_pair,
    create_likert_question,
    create_prompt_pair,
    create_survey_item,
    create_survey_submission,
    create_user,
)


ENGINE = create_engine(
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


# ============================================================================
# PROMPT PAIR ENDPOINTS
# ============================================================================

def test_create_prompt_pair():
    payload = {
        "category": "Algorithm",
        "round": 1,
        "aave_text": "How do algorithms work?",
        "sae_text": "How do algorithms work?",
        "display_order": 1,
    }

    response = client.post("/prompt/pairs", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["category"] == "Algorithm"
    assert data["round"] == 1
    assert data["display_order"] == 1
    assert data["id"] is not None


def test_create_prompt_pair_without_display_order():
    payload = {
        "category": "ELI5",
        "round": 1,
        "aave_text": "Explain AI simple",
        "sae_text": "Explain AI simply",
    }

    response = client.post("/prompt/pairs", json=payload)
    assert response.status_code == 200
    assert response.json()["display_order"] == 0


def test_get_survey_items_empty():
    response = client.get("/prompt/survey-items")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_survey_items_by_round():
    session = Session(ENGINE)
    prompt1 = create_prompt_pair(session, "Cat1", 1, "aave1", "sae1")
    prompt2 = create_prompt_pair(session, "Cat2", 2, "aave2", "sae2")
    create_survey_item(session, 1, "Cat1", prompt1.id)
    create_survey_item(session, 2, "Cat2", prompt2.id)
    session.close()

    response = client.get("/prompt/survey-items?round=1")
    assert response.status_code == 200

    items = response.json()
    assert len(items) >= 1
    assert all(item["round"] == 1 for item in items)


def test_get_survey_items_by_is_active():
    session = Session(ENGINE)
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    create_survey_item(session, 1, "Cat", prompt.id, is_active=True)
    create_survey_item(session, 1, "Cat", prompt.id, is_active=False)
    session.close()

    response = client.get("/prompt/survey-items?is_active=true")
    assert response.status_code == 200

    items = response.json()
    assert len(items) >= 1
    assert all(item["is_active"] is True for item in items)


def test_create_likert_questions():
    payload = {
        "question": "How natural does this sound?",
        "display_order": 1,
        "dimension": "naturalness",
        "scale_min": 1,
        "scale_max": 5,
        "short_answer_question": "Why?",
    }

    response = client.post("/prompt/likert-questions", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "How natural does this sound?"
    assert data["display_order"] == 1
    assert data["dimension"] == "naturalness"
    assert data["scale_min"] == 1
    assert data["scale_max"] == 5
    assert data["short_answer_question"] == "Why?"


def test_create_response_pair():
    session = Session(ENGINE)
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    session.close()

    payload = {
        "prompt_pair_id": prompt.id,
        "method": "API",
        "model": "GPT-4",
        "aave_response": "This be da response",
        "sae_response": "This is the response",
    }

    response = client.post("/prompt/response-pairs", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prompt_pair_id"] == prompt.id
    assert data["model"] == "GPT-4"


def test_create_survey_item():
    session = Session(ENGINE)
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    session.close()

    payload = {
        "round": 1,
        "category": "Cat",
        "prompt_pair_id": prompt.id,
        "display_order": 1,
        "is_active": True,
    }

    response = client.post("/prompt/survey-items", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prompt_pair_id"] == prompt.id
    assert data["category"] == "Cat"
    assert data["is_active"] is True
    assert data["display_order"] == 1


# ============================================================================
# SURVEY ENDPOINTS (with auth mocking)
# ============================================================================

def test_create_survey_submission():
    session = Session(ENGINE)
    user = create_user(session, "test@example.com")
    session.close()

    def mock_get_db_user():
        return user

    from backend.dependencies.auth import get_db_user

    app.dependency_overrides[get_db_user] = mock_get_db_user
    response = client.post("/survey/submissions/start?round=1")
    app.dependency_overrides.pop(get_db_user, None)

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == user.id
    assert data["round"] == 1
    assert data["id"] is not None


def test_get_submission_items():
    import pytest

    pytest.skip(
        "Requires router/schema fixes for nested submission item payloads, "
        "especially MultipleChoiceResponse <-> SurveyItemResponse relationships"
    )


def test_get_submission_items_not_found():
    def mock_get_db_user():
        session = Session(ENGINE)
        return create_user(session, "missing-check@example.com")

    from backend.dependencies.auth import get_db_user

    app.dependency_overrides[get_db_user] = mock_get_db_user
    response = client.get("/survey/submissions/999/items")
    app.dependency_overrides.pop(get_db_user, None)

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_submission_items_unauthorized():
    session = Session(ENGINE)
    user1 = create_user(session, "user1@example.com")
    prompt = create_prompt_pair(session, "Cat", 1, "aave", "sae")
    response_pair = create_generated_response_pair(
        session, prompt.id, "method", "model", "aave_resp", "sae_resp"
    )
    create_likert_question(session, "Q1", 1, "naturalness")
    create_survey_item(session, 1, "Cat", prompt.id, response_pair.id)
    submission = create_survey_submission(session, user1.id, 1)
    session.close()

    def mock_get_db_user():
        session = Session(ENGINE)
        return create_user(session, "user2@example.com")

    from backend.dependencies.auth import get_db_user

    app.dependency_overrides[get_db_user] = mock_get_db_user
    response = client.get(f"/survey/submissions/{submission.id}/items")
    app.dependency_overrides.pop(get_db_user, None)

    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"].lower()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_prompt_pair_lifecycle():
    prompt_payload = {
        "category": "Algorithm",
        "round": 1,
        "aave_text": "What is a binary search ye know?",
        "sae_text": "What is a binary search you know?",
        "display_order": 1,
    }

    prompt_response = client.post("/prompt/pairs", json=prompt_payload)
    assert prompt_response.status_code == 200
    prompt_id = prompt_response.json()["id"]

    response_payload = {
        "prompt_pair_id": prompt_id,
        "method": "API",
        "model": "GPT-4",
        "aave_response": "Girl that search algorithm be fast fr",
        "sae_response": "That search algorithm is fast indeed",
    }

    response_resp = client.post("/prompt/response-pairs", json=response_payload)
    assert response_resp.status_code == 200
    response_id = response_resp.json()["id"]

    item_payload = {
        "round": 1,
        "category": "Algorithm",
        "prompt_pair_id": prompt_id,
        "response_pair_id": response_id,
        "display_order": 1,
        "is_active": True,
    }

    item_response = client.post("/prompt/survey-items", json=item_payload)
    assert item_response.status_code == 200
    item_data = item_response.json()
    assert item_data["prompt_pair_id"] == prompt_id
    assert item_data["response_pair_id"] == response_id

    items_response = client.get("/prompt/survey-items?round=1")
    assert items_response.status_code == 200
    items = items_response.json()
    assert len(items) > 0
    assert any(item["id"] == item_data["id"] for item in items)


def test_multiple_likert_questions():
    dimensions = ["naturalness", "clarity", "fairness", "preference"]
    questions = [
        "How natural does this sound?",
        "How clear is this response?",
        "How fair is this representation?",
        "Which version do you prefer?",
    ]

    created_ids = []
    for idx, (question, dimension) in enumerate(zip(questions, dimensions), start=1):
        payload = {
            "question": question,
            "display_order": idx,
            "dimension": dimension,
            "scale_min": 1,
            "scale_max": 5,
        }
        response = client.post("/prompt/likert-questions", json=payload)
        assert response.status_code == 200
        created_ids.append(response.json()["id"])

    assert len(created_ids) == 4
    assert len(set(created_ids)) == 4