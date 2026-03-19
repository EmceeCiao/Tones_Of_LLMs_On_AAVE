"""Test helpers for models using an in-memory SQLModel database.

These tests are intended to provide a quick sanity check that the CRUD helpers in
`backend.models` work as expected. They use an in-memory SQLite database so tests
are isolated and have no side effects.

Run with:
    pytest backend/test_db.py
"""

from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.pool import StaticPool


from backend.models import (
    create_generated_response,
    create_likert_question,
    create_prompt_template,
    create_survey_submission,
    create_user,
    delete_generated_response,
    delete_likert_question,
    delete_prompt_template,
    delete_survey_submission,
    delete_user,
    get_generated_response,
    get_likert_question,
    get_prompt_template,
    get_survey_submission,
    get_user,
    get_user_by_email,
    update_generated_response,
    update_likert_question,
    update_prompt_template,
    update_survey_submission,
    update_user_email,
)


def _create_in_memory_session():
    engine = create_engine("sqlite:///:memory:", echo=False,  
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_user_crud_cycle():
    with _create_in_memory_session() as session:
        user = create_user(session, email="test@example.com")
        assert user.id is not None
        assert user.email == "test@example.com"

        fetched = get_user(session, user.id)
        assert fetched is not None
        assert fetched.email == "test@example.com"

        fetched_by_email = get_user_by_email(session, "test@example.com")
        assert fetched_by_email is not None
        assert fetched_by_email.id == user.id

        updated = update_user_email(session, user.id, "new@example.com")
        assert updated is not None
        assert updated.email == "new@example.com"

        assert delete_user(session, user.id) is True
        assert get_user(session, user.id) is None


def test_prompt_template_crud_cycle():
    with _create_in_memory_session() as session:
        prompt = create_prompt_template(
            session,
            category="Algorithm",
            round=1,
            variant="AAVE",
            text="Translate this into AAVE",
        )
        assert prompt.id is not None
        assert prompt.category == "Algorithm"

        fetched = get_prompt_template(session, prompt.id)
        assert fetched is not None
        assert fetched.text == "Translate this into AAVE"

        updated = update_prompt_template(session, prompt.id, text="New text")
        assert updated is not None
        assert updated.text == "New text"

        assert delete_prompt_template(session, prompt.id) is True
        assert get_prompt_template(session, prompt.id) is None


def test_likert_question_crud_cycle():
    with _create_in_memory_session() as session:
        q = create_likert_question(
            session,
            question="How natural does it sound?",
            question_type="fluency",
            scale_min=1,
            scale_max=5,
            labels=["Not at all", "Very natural"],
        )
        assert q.id is not None
        assert q.question == "How natural does it sound?"

        fetched = get_likert_question(session, q.id)
        assert fetched is not None
        assert fetched.question_type == "fluency"

        updated = update_likert_question(session, q.id, question_type="coherence")
        assert updated is not None
        assert updated.question_type == "coherence"

        assert delete_likert_question(session, q.id) is True
        assert get_likert_question(session, q.id) is None


def test_survey_submission_crud_cycle():
    with _create_in_memory_session() as session:
        user = create_user(session, email="survey@example.com")

        submission = create_survey_submission(
            session,
            user_id=user.id,
            round=1,
            prompts_resp=[
                {
                    "prompt_template_ids": [1, 2],
                    "likert_answers": [
                        {"question_id": 1, "question_type": "fluency", "value": 4}
                    ],
                }
            ],
            short_answer="Test response",
        )
        assert submission.id is not None
        assert submission.short_answer == "Test response"

        fetched = get_survey_submission(session, submission.id)
        assert fetched is not None
        assert fetched.user_id == user.id

        updated = update_survey_submission(
            session, submission.id, short_answer="Updated"
        )
        assert updated is not None
        assert updated.short_answer == "Updated"

        assert delete_survey_submission(session, submission.id) is True
        assert get_survey_submission(session, submission.id) is None
