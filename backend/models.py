"""CRUD helpers backed by SQLModel for the questionnaire app.

This module provides simple helper functions that operate against a SQLModel
session to create, read, update, and delete domain entities.

The database models themselves live in :mod:`backend.schemas`. This module is
intended for use by routers / business logic layers that need convenience
operations without having to repeat the same boilerplate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from backend.schemas import (
    GeneratedResponse,
    LikertQuestion,
    PromptTemplate,
    SurveySubmission,
    User,
)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(session: Session, email: str) -> User:
    user = User(email=email)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_user(session: Session, user_id: int) -> Optional[User]:
    return session.get(User, user_id)


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    statement = select(User).where(User.email == email)
    return session.exec(statement).first()


def update_user_email(session: Session, user_id: int, new_email: str) -> Optional[User]:
    user = get_user(session, user_id)
    if not user:
        return None
    user.email = new_email
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def delete_user(session: Session, user_id: int) -> bool:
    user = get_user(session, user_id)
    if not user:
        return False
    session.delete(user)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def create_prompt_template(
    session: Session, category: str, round: int, variant: str, text: str
) -> PromptTemplate:
    prompt = PromptTemplate(category=category, round=round, variant=variant, text=text)
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    return prompt


def get_prompt_template(session: Session, prompt_id: int) -> Optional[PromptTemplate]:
    return session.get(PromptTemplate, prompt_id)


def update_prompt_template(
    session: Session, prompt_id: int, **updates: Any
) -> Optional[PromptTemplate]:
    prompt = get_prompt_template(session, prompt_id)
    if not prompt:
        return None
    for key, value in updates.items():
        if hasattr(prompt, key) and value is not None:
            setattr(prompt, key, value)
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    return prompt


def delete_prompt_template(session: Session, prompt_id: int) -> bool:
    prompt = get_prompt_template(session, prompt_id)
    if not prompt:
        return False
    session.delete(prompt)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Generated responses
# ---------------------------------------------------------------------------

def create_generated_response(
    session: Session,
    prompt_template_id: int,
    round: int,
    method: str,
    variant: str,
    model: str,
    response: str,
) -> GeneratedResponse:
    gr = GeneratedResponse(
        prompt_template_id=prompt_template_id,
        round=round,
        method=method,
        variant=variant,
        model=model,
        response=response,
    )
    session.add(gr)
    session.commit()
    session.refresh(gr)
    return gr


def get_generated_response(session: Session, response_id: int) -> Optional[GeneratedResponse]:
    return session.get(GeneratedResponse, response_id)


def update_generated_response(
    session: Session, response_id: int, **updates: Any
) -> Optional[GeneratedResponse]:
    gr = get_generated_response(session, response_id)
    if not gr:
        return None
    for key, value in updates.items():
        if hasattr(gr, key) and value is not None:
            setattr(gr, key, value)
    session.add(gr)
    session.commit()
    session.refresh(gr)
    return gr


def delete_generated_response(session: Session, response_id: int) -> bool:
    gr = get_generated_response(session, response_id)
    if not gr:
        return False
    session.delete(gr)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Likert questions
# ---------------------------------------------------------------------------

def create_likert_question(
    session: Session,
    question: str,
    question_type: str,
    scale_min: int = 1,
    scale_max: int = 5,
    labels: Optional[List[str]] = None,
) -> LikertQuestion:
    q = LikertQuestion(
        question=question,
        question_type=question_type,
        scale_min=scale_min,
        scale_max=scale_max,
        labels=labels,
    )
    session.add(q)
    session.commit()
    session.refresh(q)
    return q


def get_likert_question(session: Session, question_id: int) -> Optional[LikertQuestion]:
    return session.get(LikertQuestion, question_id)


def update_likert_question(session: Session, question_id: int, **updates: Any) -> Optional[LikertQuestion]:
    q = get_likert_question(session, question_id)
    if not q:
        return None
    for key, value in updates.items():
        if hasattr(q, key) and value is not None:
            setattr(q, key, value)
    session.add(q)
    session.commit()
    session.refresh(q)
    return q


def delete_likert_question(session: Session, question_id: int) -> bool:
    q = get_likert_question(session, question_id)
    if not q:
        return False
    session.delete(q)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Survey submissions
# ---------------------------------------------------------------------------

def create_survey_submission(
    session: Session,
    user_id: int,
    round: int,
    prompts_resp: List[Dict[str, Any]],
    short_answer: Optional[str] = None,
) -> SurveySubmission:
    submission = SurveySubmission(
        user_id=user_id,
        round=round,
        prompts_resp=prompts_resp,
        short_answer=short_answer,
    )
    session.add(submission)
    session.commit()
    session.refresh(submission)
    return submission


def get_survey_submission(session: Session, submission_id: int) -> Optional[SurveySubmission]:
    return session.get(SurveySubmission, submission_id)


def update_survey_submission(
    session: Session, submission_id: int, **updates: Any
) -> Optional[SurveySubmission]:
    sub = get_survey_submission(session, submission_id)
    if not sub:
        return None
    for key, value in updates.items():
        if hasattr(sub, key) and value is not None:
            setattr(sub, key, value)
    session.add(sub)
    session.commit()
    session.refresh(sub)
    return sub


def delete_survey_submission(session: Session, submission_id: int) -> bool:
    sub = get_survey_submission(session, submission_id)
    if not sub:
        return False
    session.delete(sub)
    session.commit()
    return True
