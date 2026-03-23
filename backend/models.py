"""CRUD helpers backed by SQLModel for the questionnaire app.

This module provides simple helper functions that operate against a SQLModel
session to create, read, update, and delete domain entities.

The database models themselves live in :mod:`backend.schemas`. This module is
intended for use by routers / business logic layers that need convenience
operations without having to repeat the same boilerplate.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlmodel import Session, select

from backend.db_schemas import (
    GeneratedResponsePair,
    LikertQuestion,
    LikertResponse,
    MultipleChoice, 
    MultipleChoiceResponse,
    PromptPair,
    SurveyItem,
    SurveyItemResponse,
    SurveySubmission,
    User,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _apply_updates(instance: Any, updates: dict[str, Any]) -> Any:
    """Apply non-None updates to a SQLModel instance."""
    for key, value in updates.items():
        if hasattr(instance, key) and value is not None:
            setattr(instance, key, value)
    return instance


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


def list_users(session: Session) -> list[User]:
    statement = select(User)
    return list(session.exec(statement).all())


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
# Prompt pairs
# ---------------------------------------------------------------------------

def create_prompt_pair(
    session: Session,
    category: str,
    round: int,
    aave_text: str,
    sae_text: str,
    display_order: int = 0,
) -> PromptPair:
    prompt_pair = PromptPair(
        category=category,
        round=round,
        aave_text=aave_text,
        sae_text=sae_text,
        display_order=display_order,
    )
    session.add(prompt_pair)
    session.commit()
    session.refresh(prompt_pair)
    return prompt_pair


def get_prompt_pair(session: Session, prompt_pair_id: int) -> Optional[PromptPair]:
    return session.get(PromptPair, prompt_pair_id)


def list_prompt_pairs(
    session: Session,
    *,
    round: int | None = None,
    category: str | None = None,
) -> list[PromptPair]:
    statement = select(PromptPair)

    if round is not None:
        statement = statement.where(PromptPair.round == round)
    if category is not None:
        statement = statement.where(PromptPair.category == category)

    statement = statement.order_by(PromptPair.display_order, PromptPair.id)
    return list(session.exec(statement).all())


def update_prompt_pair(
    session: Session,
    prompt_pair_id: int,
    **updates: Any,
) -> Optional[PromptPair]:
    prompt_pair = get_prompt_pair(session, prompt_pair_id)
    if not prompt_pair:
        return None

    _apply_updates(prompt_pair, updates)
    session.add(prompt_pair)
    session.commit()
    session.refresh(prompt_pair)
    return prompt_pair


def delete_prompt_pair(session: Session, prompt_pair_id: int) -> bool:
    prompt_pair = get_prompt_pair(session, prompt_pair_id)
    if not prompt_pair:
        return False

    session.delete(prompt_pair)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Generated response pairs
# ---------------------------------------------------------------------------

def create_generated_response_pair(
    session: Session,
    prompt_pair_id: int,
    method: str,
    model: str,
    aave_response: str,
    sae_response: str,
) -> GeneratedResponsePair:
    response_pair = GeneratedResponsePair(
        prompt_pair_id=prompt_pair_id,
        method=method,
        model=model,
        aave_response=aave_response,
        sae_response=sae_response,
    )
    session.add(response_pair)
    session.commit()
    session.refresh(response_pair)
    return response_pair


def get_generated_response_pair(
    session: Session,
    response_pair_id: int,
) -> Optional[GeneratedResponsePair]:
    return session.get(GeneratedResponsePair, response_pair_id)


def list_generated_response_pairs(
    session: Session,
    *,
    prompt_pair_id: int | None = None,
    method: str | None = None,
    model: str | None = None,
) -> list[GeneratedResponsePair]:
    statement = select(GeneratedResponsePair)

    if prompt_pair_id is not None:
        statement = statement.where(GeneratedResponsePair.prompt_pair_id == prompt_pair_id)
    if method is not None:
        statement = statement.where(GeneratedResponsePair.method == method)
    if model is not None:
        statement = statement.where(GeneratedResponsePair.model == model)

    statement = statement.order_by(GeneratedResponsePair.id)
    return list(session.exec(statement).all())


def update_generated_response_pair(
    session: Session,
    response_pair_id: int,
    **updates: Any,
) -> Optional[GeneratedResponsePair]:
    response_pair = get_generated_response_pair(session, response_pair_id)
    if not response_pair:
        return None

    _apply_updates(response_pair, updates)
    session.add(response_pair)
    session.commit()
    session.refresh(response_pair)
    return response_pair


def delete_generated_response_pair(session: Session, response_pair_id: int) -> bool:
    response_pair = get_generated_response_pair(session, response_pair_id)
    if not response_pair:
        return False

    session.delete(response_pair)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Survey items (one rendered questionnaire page)
# ---------------------------------------------------------------------------

def create_survey_item(
    session: Session,
    round: int,
    category: str,
    prompt_pair_id: int,
    response_pair_id: int | None = None,
    display_order: int = 0,
    is_active: bool = True,
) -> SurveyItem:
    survey_item = SurveyItem(
        round=round,
        category=category,
        prompt_pair_id=prompt_pair_id,
        response_pair_id=response_pair_id,
        display_order=display_order,
        is_active=is_active,
    )
    session.add(survey_item)
    session.commit()
    session.refresh(survey_item)
    return survey_item


def get_survey_item(session: Session, survey_item_id: int) -> Optional[SurveyItem]:
    return session.get(SurveyItem, survey_item_id)


def list_survey_items(
    session: Session,
    round: int | None = None,
    category: str | None = None,
    is_active: bool | None = None,
) -> list[SurveyItem]:
    statement = select(SurveyItem)

    if round is not None:
        statement = statement.where(SurveyItem.round == round)
    if category is not None:
        statement = statement.where(SurveyItem.category == category)
    if is_active is not None:
        statement = statement.where(SurveyItem.is_active == is_active)

    statement = statement.order_by(SurveyItem.display_order, SurveyItem.id)
    return list(session.exec(statement).all())


def update_survey_item(
    session: Session,
    survey_item_id: int,
    **updates: Any,
) -> Optional[SurveyItem]:
    survey_item = get_survey_item(session, survey_item_id)
    if not survey_item:
        return None

    _apply_updates(survey_item, updates)
    session.add(survey_item)
    session.commit()
    session.refresh(survey_item)
    return survey_item


def delete_survey_item(session: Session, survey_item_id: int) -> bool:
    survey_item = get_survey_item(session, survey_item_id)
    if not survey_item:
        return False

    session.delete(survey_item)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Likert questions
# ---------------------------------------------------------------------------

def create_likert_question(
    session: Session,
    question: str,
    display_order:str,
    dimension: str,
    scale_min: int = 1,
    scale_max: int = 5,
    short_answer_question: str | None = None,
) -> LikertQuestion:
    likert_question = LikertQuestion(
        question=question,
        display_order=display_order,
        dimension=dimension,
        scale_min=scale_min,
        scale_max=scale_max,
        short_answer_question=short_answer_question
    )
    session.add(likert_question)
    session.commit()
    session.refresh(likert_question)
    return likert_question


def get_likert_question(session: Session, question_id: int) -> Optional[LikertQuestion]:
    return session.get(LikertQuestion, question_id)


def list_likert_questions(
    session: Session,
    *,
    dimension: str | None = None,
) -> list[LikertQuestion]:
    statement = select(LikertQuestion)

    if dimension is not None:
        statement = statement.where(LikertQuestion.dimension == dimension)

    statement = statement.order_by(LikertQuestion.id)
    return list(session.exec(statement).all())


def update_likert_question(
    session: Session,
    question_id: int,
    **updates: Any,
) -> Optional[LikertQuestion]:
    likert_question = get_likert_question(session, question_id)
    if not likert_question:
        return None

    _apply_updates(likert_question, updates)
    session.add(likert_question)
    session.commit()
    session.refresh(likert_question)
    return likert_question


def delete_likert_question(session: Session, question_id: int) -> bool:
    likert_question = get_likert_question(session, question_id)
    if not likert_question:
        return False

    session.delete(likert_question)
    session.commit()
    return True

# ---------------------------------------------------------------------------
# Multiple Choice Question
# ---------------------------------------------------------------------------

def create_multiple_choice_question(
    session: Session,
    question: str,
    display_order: int,
    choices: list[str],
    short_answer_question: str | None = None,
) -> MultipleChoice:
    multiple_choice_q = MultipleChoice(
        question=question,
        display_order=display_order,
        choices=choices,
        short_answer_question=short_answer_question
    )
    session.add(multiple_choice_q)
    session.commit()
    session.refresh(multiple_choice_q)
    return multiple_choice_q


def get_multiple_choice_question(session: Session, question_id: int) -> Optional[MultipleChoice]:
    return session.get(MultipleChoice, question_id)


def list_multiple_choice_questions(
    session: Session,
) -> list[MultipleChoice]:
    statement = select(MultipleChoice)

    statement = statement.order_by(MultipleChoice.id)
    return list(session.exec(statement).all())


def update_multiple_choice_questions(
    session: Session,
    question_id: int,
    **updates: Any,
) -> Optional[MultipleChoice]:
    mcq = get_multiple_choice_question(session, question_id)
    if not mcq:
        return None

    _apply_updates(mcq, updates)
    session.add(mcq)
    session.commit()
    session.refresh(mcq)
    return mcq


def delete_multiple_choice_question(session: Session, question_id: int) -> bool:
    mcq = get_multiple_choice_question(session, question_id)
    if not mcq:
        return False

    session.delete(mcq)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Survey submissions
# ---------------------------------------------------------------------------

def create_survey_submission(
    session: Session,
    user_id: int,
    round: int,
    short_answer: str | None = None,
) -> SurveySubmission:
    submission = SurveySubmission(
        user_id=user_id,
        round=round,
        short_answer=short_answer,
    )
    session.add(submission)
    session.commit()
    session.refresh(submission)
    return submission


def get_survey_submission(
    session: Session,
    submission_id: int,
) -> Optional[SurveySubmission]:
    return session.get(SurveySubmission, submission_id)


def list_survey_submissions(
    session: Session,
    *,
    user_id: int | None = None,
    round: int | None = None,
) -> list[SurveySubmission]:
    statement = select(SurveySubmission)

    if user_id is not None:
        statement = statement.where(SurveySubmission.user_id == user_id)
    if round is not None:
        statement = statement.where(SurveySubmission.round == round)

    statement = statement.order_by(SurveySubmission.created_at.desc(), SurveySubmission.id.desc())
    return list(session.exec(statement).all())


def update_survey_submission(
    session: Session,
    submission_id: int,
    **updates: Any,
) -> Optional[SurveySubmission]:
    submission = get_survey_submission(session, submission_id)
    if not submission:
        return None

    _apply_updates(submission, updates)
    session.add(submission)
    session.commit()
    session.refresh(submission)
    return submission


def delete_survey_submission(session: Session, submission_id: int) -> bool:
    submission = get_survey_submission(session, submission_id)
    if not submission:
        return False

    session.delete(submission)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Survey item responses (one participant's answer set for one page)
# ---------------------------------------------------------------------------

def create_survey_item_response(
    session: Session,
    submission_id: int,
    survey_item_id: int,
    likert_answers: list[LikertResponse] = [],
    mcq_answers: list[MultipleChoiceResponse] = [], 
    submission: Optional[SurveySubmission] = None,
    survey_item: Optional[SurveyItem] = None,
) -> SurveyItemResponse:
    item_response = SurveyItemResponse(
        submission_id=submission_id,
        survey_item_id=survey_item_id,
        likert_answers= likert_answers,
        mcq_answers = mcq_answers,
        submission = submission,
        survey_item = survey_item
    )
    session.add(item_response)
    session.commit()
    session.refresh(item_response)
    return item_response


def get_survey_item_response(
    session: Session,
    item_response_id: int,
) -> Optional[SurveyItemResponse]:
    return session.get(SurveyItemResponse, item_response_id)


def list_survey_item_responses(
    session: Session,
    *,
    submission_id: int | None = None,
    survey_item_id: int | None = None,
) -> list[SurveyItemResponse]:
    statement = select(SurveyItemResponse)

    if submission_id is not None:
        statement = statement.where(SurveyItemResponse.submission_id == submission_id)
    if survey_item_id is not None:
        statement = statement.where(SurveyItemResponse.survey_item_id == survey_item_id)

    statement = statement.order_by(SurveyItemResponse.survey_item_id)
    return list(session.exec(statement).all())


def update_survey_item_response(
    session: Session,
    item_response_id: int,
    **updates: Any,
) -> Optional[SurveyItemResponse]:
    item_response = get_survey_item_response(session, item_response_id)
    if not item_response:
        return None

    _apply_updates(item_response, updates)
    session.add(item_response)
    session.commit()
    session.refresh(item_response)
    return item_response


def delete_survey_item_response(session: Session, item_response_id: int) -> bool:
    item_response = get_survey_item_response(session, item_response_id)
    if not item_response:
        return False

    session.delete(item_response)
    session.commit()
    return True


# ---------------------------------------------------------------------------
# Likert responses (one row per answered question)
# ---------------------------------------------------------------------------

def create_likert_response(
    session: Session,
    item_response_id: int,
    question_id: int,
    value: int,
    short_answer: str | None = None,
) -> LikertResponse:
    likert_response = LikertResponse(
        item_response_id=item_response_id,
        question_id=question_id,
        value=value,
        short_answer=short_answer,
    )
    session.add(likert_response)
    session.commit()
    session.refresh(likert_response)
    return likert_response


def get_likert_response(session: Session, likert_response_id: int) -> Optional[LikertResponse]:
    return session.get(LikertResponse, likert_response_id)


def list_likert_responses(
    session: Session,
    *,
    item_response_id: int | None = None,
    question_id: int | None = None,
) -> list[LikertResponse]:
    statement = select(LikertResponse)

    if item_response_id is not None:
        statement = statement.where(LikertResponse.item_response_id == item_response_id)
    if question_id is not None:
        statement = statement.where(LikertResponse.question_id == question_id)

    statement = statement.order_by(LikertResponse.id)
    return list(session.exec(statement).all())


def update_likert_response(
    session: Session,
    likert_response_id: int,
    **updates: Any,
) -> Optional[LikertResponse]:
    likert_response = get_likert_response(session, likert_response_id)
    if not likert_response:
        return None

    _apply_updates(likert_response, updates)
    session.add(likert_response)
    session.commit()
    session.refresh(likert_response)
    return likert_response


def delete_likert_response(session: Session, likert_response_id: int) -> bool:
    likert_response = get_likert_response(session, likert_response_id)
    if not likert_response:
        return False

    session.delete(likert_response)
    session.commit()
    return True

# ---------------------------------------------------------------------------
# Multiple Choice Response
# ---------------------------------------------------------------------------
def create_multiple_choice_response(
    session: Session,
    question_id: int,
    answer_choice: str,
    short_answer: str | None = None,
) -> MultipleChoiceResponse:
    multiple_choice_resp = MultipleChoiceResponse(
        question_id=question_id,
        answer_choice=answer_choice,
        short_answer=short_answer
    )
    session.add(multiple_choice_resp)
    session.commit()
    session.refresh(multiple_choice_resp)
    return multiple_choice_resp


def get_multiple_choice_response(session: Session, resp_id: int) -> Optional[MultipleChoiceResponse]:
    return session.get(MultipleChoiceResponse, resp_id)


def list_multiple_choice_responses(
    session: Session,
    *,
    resp_id: int | None = None,
    mcq_id: int | None = None,
) -> list[MultipleChoiceResponse]:
    statement = select(MultipleChoiceResponse)
    
    if resp_id is not None:
        statement = statement.where(MultipleChoiceResponse.id == resp_id)
    if mcq_id is not None:
        statement = statement.where(MultipleChoiceResponse.question_id == mcq_id)


    statement = statement.order_by(MultipleChoiceResponse.id)
    return list(session.exec(statement).all())


def update_multiple_choice_responses(
    session: Session,
    mcq_id: int,
    **updates: Any,
) -> Optional[MultipleChoiceResponse]:
    mcq_resp = get_multiple_choice_response(session, mcq_id)
    if not mcq_resp:
        return None

    _apply_updates(mcq_resp, updates)
    session.add(mcq_resp)
    session.commit()
    session.refresh(mcq_resp)
    return mcq_resp


def delete_multiple_choice_responses(session: Session, resp_id: int) -> bool:
    mcq_resp = get_multiple_choice_response(session, resp_id)
    if not mcq_resp:
        return False

    session.delete(mcq_resp)
    session.commit()
    return True