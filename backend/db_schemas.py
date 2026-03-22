from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import EmailStr
from sqlmodel import SQLModel, Field, Relationship


class PromptVariant(str, Enum):
    AAVE = "AAVE"
    SAE = "SAE"


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    email: EmailStr

    submissions: list["SurveySubmission"] = Relationship(back_populates="user")


class PromptPair(SQLModel, table=True):
    __tablename__ = "prompt_pairs"

    id: int | None = Field(default=None, primary_key=True)
    category: str
    round: int
    aave_text: str
    sae_text: str
    display_order: int = 0

    response_pairs: list["GeneratedResponsePair"] = Relationship(back_populates="prompt_pair")
    survey_items: list["SurveyItem"] = Relationship(back_populates="prompt_pair")


class GeneratedResponsePair(SQLModel, table=True):
    __tablename__ = "generated_response_pairs"

    id: int | None = Field(default=None, primary_key=True)
    prompt_pair_id: int = Field(foreign_key="prompt_pairs.id")
    method: str
    model: str
    aave_response: str
    sae_response: str

    prompt_pair: Optional["PromptPair"] = Relationship(back_populates="response_pairs")
    survey_items: list["SurveyItem"] = Relationship(back_populates="response_pair")


class SurveyItem(SQLModel, table=True):
    __tablename__ = "survey_items"

    id: int | None = Field(default=None, primary_key=True)
    round: int
    category: str
    prompt_pair_id: int = Field(foreign_key="prompt_pairs.id")
    response_pair_id: int | None = Field(default=None, foreign_key="generated_response_pairs.id")
    display_order: int = 0
    is_active: bool = True

    prompt_pair: Optional["PromptPair"] = Relationship(back_populates="survey_items")
    response_pair: Optional["GeneratedResponsePair"] = Relationship(back_populates="survey_items")
    item_responses: list["SurveyItemResponse"] = Relationship(back_populates="survey_item")

class SurveyItemResponse(SQLModel, table=True):
    """ One row per page answered by a participant

    Fields:
        id: int 
        submission_id: int
        survey_item_id: int
        submission: SurveySubmission (optional)
        survey_item: SurveyItem (optional)
        likert_answers: list[LikertResponse]
    """
    __tablename__ = "survey_item_responses"

    id: int | None = Field(default=None, primary_key=True)
    submission_id: int = Field(foreign_key="survey_submissions.id")
    survey_item_id: int = Field(foreign_key="survey_items.id")

    submission: Optional["SurveySubmission"] = Relationship(back_populates="item_responses")
    survey_item: Optional["SurveyItem"] = Relationship(back_populates="item_responses")
    likert_answers: list["LikertResponse"] = Relationship(back_populates="item_response")

class LikertQuestion(SQLModel, table=True):
    __tablename__ = "likert_questions"

    id: int | None = Field(default=None, primary_key=True)
    question: str
    dimension: str              # e.g. naturalness, clarity, fairness, preference
    scale_min: int = 1
    scale_max: int = 5


class SurveySubmission(SQLModel, table=True):
    __tablename__ = "survey_submissions"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    round: int
    short_answer: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional["User"] = Relationship(back_populates="submissions")
    item_responses: list["SurveyItemResponse"] = Relationship(back_populates="submission")


class LikertResponse(SQLModel, table=True):
    __tablename__ = "likert_responses"

    id: int | None = Field(default=None, primary_key=True)
    item_response_id: int = Field(foreign_key="survey_item_responses.id")
    question_id: int = Field(foreign_key="likert_questions.id")
    value: int
    comment: str | None = None

    item_response: Optional["SurveyItemResponse"] = Relationship(back_populates="likert_answers")


# Response models for API payloads (read-only views of nested data)

class PromptPairDetail(SQLModel):
    """Minimal prompt pair details for survey responses."""
    aave_text: str
    sae_text: str


class ResponsePairDetail(SQLModel):
    """Minimal response pair details for survey responses."""
    aave_response: str
    sae_response: str


class LikertQuestionDetail(SQLModel):
    """Likert question for survey item responses."""
    id: int
    question: str

class SurveyItemDetail(SQLModel):
    """Rich survey item with all nested data for client consumption."""
    survey_item_id: int
    category: str
    display_order: int
    prompt_pair: PromptPairDetail
    response_pair: ResponsePairDetail
    likert_question_resp: Optional[list["LikertResponse"]]

class SubmissionItemsResponse(SQLModel):
    """Response for GET /submissions/{submission_id}/items endpoint."""
    submission_id: int
    likert_questions: list["LikertQuestion"]
    items: list[SurveyItemDetail]
