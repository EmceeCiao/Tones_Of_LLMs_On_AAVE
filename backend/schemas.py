from datetime import datetime
from enum import Enum

from pydantic import EmailStr
from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel


class PromptVariant(str, Enum):
    AAVE = "AAVE"
    SAE = "SAE"


class PromptTemplate(SQLModel, table=True):
    __tablename__ = "prompt_templates"

    id: int | None = Field(default=None, primary_key=True)
    category: str
    round: int
    variant: PromptVariant
    text: str

    generated_responses: list["GeneratedResponse"] = Relationship(
        back_populates="prompt_template"
    )


class GeneratedResponse(SQLModel, table=True):
    __tablename__ = "generated_responses"

    id: int | None = Field(default=None, primary_key=True)
    prompt_template_id: int = Field(foreign_key="prompt_templates.id")
    round: int
    method: str
    variant: PromptVariant
    model: str
    response: str

    prompt_template: "PromptTemplate" = Relationship(
        back_populates="generated_responses"
    )


class LikertQuestion(SQLModel, table=True):
    __tablename__ = "likert_questions"

    id: int | None = Field(default=None, primary_key=True)
    question: str
    question_type: str
    scale_min: int = Field(1, description="Minimum value for the Likert scale")
    scale_max: int = Field(5, description="Maximum value for the Likert scale")
    labels: list[str] | None = Field(
        default=None, sa_column=Column(JSON), description="Optional label list"
    )


class LikertAnswer(SQLModel):
    question_id: int
    question_type: str
    value: int
    comment: str | None = None


class PromptFeedback(SQLModel):
    prompt_template_ids: list[int]
    likert_answers: list[LikertAnswer]


class SurveySubmission(SQLModel, table=True):
    __tablename__ = "survey_submissions"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    round: int
    prompts_resp: list[PromptFeedback] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="List of prompt feedback entries for the round",
    )
    short_answer: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: "User" = Relationship(back_populates="submissions")


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    email: EmailStr

    submissions: list["SurveySubmission"] = Relationship(back_populates="user")