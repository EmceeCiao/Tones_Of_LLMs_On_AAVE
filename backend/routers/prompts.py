from typing import List

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from backend.db import get_session
from backend.schemas import PromptTemplate

router = APIRouter(
    prefix="/prompts",
    tags=["prompts"],
)


@router.get("/", response_model=List[PromptTemplate])
def get_prompts(session: Session = Depends(get_session)) -> List[PromptTemplate]:
    """Return all prompt templates stored in the database."""

    statement = select(PromptTemplate)
    return session.exec(statement).all()
