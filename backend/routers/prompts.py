from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from backend.core.db import get_session
from backend.models import (
    create_prompt_pair,
    create_generated_response_pair,
    create_survey_item,
    create_likert_question,
    create_multiple_choice_question,
    list_survey_items
)
from backend.db_schemas import (
    LikertQuestion,
    PromptPair,
    GeneratedResponsePair,
    SurveyItem,
    MultipleChoice,
)

router = APIRouter(
    prefix="/prompt",
    tags=["prompts"],
)


@router.post("/pairs", response_model=PromptPair)
async def create_prompt_pair_endpoint(
    prompt_pair: PromptPair, session: Session = Depends(get_session)
) -> PromptPair:
    return create_prompt_pair(
        session=session,
        category=prompt_pair.category,
        round=prompt_pair.round,
        aave_text=prompt_pair.aave_text,
        sae_text=prompt_pair.sae_text,
        display_order=prompt_pair.display_order,
    )

@router.post("/response-pairs", response_model=GeneratedResponsePair)
async def create_response_pair_endpoint(
    response_pair: GeneratedResponsePair, session: Session = Depends(get_session)
) -> GeneratedResponsePair :
    return create_generated_response_pair(
        session=session,
        prompt_pair_id=response_pair.prompt_pair_id,
        method=response_pair.method,
        model=response_pair.model,
        aave_response=response_pair.aave_response,
        sae_response=response_pair.sae_response,
        )
    
@router.post("/survey-items", response_model=SurveyItem)
# Create object to display prompt pair and response pair
# Category is for the dataset (i.e. ELI5, Algo, or Custom)

async def create_survey_items(
    survey_item: SurveyItem, session: Session = Depends(get_session)
) -> SurveyItem :
    return create_survey_item(
        session=session,
        round=survey_item.round,
        category=survey_item.category,
        prompt_pair_id=survey_item.prompt_pair_id,
        response_pair_id=survey_item.response_pair_id,
        display_order=survey_item.display_order,
        is_active=survey_item.is_active,
    )

@router.post("/likert-questions", response_model=LikertQuestion)
async def create_likert_questions(
    likert_q: LikertQuestion, session: Session = Depends(get_session)
) -> LikertQuestion :
    return create_likert_question(
        session=session,
        question=likert_q.question,
        display_order=likert_q.display_order,
        dimension=likert_q.dimension,
        scale_min=likert_q.scale_min,
        scale_max=likert_q.scale_max,
        short_answer_question=likert_q.short_answer_question
    )
    
@router.post("/mcq", response_model=MultipleChoice)
async def create_mcq(
    mcq: MultipleChoice, session: Session = Depends(get_session)
) -> MultipleChoice :
    return create_multiple_choice_question(
        session=session,
        question=mcq.question,
        display_order=mcq.display_order,
        choices=mcq.choices,
        short_answer_question=mcq.short_answer_question,
    )
    
@router.get("/survey-items", response_model=list[SurveyItem])
async def get_survey_items(
    round: int | None = None, 
    is_active: bool | None = None, 
    session: Session = Depends(get_session)
): 
    found_item=list_survey_items(
        session=session,
        round=round,
        is_active=is_active
    )
    if found_item == None:
        return []
    return found_item