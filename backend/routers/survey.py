from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from backend.core.db import get_session
from backend.dependencies.auth import get_db_user
from backend.models import (
    create_survey_submission,
    get_survey_submission,
    update_survey_item_response,
    create_survey_item_response,
    list_survey_items,
    list_likert_questions
    
)
from backend.db_schemas import (
    SurveySubmission,
    SurveyItemResponse,
    SubmissionItemsResponse,
    SurveyItemDetail,
    PromptPairDetail,
    ResponsePairDetail,
    LikertResponse,
    PromptPair,
    
)


router = APIRouter(
    prefix="/survey",
    tags=["survey"],
)

"""
    POST /survey/submissions/start
GET /survey/submissions/{submission_id}/items
PUT /survey/submissions/{submission_id}/items/{survey_item_id}/response
"""
    
@router.post("/submissions/start", response_model=SurveySubmission)
async def create_survey_submission_endpoint(
    round: int,
    current_user: dict = Depends(get_db_user),
    session: Session = Depends(get_session)
    ):
    return create_survey_submission(
        session=session,
        user_id=current_user.id,
        round=round
    )


@router.get("/submissions/{submission_id}/items", response_model=SubmissionItemsResponse)
async def get_submission_items_by_id(
    submission_id: int,
    db_user = Depends(get_db_user),
    session: Session = Depends(get_session)
):
    submission = get_survey_submission(session=session, submission_id=submission_id)

    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission.user_id != db_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this submission")

    # Use submission to pull survey item id from each item response    
    survey_item_list = list_survey_items(
        session=session,
        round=submission.round
    )
    likert_questions = list_likert_questions(
        session=session
    )
    
    survey_item_details=[]
    for item in survey_item_list:
        item_resp = next((resp for resp in submission.item_responses if resp.survey_item_id == item.id),None)
        detail_item = SurveyItemDetail(
            survey_item_id=item.id,    
            category=item.category,
            display_order=item.display_order,
            prompt_pair=PromptPairDetail(
                aave_text=item.prompt_pair.aave_text, 
                sae_text=item.prompt_pair.sae_text,    
            ),
            response_pair=ResponsePairDetail(
                aave_response=item.response_pair.aave_response,
                sae_response=item.response_pair.sae_response,
            ),
            likert_question_resp=item_resp.likert_answers if item_resp else []
        )
        survey_item_details.append(detail_item)
    return SubmissionItemsResponse(
        submission_id=submission.id,
        likert_questions=likert_questions,
        items=survey_item_details
    )

@router.put("/survey-item-response", response_model=SurveyItemResponse)
async def get_survey_item_by_id(
    survey_item_resp: SurveyItemResponse,
    db_user = Depends(get_db_user),
    session: Session = Depends(get_session)
) -> SurveyItemResponse:
    """Get specific survey item from a specific submission ID"""
    submission:SurveySubmission = get_survey_submission(
        session=session,
        submission_id=survey_item_resp.submission_id
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    if submission.user_id != db_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this submission")
    
    search = next((item for item in submission.item_responses if item.survey_item_id == survey_item_id), None)
    if search is None: 
        # Create the survey item response
        return create_survey_item_response(
            session=session,
            submission_id=survey_item_resp.submission_id,
            survey_item_id=survey_item_resp.survey_item_id,
            likert_answers=survey_item_resp.likert_answers,
            submission=survey_item_resp.submission,
            survey_item=survey_item_resp.survey_item
        )
    return update_survey_item_response(
        session=session,
        submission_id=survey_item_resp.submission_id,
        survey_item_id=survey_item_resp.survey_item_id,
        likert_answers=survey_item_resp.likert_answers,
        submission=survey_item_resp.submission,
        survey_item=survey_item_resp.survey_item
    )