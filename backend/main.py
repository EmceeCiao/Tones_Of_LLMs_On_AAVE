from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from sqlmodel import select

from backend.core.db import get_session, init_db
# from backend.db_schemas import User

from backend.routers import prompts, survey

description = """
Survey Platform API supports creation of prompts and survey content, as well as user survey submissions and responses.

## Survey Endpoints

You will be able to:

* **Start a new survey submission** for an authenticated user.
* **Retrieve all survey items** for a submission.
* **Submit or update responses** for a survey item.

## Prompt Management Endpoints

You will be able to:

* **Create prompt pairs** for survey content.
* **Create generated response pairs** linked to prompts.
* **Create survey items** used in survey rounds.
* **Create Likert questions** for evaluation.
* **List survey items** by round or active status.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    
    
app = FastAPI(
    title="Tones Of LLMs On AAVE",
    description=description,
    summary="Tones of LLMs On AAVE Questionaire API",
    version="0.0.1",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan
)

app.include_router(prompts.router)
app.include_router(survey.router)
# app.include_router(user.router)

