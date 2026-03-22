from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from sqlmodel import select

from backend.core.db import get_session, init_db
from backend.db_schemas import User

from backend.routers import prompts, survey


app = FastAPI()

app.include_router(prompts.router)
app.include_router(survey.router)
# app.include_router(user.router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield