from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from sqlmodel import select

from backend.db import get_session, init_db
from backend.schemas import User

from backend.routers import prompts, survey, user


app = FastAPI()

app.include_router(prompts.router)
# app.include_router(survey.router)
# app.include_router(user.router)

# @app.on_event("startup")
# def on_startup():
#     init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield