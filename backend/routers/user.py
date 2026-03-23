
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from backend.core.db import get_session
from backend.dependencies.auth import get_db_user

