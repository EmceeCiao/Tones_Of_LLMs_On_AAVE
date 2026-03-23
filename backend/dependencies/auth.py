from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session

from backend.core.firebase import verify_firebase_token
from backend.models import get_user_by_email, create_user
from backend.core.db import get_session

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        decoded_token = verify_firebase_token(credentials.credentials)
        return decoded_token
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid authentication token") from exc
    
def get_db_user(
    current_user: dict = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    user_email = current_user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="Invalid user identity")

    user = get_user_by_email(session, user_email)
    if user is None:
        user = create_user(session, user_email)

    return user