import firebase_admin
from firebase_admin import auth, credentials

from backend.core.config import settings

_firebase_app = None


def get_firebase_app():
    global _firebase_app
    if _firebase_app is None:
        cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT_PATH)
        _firebase_app = firebase_admin.initialize_app(cred)
    return _firebase_app


def verify_firebase_token(id_token: str) -> dict:
    get_firebase_app()
    return auth.verify_id_token(id_token)