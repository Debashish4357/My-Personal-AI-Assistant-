"""
Authentication dependency for FastAPI routes.
Uses HTTPBearer for standard Swagger UI (lock icon) support and
provides granular Firebase token error handling.
"""
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth

# HTTPBearer adds a lock button to Swagger UI at /docs automatically
security = HTTPBearer()

logger = logging.getLogger(__name__)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Validate a Firebase ID token from the Authorization: Bearer <token> header.
    Returns the decoded token payload dict on success.
    Raises 401 HTTPException on any auth failure.
    """
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token

    except auth.ExpiredIdTokenError:
        logger.warning("Rejected request: Firebase ID token is expired.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session has expired. Please log in again.",
        )
    except auth.RevokedIdTokenError:
        logger.warning("Rejected request: Firebase ID token has been revoked.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please log in again.",
        )
    except auth.InvalidIdTokenError as e:
        logger.warning(f"Rejected request: Invalid Firebase token — {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )
    except Exception as e:
        logger.error(f"Unexpected auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed.",
        )
