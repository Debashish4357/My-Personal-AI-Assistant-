"""
Simple in-process rate limiter middleware.
Limits each authenticated user to MAX_REQUESTS per TIME_WINDOW seconds.
Uses a sliding-window counter stored in memory (resets on server restart).
"""
import time
import logging
from collections import defaultdict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
MAX_REQUESTS  = 20    # per user per window
TIME_WINDOW   = 60    # seconds

# ── In-memory store: {user_key: [(timestamp, count)]} ────────────────
_request_log: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.
    Identifies users by their Firebase UID extracted from the
    Authorization header prefix (does not re-verify the token here —
    just uses the header value as a bucket key).
    Exempt: GET /  and  GET /docs  and  GET /openapi.json
    """

    EXEMPT_PATHS = {"/", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip rate limiting for health check and docs
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Use Authorization header as bucket key (cheap, no re-verification)
        auth_header = request.headers.get("authorization", "")
        if not auth_header:
            return await call_next(request)  # unauthenticated will be caught by deps

        # Use last 16 chars of token as a bucket key (avoids storing full token)
        user_key = auth_header[-16:] if len(auth_header) > 16 else auth_header

        now = time.time()
        window_start = now - TIME_WINDOW

        # Prune expired timestamps
        _request_log[user_key] = [
            ts for ts in _request_log[user_key] if ts > window_start
        ]

        if len(_request_log[user_key]) >= MAX_REQUESTS:
            logger.warning(f"Rate limit hit for key ...{user_key}")
            return Response(
                content='{"detail":"Too many requests. Please wait a moment and try again."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(TIME_WINDOW)},
            )

        _request_log[user_key].append(now)
        return await call_next(request)
