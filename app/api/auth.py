"""
API key authentication.

Checks the X-API-Key header against the API_KEY env var. If API_KEY
isn't set, auth is disabled (open access) -- this keeps local dev
frictionless, but means you MUST set API_KEY before deploying anywhere
real. The startup event in main.py logs a loud warning if it's unset,
specifically so this isn't silently forgotten in production.

This is intentionally simple (single shared key, not per-user OAuth2/
JWT/RBAC) -- adequate to stop anyone-with-the-URL from
approving/rejecting SAR escalations, which was the immediate gap. Real
multi-analyst deployments should replace this with proper OAuth2 + RBAC
per-user (see README "Recommended next steps").
"""
import os
import secrets
from fastapi import Header, HTTPException

_API_KEY = os.environ.get("API_KEY")


def auth_enabled() -> bool:
    return bool(_API_KEY)


async def require_api_key(x_api_key: str = Header(default=None)):
    if not _API_KEY:
        # No key configured -- auth disabled (dev mode). Startup already
        # warns loudly about this; don't also spam every request.
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
