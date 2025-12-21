from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import get_settings, Settings
from .db import get_user_by_email

bearer_scheme = HTTPBearer(auto_error=False)


def _sign(payload: Dict[str, Any], secret: str, exp_seconds: int) -> str:
    data = {**payload, "exp": int(time.time()) + exp_seconds}
    body = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    token = base64.urlsafe_b64encode(body).decode("ascii") + "." + sig
    return token


def _verify(token: str, secret: str) -> Optional[Dict[str, Any]]:
    try:
        body_b64, sig = token.split(".")
        body = base64.urlsafe_b64decode(body_b64.encode("ascii"))
        expected = hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(body.decode("utf-8"))
        if int(time.time()) > int(data.get("exp", 0)):
            return None
        return data
    except Exception:
        return None


def create_token(email: str, role: str, settings: Settings) -> str:
    return _sign({"email": email, "role": role}, settings.jwt_secret, settings.jwt_exp_hours * 3600)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
    role: Optional[str] = None,
) -> Dict[str, Any]:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth required")
    data = _verify(credentials.credentials, settings.jwt_secret)
    if not data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user = get_user_by_email(data.get("email", ""))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if role and user.get("role") != role:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return user


def require_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    return get_current_user(credentials, settings)


def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    return get_current_user(credentials, settings, role="admin")
