from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth_utils import get_current_user
from ..config import Settings, get_settings
from ..db import get_user_settings, save_user_settings
from ..schemas import UserSettings, UserSettingsUpdate

router = APIRouter(prefix="/users", tags=["user-settings"])


@router.get("/{user_id}/settings", response_model=UserSettings)
async def read_settings(
    user_id: str,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(get_current_user),
) -> UserSettings:
    if not user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id required")
    if current_user.get("id") != user_id and current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    data = get_user_settings(user_id)
    return UserSettings(**data)


@router.put("/{user_id}/settings", response_model=UserSettings)
async def update_settings(
    user_id: str,
    payload: UserSettingsUpdate,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(get_current_user),
) -> UserSettings:
    if not user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id required")
    if current_user.get("id") != user_id and current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    saved = save_user_settings(user_id, payload.username, payload.theme)
    return UserSettings(**saved)
