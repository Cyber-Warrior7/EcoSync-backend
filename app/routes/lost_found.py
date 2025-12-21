from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth_utils import get_current_user, require_admin
from ..config import Settings, get_settings
from ..db import (
    add_lost_found,
    delete_lost_found_record,
    find_duplicate_lost,
    get_user_settings,
    list_lost_found,
    update_lost_status,
)
from ..schemas import LostFoundCreate, LostFoundItem, PaginatedLostFound

router = APIRouter(prefix="/lost-found", tags=["lost-found"])


@router.get("", response_model=PaginatedLostFound)
async def list_items(limit: int = 25, settings: Settings = Depends(get_settings)) -> PaginatedLostFound:
    items = list_lost_found(limit)
    parsed = [
        LostFoundItem(
            id=item.get("id", ""),
            user_id=item.get("user_id", ""),
            user_email=item.get("user_email", ""),
            username=(get_user_settings(item.get("user_id", "")) or {}).get("username", ""),
            title=item.get("title", ""),
            description=item.get("description", ""),
            location=item.get("location", ""),
            contact=item.get("contact", ""),
            image_url=item.get("image_url"),
            status=item.get("status", "open"),
            credits_awarded=int(item.get("credits_awarded", 0)),
            created_at=item.get("created_at"),
        )
        for item in items
    ]
    return PaginatedLostFound(items=parsed, count=len(parsed))


@router.post("", response_model=LostFoundItem, status_code=status.HTTP_201_CREATED)
async def report_found_item(
    payload: LostFoundCreate,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(get_current_user),
) -> LostFoundItem:
    if current_user.get("id") != payload.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User mismatch")
    credits = 0  # no credits for reporting
    dup = find_duplicate_lost(payload.user_id, payload.title, payload.description)
    if dup:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate lost & found report detected for this user.",
        )
    item = {
        "id": f"lost-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        **payload.model_dump(),
        "status": "open",
        "credits_awarded": credits,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    added = add_lost_found(item)
    return LostFoundItem(**added)


@router.patch("/{item_id}/status/{status}", response_model=LostFoundItem)
async def update_status(
    item_id: str,
    status: str,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(require_admin),
) -> LostFoundItem:
    updated = update_lost_status(item_id, status)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    return LostFoundItem(**updated)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(
    item_id: str,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(require_admin),
) -> None:
    delete_lost_found_record(item_id)
