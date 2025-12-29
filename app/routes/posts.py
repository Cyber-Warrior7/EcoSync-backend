from __future__ import annotations

import base64
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth_utils import get_current_user, require_admin
from ..ai_client import call_ai_service
from ..config import Settings, get_settings
from ..db import (
    add_post,
    adjust_credits,
    delete_post_record,
    get_post,
    list_posts as db_list_posts,
    update_post,
    get_user_settings,
)
from ..schemas import PaginatedPosts, Post, PostCreate
from ..services.verification import assess_image_authenticity, embed_media, find_near_duplicate, save_embedding

router = APIRouter(prefix="/posts", tags=["posts"])


@router.get("", response_model=PaginatedPosts)
async def list_posts(limit: int = 25, settings: Settings = Depends(get_settings)) -> PaginatedPosts:
    items = db_list_posts(limit)
    posts = [
        Post(
            id=item.get("id", ""),
            user_id=item.get("user_id", ""),
            caption=item.get("caption", ""),
            user_email=item.get("user_email", ""),
            username=item.get("username", ""),
            media_url=item.get("media_url", ""),
            media_type=item.get("media_type", ""),
            media_base64=item.get("media_base64", ""),
            media_mime=item.get("media_mime", ""),
            location=item.get("location"),
            ai_summary=item.get("ai_summary"),
            credits_awarded=int(item.get("credits_awarded", 0)),
            created_at=datetime.fromisoformat(item["created_at"]) if item.get("created_at") else None,
            status=item.get("status", "pending"),
            verified=item.get("verified", False),
            review_notes=item.get("review_notes"),
        )
        for item in items
    ]
    return PaginatedPosts(items=posts, count=len(posts))


@router.post("", response_model=Post, status_code=status.HTTP_201_CREATED)
async def create_post(
    payload: PostCreate,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(get_current_user),
) -> Post:
    if current_user.get("id") != payload.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User mismatch")
    try:
        media_bytes = base64.b64decode(payload.media_base64)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 media") from exc

    post_id = f"post-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    created_at_iso = datetime.now(timezone.utc).isoformat()
    data_url = f"data:{payload.media_mime};base64,{payload.media_base64}"
    base_doc = {
        "id": post_id,
        "user_id": payload.user_id,
        "user_email": payload.user_email,
        "username": payload.username,
        "caption": payload.caption,
        "media_url": data_url,
        "media_base64": payload.media_base64,
        "media_mime": payload.media_mime,
        "media_type": payload.media_type,
        "location": payload.location,
        "ai_summary": payload.ai_summary,
        "credits_awarded": 0,
        "status": "pending",
        "verified": False,
        "review_notes": None,
        "created_at": created_at_iso,
    }

    add_post(base_doc)

    vector, _ = await embed_media(media_bytes, settings)
    dup = await find_near_duplicate(vector, settings, user_id=payload.user_id)
    if dup:
        updated = update_post(
            post_id,
            {
                "status": "rejected",
                "verified": False,
                "review_notes": f"Near-duplicate detected (similar to {dup.get('post_id')})",
                "credits_awarded": 0,
            },
        )
        return Post(**updated) if updated else Post(**base_doc)

    verdict, notes = await assess_image_authenticity(media_bytes, settings)
    gemini_issue = isinstance(notes, str) and notes.startswith("gemini-error")
    if settings.offline_mode or not settings.gemini_api_key or notes == "offline-auto-verified" or gemini_issue:
        ai_result = await call_ai_service(media_bytes, payload.user_id, post_id)
        if ai_result:
            status_val = ai_result.get("status", "pending")
            credits_val = int(ai_result.get("credits_awarded", 0) or 0)
            is_verified = status_val == "verified"
            update_post(
                post_id,
                {
                    "verified": is_verified,
                    "status": status_val,
                    "review_notes": ai_result.get("notes", notes),
                    "credits_awarded": credits_val,
                },
            )
            if is_verified and credits_val > 0:
                adjust_credits(payload.user_id, credits_val)
        else:
            update_post(
                post_id,
                {
                    "verified": False,
                    "status": "pending",
                    "review_notes": notes,
                    "credits_awarded": 0,
                },
            )
    elif verdict:
        credits = settings.default_reward_per_post
        update_post(
            post_id,
            {
                "verified": True,
                "status": "verified",
                "review_notes": notes,
                "credits_awarded": credits,
            },
        )
        adjust_credits(payload.user_id, credits)
    else:
        update_post(
            post_id,
            {
                "verified": False,
                "status": "pending",
                "credits_awarded": 0,
                "review_notes": notes,
            },
        )

    await save_embedding(post_id, payload.user_id, vector, settings)
    saved = get_post(post_id) or base_doc
    settings_row = get_user_settings(saved.get("user_id", ""))
    saved["username"] = settings_row.get("username") if settings_row else ""
    return Post(**saved)


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: str, settings: Settings = Depends(get_settings), current_user: dict = Depends(require_admin)
) -> None:
    delete_post_record(post_id)


@router.post("/{post_id}/approve", response_model=Post)
async def approve_post(
    post_id: str,
    credits: int = 10,
    review_notes: str | None = None,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(require_admin),
) -> Post:
    existing = get_post(post_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    new_credits = max(0, credits)
    old_credits = int(existing.get("credits_awarded") or 0)
    delta = new_credits - old_credits
    update_post(
        post_id,
        {
            "verified": True,
            "status": "verified",
            "review_notes": review_notes or "Approved by admin",
            "credits_awarded": new_credits,
        },
    )
    if delta != 0:
        adjust_credits(existing["user_id"], delta)
    saved = get_post(post_id)
    if saved:
        settings_row = get_user_settings(saved.get("user_id", ""))
        saved["username"] = settings_row.get("username") if settings_row else ""
    return Post(**saved) if saved else Post(**existing)


@router.post("/{post_id}/reject", response_model=Post)
async def reject_post(
    post_id: str,
    reason: str | None = None,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(require_admin),
) -> Post:
    existing = get_post(post_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    old_credits = int(existing.get("credits_awarded") or 0)
    update_post(
        post_id,
        {
            "verified": False,
            "status": "rejected",
            "review_notes": reason or "Rejected by admin",
            "credits_awarded": 0,
        },
    )
    if old_credits:
        adjust_credits(existing["user_id"], -old_credits)
    saved = get_post(post_id)
    if saved:
        settings_row = get_user_settings(saved.get("user_id", ""))
        saved["username"] = settings_row.get("username") if settings_row else ""
    return Post(**saved) if saved else Post(**existing)
