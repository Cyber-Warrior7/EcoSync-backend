from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth_utils import get_current_user
from ..config import Settings, get_settings
from ..db import adjust_credits, get_credits
from ..schemas import RedeemRequest, RedeemResponse, UserCredits

router = APIRouter(prefix="/rewards", tags=["rewards"])


@router.get("/users/{user_id}", response_model=UserCredits)
async def fetch_user_credits(
    user_id: str,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(get_current_user),
) -> UserCredits:
    if current_user.get("id") != user_id and current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    credits = get_credits(user_id)
    return UserCredits(user_id=user_id, credits=credits)


@router.post("/redeem", response_model=RedeemResponse)
async def redeem(
    request: RedeemRequest,
    settings: Settings = Depends(get_settings),
    current_user: dict = Depends(get_current_user),
) -> RedeemResponse:
    if current_user.get("id") != request.user_id and current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    current = get_credits(request.user_id)
    if request.amount > current:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough credits to redeem this reward.",
        )
    remaining = adjust_credits(request.user_id, -abs(request.amount))
    return RedeemResponse(
        user_id=request.user_id,
        amount=request.amount,
        remaining_credits=remaining,
        note=request.note,
    )
