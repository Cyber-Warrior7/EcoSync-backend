from __future__ import annotations

import hashlib
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth_utils import create_token, require_admin
from ..config import Settings, get_settings
from ..db import create_user, get_user_by_email, list_users
from ..schemas import User, UserCreate, UserList, UserLogin, UserWithToken

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserWithToken, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserCreate, settings: Settings = Depends(get_settings)) -> UserWithToken:
    email_l = payload.email.lower()
    if not email_l.endswith("@gmail.com") and not email_l.endswith("@student.mes.ac.in"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Use Gmail or student.mes.ac.in email")
    if get_user_by_email(payload.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")
    pwd_hash = hashlib.sha256(payload.password.encode("utf-8")).hexdigest()
    user = create_user(payload.email, payload.username, pwd_hash, role="user")
    token = create_token(payload.email, user.get("role", "user"), settings)
    return UserWithToken(**user, token=token)


@router.post("/login", response_model=UserWithToken)
async def login(payload: UserLogin, settings: Settings = Depends(get_settings)) -> UserWithToken:
    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    pwd_hash = hashlib.sha256(payload.password.encode("utf-8")).hexdigest()
    if user.get("password_hash") != pwd_hash:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_token(user["email"], user.get("role", "user"), settings)
    return UserWithToken(
        id=user["id"], email=user["email"], username=user.get("username") or "", role=user.get("role", "user"), token=token
    )


@router.get("/users", response_model=UserList)
async def get_users(current=Depends(require_admin)) -> UserList:
    return UserList(items=list_users())
