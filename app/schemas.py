from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "EcoSync API is running"


class PostCreate(BaseModel):
    user_id: str = Field(..., description="User ID")
    user_email: str = Field(..., description="User email for display")
    username: Optional[str] = None
    caption: str
    media_base64: str = Field(..., description="Base64-encoded media data (image or video)")
    media_mime: str = Field(..., description="MIME type of the media (e.g., image/png)")
    media_type: str = Field(..., description="image or video")
    location: Optional[str] = None
    ai_summary: Optional[str] = Field(None, description="Optional AI-generated summary/validation about the upload")


class Post(PostCreate):
    id: str
    credits_awarded: int
    created_at: Optional[datetime] = None
    status: str = "pending"
    verified: bool = False
    review_notes: Optional[str] = None
    media_url: Optional[str] = None  # data URL for display


class LostFoundCreate(BaseModel):
    user_id: str
    user_email: str
    title: str
    description: str
    location: str
    contact: str
    image_url: Optional[HttpUrl | str] = None


class LostFoundItem(LostFoundCreate):
    id: str
    credits_awarded: int
    status: str = "open"
    created_at: Optional[datetime] = None


class CreditUpdate(BaseModel):
    user_id: str
    delta: int
    reason: str


class UserCredits(BaseModel):
    user_id: str
    credits: int


class UserSettings(BaseModel):
    user_id: str
    username: str
    theme: str = "dark"


class UserSettingsUpdate(BaseModel):
    username: str
    theme: str = "dark"


class UserCreate(BaseModel):
    email: str
    username: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class User(BaseModel):
    id: str
    email: str
    username: str
    role: str = "user"


class UserWithToken(User):
    token: str


class UserList(BaseModel):
    items: List[User]


class RedeemRequest(BaseModel):
    user_id: str
    amount: int
    note: Optional[str] = None


class RedeemResponse(BaseModel):
    user_id: str
    amount: int
    remaining_credits: int
    note: Optional[str] = None


class GeminiPrompt(BaseModel):
    prompt: str
    context: Optional[str] = None


class GeminiResponse(BaseModel):
    output: str
    model: str


class PaginatedPosts(BaseModel):
    items: List[Post]
    count: int


class PaginatedLostFound(BaseModel):
    items: List[LostFoundItem]
    count: int
