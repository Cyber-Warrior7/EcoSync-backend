from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_db
from .routes import ai, auth, lost_found, posts, rewards, user_settings
from .schemas import HealthResponse

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse()


app.include_router(posts.router)
app.include_router(lost_found.router)
app.include_router(rewards.router)
app.include_router(ai.router)
app.include_router(user_settings.router)
app.include_router(auth.router)


@app.on_event("startup")
async def _startup() -> None:
    init_db()
