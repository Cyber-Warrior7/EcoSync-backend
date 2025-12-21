from __future__ import annotations

import google.generativeai as genai
from fastapi import APIRouter, Depends, HTTPException, status

from ..config import Settings, get_settings
from ..schemas import GeminiPrompt, GeminiResponse

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post("/tips", response_model=GeminiResponse)
async def generate_tips(body: GeminiPrompt, settings: Settings = Depends(get_settings)) -> GeminiResponse:
    if settings.offline_mode:
        return GeminiResponse(
            output="(Offline mode) Try reducing single-use plastics, host a cleanup, and reward helpers with credits.",
            model="offline-mock",
        )
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API key is not configured.",
        )
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    parts = [body.prompt]
    if body.context:
        parts.append(f"Context: {body.context}")
    try:
        response = model.generate_content("\n\n".join(parts))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini call failed: {exc}",
        ) from exc
    text = response.text if hasattr(response, "text") else ""
    return GeminiResponse(output=text, model=getattr(response, "model_version", "gemini-1.5-flash"))
