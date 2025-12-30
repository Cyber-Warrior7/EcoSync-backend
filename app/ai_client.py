import base64
import os
import asyncio
from typing import Any, Dict, Optional

import httpx

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL")
AI_SERVICE_KEY = os.getenv("AI_SERVICE_KEY")
AI_SERVICE_TIMEOUT = float(os.getenv("AI_SERVICE_TIMEOUT", "25"))
AI_SERVICE_RETRIES = int(os.getenv("AI_SERVICE_RETRIES", "2"))


async def call_ai_service(media_bytes: bytes, user_id: str, post_id: str) -> Optional[Dict[str, Any]]:
    if not AI_SERVICE_URL:
        return None
    payload = {
        "user_id": user_id,
        "post_id": post_id,
        "media_base64": base64.b64encode(media_bytes).decode("ascii"),
    }
    headers = {"X-AI-KEY": AI_SERVICE_KEY} if AI_SERVICE_KEY else {}
    attempts = max(1, AI_SERVICE_RETRIES)
    async with httpx.AsyncClient(timeout=AI_SERVICE_TIMEOUT) as client:
        last_error = None
        for attempt in range(attempts):
            try:
                resp = await client.post(f"{AI_SERVICE_URL.rstrip('/')}/ai/verify", json=payload, headers=headers)
                if resp.status_code == 200:
                    return resp.json()
                last_error = resp.text
            except Exception as exc:
                last_error = str(exc)
            if attempt < attempts - 1:
                await asyncio.sleep(2)
        return {"status": "pending", "notes": "Pending manual review (AI service unavailable)"}
