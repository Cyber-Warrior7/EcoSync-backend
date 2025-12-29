import base64
import os
from typing import Any, Dict, Optional

import httpx

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL")
AI_SERVICE_KEY = os.getenv("AI_SERVICE_KEY")


async def call_ai_service(media_bytes: bytes, user_id: str, post_id: str) -> Optional[Dict[str, Any]]:
    if not AI_SERVICE_URL:
        return None
    payload = {
        "user_id": user_id,
        "post_id": post_id,
        "media_base64": base64.b64encode(media_bytes).decode("ascii"),
    }
    headers = {"X-AI-KEY": AI_SERVICE_KEY} if AI_SERVICE_KEY else {}
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(f"{AI_SERVICE_URL.rstrip('/')}/ai/verify", json=payload, headers=headers)
            if resp.status_code != 200:
                return {"status": "pending", "notes": f"ai-error: {resp.text}"}
            return resp.json()
        except Exception as exc:
            return {"status": "pending", "notes": f"ai-error: {exc}"}
