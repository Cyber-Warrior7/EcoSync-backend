from __future__ import annotations

import base64
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from ..config import Settings
from ..db import get_embeddings, save_embedding as db_save_embedding


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _hash_embedding(data: bytes, dims: int = 64) -> List[float]:
    """Simple deterministic embedding from bytes for duplicate checks."""
    digest = hashlib.sha256(data).digest()
    full = (digest * ((dims // len(digest)) + 1))[:dims]
    return [byte / 255.0 for byte in full]


async def embed_media(content: bytes, settings: Settings) -> Tuple[List[float], str]:
    """Return embedding and source description."""
    if settings.offline_mode or not settings.gemini_api_key:
        return _hash_embedding(content), "offline-hash"

    genai.configure(api_key=settings.gemini_api_key)
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content={"parts": [{"inline_data": {"mime_type": "application/octet-stream", "data": content}}]},
            task_type="RETRIEVAL_DOCUMENT",
        )
        vector = result.get("embedding", [])
        if not vector:
            return _hash_embedding(content), "fallback-hash"
        return vector, result.get("model", "models/text-embedding-004")
    except Exception:
        return _hash_embedding(content), "fallback-hash"


async def assess_image_authenticity(content: bytes, settings: Settings) -> Tuple[bool, str]:
    """Use Gemini to check for human face + waste disposal action; offline auto-approve."""
    if settings.offline_mode or not settings.gemini_api_key:
        # In offline mode we cannot validate authenticity, so mark as pending without credits
        return False, "offline-auto-verified"

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are an authenticity checker for cleanup proof photos. "
        "Return only 'YES' or 'NO'. Criteria for YES: "
        "1) A human face is visible (selfie or partial face counts). "
        '2) The person is actively placing waste into a bin/trash can/recycling container. '
        "3) The image is not a stock illustration or obvious duplicate/edited stock. "
        "If uncertain, respond 'NO'."
    )
    try:
        # Inline base64 image
        image_b64 = base64.b64encode(content).decode("ascii")
        response = model.generate_content(
            [prompt, {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]
        )
        text = (response.text or "").strip().lower()
        verdict = "yes" in text and "no" not in text[:10]
        return verdict, text
    except Exception as exc:
        return False, f"gemini-error: {exc}"


async def find_near_duplicate(
    vector: List[float], settings: Settings, threshold: float = 0.92, user_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    candidates = [c for c in get_embeddings() if (user_id is None or c["user_id"] == user_id)]
    best = None
    best_score = 0.0
    for item in candidates:
        score = _cosine(vector, item["vector"])
        if score > threshold and score > best_score:
            best = {**item, "score": score}
            best_score = score
    return best


async def save_embedding(post_id: str, user_id: str, vector: List[float], settings: Settings) -> None:
    db_save_embedding(post_id, user_id, vector)
