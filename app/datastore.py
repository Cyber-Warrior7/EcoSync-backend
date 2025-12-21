from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "store.json"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


class DataStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state: Dict[str, Any] = {
            "posts": [],
            "lost_found": [],
            "credits": {},
            "embeddings": [],
        }
        self._load()

    def _load(self) -> None:
        if DATA_PATH.exists():
            try:
                with DATA_PATH.open("r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except Exception:
                self._state = {
                    "posts": [],
                    "lost_found": [],
                    "credits": {},
                    "embeddings": [],
                }

    def _save(self) -> None:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATA_PATH.open("w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2, default=str)

    # Posts
    def list_posts(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            posts = sorted(self._state["posts"], key=lambda x: x.get("created_at", ""), reverse=True)
            return posts[:limit]

    def add_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._state["posts"].append(post)
            self._save()
            return post

    def update_post(self, post_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            for p in self._state["posts"]:
                if p.get("id") == post_id:
                    p.update(updates)
                    self._save()
                    return p
        return None

    # Lost & found
    def list_lost_found(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            items = sorted(self._state["lost_found"], key=lambda x: x.get("created_at", ""), reverse=True)
            return items[:limit]

    def add_lost_found(self, item: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._state["lost_found"].append(item)
            self._save()
            return item

    def update_lost_status(self, item_id: str, status: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for item in self._state["lost_found"]:
                if item.get("id") == item_id:
                    item["status"] = status
                    self._save()
                    return item
        return None

    # Credits
    def adjust_credits(self, user_id: str, delta: int) -> int:
        with self._lock:
            current = int(self._state["credits"].get(user_id, 0))
            new_val = max(0, current + delta)
            self._state["credits"][user_id] = new_val
            self._save()
            return new_val

    def get_credits(self, user_id: str) -> int:
        with self._lock:
            return int(self._state["credits"].get(user_id, 0))

    # Embeddings
    def save_embedding(self, post_id: str, user_id: str, vector: List[float]) -> None:
        with self._lock:
            self._state["embeddings"].append({"post_id": post_id, "user_id": user_id, "vector": vector})
            self._save()

    def get_embeddings(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._state["embeddings"])


store = DataStore()

