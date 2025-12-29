from __future__ import annotations

import sqlite3
from array import array
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ecosync.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    def _ensure_column(table: str, column: str, definition: str) -> None:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row["name"] for row in cur.fetchall()]
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            user_email TEXT NOT NULL,
            caption TEXT,
            media_base64 TEXT,
            media_mime TEXT,
            media_type TEXT,
            media_url TEXT,
            location TEXT,
            ai_summary TEXT,
            credits_awarded INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            verified INTEGER DEFAULT 0,
            review_notes TEXT,
            created_at TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS lost_found (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            user_email TEXT NOT NULL,
            title TEXT,
            description TEXT,
            location TEXT,
            contact TEXT,
            image_url TEXT,
            status TEXT DEFAULT 'open',
            credits_awarded INTEGER DEFAULT 0,
            created_at TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS credits (
            user_id TEXT PRIMARY KEY,
            credits INTEGER DEFAULT 0
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            post_id TEXT,
            user_id TEXT,
            vector BLOB
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mobilenet_embeddings (
            post_id TEXT,
            user_id TEXT,
            vector BLOB
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            theme TEXT DEFAULT 'dark'
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            username TEXT,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT
        );
        """
    )
    _ensure_column("user_settings", "username", "TEXT")
    _ensure_column("user_settings", "theme", "TEXT DEFAULT 'dark'")
    # Backward compatibility for existing local DBs that were created without these columns
    _ensure_column("posts", "user_email", "TEXT NOT NULL DEFAULT ''")
    _ensure_column("lost_found", "user_email", "TEXT NOT NULL DEFAULT ''")
    # Seed admin account if missing
    cur.execute("SELECT 1 FROM users WHERE email = ?", ("admin@ecosync.local",))
    if not cur.fetchone():
        import hashlib
        cur.execute(
            """
            INSERT INTO users (id, email, username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                "user-admin",
                "admin@ecosync.local",
                "admin",
                hashlib.sha256("Admin@123".encode("utf-8")).hexdigest(),
                "admin",
            ),
        )
    conn.commit()
    conn.close()

def purge_rejected_posts(max_age_hours: int = 24) -> None:
    """Delete rejected posts older than the given age (also removes embeddings)."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    stale_ids: List[str] = []
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, created_at FROM posts WHERE status = 'rejected'")
    for row in cur.fetchall():
        created_raw = row["created_at"]
        if not created_raw:
            continue
        try:
            dt = datetime.fromisoformat(created_raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            if dt < cutoff:
                stale_ids.append(row["id"])
        except Exception:
            continue
    conn.close()
    for post_id in stale_ids:
        delete_post_record(post_id)


def list_posts(limit: int = 25) -> List[Dict[str, Any]]:
    purge_rejected_posts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM posts ORDER BY datetime(created_at) DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def add_post(doc: Dict[str, Any]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO posts (
            id, user_id, user_email, caption, media_base64, media_mime, media_type, media_url,
            location, ai_summary, credits_awarded, status, verified, review_notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc["id"],
            doc["user_id"],
            doc["user_email"],
            doc.get("caption"),
            doc.get("media_base64"),
            doc.get("media_mime"),
            doc.get("media_type"),
            doc.get("media_url"),
            doc.get("location"),
            doc.get("ai_summary"),
            int(doc.get("credits_awarded", 0)),
            doc.get("status", "pending"),
            1 if doc.get("verified") else 0,
            doc.get("review_notes"),
            doc.get("created_at"),
        ),
    )
    conn.commit()
    conn.close()


def update_post(post_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not updates:
        return None
    fields = []
    values = []
    for key, value in updates.items():
        fields.append(f"{key} = ?")
        values.append(value)
    values.append(post_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"UPDATE posts SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()
    cur.execute("SELECT * FROM posts WHERE id = ?", (post_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_post(post_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM posts WHERE id = ?", (post_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_lost_found(limit: int = 25) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM lost_found ORDER BY datetime(created_at) DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def add_lost_found(item: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO lost_found (
            id, user_id, user_email, title, description, location, contact, image_url,
            status, credits_awarded, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item["id"],
            item["user_id"],
            item["user_email"],
            item.get("title"),
            item.get("description"),
            item.get("location"),
            item.get("contact"),
            item.get("image_url"),
            item.get("status", "open"),
            int(item.get("credits_awarded", 0)),
            item.get("created_at"),
        ),
    )
    conn.commit()
    conn.close()
    return item


def find_duplicate_lost(user_id: str, title: str, description: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM lost_found
        WHERE user_id = ? AND (LOWER(title) = LOWER(?) OR LOWER(description) = LOWER(?))
        ORDER BY datetime(created_at) DESC
        """,
        (user_id, title, description),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def update_lost_status(item_id: str, status: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE lost_found SET status = ? WHERE id = ?", (status, item_id))
    conn.commit()
    cur.execute("SELECT * FROM lost_found WHERE id = ?", (item_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def adjust_credits(user_id: str, delta: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT credits FROM credits WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    current = int(row["credits"]) if row else 0
    new_val = max(0, current + delta)
    if row:
        cur.execute("UPDATE credits SET credits = ? WHERE user_id = ?", (new_val, user_id))
    else:
        cur.execute("INSERT INTO credits (user_id, credits) VALUES (?, ?)", (user_id, new_val))
    conn.commit()
    conn.close()
    return new_val


def get_credits(user_id: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT credits FROM credits WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return int(row["credits"]) if row else 0


def save_embedding(post_id: str, user_id: str, vector: List[float]) -> None:
    buf = array("f", vector).tobytes()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO embeddings (post_id, user_id, vector) VALUES (?, ?, ?)",
        (post_id, user_id, sqlite3.Binary(buf)),
    )
    conn.commit()
    conn.close()


def get_embeddings() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT post_id, user_id, vector FROM embeddings")
    rows = []
    for row in cur.fetchall():
        vec_bytes = row["vector"]
        arr = array("f")
        arr.frombytes(vec_bytes)
        rows.append({"post_id": row["post_id"], "user_id": row["user_id"], "vector": list(arr)})
    conn.close()
    return rows


def save_mobilenet_embedding(post_id: str, user_id: str, vector: List[float]) -> None:
    buf = array("f", vector).tobytes()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mobilenet_embeddings (post_id, user_id, vector) VALUES (?, ?, ?)",
        (post_id, user_id, sqlite3.Binary(buf)),
    )
    conn.commit()
    conn.close()


def get_mobilenet_embeddings() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT post_id, user_id, vector FROM mobilenet_embeddings")
    rows = []
    for row in cur.fetchall():
        vec_bytes = row["vector"]
        arr = array("f")
        arr.frombytes(vec_bytes)
        rows.append({"post_id": row["post_id"], "user_id": row["user_id"], "vector": list(arr)})
    conn.close()
    return rows


def delete_post_record(post_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    cur.execute("DELETE FROM embeddings WHERE post_id = ?", (post_id,))
    conn.commit()
    conn.close()


def delete_lost_found_record(item_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM lost_found WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()


def create_user(email: str, username: str, password_hash: str, role: str = "user") -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    user_id = f"user-{abs(hash(email))}"
    cur.execute(
        """
        INSERT INTO users (id, email, username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        """,
        (user_id, email.lower(), username, password_hash, role),
    )
    conn.commit()
    cur.execute("SELECT id, email, username, role FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, email, username, password_hash, role FROM users WHERE email = ?", (email.lower(),))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_users() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, email, username, role, created_at FROM users ORDER BY datetime(created_at) DESC")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_user_settings(user_id: str) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id, username, theme FROM user_settings WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {"user_id": user_id, "username": "", "theme": "dark"}


def save_user_settings(user_id: str, username: str, theme: str) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_settings (user_id, username, theme)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET username = excluded.username, theme = excluded.theme
        """,
        (user_id, username, theme),
    )
    conn.commit()
    conn.close()
    return {"user_id": user_id, "username": username, "theme": theme}
