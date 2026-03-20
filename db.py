"""
PostgreSQL database layer for Ultron v3 — Agent-based chat.
Falls back to local JSON file when DATABASE_URL is not set (local dev).

Schema:
  projects  — brief + research (one-time analysis)
  messages  — full chat history with optional artifacts
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras


def _get_database_url():
    return os.getenv("DATABASE_URL")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    brief       TEXT NOT NULL DEFAULT '',
    research    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    agent           TEXT,
    artifact_type   TEXT,
    artifact_json   TEXT,
    metadata        TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id);
"""

MIGRATE_SQL = [
    """CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        agent TEXT,
        artifact_type TEXT,
        artifact_json TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id)",
]


def _get_conn():
    conn = psycopg2.connect(_get_database_url())
    conn.autocommit = True
    return conn


def init_db():
    if not _get_database_url():
        return
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            for sql in MIGRATE_SQL:
                try:
                    cur.execute(sql)
                except Exception:
                    pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# JSON fallback (local dev)
# ---------------------------------------------------------------------------
HISTORY_FILE = Path(__file__).parent / "data" / "history.json"
HISTORY_FILE.parent.mkdir(exist_ok=True)


def _load_json() -> list[dict]:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text("utf-8"))
    return []


def _save_json(history: list[dict]):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")


def _parse_message(msg: dict) -> dict:
    """Parse JSON strings in a message dict."""
    m = dict(msg)
    if isinstance(m.get("artifact_json"), str):
        try:
            m["artifact"] = json.loads(m["artifact_json"])
        except Exception:
            m["artifact"] = None
    else:
        m["artifact"] = m.get("artifact_json")
    if isinstance(m.get("metadata"), str):
        try:
            m["metadata"] = json.loads(m["metadata"])
        except Exception:
            m["metadata"] = {}
    elif not isinstance(m.get("metadata"), dict):
        m["metadata"] = {}
    return m


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def save_project(entry: dict):
    if not _get_database_url():
        proj = {
            "id": entry["id"],
            "title": entry.get("title", ""),
            "created_at": entry["created_at"],
            "brief": entry.get("brief", ""),
            "research": entry.get("research", ""),
            "messages": [],
        }
        history = _load_json()
        history.insert(0, proj)
        _save_json(history)
        return

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO projects (id, title, created_at, brief, research) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (entry["id"], entry.get("title", ""), entry["created_at"],
                 entry.get("brief", ""), entry.get("research", "")),
            )
    finally:
        conn.close()


def update_project(entry_id: str, **fields):
    if not _get_database_url():
        history = _load_json()
        for e in history:
            if e["id"] == entry_id:
                for k, v in fields.items():
                    if k in ("title", "brief", "research"):
                        e[k] = v
                break
        _save_json(history)
        return

    allowed = {"title", "brief", "research"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sets = ", ".join(f"{k} = %s" for k in updates)
            vals = list(updates.values()) + [entry_id]
            cur.execute(f"UPDATE projects SET {sets} WHERE id = %s", vals)
    finally:
        conn.close()


def get_project(entry_id: str) -> dict | None:
    if not _get_database_url():
        for e in _load_json():
            if e["id"] == entry_id:
                return {k: v for k, v in e.items() if k != "messages"}
        return None

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, created_at, brief, research FROM projects WHERE id = %s",
                (entry_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def load_history() -> list[dict]:
    if not _get_database_url():
        return [{k: v for k, v in e.items() if k != "messages"} for e in _load_json()]

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, created_at FROM projects ORDER BY created_at DESC"
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def delete_project(entry_id: str):
    if not _get_database_url():
        history = _load_json()
        history = [e for e in history if e["id"] != entry_id]
        _save_json(history)
        return

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM projects WHERE id = %s", (entry_id,))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def save_message(project_id: str, role: str, content: str,
                 agent: str | None = None, artifact_type: str | None = None,
                 artifact=None, metadata: dict | None = None) -> dict:
    now = datetime.now().isoformat()
    artifact_str = json.dumps(artifact, ensure_ascii=False) if artifact is not None else None
    metadata_str = json.dumps(metadata or {}, ensure_ascii=False)

    msg = {
        "role": role, "content": content, "agent": agent,
        "artifact_type": artifact_type, "artifact": artifact,
        "metadata": metadata or {}, "created_at": now,
    }

    if not _get_database_url():
        history = _load_json()
        for e in history:
            if e["id"] == project_id:
                if "messages" not in e:
                    e["messages"] = []
                e["messages"].append({
                    "role": role, "content": content, "agent": agent,
                    "artifact_type": artifact_type, "artifact_json": artifact_str,
                    "metadata": metadata_str, "created_at": now,
                })
                break
        _save_json(history)
        return msg

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages "
                "(project_id, role, content, agent, artifact_type, artifact_json, metadata, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (project_id, role, content, agent, artifact_type, artifact_str, metadata_str, now),
            )
    finally:
        conn.close()
    return msg


def get_messages(project_id: str) -> list[dict]:
    if not _get_database_url():
        for e in _load_json():
            if e["id"] == project_id:
                return [_parse_message(m) for m in e.get("messages", [])]
        return []

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, agent, artifact_type, artifact_json, metadata, created_at "
                "FROM messages WHERE project_id = %s ORDER BY id ASC",
                (project_id,),
            )
            return [_parse_message(dict(row)) for row in cur.fetchall()]
    finally:
        conn.close()


def get_latest_artifacts(project_id: str) -> dict:
    """Return the most recent artifact of each type: {directions: ..., lyrics: ..., suno: ...}"""
    messages = get_messages(project_id)
    artifacts: dict = {}
    for msg in reversed(messages):
        atype = msg.get("artifact_type")
        if atype and atype not in artifacts:
            artifacts[atype] = msg.get("artifact")
    return artifacts
