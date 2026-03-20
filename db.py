"""
PostgreSQL database layer for Ultron v2.
Falls back to local JSON file when DATABASE_URL is not set (local dev).

5-step workflow schema:
  ① brief → ② research → ③ directions → ④ select → ⑤ lyrics+suno
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
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    step                INTEGER NOT NULL DEFAULT 1,
    brief               TEXT NOT NULL DEFAULT '',
    research            TEXT NOT NULL DEFAULT '',
    summary             TEXT NOT NULL DEFAULT '{}',
    directions          TEXT NOT NULL DEFAULT '[]',
    selected_direction  INTEGER,
    lyrics              TEXT NOT NULL DEFAULT '',
    suno_prompt         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS versions (
    id          SERIAL PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    lyrics      TEXT NOT NULL DEFAULT '',
    parent      TEXT,
    feedback    TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_versions_project ON versions(project_id);
"""

MIGRATE_SQL = [
    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS step INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS directions TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS selected_direction INTEGER",
    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS suno_prompt TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT '{}'",
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
        data = json.loads(HISTORY_FILE.read_text("utf-8"))
        for d in data:
            if isinstance(d.get("directions"), str):
                try:
                    d["directions"] = json.loads(d["directions"])
                except Exception:
                    d["directions"] = []
            if isinstance(d.get("summary"), str):
                try:
                    d["summary"] = json.loads(d["summary"])
                except Exception:
                    d["summary"] = {}
        return data
    return []


def _save_json(history: list[dict]):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")


def _row_to_dict(row: dict) -> dict:
    """Convert DB row to API-compatible dict (parse JSON fields)."""
    d = dict(row)
    if isinstance(d.get("directions"), str):
        try:
            d["directions"] = json.loads(d["directions"])
        except Exception:
            d["directions"] = []
    if isinstance(d.get("summary"), str):
        try:
            d["summary"] = json.loads(d["summary"])
        except Exception:
            d["summary"] = {}
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    if not _get_database_url():
        return _load_json()

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
            projects = [_row_to_dict(row) for row in cur.fetchall()]
            for proj in projects:
                cur.execute(
                    "SELECT label, lyrics, parent, feedback, created_at as timestamp "
                    "FROM versions WHERE project_id = %s ORDER BY id",
                    (proj["id"],),
                )
                proj["versions"] = [dict(r) for r in cur.fetchall()]
            return projects
    finally:
        conn.close()


def save_project(entry: dict):
    if not _get_database_url():
        history = _load_json()
        history.insert(0, entry)
        _save_json(history)
        return

    conn = _get_conn()
    try:
        directions_json = json.dumps(entry.get("directions", []), ensure_ascii=False)
        summary_json = json.dumps(entry.get("summary", {}), ensure_ascii=False)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO projects (id, title, created_at, step, brief, research, summary, directions, selected_direction, lyrics, suno_prompt) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    entry["id"],
                    entry.get("title", ""),
                    entry["created_at"],
                    entry.get("step", 1),
                    entry.get("brief", ""),
                    entry.get("research", ""),
                    summary_json,
                    directions_json,
                    entry.get("selected_direction"),
                    entry.get("lyrics", ""),
                    entry.get("suno_prompt", ""),
                ),
            )
            for ver in entry.get("versions", []):
                cur.execute(
                    "INSERT INTO versions (project_id, label, lyrics, parent, feedback, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (entry["id"], ver["label"], ver["lyrics"],
                     ver.get("parent"), ver.get("feedback"),
                     ver.get("timestamp", entry["created_at"])),
                )
    finally:
        conn.close()


def update_project(entry_id: str, **fields):
    """Update arbitrary fields on a project."""
    if not _get_database_url():
        history = _load_json()
        for e in history:
            if e["id"] == entry_id:
                e.update(fields)
                break
        _save_json(history)
        return

    allowed = {"title", "step", "brief", "research", "summary", "directions",
               "selected_direction", "lyrics", "suno_prompt"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return

    if "directions" in updates and not isinstance(updates["directions"], str):
        updates["directions"] = json.dumps(updates["directions"], ensure_ascii=False)
    if "summary" in updates and not isinstance(updates["summary"], str):
        updates["summary"] = json.dumps(updates["summary"], ensure_ascii=False)

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
                return e
        return None

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM projects WHERE id = %s", (entry_id,))
            row = cur.fetchone()
            if not row:
                return None
            proj = _row_to_dict(row)
            cur.execute(
                "SELECT label, lyrics, parent, feedback, created_at as timestamp "
                "FROM versions WHERE project_id = %s ORDER BY id",
                (entry_id,),
            )
            proj["versions"] = [dict(r) for r in cur.fetchall()]
            return proj
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


def add_version(entry_id: str, version: dict, new_lyrics: str):
    if not _get_database_url():
        history = _load_json()
        for entry in history:
            if entry["id"] == entry_id:
                if "versions" not in entry:
                    entry["versions"] = []
                entry["versions"].append(version)
                entry["lyrics"] = new_lyrics
                break
        _save_json(history)
        return

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO versions (project_id, label, lyrics, parent, feedback, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (entry_id, version["label"], version["lyrics"],
                 version.get("parent"), version.get("feedback"),
                 version.get("timestamp", datetime.now().isoformat())),
            )
            cur.execute("UPDATE projects SET lyrics = %s WHERE id = %s", (new_lyrics, entry_id))
    finally:
        conn.close()


def get_versions(entry_id: str) -> list[dict]:
    if not _get_database_url():
        for e in _load_json():
            if e["id"] == entry_id:
                return e.get("versions", [])
        return []

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT label, lyrics, parent, feedback, created_at as timestamp "
                "FROM versions WHERE project_id = %s ORDER BY id",
                (entry_id,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
