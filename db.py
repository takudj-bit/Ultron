"""
PostgreSQL database layer for Ultron.
Falls back to local JSON file when DATABASE_URL is not set (local dev).
"""

from __future__ import annotations

import json
import os
import uuid
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
    research    TEXT NOT NULL DEFAULT '',
    distilled   TEXT NOT NULL DEFAULT '',
    lyrics      TEXT NOT NULL DEFAULT ''
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


def _get_conn():
    """Get a new database connection."""
    conn = psycopg2.connect(_get_database_url())
    conn.autocommit = True
    return conn


def init_db():
    """Create tables if they don't exist."""
    if not _get_database_url():
        return
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
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


# ---------------------------------------------------------------------------
# Public API — same interface, DB or JSON backend
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    """Return all projects with their versions, newest first."""
    if not _get_database_url():
        return _load_json()

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
            projects = [dict(row) for row in cur.fetchall()]

            for proj in projects:
                cur.execute(
                    "SELECT label, lyrics, parent, feedback, created_at as timestamp "
                    "FROM versions WHERE project_id = %s ORDER BY id",
                    (proj["id"],),
                )
                proj["versions"] = [dict(row) for row in cur.fetchall()]
            return projects
    finally:
        conn.close()


def save_project(entry: dict):
    """Insert a new project with its initial version."""
    if not _get_database_url():
        history = _load_json()
        history.insert(0, entry)
        _save_json(history)
        return

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO projects (id, title, created_at, brief, research, distilled, lyrics) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    entry["id"],
                    entry["title"],
                    entry["created_at"],
                    entry["brief"],
                    entry["research"],
                    entry["distilled"],
                    entry["lyrics"],
                ),
            )
            for ver in entry.get("versions", []):
                cur.execute(
                    "INSERT INTO versions (project_id, label, lyrics, parent, feedback, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        entry["id"],
                        ver["label"],
                        ver["lyrics"],
                        ver.get("parent"),
                        ver.get("feedback"),
                        ver.get("timestamp", entry["created_at"]),
                    ),
                )
    finally:
        conn.close()


def get_project(entry_id: str) -> dict | None:
    """Get a single project by ID."""
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
            proj = dict(row)
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
    """Delete a project and its versions."""
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


def update_project_title(entry_id: str, title: str):
    """Update a project's title."""
    if not _get_database_url():
        history = _load_json()
        for e in history:
            if e["id"] == entry_id:
                e["title"] = title
                break
        _save_json(history)
        return

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE projects SET title = %s WHERE id = %s", (title, entry_id))
    finally:
        conn.close()


def add_version(entry_id: str, version: dict, new_lyrics: str):
    """Add a new version to a project and update its current lyrics."""
    if not _get_database_url():
        history = _load_json()
        for entry in history:
            if entry["id"] == entry_id:
                if "versions" not in entry:
                    entry["versions"] = []
                entry["versions"].append(version)
                entry["lyrics"] = new_lyrics
                # Legacy revisions compat
                if "revisions" not in entry:
                    entry["revisions"] = []
                entry["revisions"].append({
                    "feedback": version.get("feedback", ""),
                    "previous_lyrics": "",
                    "timestamp": version.get("timestamp", ""),
                })
                break
        _save_json(history)
        return

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO versions (project_id, label, lyrics, parent, feedback, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    entry_id,
                    version["label"],
                    version["lyrics"],
                    version.get("parent"),
                    version.get("feedback"),
                    version.get("timestamp", datetime.now().isoformat()),
                ),
            )
            cur.execute(
                "UPDATE projects SET lyrics = %s WHERE id = %s",
                (new_lyrics, entry_id),
            )
    finally:
        conn.close()


def get_versions(entry_id: str) -> list[dict]:
    """Get all versions for a project."""
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
