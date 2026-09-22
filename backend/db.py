"""
db.py - SQLite Session & Message Storage for Agent OTG

Replaces PostgreSQL with Python's built-in sqlite3 module.
All session history and messages are stored locally in agent_otg.db in the
project root, created automatically with zero external dependencies or configuration.
"""
from __future__ import annotations

import datetime
import os
import sqlite3
import threading
from pathlib import Path

# Database file in the project root directory
_DB_PATH = Path(__file__).resolve().parent / "agent_otg.db"
_local = threading.local()
_ready = False
_error = ""


def _now_iso() -> str:
    """Return current timestamp as ISO-8601 formatted text string."""
    return datetime.datetime.now().isoformat()


def _connection() -> sqlite3.Connection:
    """Thread-local SQLite connection with WAL mode and foreign keys enabled."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        _local.conn = conn
    return conn


def init_db() -> bool:
    """Initialize database tables. Returns True on success, False on error."""
    global _ready, _error
    try:
        conn = _connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_name TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL REFERENCES agent_sessions(session_name) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON agent_messages(session_name, id);
        """)
        conn.commit()
        _ready, _error = True, ""
        return True
    except Exception as exc:
        _ready, _error = False, str(exc)
        return False


def is_ready() -> bool:
    """Check if the database connection is initialized and ready."""
    return _ready or init_db()


def get_error() -> str:
    """Return the last database error message, if any."""
    return _error


def create_session(session_name: str) -> bool:
    """Create a new session record if it doesn't already exist."""
    try:
        if not is_ready():
            return False
        conn = _connection()
        conn.execute(
            "INSERT OR IGNORE INTO agent_sessions (session_name, created_at) VALUES (?, ?)",
            (session_name, _now_iso()),
        )
        conn.commit()
        return True
    except Exception as exc:
        global _error
        _error = str(exc)
        return False


def save_message(session_name: str, role: str, content: str) -> bool:
    """Save a user or assistant message to the session."""
    try:
        if not create_session(session_name):
            return False
        conn = _connection()
        conn.execute(
            "INSERT INTO agent_messages (session_name, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_name, role, content, _now_iso()),
        )
        conn.commit()
        return True
    except Exception as exc:
        global _error
        _error = str(exc)
        return False


def get_all_sessions(limit: int = 20) -> list[dict]:
    """
    Retrieve recent sessions with message count and last activity.
    Returns dicts with keys: session_name, created_at, message_count, last_activity.
    """
    if not is_ready():
        return []
    try:
        conn = _connection()
        cursor = conn.execute("""
            SELECT 
                s.session_name, 
                s.created_at, 
                COUNT(m.id) AS message_count,
                COALESCE(MAX(m.created_at), s.created_at) AS last_activity
            FROM agent_sessions s 
            LEFT JOIN agent_messages m ON m.session_name = s.session_name
            GROUP BY s.session_name 
            ORDER BY last_activity DESC 
            LIMIT ?
        """, (max(1, min(limit, 200)),))
        return [dict(row) for row in cursor.fetchall()]
    except Exception as exc:
        global _error
        _error = str(exc)
        return []


def get_session_messages(session_name: str, limit: int = 50) -> list[dict]:
    """
    Retrieve message history for a specific session in chronological order.
    Returns dicts with keys: role, content, created_at.
    """
    if not is_ready():
        return []
    try:
        conn = _connection()
        cursor = conn.execute("""
            SELECT role, content, created_at FROM (
                SELECT role, content, created_at, id 
                FROM agent_messages
                WHERE session_name = ? 
                ORDER BY id DESC 
                LIMIT ?
            ) ORDER BY id ASC
        """, (session_name, max(1, min(limit, 500))))
        return [dict(row) for row in cursor.fetchall()]
    except Exception as exc:
        global _error
        _error = str(exc)
        return []


def get_session_count() -> int:
    """Return total number of sessions stored."""
    if not is_ready():
        return 0
    try:
        conn = _connection()
        row = conn.execute("SELECT COUNT(*) FROM agent_sessions").fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:
        global _error
        _error = str(exc)
        return 0


def delete_session(session_name: str) -> bool:
    """Delete a session and all its messages (via ON DELETE CASCADE)."""
    try:
        if not is_ready():
            return False
        conn = _connection()
        conn.execute("DELETE FROM agent_sessions WHERE session_name = ?", (session_name,))
        conn.commit()
        return True
    except Exception as exc:
        global _error
        _error = str(exc)
        return False
