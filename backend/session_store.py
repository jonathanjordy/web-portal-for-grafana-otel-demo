"""In-memory store for chatbot conversation sessions.

Sessions live only in process memory (no database), so they are shared across
all clients (the app has no auth) and are lost when the backend restarts. This
is intentional for the demo. All mutations are guarded by a lock and reads
return deep copies so callers can't mutate stored state.
"""

import copy
import threading
import time
import uuid

DEFAULT_TITLE = "New session"

_lock = threading.Lock()
_sessions: dict[str, dict] = {}


def _now() -> float:
    return time.time()


def create() -> dict:
    """Create an empty session and return a copy of it."""
    session = {
        "id":         uuid.uuid4().hex,
        "title":      DEFAULT_TITLE,
        "turns":      [],
        "created_at": _now(),
        "updated_at": _now(),
    }
    with _lock:
        _sessions[session["id"]] = session
        return copy.deepcopy(session)


def get(session_id: str) -> dict | None:
    """Return a deep copy of the full session, or None if it doesn't exist."""
    with _lock:
        session = _sessions.get(session_id)
        return copy.deepcopy(session) if session else None


def _summary(session: dict) -> dict:
    return {
        "id":            session["id"],
        "title":         session["title"],
        "message_count": len(session["turns"]),
        "updated_at":    session["updated_at"],
    }


def list_summaries() -> list[dict]:
    """Return session summaries (no turns), newest-updated first."""
    with _lock:
        sessions = sorted(
            _sessions.values(),
            key=lambda s: s["updated_at"],
            reverse=True,
        )
        return [_summary(s) for s in sessions]


def append_turn(session_id: str, role: str, content: str, html: str | None = None) -> bool:
    """Append a turn to a session and bump updated_at. Returns False if missing."""
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return False
        session["turns"].append({"role": role, "content": content, "html": html})
        session["updated_at"] = _now()
        return True


def set_title(session_id: str, title: str) -> bool:
    """Set a session's title without bumping updated_at. Returns False if missing."""
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return False
        session["title"] = title
        return True


def rename(session_id: str, title: str) -> dict | None:
    """Rename a session (bumps updated_at). Returns the updated summary or None."""
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return None
        session["title"] = title
        session["updated_at"] = _now()
        return _summary(session)


def delete(session_id: str) -> bool:
    """Delete a session. Returns True if it existed."""
    with _lock:
        return _sessions.pop(session_id, None) is not None


def has_default_title(session_id: str) -> bool:
    """True if the session exists and still has the auto-generated default title."""
    with _lock:
        session = _sessions.get(session_id)
        return bool(session and session["title"] == DEFAULT_TITLE)
