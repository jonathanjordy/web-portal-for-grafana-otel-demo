# Chatbot Session History — Design

**Date:** 2026-06-03
**Status:** Approved (design), pending spec review
**Area:** `backend/` (FastAPI) + `frontend/` (React SPA, Page 4 — AIOps Assistant)

## Context

The AIOps chatbot (Page 4) is currently a single stateless exchange: the React page holds
all messages in component state and posts `{message, history}` to `POST /api/chatbot/chat`.
There is no concept of a conversation/session — refreshing or navigating away loses the chat,
and there is no way to keep multiple conversations.

We want a **session history panel on the right side of the chat area** (per the user's
reference screenshot): a **+ New Chat** button, a list of past sessions grouped by time,
a per-session message-count badge, edit (rename) and delete icons on hover, an active-session
highlight, and a `N sessions saved` footer. Sessions are stored in an **in-memory backend
store** (no real database), and the backend owns the full conversation so a session can be
re-opened and resumed.

**Forward-looking constraint:** the app will later migrate from the current manual
`activePage` SPA switching to a real router. So session logic must stay self-contained in the
chatbot page and communicate purely over REST — no coupling to the SPA navigation mechanism —
so it ports cleanly to routed pages/URLs later.

## Decisions (confirmed with user)

- **Storage:** full conversations stored server-side in an in-memory store. Clicking a session
  reloads its messages from the backend. Consequences (accepted): sessions are **global**
  (the app has no auth) and **lost on backend restart**.
- **Title:** **LLM-generated** short (3–5 word) summary, created once after the first exchange;
  editable via the pencil icon.
- **Chat integration:** **session-aware chat endpoint** (Approach A). The backend is the single
  source of truth for conversation history; the frontend stops maintaining its own `history`
  array.

## Architecture

```
ChatbotPage (React)
  ├─ chat column        → POST /api/chatbot/chat { message, session_id }
  └─ SessionHistory      → GET/POST/PATCH/DELETE /api/chatbot/sessions[...]
        (right panel)

FastAPI (backend)
  routers/chatbot.py  ──uses──>  session_store.py  (in-memory SessionStore + Lock)
        └─ /chat reads prior turns, runs Gemini (existing structured-output flow),
           appends user+assistant turns, generates title on first exchange
```

### Data model (backend, in-memory)

```python
Turn = {
  "role":    "user" | "assistant",
  "content": str,          # plain text — used for LLM context AND user-bubble display
  "html":    str | None,   # rich assistant render (tables / SQL details); None for user turns
}

Session = {
  "id":         str,        # uuid4
  "title":      str,        # "New session" until the LLM title is generated
  "turns":      list[Turn],
  "created_at": float,      # epoch seconds
  "updated_at": float,
}
```

- `message_count` (badge) = `len(turns)`. The static greeting is a frontend constant and is
  **not** stored, so a brand-new session shows count `0`.
- Assistant `content` is the plain-text summary the backend already computes
  (`business_interpretation`, or `answer.message` for conversational turns) — not stripped HTML.
  This is what feeds back into Gemini as context on later turns.

## Components

### Backend

1. **`backend/session_store.py`** (new) — thread-safe in-memory store.
   - Module-level `dict[str, Session]` guarded by a `threading.Lock`.
   - `create() -> Session` (title `"New session"`, empty turns)
   - `get(session_id) -> Session | None`
   - `list_summaries() -> list[dict]` → `{id, title, message_count, updated_at}`, newest-first
   - `rename(session_id, title) -> Session`
   - `delete(session_id) -> bool`
   - `append_turn(session_id, role, content, html=None)` — appends and bumps `updated_at`
   - `set_title(session_id, title)`
   - 404 handling lives in the router; the store returns `None`/bools.

2. **`backend/routers/chatbot.py`** (modified) — session endpoints under the existing
   `/api/chatbot` prefix, plus a session-aware `/chat`:
   - `GET  /sessions` → `{ "sessions": [ {id, title, message_count, updated_at}, ... ] }`
   - `POST /sessions` → create empty, returns the new session summary
   - `GET  /sessions/{id}` → full session (`{id, title, turns, created_at, updated_at}`) or 404
   - `PATCH /sessions/{id}` body `{title}` → rename, returns updated summary, or 404
   - `DELETE /sessions/{id}` → `{deleted: true}` or 404
   - `POST /chat` body `{message, session_id}`:
     1. Resolve session; if `session_id` missing/unknown, create a new one (robustness).
     2. Build LLM history from stored `turns` (reusing the existing `{role, content}` shape that
        `llm.generate_structured` already consumes).
     3. Run the existing structured-output flow (`ChatAnswer` → optional SQL exec → business
        interpretation). Unchanged.
     4. Append the user turn `{user, message}` and the assistant turn
        `{assistant, content=plain_summary, html=response_html}`.
     5. If this was the **first** exchange (session still has the default title), call
        `llm.generate()` once to produce a 3–5 word title and store it.
     6. Return the existing response object **plus** `session_id` and `title`.
   - The `history` field is removed from `ChatRequest`; `session_id: str | None` is added.

### Frontend

3. **`frontend/src/components/SessionHistory.jsx`** (new) — presentational + handlers via props:
   - Props: `sessions`, `activeSessionId`, `onNewChat`, `onSelect`, `onRename`, `onDelete`.
   - Renders the **+ New Chat** button (magenta), `SESSION HISTORY` label, time-grouped list
     (Today / Yesterday / This Week / Older from `updated_at`), each row = title + relative
     time + count badge; pencil + trash icons shown on hover; active row highlighted; footer
     `N sessions saved`.
   - Rename swaps the title row into an inline text input (Enter = save, Esc/blur = cancel);
     delete asks for a lightweight `window.confirm` before calling the endpoint.
   - A small local helper formats relative time (`Just now`, `N min ago`, `N hr ago`, date) and
     assigns the time-group bucket.

4. **`frontend/src/pages/ChatbotPage.jsx`** (modified) — two-column layout (`.chat-layout`):
   chat on the left, `<SessionHistory>` on the right.
   - State: `sessions`, `activeSessionId`, `messages`, `sending`.
   - On mount: `GET /sessions`; if any exist, select the newest and load its turns; otherwise
     create one. Always render the static greeting first, then the session's turns.
   - `sendMessage`: `POST /chat {message, session_id}`; optimistically append the user bubble +
     a "Thinking…" placeholder; on response, replace placeholder with `data.response`; then
     refresh the session summaries (for updated count/title/time).
   - New Chat / select / rename / delete call the matching endpoints and update state. Deleting
     the active session selects the newest remaining session, or creates a fresh one if none.
   - No dependency on `activePage`/SPA nav — all state is internal + REST, router-ready.

5. **`frontend/src/services/api.js`** (modified) — add `apiPatch(path, body, options)` and
   `apiDelete(path, options)` mirroring the existing `apiGet`/`apiPost` helpers.

6. **`frontend/src/styles.css`** (modified) — add `.chat-layout` (flex row), `.session-panel`
   (right column, ~260px), `.session-row` (+ `.active`, hover-revealed `.session-actions`),
   `.session-badge`, `.new-chat-btn`, group labels, and footer styles. Reuse existing tokens
   (`--indosat-magenta`, `--surface`, `--border`, `--text-3`). The existing `.chat-wrap` keeps
   its current styling inside the left column.

## Error handling

- Unknown `session_id` on any `/sessions/{id}` route → HTTP 404 `{detail}`.
- `/chat` with a missing/unknown `session_id` → backend creates a fresh session (never 500s on
  a stale client id) and returns its `session_id` so the client can re-sync.
- LLM title generation failure is non-fatal: keep `"New session"` (wrapped in try/except, like
  the existing business-interpretation fallback).
- Frontend: failed session API calls surface a small inline error and leave existing state
  intact; the chat path keeps the current `"Backend unreachable."` fallback.
- Concurrency: all store mutations happen under the `Lock`; reads return copies to avoid
  callers mutating stored state.

## Testing / verification

End-to-end via the running stack (`docker compose up -d --build aiops-backend`):

1. `POST /api/chatbot/sessions` → returns a session with `title: "New session"`, count `0`.
2. `POST /api/chatbot/chat {message, session_id}` (a data question) → HTTP 200, executes SQL;
   `GET /api/chatbot/sessions` shows that session with `message_count: 2` and an LLM title
   (no longer "New session").
3. Second `POST /chat` on the same session referencing the prior turn → context carries over
   (proves history is rebuilt from the store).
4. `GET /api/chatbot/sessions/{id}` → returns both turns with assistant `html` populated.
5. `PATCH /sessions/{id} {title:"renamed"}` then `GET /sessions` → title updated.
6. `DELETE /sessions/{id}` → removed from the list.
7. UI smoke test at http://localhost:3001 → Page 4: New Chat creates a row; sending messages
   updates the count badge and title; switching sessions reloads messages; hover shows
   edit/delete; delete works; footer count is correct.

## Out of scope

- Persistence across backend restarts (explicitly in-memory) and any real database.
- Authentication / per-user isolation (sessions are global).
- Streaming responses, message editing, search across sessions, pagination of the list.
- The future router migration itself (design is kept compatible, but not implemented here).
