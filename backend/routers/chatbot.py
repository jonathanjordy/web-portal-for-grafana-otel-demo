import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from db import query_rows, get_clickhouse_schema
from llm import generate, generate_structured
import session_store

router = APIRouter()


class ChatAnswer(BaseModel):
    """Structured answer the model must return for every chat turn."""
    sql: str | None = Field(
        default=None,
        description=(
            "A single valid ClickHouse SELECT query that answers the user's data "
            "question, with NO markdown fences. Set to null if the question is "
            "purely conversational and needs no data."
        ),
    )
    message: str = Field(
        description=(
            "If `sql` is provided: a 1-2 sentence technical explanation of what the "
            "query does. If `sql` is null: a natural-language reply to the user."
        ),
    )


def format_results(rows: list[dict], limit: int = 50) -> str:
    """Format query results as a readable HTML table."""
    if not rows:
        return "<em>Query returned no results.</em>"

    rows = rows[:limit]
    cols = list(rows[0].keys())

    header = "".join(f"<th>{c}</th>" for c in cols)
    body   = ""
    for row in rows:
        cells = "".join(
            f"<td>{str(v)[:80] if v is not None else '—'}</td>"
            for v in row.values()
        )
        body += f"<tr>{cells}</tr>"

    return f"""
<div style="overflow-x:auto;margin-top:0.5rem">
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;font-family:monospace">
<thead><tr style="background:var(--surface2);font-family:var(--sans)">{header}</tr></thead>
<tbody>{body}</tbody>
</table>
</div>
<div style="font-size:0.75rem;color:var(--text-3);margin-top:0.4rem;font-family:var(--sans)">
  Showing {len(rows)} row{'s' if len(rows) != 1 else ''}
</div>"""


def build_system_prompt() -> str:
    return f"""You are an AIOps assistant with access to a ClickHouse database called `otel`.
Your job is to help engineers query their observability data using natural language.

{get_clickhouse_schema()}

RULES:
1. When the user asks a data question, put a single valid ClickHouse SELECT query in the `sql` field (plain SQL, NO markdown code fences) and a 1-2 sentence technical explanation in the `message` field.
2. Always use proper ClickHouse syntax (e.g. toStartOfInterval, quantile(), countIf()).
3. Keep queries efficient — always include a time filter like: AND TimeUnix >= now() - INTERVAL 1 HOUR
4. For duration fields in otel_traces, Duration is stored in nanoseconds. Divide by 1e6 for milliseconds.
5. For p95 latency use: quantile(0.95)(Duration) / 1e6
6. If the question is purely conversational, set `sql` to null and put your natural-language reply in `message`.
7. Never query more than 10000 rows. Always add LIMIT clauses.
"""


class ChatRequest(BaseModel):
    message:    str
    session_id: str | None = None


async def _generate_title(user_message: str, assistant_summary: str) -> str:
    """Ask the model for a short 3-5 word session title."""
    prompt = (
        "Summarize the topic of this conversation as a short title of 3-5 words. "
        "Respond with ONLY the title — no quotes, no trailing punctuation.\n\n"
        f"User: {user_message}\n"
        f"Assistant: {assistant_summary}\n"
        "Title:"
    )
    title = await generate(prompt, temperature=0.2, max_tokens=512)
    title = title.strip().strip('"').splitlines()[0].strip() if title.strip() else ""
    return title[:60] or session_store.DEFAULT_TITLE


@router.post("/chat")
async def chat(req: ChatRequest):
    # Resolve the session; create a fresh one if the id is missing or stale so a
    # client with an unknown id never 500s — it just gets a new session back.
    session = session_store.get(req.session_id) if req.session_id else None
    if not session:
        session = session_store.create()
    session_id = session["id"]
    is_first_exchange = len(session["turns"]) == 0

    system = build_system_prompt()

    # Build LLM history from the session's stored turns (backend owns history now)
    history = [{"role": t["role"], "content": t["content"]} for t in session["turns"][-6:]]

    # Step 1 — Generate structured SQL + technical explanation
    answer = await generate_structured(req.message, ChatAnswer, history=history, system=system)
    sql = answer.sql.strip() if answer.sql else None

    # plain_summary is the plain-text the assistant "said" — stored for LLM context.
    plain_summary = answer.message

    if sql and not sql.upper().startswith("SELECT"):
        result = {"response": answer.message, "sql": sql, "executed": False}

    elif sql:
        try:
            # Execute SQL
            rows  = query_rows(sql)
            table = format_results(rows)

            tech_explanation = answer.message or "SQL Query executed successfully."

            # Step 2 — Generate Business Interpretation based on returned data
            if rows:
                safe_data_subset = [
                    {key: str(value) for key, value in row.items()}
                    for row in rows[:10]
                ]
                interp_prompt = f"""
                                The user asked: "{req.message}"
                                The database returned this data: {json.dumps(safe_data_subset)}

                                Provide a concise, 1-2 sentence business-oriented interpretation of this data.
                                Focus on what the numbers mean for the system's health or business.
                                Do NOT explain the SQL query here. Just give the insight.
                                """
                try:
                    business_interpretation = await generate(interp_prompt)
                except Exception:
                    business_interpretation = "Data retrieved successfully."
            else:
                business_interpretation = "The query executed successfully, but no data matched the criteria for this timeframe."

            response_html = f"""
<div style="margin-bottom:1rem; font-size: 0.95rem; color: var(--text); font-weight: 500;">
    {business_interpretation}
</div>

<details>
  <summary style="cursor:pointer;font-size:0.78rem;color:var(--text-3);font-family:monospace;margin-bottom:0.4rem;outline:none;">
    View SQL & Technical Details
  </summary>
  <div style="background:var(--surface2); padding:0.85rem; border-radius:8px; border:1px solid var(--border); margin-top:0.4rem;">
      <div style="font-size:0.82rem; color:var(--text-2); margin-bottom:0.75rem; line-height: 1.5;">
        <em>{tech_explanation}</em>
      </div>
      <pre style="background:#1e1e1e;color:#d4d4d4;padding:0.75rem;border-radius:6px;font-size:0.75rem;overflow-x:auto;margin:0;">{sql}</pre>
  </div>
</details>

{table}"""

            plain_summary = business_interpretation
            result = {
                "response": response_html,
                "sql":      sql,
                "executed": True,
                "row_count": len(rows),
            }

        except Exception as e:
            error_html = f"""
<div style="margin-bottom:0.75rem">{answer.message}</div>
<div style="margin-top:0.75rem;padding:0.75rem;background:var(--red-bg);border-radius:6px;font-size:0.82rem;color:var(--red);border:1px solid rgba(235,0,140,0.2)">
  <strong>Query execution failed:</strong><br>{str(e)[:200]}
</div>"""
            result = {"response": error_html, "sql": sql, "executed": False, "error": str(e)}

    else:
        # Pure conversation
        result = {"response": answer.message, "sql": None, "executed": False}

    # Persist this exchange to the session
    session_store.append_turn(session_id, "user", req.message)
    session_store.append_turn(session_id, "assistant", plain_summary, html=result["response"])

    # Title the session on its first exchange (unless the user already renamed it)
    if is_first_exchange and session["title"] == session_store.DEFAULT_TITLE:
        try:
            session_store.set_title(session_id, await _generate_title(req.message, plain_summary))
        except Exception:
            pass  # keep the default title if titling fails

    updated = session_store.get(session_id)
    result["session_id"] = session_id
    result["title"] = updated["title"] if updated else None
    return result


# ─────────────────────────────────────────────────────────────
# SESSION HISTORY (in-memory)
# ─────────────────────────────────────────────────────────────
class RenameRequest(BaseModel):
    title: str


@router.get("/sessions")
async def list_sessions():
    return {"sessions": session_store.list_summaries()}


@router.post("/sessions")
async def create_session():
    session = session_store.create()
    return {
        "id":            session["id"],
        "title":         session["title"],
        "message_count": 0,
        "updated_at":    session["updated_at"],
    }


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameRequest):
    summary = session_store.rename(session_id, req.title.strip() or session_store.DEFAULT_TITLE)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_store.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}




# ─────────────────────────────────────────────────────────────
# SUGGESTED QUESTIONS (used to populate chatbot UI chips)
# ─────────────────────────────────────────────────────────────
@router.get("/suggestions")
async def get_suggestions():
    return {
        "suggestions": [
            "How many orders in the last hour?",
            "Show payment failure rate by item",
            "Which service had the most errors today?",
            "What is the p95 order duration in ms?",
            "Show node memory usage over the last 6 hours",
            "Which trace had the highest latency today?",
            "How many cache misses in the last 30 minutes?",
            "Show error rate trend for payment-service",
        ]
    }