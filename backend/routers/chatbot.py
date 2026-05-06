import os
import json
import re
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db import query_rows, get_clickhouse_schema

router = APIRouter()

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


async def call_gemini(prompt: str, system: str = "") -> str:
    """Call Gemini API and return the text response."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not set in .env."
        )

    contents = []
    if system:
        contents.append({"role": "user",  "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood. I will follow those instructions."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{GEMINI_URL}?key={api_key}",
            headers={"content-type": "application/json"},
            json={
                "contents": contents,
                "generationConfig": {
                    "temperature":     0.1,
                    "maxOutputTokens": 1500,
                }
            }
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {response.status_code} — {response.text[:200]}"
        )

    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


def extract_sql(text: str) -> str | None:
    """Extract SQL from a markdown code block or plain text."""
    # Try ```sql ... ``` block first
    match = re.search(r"```(?:sql)?\s*(SELECT[\s\S]+?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try plain SELECT statement
    match = re.search(r"(SELECT[\s\S]+?;)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


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


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    return f"""You are an AIOps assistant with access to a ClickHouse database called `otel`.
Your job is to help engineers query their observability data using natural language.

{get_clickhouse_schema()}

RULES:
1. When the user asks a data question, respond with a valid ClickHouse SQL query inside a ```sql code block.
2. Always use proper ClickHouse syntax (e.g. toStartOfInterval, quantile(), countIf()).
3. Keep queries efficient — always include a time filter like: AND TimeUnix >= now() - INTERVAL 1 HOUR
4. For duration fields in otel_traces, Duration is stored in nanoseconds. Divide by 1e6 for milliseconds.
5. For p95 latency use: quantile(0.95)(Duration) / 1e6
6. If the question is conversational (greetings, clarifications), respond naturally without SQL.
7. After providing SQL, briefly explain what the query does in 1-2 sentences.
8. If a question is ambiguous, make a reasonable assumption and state it.
9. Never query more than 10000 rows. Always add LIMIT clauses.
10. Format numbers nicely in SELECT aliases (e.g. round(avg(Duration)/1e6, 1) AS avg_ms).
"""


# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


# ─────────────────────────────────────────────────────────────
# MAIN CHAT ENDPOINT
# ─────────────────────────────────────────────────────────────
@router.post("/chat")
async def chat(req: ChatRequest):
    """
    Accepts a natural language message, generates SQL via Gemini,
    executes it against ClickHouse, and returns the result.
    """
    system = build_system_prompt()

    # Build conversation context (last 6 messages for context window)
    context = ""
    for msg in req.history[-6:]:
        role = "User" if msg.role == "user" else "Assistant"
        context += f"{role}: {msg.content}\n\n"

    full_prompt = f"{context}User: {req.message}"

    # Step 1 — Ask Gemini to generate SQL or respond conversationally
    gemini_response = await call_gemini(full_prompt, system)

    # Step 2 — Try to extract and execute SQL
    sql = extract_sql(gemini_response)

    if sql:
        try:
            # Safety check — only allow SELECT statements
            clean = sql.strip().upper()
            if not clean.startswith("SELECT"):
                return {"response": gemini_response, "sql": sql, "executed": False}

            rows   = query_rows(sql)
            table  = format_results(rows)

            # Build response with explanation + results
            # Strip the SQL block from the explanation to avoid duplication
            explanation = re.sub(r"```(?:sql)?[\s\S]+?```", "", gemini_response).strip()
            if not explanation:
                explanation = f"Query returned {len(rows)} result{'s' if len(rows) != 1 else ''}."

            response_html = f"""<div style="margin-bottom:0.75rem">{explanation}</div>
<details open>
  <summary style="cursor:pointer;font-size:0.78rem;color:var(--text-3);font-family:monospace;margin-bottom:0.4rem">▶ SQL query</summary>
  <pre style="background:#1e1e1e;color:#d4d4d4;padding:0.75rem;border-radius:6px;font-size:0.78rem;overflow-x:auto;margin-top:0.4rem">{sql}</pre>
</details>
{table}"""

            return {
                "response": response_html,
                "sql":      sql,
                "executed": True,
                "row_count": len(rows),
            }

        except Exception as e:
            # SQL failed — return Gemini's explanation + error
            error_html = f"""{gemini_response}
<div style="margin-top:0.75rem;padding:0.75rem;background:var(--red-bg);border-radius:6px;font-size:0.82rem;color:var(--red)">
  <strong>Query error:</strong> {str(e)[:200]}
</div>"""
            return {"response": error_html, "sql": sql, "executed": False, "error": str(e)}

    # No SQL — pure conversational response
    return {"response": gemini_response, "sql": None, "executed": False}


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