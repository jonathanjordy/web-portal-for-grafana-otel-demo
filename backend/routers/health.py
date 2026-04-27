from fastapi import APIRouter
from db import query_scalar

router = APIRouter()


@router.get("")
async def health():
    """Check API and ClickHouse connectivity."""
    try:
        result = query_scalar("SELECT 1")
        ch_status = "ok" if result == 1 else "error"
    except Exception as e:
        ch_status = f"error: {str(e)}"

    return {
        "api":        "ok",
        "clickhouse": ch_status,
    }