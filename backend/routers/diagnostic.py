from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def status():
    return {"page": "diagnostic", "status": "coming in phase 4"}