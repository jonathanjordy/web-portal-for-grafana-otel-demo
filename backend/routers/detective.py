from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def status():
    return {"page": "detective", "status": "coming in phase 3"}