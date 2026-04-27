from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def status():
    return {"page": "chatbot", "status": "coming in phase 5"}