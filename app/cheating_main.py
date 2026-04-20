from fastapi import APIRouter

router = APIRouter(prefix="/cheating", tags=["Cheating Detection"])


@router.get("/")
async def cheating_root():
    return {"message": "Cheating Detection service - coming soon"}
