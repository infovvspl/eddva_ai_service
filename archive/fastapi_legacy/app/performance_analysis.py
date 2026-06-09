from fastapi import APIRouter

router = APIRouter(prefix="/performance", tags=["Performance Analysis"])


@router.get("/")
async def performance_root():
    return {"message": "Performance Analysis service - coming soon"}
