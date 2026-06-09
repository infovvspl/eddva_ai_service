import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import lifespan from feedback service as it's the most complex
from app.feedback import lifespan, router as feedback_router
from app.ai_notes import router as notes_router
from app.ai_content import router as content_router
from app.ai_test import router as test_router
from app.career_roadmap import router as career_router
from app.performance_analysis import router as performance_router
from app.personalization import router as personalization_router
from app.cheating_main import router as cheating_router
from app.translate import router as translate_router
from app.doubt_resolve import router as doubt_router

app = FastAPI(
    title="AI Study Unified API",
    description="Production-ready consolidated API for the AI Study project.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(feedback_router)
app.include_router(notes_router)
app.include_router(content_router)
app.include_router(test_router)
app.include_router(career_router)
app.include_router(performance_router)
app.include_router(personalization_router)
app.include_router(cheating_router)
app.include_router(translate_router)
app.include_router(doubt_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the AI Study Unified API",
        "services": [
            "/feedback",
            "/notes",
            "/content",
            "/test",
            "/career",
            "/performance",
            "/personalization",
            "/cheating",
        ],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
