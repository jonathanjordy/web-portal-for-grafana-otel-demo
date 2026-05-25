import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import health, predictive, detective, diagnostic, chatbot, filters

load_dotenv()

app = FastAPI(
    title="AIOps Portal API",
    description="AI-powered observability portal backed by ClickHouse",
    version="1.0.0",
)

# Allow frontend HTML files to call the API
app.add_middleware(
    CORSMiddleware,
    # Explicitly allow the browser's local file origin string
    allow_origins=["*", "null"], 
    allow_credentials=False,  # Must be False if using "*" wildcards
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(health.router,      prefix="/api/health",      tags=["Health"])
app.include_router(predictive.router,  prefix="/api/predictive",  tags=["Predictive"])
app.include_router(detective.router,   prefix="/api/detective",   tags=["Detective"])
app.include_router(diagnostic.router,  prefix="/api/diagnostic",  tags=["Diagnostic"])
app.include_router(chatbot.router,     prefix="/api/chatbot",     tags=["Chatbot"])
app.include_router(filters.router,     prefix="/api/filters",     tags=["Filters"])


@app.get("/")
async def root():
    return {
        "service": "AIOps Portal API",
        "version": "1.0.0",
        "pages": {
            "predictive":  "/api/predictive",
            "detective":   "/api/detective",
            "diagnostic":  "/api/diagnostic",
            "chatbot":     "/api/chatbot",
        }
    }
