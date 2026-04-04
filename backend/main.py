import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env FIRST
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'), override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from api.routes import upload, documents, scrape, chat, history
from core.firebase import init_firebase
from core.rate_limiter import RateLimitMiddleware
from services.embedding_scheduler import process_pending_chunks

import logging
logging.basicConfig(level=logging.INFO)

# ── APScheduler: batch embed every 2 minutes ──────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(
    process_pending_chunks,
    trigger="interval",
    minutes=2,
    id="embed_pending_chunks",
    max_instances=1,          # prevent overlap if a run takes > 2 min
    misfire_grace_time=30,    # tolerate up to 30s late start
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_firebase()
    scheduler.start()
    logging.info("✅ Embedding scheduler started — runs every 2 minutes.")
    yield
    # Shutdown
    scheduler.shutdown(wait=False)
    logging.info("🛑 Embedding scheduler stopped.")

app = FastAPI(
    title="AI Research Assistant API",
    description="RAG-powered AI assistant with document search, web grounding, and conversation memory.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware (order matters: CORS first, then rate limiter) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)

# ── Routers ───────────────────────────────────────────────────────────
app.include_router(upload.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(scrape.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(history.router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "FastAPI is running!"}
