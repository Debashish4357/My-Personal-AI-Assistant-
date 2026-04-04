"""
Chat route — Improved RAG pipeline with:
  - Conversation Memory (fetches last 5 turns from Firestore by session_id)
  - Query Expansion (rephrases query in 2 ways, merges results)
  - Dynamic Similarity Threshold (auto-lowers if no results found)
  - Top-8 context chunks (up from 5)
  - AI-generated session titles (4-5 word summary of first question)
  - Streaming-ready structure
  - Returns sources + follow-up questions to the frontend
"""
import math
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from firebase_admin import firestore

from api.deps import verify_token
from services.embedder import generate_embedding
from services.llm import generate_answer, expand_query, get_client, _get_model_name
from google.genai import types

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Tunable RAG constants ─────────────────────────────────────────────
TOP_K = 8                         # Retrieve top-8 chunks (was 5)
SIMILARITY_THRESHOLD_HIGH = 0.35  # Primary threshold
SIMILARITY_THRESHOLD_LOW  = 0.20  # Fallback if no chunks pass primary
EMBEDDING_CACHE: dict[str, list[float]] = {}  # Simple in-process cache


# ── Request/Response schemas ──────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def _get_embedding_cached(text: str) -> list[float]:
    """Return cached embedding if available, otherwise compute and cache it."""
    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]
    emb = generate_embedding(text)
    # Keep cache bounded to ~500 entries
    if len(EMBEDDING_CACHE) > 500:
        first_key = next(iter(EMBEDDING_CACHE))
        del EMBEDDING_CACHE[first_key]
    EMBEDDING_CACHE[text] = emb
    return emb


def _fetch_history(db, user_email: str, session_id: str, limit: int = 5) -> list[dict]:
    """Fetch the last `limit` Q&A turns for a session, oldest-first."""
    try:
        docs = (
            db.collection("chat_history")
            .where("user_email", "==", user_email)
            .where("session_id", "==", session_id)
            .stream()
        )
        turns = []
        for doc in docs:
            d = doc.to_dict()
            turns.append(d)
        # Sort ascending by timestamp, take last N
        turns.sort(key=lambda x: x.get("timestamp", datetime.min))
        return turns[-limit:]
    except Exception as e:
        logger.warning(f"Could not fetch history for session {session_id}: {e}")
        return []


def _generate_session_title(question: str) -> str:
    """Ask Gemini to produce a 4-5 word title for the session."""
    try:
        client = get_client()
        model_name = _get_model_name()
        response = client.models.generate_content(
            model=model_name,
            contents=(
                f"Create a concise 4-5 word title for a chat session that starts with "
                f"this question. Output ONLY the title, no punctuation, no quotes:\n\n"
                f"{question}"
            ),
            config=types.GenerateContentConfig(temperature=0.4),
        )
        title = response.text.strip()[:80]
        return title if title else question[:60].strip()
    except Exception:
        return question[:60].strip()


def _retrieve_chunks(
    db, user_email: str, query_embeddings: list[list[float]]
) -> list[dict]:
    """
    Retrieve all processed chunks for the user,
    score against ALL query embeddings (multi-query), deduplicate,
    and apply dynamic threshold.
    """
    # Fetch all processed chunks for this user
    docs = (
        db.collection("document_chunks")
        .where("user_email", "==", user_email)
        .where("status", "==", "processed")
        .stream()
    )

    chunk_scores: dict[str, dict] = {}  # chunk_id -> best scored chunk

    for doc in docs:
        data = doc.to_dict()
        doc_embedding = data.get("embedding")
        if not doc_embedding:
            continue

        # Score against each query variant, keep the best score
        best_score = max(
            cosine_similarity(qe, doc_embedding) for qe in query_embeddings
        )

        chunk_id = doc.id
        if chunk_id not in chunk_scores or best_score > chunk_scores[chunk_id]["score"]:
            chunk_scores[chunk_id] = {
                "id": chunk_id,
                "text": data.get("text", ""),
                "document_name": data.get("document_name", "Unknown"),
                "filename": data.get("filename", ""),
                "page_number": data.get("page_number", 1),
                "source_url": data.get("source_url", ""),
                "doc_type": data.get("doc_type", "pdf"),
                "score": best_score,
            }

    all_chunks = list(chunk_scores.values())
    all_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Dynamic similarity threshold: try primary, fall back if empty
    relevant = [c for c in all_chunks if c["score"] >= SIMILARITY_THRESHOLD_HIGH]
    if not relevant:
        logger.info("No chunks above primary threshold — relaxing to lower threshold.")
        relevant = [c for c in all_chunks if c["score"] >= SIMILARITY_THRESHOLD_LOW]

    return relevant[:TOP_K]


# ── Main chat endpoint ────────────────────────────────────────────────
@router.post("/chat")
async def chat_with_docs(
    payload: ChatRequest,
    user_token: dict = Depends(verify_token),
):
    user_email = user_token.get("email")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not found in token")

    query_text = payload.message.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = payload.session_id or str(uuid.uuid4())
    db = firestore.client()

    # ── 1. Fetch conversation history ──────────────────────────────────
    history = _fetch_history(db, user_email, session_id)
    is_new_session = len(history) == 0

    # ── 2. Query expansion ─────────────────────────────────────────────
    query_variants = expand_query(query_text)
    logger.info(f"Query variants: {query_variants}")

    # ── 3. Embed all query variants (with cache) ───────────────────────
    try:
        query_embeddings = [_get_embedding_cached(q) for q in query_variants]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {str(e)}")

    # ── 4. Retrieve & re-rank top-K chunks with dynamic threshold ──────
    top_chunks = _retrieve_chunks(db, user_email, query_embeddings)
    logger.info(f"Retrieved {len(top_chunks)} relevant chunks for query.")

    # ── 5. Generate answer with memory ────────────────────────────────
    result = generate_answer(
        query_text=query_text,
        context_chunks=top_chunks,
        history=history,
    )

    # ── 6. Generate smart session title for new sessions ──────────────
    if is_new_session:
        session_title = _generate_session_title(query_text)
    else:
        session_title = history[0].get("title", query_text[:60].strip()) if history else query_text[:60].strip()

    # ── 7. Persist Q&A turn to Firestore ─────────────────────────────
    now = datetime.now(timezone.utc)
    db.collection("chat_history").add({
        "session_id": session_id,
        "user_email": user_email,
        "query": query_text,
        "reply": result["text"],
        "sources": result["sources"],
        "follow_up": result["follow_up"],
        "timestamp": now,
        "title": session_title,
    })

    # ── 8. Return enriched response ────────────────────────────────────
    return {
        "reply": result["text"],
        "sources": result["sources"],          # [{title, url}, ...]
        "follow_up": result["follow_up"],      # ["Question 1?", ...]
        "session_id": session_id,
        "session_title": session_title,
        "context_chunks": [
            {k: v for k, v in c.items() if k != "score"}
            for c in top_chunks
        ],
    }
