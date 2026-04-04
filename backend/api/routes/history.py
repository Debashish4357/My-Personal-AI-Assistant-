from fastapi import APIRouter, Depends, HTTPException
from api.deps import verify_token
from firebase_admin import firestore

router = APIRouter()


@router.get("/chats")
async def get_chat_sessions(user_token: dict = Depends(verify_token)):
    """
    Return the list of unique chat sessions for the user,
    ordered by most recent first. Each session has:
      - session_id
      - title  (first message's text, trimmed)
      - timestamp (of the most recent message in that session)
    """
    user_email = user_token.get("email")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not found in token")

    db = firestore.client()
    docs = (
        db.collection("chat_history")
        .where("user_email", "==", user_email)
        .stream()
    )

    # Sort in memory to avoid Firebase Composite Index requirement
    all_data = []
    for doc in docs:
        all_data.append(doc.to_dict())
    
    # Descending sort by timestamp
    all_data.sort(key=lambda x: x.get("timestamp"), reverse=True)

    # Deduplicate: keep only the first (latest) doc per session_id
    seen: set[str] = set()
    sessions = []
    for data in all_data:
        sid = data.get("session_id")
        if sid and sid not in seen:
            seen.add(sid)
            ts = data.get("timestamp")
            sessions.append({
                "session_id": sid,
                "title": data.get("title", "Untitled Chat"),
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            })

    return {"sessions": sessions}


@router.get("/chats/{session_id}")
async def get_session_messages(session_id: str, user_token: dict = Depends(verify_token)):
    """
    Return all Q&A turns for a given session_id, ordered oldest first.
    """
    user_email = user_token.get("email")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not found in token")

    db = firestore.client()
    docs = (
        db.collection("chat_history")
        .where("user_email", "==", user_email)
        .where("session_id", "==", session_id)
        .stream()
    )

    all_data = []
    for doc in docs:
        all_data.append(doc.to_dict())

    # Ascending sort by timestamp
    all_data.sort(key=lambda x: x.get("timestamp"))

    messages = []
    for data in all_data:
        ts = data.get("timestamp")
        messages.append({
            "query":     data.get("query", ""),
            "reply":     data.get("reply", ""),
            "sources":   data.get("sources", []),       # [{title, url}, ...]
            "follow_up": data.get("follow_up", []),     # ["Question?", ...]
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        })

    return {"messages": messages}
