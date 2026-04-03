from fastapi import APIRouter, Depends, HTTPException
from api.deps import verify_token
from firebase_admin import firestore
import os

router = APIRouter()



@router.get("/documents")
async def list_documents(user_token: dict = Depends(verify_token)):
    user_email = user_token.get("email")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not found in token")

    db = firestore.client()
    chunks_ref = db.collection("document_chunks")
    
    # We will fetch all chunks for the user to determine the unique document list
    docs = chunks_ref.where("user_email", "==", user_email).stream()
    
    docs_map = {}
    
    for doc in docs:
        data = doc.to_dict()
        filename = data.get("filename") or data.get("document_name", "Unknown File")
        
        if filename not in docs_map:
            # Fallback if created_at is missing for some reason
            created_at = data.get("created_at")
            try:
                timestamp = created_at.timestamp() if hasattr(created_at, "timestamp") else 0
            except:
                timestamp = 0
                
            docs_map[filename] = {
                "name": data.get("document_name") or filename,
                "filename": filename,
                "type": data.get("doc_type", "pdf"),
                "source_url": data.get("source_url", ""),
                "size": data.get("char_count", 0),
                "created": timestamp,
                "status": "ready"
            }
        else:
            # Aggregate chunk sizes to give total doc size
            docs_map[filename]["size"] += data.get("char_count", 0)
            
        # If any chunk is processing, the document is processing
        status = data.get("status", "new")
        if status in ["new", "embedding_failed"]:
            docs_map[filename]["status"] = "processing"

    documents = list(docs_map.values())
    documents.sort(key=lambda x: x["created"], reverse=True)
    return {"documents": documents}


@router.delete("/documents/{filename}")
async def delete_document(filename: str, user_token: dict = Depends(verify_token)):
    user_email = user_token.get("email")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email not found in token")

    # Cascade delete all matching document_chunks from Firestore
    db = firestore.client()
    chunks_ref = db.collection("document_chunks")

    deleted_count = 0
    batch = db.batch()
    
    # Use single-field query (no composite index needed) then filter in Python
    # This avoids composite index build-time race conditions entirely
    print(f"[DELETE] Searching chunks: user={user_email}, filename={filename}")
    all_user_chunks = chunks_ref.where("user_email", "==", user_email).stream()
    for doc in all_user_chunks:
        data = doc.to_dict()
        stored_filename = data.get("filename", "")
        stored_doc_name = data.get("document_name", "")
        # Match the dedicated filename field OR document_name (fallback for old chunks without filename)
        if stored_filename == filename or stored_doc_name == filename:
            batch.delete(doc.reference)
            deleted_count += 1
            print(f"  ⏳ Queued chunk {doc.id[:20]} for deletion")
            
            # Firestore limit: 499 operations per batch, flush when reaching 400 for safety
            if deleted_count % 400 == 0:
                batch.commit()
                batch = db.batch()

    # Commit any remaining deletes in the batch
    if deleted_count % 400 != 0:
        batch.commit()

    print(f"[DELETE] Total deleted: {deleted_count} chunks")

    return {
        "message": f"Document '{filename}' and {deleted_count} chunks deleted successfully",
        "filename": filename,
        "chunks_deleted": deleted_count
    }
