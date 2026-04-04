"""
Shared document chunking service.
Improvements:
  - Better semantic separators: splits on paragraph boundaries first
  - Smarter _clean_text: fixes PDF garbling, glued words, and whitespace
  - Duplicate chunk detection: hashes text to skip saving identical chunks
  - Section heading detection: tags each chunk with nearest heading
  - Saves section_heading field alongside each chunk for richer context
"""
import re
import hashlib
import logging
from datetime import datetime, timezone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from firebase_admin import firestore

logger = logging.getLogger(__name__)

# ── Chunking parameters ───────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150   # Slightly reduced for faster search matching


def _clean_text(text: str) -> str:
    """
    Pre-process text extracted from PDFs to fix common garbling issues:
      1. Missing spaces before/after bullet characters
      2. CamelCase word boundaries (e.g. 'fromYour' → 'from Your')
      3. Missing space after sentence-ending period before capitals
      4. Normalise excessive newlines
      5. Strip non-printable control characters
    """
    # Remove non-printable control chars (except newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Space before/after bullet-style characters
    text = re.sub(r'(?<=[^\s])([•·▸▪▸‣⁃])', r' \1', text)
    text = re.sub(r'([•·▸▪▸‣⁃])(?=[^\s])', r'\1 ', text)
    # CamelCase boundary: 'fromYour' → 'from Your'
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    # Missing space after period before capital: 'end.Start' → 'end. Start'
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    # Collapse 3+ newlines to paragraph break
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def _infer_page_number(text: str, chunk_text: str) -> int:
    """
    Infer page number by finding the nearest '--- Page N ---' marker
    before the chunk's position in the full text.
    """
    try:
        chunk_pos = text.find(chunk_text[:80])
        markers = [
            (m.start(), int(m.group(1)))
            for m in re.finditer(r'--- Page (\d+) ---', text[:chunk_pos])
        ]
        if markers:
            return markers[-1][1]
    except Exception:
        pass
    return 1


def _infer_section_heading(text: str, chunk_text: str) -> str:
    """
    Detect the nearest section heading before this chunk.
    Heuristic: any line that is ALL CAPS, or ends with a colon,
    or matches 'Chapter N / Section N' patterns.
    Returns the heading string or empty string.
    """
    try:
        chunk_pos = text.find(chunk_text[:80])
        search_area = text[:chunk_pos]
        # Find lines that look like headings
        heading_pattern = re.compile(
            r'^(?:[A-Z][A-Z\s\d\-:]{4,}|(?:Chapter|Section|Part)\s+[\dIVX]+.*|.*:)\s*$',
            re.MULTILINE
        )
        matches = list(heading_pattern.finditer(search_area))
        if matches:
            return matches[-1].group(0).strip()[:120]
    except Exception:
        pass
    return ""


def _hash_chunk(text: str) -> str:
    """SHA-256 hash of chunk text for duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_text_exists(db, chunk_hash: str, user_email: str) -> bool:
    """
    Check if a chunk with this hash already exists in Firestore for this user.
    Prevents re-uploading duplicate documents.
    """
    try:
        docs = (
            db.collection("document_chunks")
            .where("user_email", "==", user_email)
            .where("chunk_hash", "==", chunk_hash)
            .limit(1)
            .stream()
        )
        return any(True for _ in docs)
    except Exception:
        return False


def chunk_and_save(
    text: str,
    document_name: str,
    user_email: str,
    filename: str = "",
    source_url: str = "",
    doc_type: str = "pdf",
) -> int:
    """
    Clean, split, and persist text chunks to Firestore.
    - Semantic separators prioritise paragraph breaks
    - Each chunk gets page_number, section_heading, chunk_hash
    - Duplicate chunks are silently skipped
    Returns the number of NEW chunks saved.
    """
    cleaned = _clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Paragraph → sentence → word → character, in priority order
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_text(cleaned)

    db = firestore.client()
    collection = db.collection("document_chunks")
    created_at = datetime.now(timezone.utc)

    saved = 0
    skipped = 0

    for idx, chunk_text in enumerate(chunks):
        chunk_hash = _hash_chunk(chunk_text)

        # ── Duplicate detection ────────────────────────────────────────
        if _chunk_text_exists(db, chunk_hash, user_email):
            skipped += 1
            continue

        page_number = _infer_page_number(cleaned, chunk_text)
        section_heading = _infer_section_heading(cleaned, chunk_text)

        doc_data = {
            "text": chunk_text,
            "chunk_index": idx,
            "page_number": page_number,
            "section_heading": section_heading,
            "chunk_hash": chunk_hash,
            "document_name": document_name,
            "filename": filename or document_name,
            "source_url": source_url,
            "user_email": user_email,
            "doc_type": doc_type,
            "created_at": created_at,
            "status": "new",         # Picked up by embedding_scheduler
            "char_count": len(chunk_text),
        }
        collection.add(doc_data)
        saved += 1

    logger.info(
        f"[Chunker] '{document_name}': {saved} new chunks saved, "
        f"{skipped} duplicates skipped → pending embedding."
    )
    return saved
