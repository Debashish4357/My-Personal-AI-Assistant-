"""
LLM service — Gemini Flash with:
  - Conversation memory (past Q&A history injected into the prompt)
  - Query expansion (rephrase query 2 ways before embedding search)
  - Google Search grounding with source URL extraction
  - Follow-up question suggestions
  - Returns a rich dict instead of a plain string
"""
import logging
from google import genai
from google.genai import types
from firebase_admin import firestore

logger = logging.getLogger(__name__)

_client = None


def get_client() -> genai.Client:
    global _client
    if not _client:
        _client = genai.Client()
    return _client


def _get_model_name() -> str:
    """Fetch the active model name from Firestore, with fallback."""
    try:
        db = firestore.client()
        docs = db.collection("models").limit(1).stream()
        for doc in docs:
            data = doc.to_dict()
            name = (
                data.get("name")
                or data.get("model_name")
                or data.get("modelId")
                or doc.id
            )
            if name:
                return name
    except Exception as e:
        logger.warning(f"Could not fetch model from Firestore: {e}")
    return "gemini-2.5-flash"


def expand_query(query: str) -> list[str]:
    """
    Use Gemini to rephrase the user query in 2 alternative ways.
    Returns the original query plus up to 2 rephrased variations.
    This improves RAG recall for vague or short queries.
    """
    client = get_client()
    model_name = _get_model_name()
    try:
        prompt = (
            f"Rephrase the following question in exactly 2 alternative ways "
            f"to improve document search recall. Output ONLY the 2 rephrased "
            f"questions, each on its own line, with NO numbering or extra text.\n\n"
            f"Question: {query}"
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        lines = [l.strip() for l in response.text.strip().split("\n") if l.strip()]
        rephrased = lines[:2]
        # Return original + rephrased, deduplicated
        all_queries = [query] + [q for q in rephrased if q != query]
        return all_queries
    except Exception as e:
        logger.warning(f"Query expansion failed, using original: {e}")
        return [query]


def _extract_sources(response) -> list[dict]:
    """
    Safely extract web source URLs from Gemini grounding metadata.
    Returns a list of {title, url} dicts, or empty list if
    Google Search was not invoked.
    """
    sources = []
    try:
        candidates = response.candidates or []
        for candidate in candidates:
            gm = getattr(candidate, "grounding_metadata", None)
            if not gm:
                continue
            chunks = getattr(gm, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    uri = getattr(web, "uri", None)
                    title = getattr(web, "title", uri or "Source")
                    if uri:
                        sources.append({"title": title, "url": uri})
    except Exception as e:
        logger.warning(f"Could not parse grounding metadata: {e}")
    # Deduplicate by URL
    seen = set()
    unique = []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            unique.append(s)
    return unique


def _format_history(history: list[dict]) -> str:
    """Format past Q&A turns as a readable conversation block for the prompt."""
    if not history:
        return ""
    lines = ["PREVIOUS CONVERSATION (for context only — do NOT repeat these verbatim):"]
    for turn in history:
        q = turn.get("query", "").strip()
        a = turn.get("reply", "").strip()
        if q:
            lines.append(f"User: {q}")
        if a:
            # Truncate very long previous replies to avoid blowing up context
            lines.append(f"Assistant: {a[:600]}{'...' if len(a) > 600 else ''}")
    return "\n".join(lines)


def generate_answer(
    query_text: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
) -> dict:
    """
    Generate an enriched answer using Gemini Flash.

    Returns:
        {
            "text": str,               # Markdown answer
            "sources": list[dict],     # [{title, url}, ...] from Google Search
            "follow_up": list[str],    # 3 suggested follow-up questions
        }
    """
    history = history or []

    # ── Build context block from retrieved chunks ──────────────────────
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("document_name", "Unknown Document")
        page = chunk.get("page_number", "")
        text = chunk.get("text", "")
        page_tag = f" (Page {page})" if page else ""
        context_text += f"\n--- Source {i}: {source}{page_tag} ---\n{text}\n"

    # ── System instruction ─────────────────────────────────────────────
    system_instruction = (
        "You are a highly helpful, precise, and conversational AI Research Assistant.\n\n"
        "ANSWER STRATEGY:\n"
        "1. First, use the provided Context excerpts to answer the question.\n"
        "2. If the Context does NOT contain the answer, use Google Search to find "
        "reliable, up-to-date web information.\n"
        "3. If the user refers to something said earlier in the conversation, use the "
        "PREVIOUS CONVERSATION section to understand the context.\n\n"
        "FORMATTING RULES:\n"
        "- ALWAYS use markdown formatting.\n"
        "- Use bullet points `- ` or numbered lists `1.` for steps and lists.\n"
        "- Use **bold** for key terms and concepts.\n"
        "- Use `code blocks` for code, commands, or technical strings.\n"
        "- Keep paragraphs short. NEVER output walls of text.\n"
        "- When citing your own documents, add `(Source: DocumentName)` inline.\n"
        "- When citing Google Search results, add `[Read more](URL)` inline.\n\n"
        "FOLLOW-UP QUESTIONS:\n"
        "After your main answer, always end with a section:\n"
        "**💡 You might also want to ask:**\n"
        "- Question 1\n"
        "- Question 2\n"
        "- Question 3"
    )

    # ── Build full prompt ──────────────────────────────────────────────
    history_block = _format_history(history)
    prompt_parts = []
    if history_block:
        prompt_parts.append(history_block)
    if context_text.strip():
        prompt_parts.append(f"DOCUMENT CONTEXT:\n{context_text}")
    prompt_parts.append(f"User Question:\n{query_text}")
    prompt = "\n\n".join(prompt_parts)

    # ── Call Gemini ────────────────────────────────────────────────────
    model_name = _get_model_name()
    client = get_client()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.15,
                tools=[types.Tool(google_search=types.GoogleSearchRetrieval())],
            ),
        )
        answer_text = response.text or ""
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        return {
            "text": f"⚠️ Error generating answer: {str(e)}",
            "sources": [],
            "follow_up": [],
        }

    # ── Extract grounding source URLs ──────────────────────────────────
    sources = _extract_sources(response)

    # ── Parse follow-up questions from the response text ──────────────
    follow_up = []
    try:
        if "You might also want to ask:" in answer_text:
            section = answer_text.split("You might also want to ask:")[-1]
            for line in section.strip().split("\n"):
                line = line.strip().lstrip("-•* ").strip()
                if line and "?" in line:
                    follow_up.append(line)
                    if len(follow_up) == 3:
                        break
    except Exception:
        pass

    return {
        "text": answer_text,
        "sources": sources,
        "follow_up": follow_up,
    }
