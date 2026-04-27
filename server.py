"""
CKD Chatbot — FastAPI Backend
-------------------------------
Exposes a POST /chat endpoint that:
1. Takes a patient question
2. Searches FAISS for relevant NHS CKD content
3. Sends context + question to Llama 3.1 via Ollama
4. Returns the response with sources

Run: uvicorn server:app --reload --port 8000
Requires: Ollama running with llama3.1:8b loaded
"""

import json
import numpy as np
import faiss
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────
FAISS_INDEX_PATH = "data/index/faiss_index.bin"
METADATA_PATH = "data/index/faiss_metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
TOP_K = 5
MIN_SCORE = 0.45

# ── Global state (loaded once at startup) ──────────────────────
faiss_index = None
chunks = None
embed_model = None


# ── Lifespan: load models on startup ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global faiss_index, chunks, embed_model

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    chunks = metadata["chunks"]
    print(f"  ✓ Loaded {faiss_index.ntotal} vectors")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("  ✓ Model loaded")

    print("\n🟢 Server ready — FAISS + embeddings loaded\n")

    yield  # app runs here

    # Cleanup on shutdown (nothing needed for now)
    print("Shutting down server...")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(title="CKD Chatbot API", lifespan=lifespan)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from typing import List


# ── Request/Response models ────────────────────────────────────
class HistoryMessage(BaseModel):
    role: str  # "user" or "bot"
    text: str


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryMessage] = []  # Previous messages in conversation


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    retrieved_chunks: List[dict]


# ── Core functions ─────────────────────────────────────────────
def search_faiss(query: str):
    """Search FAISS index and return relevant chunks above threshold."""
    query_vector = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = faiss_index.search(query_vector, k=TOP_K)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if float(score) >= MIN_SCORE:
            results.append((chunks[idx], float(score)))

    return results


def build_prompt(query: str, retrieved_chunks, history: List["HistoryMessage"] = None):
    """Build the prompt with retrieved context and conversation history for Llama."""
    context_parts = []

    for chunk, score in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['page_title']} — {chunk['section_title']}]\n"
            f"{chunk['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Build conversation history string (last few exchanges for context)
    history_text = ""
    if history:
        # Take last 6 messages (3 exchanges) to keep prompt manageable
        recent = history[-6:]
        history_lines = []
        for msg in recent:
            role_label = "Patient" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg.text}")
        history_text = "\n".join(history_lines)

    prompt = f"""You are an NHS educational assistant helping patients understand chronic kidney disease (CKD).

RULES:
- Your response will be shown directly to a patient.
- Provide general educational information ONLY.
- Do NOT give personalised medical advice.
- Do NOT tell the patient what they should do — use phrases like "patients are generally advised to" or "NHS guidance suggests".
- If the patient asks about something not covered in the context, say clearly that you do not have that information and suggest they speak to their GP.
- Use simple, clear language suitable for a general audience.
- Use ONLY the information provided in the context below.
- Consider the previous conversation when answering follow-up questions.

Context:
{context}

Previous Conversation:
{history_text if history_text else "(No previous messages)"}

Patient Question:
{query}

Answer:"""

    return prompt


def ask_ollama(prompt: str) -> str:
    """Send prompt to Ollama and return the response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 400,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure it's running.",
        )
    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama took too long to respond.",
        )


# ── API endpoint ───────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Handle greetings without calling FAISS or the LLM
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hiya", "howdy","heya"}
    if query.lower().strip("!., ") in greetings:
        return ChatResponse(
            answer="Hello! I'm a CKD assistant powered by NHS guidance. You can ask me about chronic kidney disease — symptoms, diagnosis, treatment, prevention, or living with the condition. How can I help?",
            sources=[],
            retrieved_chunks=[],
        )

    # Step 1: Search FAISS
    # For follow-up questions, we expand the search query with the
    # user's previous messages from this conversation. This preserves
    # the conversation topic across multiple turns. Without this,
    # short follow-ups like "tell me more" or affirmations like "yes"
    # would fall below the relevance threshold and trigger the
    # off-topic fallback.
    #
    # Strategy: concatenate recent user messages with the current
    # query, then REPEAT the current query to give its terms more
    # weight in the resulting embedding (term repetition weighting).
    # This keeps the conversation topic alive while ensuring the
    # latest intent dominates retrieval.
    search_query = query
    if req.history:
        previous_user_msgs = [m.text for m in req.history if m.role == "user"]
        if previous_user_msgs:
            recent_context = " ".join(previous_user_msgs[-3:])
            # Current query repeated twice weights it more heavily
            # than the historical context in the embedding space
            search_query = f"{recent_context} {query} {query}"

    results = search_faiss(search_query)

    if not results:
        return ChatResponse(
            answer="That question doesn't appear to be related to chronic kidney disease (CKD), which is what I'm designed to help with. Feel free to ask me anything about CKD symptoms, diagnosis, treatment, prevention, or living with the condition.",
            sources=[],
            retrieved_chunks=[],
        )

    # Step 2: Build prompt with conversation history
    prompt = build_prompt(query, results, req.history)

    # Step 3: Get response from Llama
    answer = ask_ollama(prompt)

    # Step 4: Build source list (deduplicated)
    seen_urls = set()
    sources = []
    for chunk, score in results:
        url = chunk["source_url"]
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "title": chunk["page_title"],
                "url": url,
            })

    # Step 5: Build retrieved chunks info (useful for frontend debugging)
    retrieved_info = [
        {
            "section_title": chunk["section_title"],
            "score": round(score, 3),
            "source_url": chunk["source_url"],
        }
        for chunk, score in results
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved_info,
    )


# ── Health check endpoint ──────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": faiss_index is not None,
        "chunks_loaded": chunks is not None,
        "model_loaded": embed_model is not None,
        "total_chunks": len(chunks) if chunks else 0,
    }