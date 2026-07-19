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
import re  # NEW
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


# ── NEW: Text normalisation ────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[!?.]+$", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


# ── NEW: Expanded intent sets ──────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "hiya", "howdy", "heya",
    "helo", "hii", "heyy"
}

FILLERS = {
    "ok", "okay", "kk", "k",
    "cool", "alright", "right",
    "thanks", "thank you", "thx", "ty",
    "yes", "yeah", "yep", "yup",
    "no", "nope"
}


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

    print("Shutting down server...")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(title="CKD Chatbot API", lifespan=lifespan)

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
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryMessage] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    retrieved_chunks: List[dict]


# ── Core functions ─────────────────────────────────────────────
def search_faiss(query: str):
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
    context_parts = []

    for chunk, score in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['page_title']} — {chunk['section_title']}]\n"
            f"{chunk['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    history_text = ""
    if history:
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

    # ── NEW: Normalised intent handling ────────────────────────
    normalized = normalize_text(query)

    if normalized in GREETINGS:
        return ChatResponse(
            answer="Hello! I'm a CKD assistant powered by NHS guidance. You can ask me about chronic kidney disease — symptoms, diagnosis, treatment, prevention, or living with the condition. How can I help?",
            sources=[],
            retrieved_chunks=[],
        )

    if normalized in FILLERS and req.history:
        return ChatResponse(
            answer="No problem — feel free to ask any questions about chronic kidney disease whenever you're ready.",
            sources=[],
            retrieved_chunks=[],
        )

    # ── ORIGINAL LOGIC CONTINUES UNCHANGED ─────────────────────
    search_query = query
    if req.history:
        previous_user_msgs = [m.text for m in req.history if m.role == "user"]
        if previous_user_msgs:
            recent_context = " ".join(previous_user_msgs[-3:])
            search_query = f"{recent_context} {query} {query}"

    results = search_faiss(search_query)

    if not results:
        return ChatResponse(
            answer="That question doesn't appear to be related to chronic kidney disease (CKD), which is what I'm designed to help with. Feel free to ask me anything about CKD symptoms, diagnosis, treatment, prevention, or living with the condition.",
            sources=[],
            retrieved_chunks=[],
        )

    prompt = build_prompt(query, results, req.history)
    answer = ask_ollama(prompt)

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