"""
CKD Chatbot — FastAPI Backend (Supabase + Groq edition)
--------------------------------------------------------
Exposes a POST /chat endpoint that:
1. Takes a patient question
2. Searches Supabase (pgvector) for relevant NHS CKD content
3. Sends context + question to a hosted LLM (Groq, llama-3.3-70b) via an
   OpenAI-compatible API
4. Returns the response with sources

The LLM is now a hosted API instead of local Ollama. The call goes through
ask_llm(), the single place to change if you later swap providers (e.g. Azure
OpenAI for clinical deployment) — just update the LLM_* config and client below.

Run: uvicorn server_new:app --reload --port 8000
Setup:
    pip install openai supabase sentence-transformers fastapi uvicorn python-dotenv
    .env must contain: SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY
"""

import os
import re
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

load_dotenv()

# ── Configuration ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
RAG_TABLE = "rag_chunks"
MATCH_FN = "match_rag_chunks"

# LLM provider config — this block is the swap point for other providers.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 512

TOP_K = 5
MIN_SCORE = 0.45

# ── Global state (loaded once at startup) ──────────────────────
sb = None
embed_model = None
llm_client = None


# ── Text normalisation ─────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[!?.]+$", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


# ── Intent sets ────────────────────────────────────────────────
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


# ── Lifespan: load model + connect to Supabase on startup ──────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sb, embed_model, llm_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY must be set in .env")

    print("Connecting to Supabase...")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    count = sb.table(RAG_TABLE).select("id", count="exact").execute().count
    print(f"  ✓ Connected — {count} chunks in '{RAG_TABLE}'")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("  ✓ Model loaded")

    print(f"Initialising LLM client (Groq — {LLM_MODEL})...")
    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=GROQ_API_KEY)
    print("  ✓ LLM client ready")

    print("\n🟢 Server ready — Supabase + embeddings + Groq loaded\n")

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
def search_supabase(query: str):
    """Embed the query and retrieve the closest NHS chunks from Supabase.

    Returns the same shape the old FAISS path returned — a list of
    (chunk_dict, score) — so build_prompt() and the response code below
    stay unchanged. chunk_dict keys mirror the old metadata keys.
    """
    vec = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].tolist()
    emb_literal = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

    resp = sb.rpc(
        MATCH_FN,
        {"query_embedding": emb_literal, "match_count": TOP_K},
    ).execute()

    results = []
    for row in (resp.data or []):
        score = float(row["similarity"])
        if score >= MIN_SCORE:
            chunk = {
                "page_title": row["title"],
                "section_title": row["section"],
                "content": row["content"],
                "source_url": row["source_url"],
            }
            results.append((chunk, score))

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


def ask_llm(prompt: str) -> str:
    """Send the prompt to the hosted LLM (Groq, OpenAI-compatible).
    This is the single swap point for changing providers later.
    """
    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="The model is rate-limited (free-tier tokens-per-minute limit). Please wait a moment and try again.",
        )
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="The language model took too long to respond.")
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach the language model service.")


# ── API endpoint ───────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ── Normalised intent handling ─────────────────────────────
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

    # ── Retrieval + generation ─────────────────────────────────
    search_query = query
    if req.history:
        previous_user_msgs = [m.text for m in req.history if m.role == "user"]
        if previous_user_msgs:
            recent_context = " ".join(previous_user_msgs[-3:])
            search_query = f"{recent_context} {query} {query}"

    results = search_supabase(search_query)

    if not results:
        return ChatResponse(
            answer="That question doesn't appear to be related to chronic kidney disease (CKD), which is what I'm designed to help with. Feel free to ask me anything about CKD symptoms, diagnosis, treatment, prevention, or living with the condition.",
            sources=[],
            retrieved_chunks=[],
        )

    prompt = build_prompt(query, results, req.history)
    answer = ask_llm(prompt)

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