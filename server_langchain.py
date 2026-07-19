"""
CKD Chatbot — FastAPI Backend (LangChain edition)
--------------------------------------------------
One server, two deployments, selected with the CHATBOT_MODE env var:

    CHATBOT_MODE=local   →  FAISS index on disk + Llama 3.1 8B via Ollama
    CHATBOT_MODE=cloud   →  Supabase pgvector    + llama-3.3-70b via Groq

Every component is a LangChain primitive, so swapping providers is a config
change instead of a parallel server file (compare server.py / server_groq.py,
which duplicate the whole pipeline per provider):

    retrieval  →  BaseRetriever subclasses (FaissNHSRetriever / SupabaseNHSRetriever)
    embeddings →  HuggingFaceEmbeddings (all-MiniLM-L6-v2, normalised)
    LLM        →  ChatOllama / ChatGroq behind the shared BaseChatModel interface
    prompt     →  ChatPromptTemplate
    pipeline   →  LCEL chain: prompt | llm | StrOutputParser

The /chat contract (request, response, thresholds, prompt wording) is identical
to the original servers so the frontend needs no changes and answers stay
comparable in the retrieval evaluation harness.

Run: uvicorn server_langchain:app --reload --port 8000
.env must contain SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY for cloud mode.
"""

import json
import os
import re
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Configuration ──────────────────────────────────────────────
MODE = os.environ.get("CHATBOT_MODE", "local")  # "local" | "cloud"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
MIN_SCORE = 0.45
LLM_TEMPERATURE = 0.3

FAISS_INDEX_PATH = "data/index/faiss_index.bin"
METADATA_PATH = "data/index/faiss_metadata.json"
OLLAMA_MODEL = "llama3.1:8b"

RAG_TABLE = "rag_chunks"
MATCH_FN = "match_rag_chunks"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Text normalisation + intent sets (same as server_groq) ─────
GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "hiya", "howdy", "heya", "helo", "hii", "heyy",
}

FILLERS = {
    "ok", "okay", "kk", "k", "cool", "alright", "right",
    "thanks", "thank you", "thx", "ty",
    "yes", "yeah", "yep", "yup", "no", "nope",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[!?.]+$", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


# ── Retrievers ─────────────────────────────────────────────────
# Both return LangChain Documents with identical metadata keys, so the rest
# of the pipeline cannot tell which backend produced them.

class FaissNHSRetriever(BaseRetriever):
    """Retrieves NHS CKD chunks from the on-disk FAISS index (local mode)."""

    k: int = TOP_K
    min_score: float = MIN_SCORE

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        vec = _STATE["embeddings"].embed_query(query)
        import numpy as np
        scores, indices = _STATE["faiss_index"].search(
            np.array([vec], dtype=np.float32), k=self.k
        )
        docs = []
        for score, idx in zip(scores[0], indices[0]):
            if float(score) >= self.min_score:
                chunk = _STATE["chunks"][idx]
                docs.append(Document(
                    page_content=chunk["content"],
                    metadata={
                        "page_title": chunk["page_title"],
                        "section_title": chunk["section_title"],
                        "source_url": chunk["source_url"],
                        "score": float(score),
                    },
                ))
        return docs


class SupabaseNHSRetriever(BaseRetriever):
    """Retrieves NHS CKD chunks from Supabase pgvector via RPC (cloud mode)."""

    k: int = TOP_K
    min_score: float = MIN_SCORE

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        vec = _STATE["embeddings"].embed_query(query)
        emb_literal = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
        resp = _STATE["supabase"].rpc(
            MATCH_FN, {"query_embedding": emb_literal, "match_count": self.k}
        ).execute()
        docs = []
        for row in (resp.data or []):
            score = float(row["similarity"])
            if score >= self.min_score:
                docs.append(Document(
                    page_content=row["content"],
                    metadata={
                        "page_title": row["title"],
                        "section_title": row["section"],
                        "source_url": row["source_url"],
                        "score": score,
                    },
                ))
        return docs


# ── Prompt (wording identical to the original servers) ─────────
PROMPT = ChatPromptTemplate.from_template("""You are an NHS educational assistant helping patients understand chronic kidney disease (CKD).

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
{history}

Patient Question:
{question}

Answer:""")


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata['page_title']} — {d.metadata['section_title']}]\n{d.page_content}"
        for d in docs
    )


def format_history(history: List["HistoryMessage"]) -> str:
    if not history:
        return "(No previous messages)"
    lines = []
    for msg in history[-6:]:
        role_label = "Patient" if msg.role == "user" else "Assistant"
        lines.append(f"{role_label}: {msg.text}")
    return "\n".join(lines)


# ── Global state (loaded once at startup) ──────────────────────
_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting in '{MODE}' mode...")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    _STATE["embeddings"] = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    if MODE == "local":
        import faiss
        print("Loading FAISS index...")
        _STATE["faiss_index"] = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _STATE["chunks"] = json.load(f)["chunks"]

        from langchain_ollama import ChatOllama
        _STATE["retriever"] = FaissNHSRetriever()
        _STATE["llm"] = ChatOllama(
            model=OLLAMA_MODEL, temperature=LLM_TEMPERATURE, num_predict=400
        )
    elif MODE == "cloud":
        from supabase import create_client
        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key or not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("Cloud mode needs SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY in .env")
        print("Connecting to Supabase...")
        _STATE["supabase"] = create_client(url, key)

        from langchain_groq import ChatGroq
        _STATE["retriever"] = SupabaseNHSRetriever()
        _STATE["llm"] = ChatGroq(
            model=GROQ_MODEL, temperature=LLM_TEMPERATURE, max_tokens=512
        )
    else:
        raise RuntimeError(f"Unknown CHATBOT_MODE '{MODE}' — use 'local' or 'cloud'")

    # The LCEL pipeline: prompt template → chat model → plain string out.
    _STATE["chain"] = PROMPT | _STATE["llm"] | StrOutputParser()

    print(f"\nServer ready — LangChain pipeline loaded ({MODE} mode)\n")
    yield
    print("Shutting down server...")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(title="CKD Chatbot API (LangChain)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models (same contract as the originals) ───
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


# ── API endpoint ───────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

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

    # Query expansion for follow-ups: recent user messages keep the topic
    # alive, current query repeated to dominate the embedding.
    search_query = query
    if req.history:
        previous_user_msgs = [m.text for m in req.history if m.role == "user"]
        if previous_user_msgs:
            recent_context = " ".join(previous_user_msgs[-3:])
            search_query = f"{recent_context} {query} {query}"

    docs = _STATE["retriever"].invoke(search_query)

    if not docs:
        return ChatResponse(
            answer="That question doesn't appear to be related to chronic kidney disease (CKD), which is what I'm designed to help with. Feel free to ask me anything about CKD symptoms, diagnosis, treatment, prevention, or living with the condition.",
            sources=[],
            retrieved_chunks=[],
        )

    answer = _STATE["chain"].invoke({
        "context": format_docs(docs),
        "history": format_history(req.history),
        "question": query,
    })

    seen_urls = set()
    sources = []
    for d in docs:
        url = d.metadata["source_url"]
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({"title": d.metadata["page_title"], "url": url})

    retrieved_info = [
        {
            "section_title": d.metadata["section_title"],
            "score": round(d.metadata["score"], 3),
            "source_url": d.metadata["source_url"],
        }
        for d in docs
    ]

    return ChatResponse(answer=answer, sources=sources, retrieved_chunks=retrieved_info)


# ── Health check endpoint ──────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": MODE,
        "retriever": type(_STATE.get("retriever")).__name__ if _STATE.get("retriever") else None,
        "llm": type(_STATE.get("llm")).__name__ if _STATE.get("llm") else None,
    }


# ── Entry point ────────────────────────────────────────────────
# Lets `python server_langchain.py` work directly. Without this block the file
# only defines the app and exits silently — uvicorn is what actually serves it.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
