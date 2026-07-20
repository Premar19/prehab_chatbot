"""
CKD Chatbot — FastAPI Backend (LangGraph edition)
--------------------------------------------------
The chat flow as an explicit LangGraph state machine instead of the if/else
chain in the earlier servers. Same /chat contract, same LangChain components
(imported from server_langchain, so both servers share one implementation),
same CHATBOT_MODE=local|cloud switch.

The graph:

    message ──► triage ──red flag──────────► urgent-care response ─► END
                  │  ├───greeting──────────► canned greeting ──────► END
                  │  └───filler────────────► canned filler ────────► END
                  ▼ normal question
              rewrite_query   (LLM turns follow-ups into standalone queries)
                  ▼
               retrieve   ──nothing relevant──► out-of-scope reply ─► END
                  ▼
               generate   (grounded answer + NHS citations) ───────► END

New over the previous servers:
  * triage node    — deterministic red-flag symptom gate. Emergencies get an
    urgent-care response instead of an educational RAG answer, and never reach
    the LLM at all.
  * rewrite node   — an LLM call rewrites follow-ups ("is it hereditary?")
    into standalone retrieval queries ("is chronic kidney disease
    hereditary?"), replacing the old concatenate-and-repeat heuristic.

Run: python server_langgraph.py         (or uvicorn server_langgraph:app)
GET /graph returns the flow as Mermaid text for documentation/demos.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional, TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Shared components — one implementation for both servers.
from server_langchain import (
    _STATE,
    FILLERS,
    GREETINGS,
    MODE,
    format_docs,
    format_history,
    load_components,
    normalize_text,
)

# ── Red-flag triage ────────────────────────────────────────────
# Deliberately deterministic (keyword-based, no LLM): a safety gate should be
# predictable and auditable, and must work even when the model is down.
RED_FLAGS = [
    "chest pain", "chest tightness", "heart attack",
    "can't breathe", "cannot breathe", "cant breathe", "struggling to breathe",
    "short of breath", "shortness of breath",
    "collapsed", "collapse", "passed out", "unconscious", "fainted",
    "seizure", "fitting",
    "not urinated", "haven't urinated", "no urine", "stopped urinating",
    "coughing up blood", "vomiting blood",
    "suicidal", "kill myself", "end my life", "self harm", "self-harm",
    "overdose",
]

URGENT_ANSWER = (
    "Your message mentions symptoms that may need urgent medical attention. "
    "I'm only able to provide general information about chronic kidney disease, "
    "so please don't wait for advice here.\n\n"
    "- If this is an emergency (such as chest pain, difficulty breathing, or "
    "loss of consciousness), call 999 or go to A&E now.\n"
    "- If you need urgent advice but it is not life-threatening, call NHS 111.\n"
    "- If you are struggling with thoughts of harming yourself, call 999, or "
    "the Samaritans on 116 123 — they are available 24/7."
)

GREETING_ANSWER = (
    "Hello! I'm a CKD assistant powered by NHS guidance. You can ask me about "
    "chronic kidney disease — symptoms, diagnosis, treatment, prevention, or "
    "living with the condition. How can I help?"
)

FILLER_ANSWER = (
    "No problem — feel free to ask any questions about chronic kidney disease "
    "whenever you're ready."
)

OUT_OF_SCOPE_ANSWER = (
    "That question doesn't appear to be related to chronic kidney disease "
    "(CKD), which is what I'm designed to help with. Feel free to ask me "
    "anything about CKD symptoms, diagnosis, treatment, prevention, or living "
    "with the condition."
)

# ── Query rewriting ────────────────────────────────────────────
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the patient's latest message as one standalone question about
chronic kidney disease, using the conversation for context. The rewritten
question will be used for a document search, so include the specific topic
being discussed. If the message is already self-contained, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation:
{history}

Latest message:
{question}

Standalone question:"""
)


# ── Graph state ────────────────────────────────────────────────
class ChatState(TypedDict, total=False):
    question: str
    history_text: str          # pre-formatted, "(No previous messages)" if none
    has_history: bool
    search_query: str
    docs: List[Document]
    answer: str
    route: str                 # set by triage: urgent|greeting|filler|normal


# ── Nodes ──────────────────────────────────────────────────────
def triage(state: ChatState) -> ChatState:
    """Classify the message before anything else touches it."""
    q = state["question"].lower()
    if any(flag in q for flag in RED_FLAGS):
        return {"route": "urgent"}
    normalized = normalize_text(state["question"])
    if normalized in GREETINGS:
        return {"route": "greeting"}
    if normalized in FILLERS and state.get("has_history"):
        return {"route": "filler"}
    return {"route": "normal"}


def urgent(state: ChatState) -> ChatState:
    return {"answer": URGENT_ANSWER, "docs": []}


def greeting(state: ChatState) -> ChatState:
    return {"answer": GREETING_ANSWER, "docs": []}


def filler(state: ChatState) -> ChatState:
    return {"answer": FILLER_ANSWER, "docs": []}


def rewrite_query(state: ChatState) -> ChatState:
    """Turn follow-ups into standalone retrieval queries via the LLM."""
    if not state.get("has_history"):
        return {"search_query": state["question"]}
    try:
        chain = REWRITE_PROMPT | _STATE["llm"] | StrOutputParser()
        rewritten = chain.invoke({
            "history": state["history_text"],
            "question": state["question"],
        }).strip()
        # Guard against a chatty model returning prose instead of a query.
        if not rewritten or len(rewritten) > 300:
            rewritten = state["question"]
        return {"search_query": rewritten}
    except Exception:
        # Retrieval must not die because rewriting did.
        return {"search_query": state["question"]}


def retrieve(state: ChatState) -> ChatState:
    docs = _STATE["retriever"].invoke(state["search_query"])
    return {"docs": docs}


def out_of_scope(state: ChatState) -> ChatState:
    return {"answer": OUT_OF_SCOPE_ANSWER, "route": "out_of_scope"}


def generate(state: ChatState) -> ChatState:
    answer = _STATE["chain"].invoke({
        "context": format_docs(state["docs"]),
        "history": state["history_text"],
        "question": state["question"],
    })
    return {"answer": answer}


# ── Graph wiring ───────────────────────────────────────────────
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("triage", triage)
    g.add_node("urgent", urgent)
    g.add_node("greeting", greeting)
    g.add_node("filler", filler)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve", retrieve)
    g.add_node("out_of_scope", out_of_scope)
    g.add_node("generate", generate)

    g.add_edge(START, "triage")
    g.add_conditional_edges(
        "triage",
        lambda s: s["route"],
        {
            "urgent": "urgent",
            "greeting": "greeting",
            "filler": "filler",
            "normal": "rewrite_query",
        },
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_conditional_edges(
        "retrieve",
        lambda s: "generate" if s["docs"] else "out_of_scope",
        {"generate": "generate", "out_of_scope": "out_of_scope"},
    )
    for terminal in ("urgent", "greeting", "filler", "out_of_scope", "generate"):
        g.add_edge(terminal, END)

    return g.compile()


# ── FastAPI app ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_components(MODE)
    _STATE["graph"] = build_graph()
    print(f"\nServer ready — LangGraph state machine compiled ({MODE} mode)\n")
    yield
    print("Shutting down server...")


app = FastAPI(title="CKD Chatbot API (LangGraph)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models (same contract as the other servers) ─
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
    route: Optional[str] = None      # which graph path answered (debug/demo)
    search_query: Optional[str] = None  # the rewritten retrieval query


# ── API endpoints ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    result = _STATE["graph"].invoke({
        "question": query,
        "history_text": format_history(req.history),
        "has_history": bool(req.history),
    })

    docs = result.get("docs") or []
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

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        retrieved_chunks=retrieved_info,
        route=result.get("route"),
        search_query=result.get("search_query"),
    )


@app.get("/graph", response_class=PlainTextResponse)
def graph_diagram():
    """The compiled graph as Mermaid text — paste into mermaid.live to render."""
    return _STATE["graph"].get_graph().draw_mermaid()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": MODE,
        "graph_compiled": "graph" in _STATE,
        "nodes": list(_STATE["graph"].get_graph().nodes) if "graph" in _STATE else [],
    }


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
