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
# Deliberately deterministic (pattern-based, no LLM): a safety gate should be
# predictable and auditable, and must work even when the model is down.
#
# Patterns, not exact phrases: "chest pain", "pain in my chest", and "my chest
# really hurts" must all match, as must "can't breathe" / "unable to breathe" /
# "hard to breathe". Each entry pairs the symptom word with its qualifiers in
# either order, within a short window.
import re as _re

_W = r"[^.!?]{0,40}?"  # words in between, but never across sentence boundaries

RED_FLAG_PATTERNS = [
    # chest pain / tightness / pressure, either word order
    rf"\bchest\b{_W}\b(pain|hurt\w*|ach\w*|tight\w*|pressure|crush\w*)",
    rf"\b(pain\w*|hurt\w*|ach\w*|tight\w*|pressure|crush\w*)\b{_W}\bchest\b",
    r"\bheart attack\b",
    # breathing difficulty: negation/struggle word + any breath variant
    rf"\b(can'?t|can ?not|cannot|unable|struggl\w*|difficult\w*|hard|short(ness)?)\b{_W}\bbreath\w*",
    rf"\bbreath\w*{_W}\b(difficult\w*|problem\w*|trouble|struggl\w*)",
    # collapse / consciousness
    r"\b(collaps\w*|passed out|pass out|unconscious|faint\w*|seizure|fitting)\b",
    # anuria (no urine output)
    rf"\b(no|not|stop\w*|haven'?t|unable)\b{_W}\burin\w*",
    # blood where it shouldn't be
    rf"\b(cough\w*|vomit\w*|throw\w* up|spit\w*)\b{_W}\bblood\b",
    # self-harm / crisis
    r"\b(suicid\w*|kill myself|end my life|self.?harm|overdose|want to die|don'?t want to live)\b",
]

_RED_FLAG_RE = [_re.compile(p, _re.IGNORECASE) for p in RED_FLAG_PATTERNS]


def is_red_flag(text: str) -> bool:
    return any(rx.search(text) for rx in _RED_FLAG_RE)


# ── Layer 2: LLM safety classifier ─────────────────────────────
# Catches phrasings the rules can't enumerate ("an elephant is sitting on my
# chest"). Runs only when the rules did NOT flag, so the deterministic layer
# stays the fast path and keeps working if the model is down.
TRIAGE_CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are a safety classifier for an NHS kidney-disease chatbot.

Decide whether the patient's message describes a possible medical emergency or
crisis they may be experiencing RIGHT NOW — for example chest pain, trouble
breathing, signs of stroke or heart attack, collapse, severe bleeding,
inability to urinate, or thoughts of self-harm.

General or educational questions ABOUT symptoms (e.g. "does CKD cause chest
pain?") are NOT emergencies. If it is genuinely unclear whether the patient is
describing their own current emergency, treat it as an emergency.

Answer with exactly one word: EMERGENCY or OK.

Patient message:
{question}

Answer:"""
)


def llm_flags_emergency(text: str) -> bool:
    """LLM opinion on messages the rules didn't flag. Fails open to the normal
    flow: the rules layer already ran, and the generation prompt's grounding
    guardrail remains behind everything."""
    try:
        chain = TRIAGE_CLASSIFIER_PROMPT | _STATE["llm"] | StrOutputParser()
        verdict = chain.invoke({"question": text}).strip().upper()
        return verdict.startswith("EMERGENCY")
    except Exception:
        return False


URGENT_ANSWER = (
    "Your message mentions symptoms that may need urgent medical attention. "
    "I'm only able to provide general information about chronic kidney disease, "
    "so please don't wait for advice here.\n\n"
    "- If this is an emergency (such as chest pain, difficulty breathing, or "
    "loss of consciousness), call 999 or go to A&E now.\n"
    "- If you need urgent advice but it is not life-threatening, call NHS 111.\n"
    "- If you are struggling with thoughts of harming yourself, call 999, or "
    "the Samaritans on 116 123 — they are available 24/7.\n\n"
    "If you were asking a general question rather than describing how you feel "
    "right now, please rephrase it and I'll do my best to help."
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
    "I'm sorry — I don't have good information about that, as I can only help "
    "with questions about chronic kidney disease (CKD).\n\n"
    "If your question is about another health concern, the NHS website "
    "(nhs.uk) or your GP practice are the best places to start — and if you "
    "need urgent advice, you can always call NHS 111.\n\n"
    "I'm happy to help with anything about CKD: symptoms, diagnosis, "
    "treatment, prevention, or day-to-day life with the condition."
)

# ── Query rewriting ────────────────────────────────────────────
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the patient's latest message as one standalone question, using the
conversation only to resolve what words like "it", "that", or "this" refer to.
Preserve the patient's intended topic exactly — never change the subject of
their question, even if it is not about kidney disease. (Scope filtering
happens later; your job is only to make the question self-contained.)
If the message is already self-contained, return it unchanged.
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
    triage_method: str         # rules | llm_classifier | none


# ── Nodes ──────────────────────────────────────────────────────
def triage(state: ChatState) -> ChatState:
    """Layered safety gate, cheapest check first:
    1. deterministic red-flag rules (no LLM, works offline, auditable)
    2. exact-match greeting/filler shortcuts (no LLM)
    3. LLM safety classifier for phrasings the rules can't enumerate
    """
    if is_red_flag(state["question"]):
        return {"route": "urgent", "triage_method": "rules"}
    normalized = normalize_text(state["question"])
    if normalized in GREETINGS:
        return {"route": "greeting", "triage_method": "none"}
    if normalized in FILLERS and state.get("has_history"):
        return {"route": "filler", "triage_method": "none"}
    if llm_flags_emergency(state["question"]):
        return {"route": "urgent", "triage_method": "llm_classifier"}
    return {"route": "normal", "triage_method": "none"}


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
    triage_method: Optional[str] = None  # rules | llm_classifier | none


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
        triage_method=result.get("triage_method"),
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
