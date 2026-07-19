"""
CKD Chatbot — Terminal Client
------------------------------
Interactive terminal client for testing the chatbot by hand.

Talks to the /chat HTTP endpoint, so it works against whichever backend is
running (server.py / Ollama, or server_groq.py / Groq) with no changes.

Run:  python chat_cli.py            (defaults to http://localhost:8000)
      python chat_cli.py --no-rag   (hit the ablation endpoint instead)
      python chat_cli.py --port 8001

Commands: /bye to exit, /reset to clear conversation history.
"""

import argparse
import time

import requests

parser = argparse.ArgumentParser(description="Terminal client for the CKD chatbot")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--host", default="localhost")
parser.add_argument("--no-rag", action="store_true", help="use the /chat-no-rag ablation endpoint")
parser.add_argument("--quiet", action="store_true", help="hide sources and retrieval scores")
args = parser.parse_args()

endpoint = "/chat-no-rag" if args.no_rag else "/chat"
url = f"http://{args.host}:{args.port}{endpoint}"

# Confirm the server is up before prompting, so a dead backend gives a clear
# message rather than a traceback on the first question.
try:
    health = requests.get(f"http://{args.host}:{args.port}/health", timeout=5).json()
    print(f"Connected to {url}")
    if health.get("total_chunks"):
        print(f"  {health['total_chunks']} chunks loaded")
except requests.RequestException:
    print(f"Cannot reach the server at http://{args.host}:{args.port}")
    print("Start one first, e.g.:")
    print("  uvicorn server:app --reload --port 8000        (Ollama)")
    print("  uvicorn server_groq:app --reload --port 8000   (Groq)")
    raise SystemExit(1)

print("Type /bye to exit, /reset to clear history.\n")

history = []

while True:
    try:
        query = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not query:
        continue
    if query.lower() == "/bye":
        print("Exiting.")
        break
    if query.lower() == "/reset":
        history = []
        print("History cleared.\n")
        continue

    t0 = time.time()
    try:
        resp = requests.post(
            url,
            json={"message": query, "history": history},
            timeout=300,
        )
    except requests.RequestException as e:
        print(f"\nRequest failed: {e}\n")
        continue

    elapsed = time.time() - t0

    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text)
        print(f"\nError {resp.status_code}: {detail}\n")
        continue

    data = resp.json()
    print(f"\nBot ({elapsed:.1f}s): {data['answer']}\n")

    if not args.quiet:
        if data.get("retrieved_chunks"):
            print("  Retrieved:")
            for c in data["retrieved_chunks"]:
                print(f"    {c['score']:.3f}  {c['section_title'][:60]}")
        if data.get("sources"):
            print("  Sources:")
            for s in data["sources"]:
                print(f"    {s['title']} — {s['url']}")
        print()

    history.append({"role": "user", "text": query})
    history.append({"role": "bot", "text": data["answer"]})
