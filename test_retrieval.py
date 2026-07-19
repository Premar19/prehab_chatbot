import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
model = SentenceTransformer("all-MiniLM-L6-v2")

q = "What are the symptoms of chronic kidney disease?"
vec = model.encode([q], normalize_embeddings=True)[0].tolist()
emb = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

res = sb.rpc("match_rag_chunks", {"query_embedding": emb, "match_count": 5}).execute()

print("ROWS RETURNED:", len(res.data))
print("RAW:", res.data)
print("---")
for r in res.data:
    print(round(r["similarity"], 3), "|", r["section"][:60])