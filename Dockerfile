FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY server_groq.py .

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn server_groq:app --host 0.0.0.0 --port ${PORT}"]