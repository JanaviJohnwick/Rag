"""
Centralized configuration for the RAG Pipeline.
All tunable parameters and environment variables are managed here.
"""

import os
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── API Keys ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER = os.path.join(BASE_DIR, "data")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# ── Text Splitting ──────────────────────────────────────────────────────────
# Tuned to keep semantic sentences fully intact across chunk boundaries.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"

# ── LLM (Groq + Llama) ─────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0   # Deterministic for factual answers
LLM_MAX_TOKENS = 1024

# ── Retriever ───────────────────────────────────────────────────────────────
RETRIEVER_TOP_K = 4      # Number of relevant chunks to retrieve
