"""
Central configuration for the agentic RAG pipeline.
Keeping all tunables in one place makes it easy to defend
design choices in an interview (e.g. "why 500-token chunks?").
"""

import os

from dotenv import load_dotenv

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

load_dotenv(os.path.join(BASE_DIR, ".env"))

# --- Embedding model ---
# all-MiniLM-L6-v2: 384-dim, fast, good enough for small personal corpora.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM (OpenRouter) ---
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# --- Chunking ---
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap to preserve context across chunk boundaries

# --- Retrieval ---
TOP_K = 4              # number of chunks to retrieve per query
