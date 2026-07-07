"""
Central configuration for the agentic RAG pipeline.
Keeping all tunables in one place makes it easy to defend
design choices in an interview (e.g. "why 500-token chunks?").
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

# --- Embedding model ---
# all-MiniLM-L6-v2: 384-dim, fast, good enough for small personal corpora.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM (Ollama) ---
OLLAMA_MODEL = "gemma:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Chunking ---
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap to preserve context across chunk boundaries

# --- Retrieval ---
TOP_K = 4              # number of chunks to retrieve per query
