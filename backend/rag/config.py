import os
from pathlib import Path

# ── Load .env (idempotent — safe to call multiple times) ──────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

# All values are configurable through environment variables so the backend can be
# changed at deployment time without rewriting the RAG code.
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")
RAG_LLM_MODEL       = os.getenv("RAG_LLM_MODEL",       "qwen2.5:7b")
RAG_VISION_MODEL    = os.getenv("RAG_VISION_MODEL",    "qwen2.5vl:7b")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR",  str(_BACKEND_ROOT / "chroma_db"))
# New collection avoids dimension conflicts with the old hand-written Chroma code.
CHROMA_COLLECTION   = os.getenv("CHROMA_COLLECTION",   "knowledge_base_v2")

# ── Retrieval tuning ──────────────────────────────────────────────────────────
# Higher TOP_K = more candidates retrieved → better recall at slight latency cost
TOP_K             = int(os.getenv("RAG_TOP_K",            "8"))   # was 5
FETCH_K           = int(os.getenv("RAG_FETCH_K",          "25"))  # was 20
MAX_CONTEXT_DOCS  = int(os.getenv("RAG_MAX_CONTEXT_DOCS", "8"))   # was 6

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = int(os.getenv("RAG_CHUNK_SIZE",    "900"))
CHUNK_OVERLAP     = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))

# ── Feature flags (all ON by default for best out-of-box experience) ──────────
# Multi-query: generates multiple query phrasings via the LLM to boost recall
ENABLE_MULTI_QUERY       = os.getenv("RAG_ENABLE_MULTI_QUERY",       "false").lower()  == "true"
# Semantic splitter: uses embeddings to split at topic boundaries (slower, better)
ENABLE_SEMANTIC_SPLITTER = os.getenv("RAG_ENABLE_SEMANTIC_SPLITTER", "false").lower() == "true"
# Query expansion: rewrites the user query into 2 semantic variants before retrieval
ENABLE_QUERY_EXPANSION   = os.getenv("RAG_QUERY_EXPANSION",          "false").lower()  == "true"

# The application can later pass RBAC filters into ingest/query without changing
# the vector-store API.
DEFAULT_ACCESS_LEVEL = os.getenv("RAG_DEFAULT_ACCESS_LEVEL", "public")
