"""
config.py — Central configuration loader for Agent OTG
=======================================================

Loads settings from .env first, then falls back to system environment variables.
All other modules import model names and settings from here — never hardcode them.

Usage:
    from config import MAIN_MODEL, FAST_MODEL, CODER_MODEL, IMAGE_MODEL
"""

import os
from pathlib import Path

# ── Load .env file ─────────────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH, override=False)   # override=False: system env vars take priority
except ImportError:
    # python-dotenv not installed — values will come from system env or defaults below
    pass

# ── Model Configuration ────────────────────────────────────────────────────────
# MAIN_MODEL and FAST_MODEL use qwen2.5:7b.
# CODER_MODEL (qwen2.5-coder) handles ALL coding questions exclusively.
CODER_MODEL  = os.getenv("CODER_MODEL",  "qwen2.5-coder:latest")
MAIN_MODEL   = os.getenv("MAIN_MODEL",   "qwen2.5:7b")
FAST_MODEL   = os.getenv("FAST_MODEL",   "qwen2.5:7b")
IMAGE_MODEL  = os.getenv("IMAGE_MODEL",  "qwen2.5vl:7b")

# The 14B model is reserved for an explicit `/complex` terminal/API command.
# Normal work continues through the fast 7B/coder/vision routing path.
COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", "qwen2.5:14b")
DIRECT_14B_MODE = os.getenv("DIRECT_14B_MODE", "true").lower() == "true"

# Generation limits are intentionally conservative.  They keep the local
# process responsive and prevent a large context/history from consuming RAM.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1536"))
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "6"))
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "12000"))

# Local image generation is separate from IMAGE_MODEL (which is a vision
# model).  This must name weights that are already available locally.
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "stabilityai/sd-turbo")
IMAGE_GEN_LOCAL_ONLY = os.getenv("IMAGE_GEN_LOCAL_ONLY", "false").lower() == "true"
IMAGE_GEN_SIZE = int(os.getenv("IMAGE_GEN_SIZE", "512"))

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── RAG ────────────────────────────────────────────────────────────────────────
RAG_LLM_MODEL          = os.getenv("RAG_LLM_MODEL",          MAIN_MODEL)
RAG_EMBEDDING_MODEL    = os.getenv("RAG_EMBEDDING_MODEL",    "nomic-embed-text")
RAG_VISION_MODEL       = os.getenv("RAG_VISION_MODEL",       IMAGE_MODEL)
RAG_TOP_K              = int(os.getenv("RAG_TOP_K",          "8"))
RAG_FETCH_K            = int(os.getenv("RAG_FETCH_K",        "20"))
RAG_MAX_CONTEXT_DOCS   = int(os.getenv("RAG_MAX_CONTEXT_DOCS","8"))
RAG_CHUNK_SIZE         = int(os.getenv("RAG_CHUNK_SIZE",     "900"))
RAG_CHUNK_OVERLAP      = int(os.getenv("RAG_CHUNK_OVERLAP",  "150"))
RAG_ENABLE_MULTI_QUERY       = os.getenv("RAG_ENABLE_MULTI_QUERY",       "false").lower() == "true"
RAG_ENABLE_SEMANTIC_SPLITTER = os.getenv("RAG_ENABLE_SEMANTIC_SPLITTER", "false").lower() == "true"
RAG_QUERY_EXPANSION          = os.getenv("RAG_QUERY_EXPANSION",          "false").lower() == "true"

# ── Output / Files ─────────────────────────────────────────────────────────────
# All generated files (PDFs, Word docs, etc.) go to ~/Downloads/AgentOTG/
DOWNLOADS_DIR = Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads" / "AgentOTG"

# ── Smart Routing ──────────────────────────────────────────────────────────────
# When True, the system auto-detects intent (RAG / agent / image / normal)
# without requiring the user to type /rag, /agent, /image etc.
SMART_ROUTING_ENABLED = os.getenv("SMART_ROUTING_ENABLED", "true").lower() == "true"

# ── PostgreSQL ─────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "agent_otg")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ── App Metadata ───────────────────────────────────────────────────────────────
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
APP_TEAM    = "DWE Team"
APP_NAME    = "Agent OTG"
