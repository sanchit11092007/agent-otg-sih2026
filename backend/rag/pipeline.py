from __future__ import annotations

import json
import logging
import re
import traceback
from functools import lru_cache

logger = logging.getLogger("rag.pipeline")

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from .config import (
    ENABLE_QUERY_EXPANSION,
    OLLAMA_BASE_URL,
    RAG_LLM_MODEL,
    MAX_CONTEXT_DOCS,
)
from .loaders import load_file, load_files
from .prompts import RAG_PROMPT
from .retriever import retrieve, retrieve_expanded
from .splitter import split_documents
from .vectorstore import add_documents, delete_source


# ── LLM (cached) ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm() -> ChatOllama:
    return ChatOllama(model=RAG_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def _get_chain():
    return RAG_PROMPT | _get_llm() | StrOutputParser()


# ── Query expansion ───────────────────────────────────────────────────────────

def _expand_query(query: str) -> list[str]:
    """
    Use the LLM to rewrite the user query into 2 semantic variants.

    This significantly improves retrieval recall because the same information
    can be expressed in many ways in source documents. Falls back to the
    original query if the LLM call fails or returns invalid JSON.

    Example:
        Input:  "refund policy"
        Output: ["refund policy",
                 "return and refund conditions",
                 "how to get a refund or return a product"]
    """
    if not ENABLE_QUERY_EXPANSION:
        return [query]

    prompt = (
        "Generate exactly 2 alternative phrasings of this search query to help "
        "find relevant information in a document knowledge base.\n"
        "Reply with ONLY a JSON array of 2 strings, no explanation, no markdown.\n"
        f"Original query: {query}\n"
        'Example format: ["alternative phrasing one", "alternative phrasing two"]'
    )
    try:
        response = _get_llm().invoke(prompt)
        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()
        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        variants = json.loads(raw)
        if isinstance(variants, list) and variants:
            all_q = [query] + [str(v).strip() for v in variants if str(v).strip()]
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique = []
            for q in all_q:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique[:3]   # original + up to 2 variants
    except Exception:
        pass
    return [query]


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_docs(docs) -> str:
    """Format retrieved documents into a rich context block with full metadata."""
    if not docs:
        return "[NO_RELEVANT_CONTEXT]"
    blocks = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        source    = m.get("file_name") or m.get("source", "unknown")
        page      = m.get("page")
        sheet     = m.get("sheet")
        row       = m.get("row")
        chunk_idx = m.get("chunk_index", "")
        ctype     = m.get("content_type", "text")

        # Build location string
        location_parts = []
        if page:
            location_parts.append(f"page {page}")
        if sheet:
            location_parts.append(f"sheet '{sheet}'")
        if row:
            location_parts.append(f"row {row}")
        location = (", " + ", ".join(location_parts)) if location_parts else ""

        header = f"[{i}] Source: {source}{location} | Type: {ctype}"
        blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(blocks)


def categorize_ingestion_error(exc: Exception) -> str:
    err_msg = str(exc)
    exc_type = type(exc).__name__
    err_lower = err_msg.lower()

    if isinstance(exc, FileNotFoundError) or "file not found" in err_lower:
        return f"File not found: {err_msg}"
    if isinstance(exc, PermissionError) or "permission denied" in err_lower:
        return f"Permission denied: {err_msg}"
    if isinstance(exc, UnicodeDecodeError) or "encoding error" in err_lower:
        return f"Encoding error: {err_msg}"
    if "csv parsing error" in err_lower or "parsererror" in exc_type.lower() or "parsererror" in err_lower:
        return f"CSV parsing error: {err_msg}"
    if isinstance(exc, ValueError) and "unsupported format" in err_lower:
        return f"Unsupported format: {err_msg}"
    if "vectorization" in err_lower or "ollama" in exc_type.lower() or "ollama" in err_lower or "embedding" in err_lower:
        return f"Vectorization error (check that nomic-embed-text is pulled): {err_msg}"
    if "chroma" in exc_type.lower() or "chroma" in err_lower or "database" in err_lower or "storage" in err_lower:
        return f"Database storage error: {err_msg}"

    return f"Ingestion error: {err_msg}"


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_paths(paths: list[str], replace_existing: bool = True) -> dict:
    try:
        raw_docs = load_files(paths)
        if replace_existing:
            sources = {d.metadata.get("source") for d in raw_docs if d.metadata.get("source")}
            for source in sources:
                try:
                    delete_source(source)
                except Exception as del_err:
                    logger.debug(f"Delete source failed: {del_err}\n{traceback.format_exc()}")

        chunks = split_documents(raw_docs)
        if not raw_docs:
            raise ValueError("No readable content was found in the supplied file(s).")
        if not chunks:
            raise ValueError("The supplied file(s) contained no indexable text.")

        try:
            count = add_documents(chunks)
        except Exception as embed_err:
            logger.debug(f"Vectorization/Storage failed:\n{traceback.format_exc()}")
            err_cat = categorize_ingestion_error(embed_err)
            raise RuntimeError(err_cat) from embed_err

        loader_name = raw_docs[0].metadata.get("loader_name", "DocumentLoader") if raw_docs else "UnknownLoader"
        rows_loaded = raw_docs[0].metadata.get("rows_loaded", len(raw_docs)) if raw_docs else 0

        first_path = paths[0] if paths else ""
        return {
            "path":               first_path,
            "file_exists":        True,
            "loader":             loader_name,
            "rows_loaded":        rows_loaded,
            "documents_loaded":   len(raw_docs),
            "chunks_generated":   len(chunks),
            "embeddings_created": count,
            "status":             "SUCCESS",
            "files":              len(paths),
            "chunks_indexed":     count,
        }
    except RuntimeError:
        # Already categorized — re-raise as-is to avoid double-prefixing the message.
        logger.debug(f"Ingestion error trace:\n{traceback.format_exc()}")
        raise
    except Exception as exc:
        logger.debug(f"Ingestion error trace:\n{traceback.format_exc()}")
        cat_msg = categorize_ingestion_error(exc)
        raise RuntimeError(cat_msg) from exc


def ingest_path(path: str, replace_existing: bool = True) -> dict:
    return ingest_paths([path], replace_existing=replace_existing)


# ── RAG Answer (with query expansion loop) ───────────────────────────────────

def answer(
    query: str,
    access_filter: dict | None = None,
    strategy: str | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Full RAG pipeline with optimized query expansion loop:

      1. Expand query → 1–3 semantic variants
      2. Retrieve for ALL variants, deduplicate by content hash
      3. If expansion gave 0 docs, fall back to direct similarity search
      4. Re-rank retrieved docs by keyword overlap
      5. Format rich context and invoke the LLM chain
      6. Return answer + sources + strategy used

    The expansion loop dramatically improves recall without requiring
    MultiQueryRetriever or any extra dependencies.
    """
    # Safely resolve top_k — fix the bug where None caused int() crash
    requested_k: int | None = None
    if top_k is not None:
        try:
            requested_k = max(1, min(int(top_k), 20))
        except (TypeError, ValueError):
            requested_k = None

    queries = _expand_query(query)

    if len(queries) > 1:
        docs, selected_strategy = retrieve_expanded(
            queries,
            strategy=strategy,
            access_filter=access_filter,
            **( {"top_k": requested_k} if requested_k else {}),
        )
    else:
        docs, selected_strategy = retrieve(
            query,
            strategy=strategy,
            access_filter=access_filter,
            **( {"top_k": requested_k} if requested_k else {}),
        )

    # Fallback: if expansion found nothing, try direct similarity
    if not docs and len(queries) > 1:
        docs, selected_strategy = retrieve(
            query,
            strategy="similarity",
            access_filter=access_filter,
            **( {"top_k": requested_k} if requested_k else {}),
        )

    # No docs at all → early return with clear message
    if not docs:
        return {
            "answer":             "I couldn't find sufficient information in the provided knowledge base to answer this question. Please make sure the relevant documents have been ingested using /doc <file>.",
            "sources":            [],
            "retrieval_strategy": selected_strategy,
            "documents":          [],
        }

    # Invoke LLM with formatted, rich context
    context_text = _format_docs(docs)

    # Guard against [NO_RELEVANT_CONTEXT] pass-through to LLM
    if context_text == "[NO_RELEVANT_CONTEXT]":
        return {
            "answer":             "No relevant context was found in the knowledge base for this query. Try rephrasing or ingest more documents.",
            "sources":            [],
            "retrieval_strategy": selected_strategy,
            "documents":          [],
        }

    try:
        result = _get_chain().invoke({"question": query, "context": context_text})
        result = str(result).strip()
        if not result:
            raise ValueError("The local model returned an empty response")
    except Exception as exc:
        # Retrieval has already succeeded.  Preserve a useful, grounded result
        # rather than turning a temporary generation-model failure into a 500.
        logger.warning("RAG generation failed after retrieval: %s", exc)
        excerpts = []
        for doc in docs[:3]:
            source = doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
            text = " ".join(doc.page_content.split())
            excerpts.append(f"[{source}] {text[:700]}")
        result = (
            "I found relevant material in the knowledge base, but the local "
            "answer model is unavailable. Retrieved excerpts:\n\n" + "\n\n".join(excerpts)
        )

    sources = []
    for d in docs:
        entry = {
            "source":      d.metadata.get("file_name") or d.metadata.get("source", "unknown"),
            "page":        d.metadata.get("page"),
            "sheet":       d.metadata.get("sheet"),
            "row":         d.metadata.get("row"),
            "document_id": d.metadata.get("document_id"),
            "chunk_index": d.metadata.get("chunk_index"),
        }
        # Deduplicate sources by source name + page
        sources.append(entry)

    return {
        "answer":             result,
        "sources":            sources,
        "retrieval_strategy": selected_strategy,
        "documents":          docs,
    }


def count() -> int:
    """Return total number of chunks currently stored in the knowledge base."""
    try:
        from .vectorstore import get_vector_store
        return get_vector_store()._collection.count()
    except Exception:
        return 0


def list_sources() -> list[str]:
    """Return list of distinct source document names stored in the knowledge base."""
    try:
        from .vectorstore import get_vector_store
        store = get_vector_store()
        res = store._collection.get(include=["metadatas"])
        metadatas = res.get("metadatas") or []
        seen = set()
        sources = []
        for m in metadatas:
            if not m:
                continue
            s = m.get("file_name") or m.get("source")
            if s and s not in seen:
                seen.add(s)
                sources.append(s)
        return sources
    except Exception:
        return []


def clear_knowledge_base() -> dict:
    """Purge all stored vectors and delete all files in uploads directory."""
    from pathlib import Path
    from .vectorstore import clear_vector_store
    clear_vector_store()
    upload_dir = Path(__file__).resolve().parent.parent / "uploads"
    deleted = 0
    if upload_dir.exists():
        for file in upload_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                    deleted += 1
                except Exception:
                    pass
    return {"status": "success", "files_deleted": deleted}

