from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import ENABLE_MULTI_QUERY, FETCH_K, MAX_CONTEXT_DOCS, OLLAMA_BASE_URL, RAG_LLM_MODEL, TOP_K
from .vectorstore import get_vector_store


@lru_cache(maxsize=1)
def get_query_llm():
    return ChatOllama(model=RAG_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def choose_strategy(query: str) -> str:
    """Choose the retrieval strategy based on query characteristics."""
    q = query.lower().strip()

    # Comparative / diverse queries → MMR for diversity
    if any(x in q for x in [
        "compare", "difference", "different", "alternatives", "pros and cons",
        "various", "all methods", "types of", "list all", "enumerate",
    ]):
        return "mmr"

    # Ambiguous / explanatory queries → multi-query if enabled
    if any(x in q for x in [
        "ambiguous", "another way", "different wording", "what does this mean",
        "why does", "how does", "explain", "what is",
    ]):
        return "multi_query" if ENABLE_MULTI_QUERY else "similarity"

    # Precise identifiers/numbers → direct similarity (exact match matters)
    if re.search(r"\b[A-Z]{2,}[\-_]?\d{2,}\b|\b\d{4,}\b", query):
        return "similarity"

    # Long, complex queries → multi-query for broader coverage
    if len(q.split()) > 18:
        return "multi_query" if ENABLE_MULTI_QUERY else "mmr"

    return "similarity"


def _base_retriever(strategy: str, top_k: int = TOP_K):
    store = get_vector_store()
    if strategy == "mmr":
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": FETCH_K, "lambda_mult": 0.7},
        )
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def _rerank_docs(docs: list[Document], query: str) -> list[Document]:
    """
    Simple keyword-overlap re-ranking: boost documents whose content
    contains more query terms. Stable sort (preserves retrieval order
    for ties) so retrieval ranking still dominates.
    """
    if not docs or not query:
        return docs

    query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
    if not query_terms:
        return docs

    def _score(doc: Document) -> float:
        text = doc.page_content.lower()
        hits = sum(1 for term in query_terms if term in text)
        return hits / len(query_terms)

    return sorted(docs, key=_score, reverse=True)


def retrieve(
    query: str,
    strategy: str | None = None,
    access_filter: dict | None = None,
    top_k: int | None = None,
) -> tuple[list[Document], str]:
    """Retrieve documents for a single query with optional re-ranking."""
    # Safely resolve top_k — handles None, str, or int inputs
    effective_top_k = int(top_k) if top_k is not None else TOP_K
    effective_top_k = max(1, min(effective_top_k, 30))

    strategy = strategy or choose_strategy(query)
    try:
        if strategy == "multi_query" and ENABLE_MULTI_QUERY:
            from langchain_community.retrievers import MultiQueryRetriever
            base = _base_retriever("mmr", effective_top_k)
            if access_filter:
                base = get_vector_store().as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": effective_top_k, "fetch_k": FETCH_K, "filter": access_filter},
                )
            retriever = MultiQueryRetriever.from_llm(retriever=base, llm=get_query_llm())
        else:
            retriever = _base_retriever(strategy, effective_top_k)
            if access_filter:
                retriever = get_vector_store().as_retriever(
                    search_type="mmr" if strategy == "mmr" else "similarity",
                    search_kwargs={
                        "k": effective_top_k,
                        **( {"fetch_k": FETCH_K} if strategy == "mmr" else {}),
                        "filter": access_filter,
                    },
                )

        docs = retriever.invoke(query)
        # Re-rank by keyword overlap for better precision
        docs = _rerank_docs(docs, query)
        return docs[:MAX_CONTEXT_DOCS], strategy

    except Exception:
        # Reliability rule: advanced retrieval must never make RAG unavailable.
        try:
            fallback = _base_retriever("similarity", effective_top_k)
            if access_filter:
                fallback = get_vector_store().as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": effective_top_k, "filter": access_filter},
                )
            raw = fallback.invoke(query)
            return _rerank_docs(raw, query)[:MAX_CONTEXT_DOCS], "similarity_fallback"
        except Exception:
            return [], "unavailable"


def retrieve_expanded(
    queries: list[str],
    strategy: str | None = None,
    access_filter: dict | None = None,
    top_k: int | None = None,
) -> tuple[list[Document], str]:
    """
    Retrieve documents for multiple query variants and merge results,
    deduplicating by MD5 hash of page_content.

    This is the backend for the query-expansion optimization loop in pipeline.py.
    It gives much higher recall than a single-query retrieval because the same
    information may be phrased differently across source documents.
    """
    # Safely resolve top_k
    effective_top_k = int(top_k) if top_k is not None else TOP_K
    effective_top_k = max(1, min(effective_top_k, 30))

    seen: set[str] = set()
    all_docs: list[Document] = []
    last_strategy = "similarity"
    # Use the primary query (first) for re-ranking
    primary_query = queries[0] if queries else ""

    for q in queries:
        try:
            docs, strat = retrieve(
                q,
                strategy=strategy,
                access_filter=access_filter,
                top_k=effective_top_k,
            )
            last_strategy = strat
            for doc in docs:
                key = hashlib.md5(doc.page_content.encode("utf-8", errors="replace")).hexdigest()
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
        except Exception:
            continue

    # Final re-rank of merged results against the primary query
    all_docs = _rerank_docs(all_docs, primary_query)
    return all_docs[:MAX_CONTEXT_DOCS], last_strategy or "unavailable"
