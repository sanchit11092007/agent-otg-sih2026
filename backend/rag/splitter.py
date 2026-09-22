from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, ENABLE_SEMANTIC_SPLITTER


def _recursive_for(metadata: dict) -> RecursiveCharacterTextSplitter:
    content_type = metadata.get("content_type", "text")
    file_type = metadata.get("file_type", "")

    # Code and markdown benefit from structure-aware separators; normal prose
    # uses the generic recommended recursive splitter.
    if file_type in {"py", "js", "ts", "java", "cpp", "c", "h", "sql"}:
        separators = ["\nclass ", "\ndef ", "\nfunction ", "\nSELECT ", "\n\n", "\n", " ", ""]
    elif file_type in {"md", "markdown"}:
        separators = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""]
    elif content_type == "table":
        # Keep rows intact as much as possible.
        separators = ["\n\n", "\n", " ", ""]
    else:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        add_start_index=True,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Content-aware splitting with a safe recursive fallback.

    Semantic splitting is intentionally opt-in because it requires more model
    calls and can make ingestion much slower. If enabled but unavailable, the
    system automatically falls back to recursive splitting.
    """
    out: list[Document] = []
    for doc in documents:
        if not doc.page_content.strip():
            continue
        splitter = None
        if ENABLE_SEMANTIC_SPLITTER:
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from .embeddings import get_embeddings
                splitter = SemanticChunker(get_embeddings(), breakpoint_threshold_type="percentile")
            except Exception:
                splitter = None
        if splitter is None:
            splitter = _recursive_for(doc.metadata)
        chunks = splitter.split_documents([doc])
        for index, chunk in enumerate(chunks):
            chunk.metadata = {
                **doc.metadata,
                **chunk.metadata,
                "chunk_index": index,
            }
        out.extend(chunks)
    return out
