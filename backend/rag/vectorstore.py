from __future__ import annotations

import hashlib
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR
from .embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def _stable_id(doc: Document) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    start = str(doc.metadata.get("start_index", "0"))
    page = str(doc.metadata.get("page", ""))
    sheet = str(doc.metadata.get("sheet", ""))
    raw = "|".join([source, page, sheet, start, doc.page_content])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def delete_source(source: str) -> None:
    store = get_vector_store()
    # Chroma supports metadata filters through its underlying collection. We use
    # this only for source replacement, keeping normal app retrieval abstracted.
    try:
        result = store._collection.get(where={"source": source}, include=[])
        ids = result.get("ids", []) if result else []
        if ids:
            store.delete(ids=ids)
    except Exception:
        # Deletion failure must not destroy ingestion. The caller can still add
        # new chunks; source replacement is best-effort.
        pass


def clear_vector_store() -> None:
    store = get_vector_store()
    try:
        result = store._collection.get(include=[])
        ids = result.get("ids", []) if result else []
        if ids:
            store.delete(ids=ids)
    except Exception:
        pass



def add_documents(documents: list[Document]) -> int:
    if not documents:
        return 0
    store = get_vector_store()
    ids = [_stable_id(d) for d in documents]
    store.add_documents(documents=documents, ids=ids)
    return len(documents)
