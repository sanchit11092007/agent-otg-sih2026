from functools import lru_cache
from langchain_ollama import OllamaEmbeddings
from .config import OLLAMA_BASE_URL, RAG_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings():
    return OllamaEmbeddings(
        model=RAG_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
