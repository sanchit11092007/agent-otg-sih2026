# Final RAG integration

## What changed

- `tools.py` is the only existing Python file that must be replaced.
- The old direct `chromadb.PersistentClient(...).get_or_create_collection("knowledge_base")` code is removed.
- RAG is lazy-loaded, so the FastAPI application can still start for normal/non-RAG requests if RAG dependencies are missing.
- New RAG uses `./chroma_db` with collection `knowledge_base_v2` by default.
- `router.py` does not need to be replaced. Its existing `TOOL_FUNCTIONS`/`TOOL_SCHEMAS` imports continue to work.
- No PostgreSQL and no RBAC are added.

## Project layout

```text
project/
├── main.py
├── router.py
├── langgraph_agent.py
├── tools.py                 # REPLACE with this package version
├── file_readers.py
├── ask.py
├── offline_guard.py
├── system_prompt.txt
├── requirements.txt         # ADD requirements-rag.txt entries
├── rag/                     # COPY this entire folder
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── loaders.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── retriever.py
│   ├── splitter.py
│   └── vectorstore.py
├── chroma_db/               # KEEP for now
└── chat_sessions/           # KEEP for now
```

## Old Chroma DB

Do not delete `chroma_db/` before testing. The new RAG uses a separate collection named `knowledge_base_v2`, so the old `knowledge_base` collection can remain untouched.

After the new RAG is verified, the old collection/data can be removed during cleanup. You do not need to delete the whole `chroma_db/` directory while testing.

## Install

Create/activate your virtual environment, then install the project's existing requirements plus `requirements-rag.txt`.

Also make sure Ollama is running and pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Your existing Qwen models should remain available:

```text
qwen2.5-coder:latest
qwen2.5:7b
qwen2.5vl:7b
```

## Test order

### 1. Test that the application still starts

```bash
uvicorn main:app --reload
```

If the server starts, the non-RAG application path is intact.

### 2. Test a normal chat request

Use your existing `ask.py` or existing frontend/client. This confirms that `router.py`, model routing, history and the tool registry still work.

### 3. Put a test document in the project

For example:

```text
test_data/company_policy.txt
```

Put a distinctive sentence in it, e.g.:

```text
The SIH prototype knowledge base test policy says the support window is 10:30 AM to 4:30 PM on working days.
```

### 4. Ingest it

From Python:

```python
from rag.pipeline import ingest_path

print(ingest_path("test_data/company_policy.txt"))
```

You should get counts for files, documents loaded and chunks indexed.

### 5. Query it

```python
from rag.pipeline import answer

result = answer("What is the support window in the test policy?")
print(result["answer"])
print(result["retrieval_strategy"])
print(result["sources"])
```

The answer should be grounded in the sentence you inserted, and the source should identify the test file.

### 6. Test strategy selection

```python
from rag.retriever import choose_strategy

print(choose_strategy("What is the support window?"))
print(choose_strategy("Compare the two support policies and their differences"))
print(choose_strategy("What does this policy mean and how does it apply?"))
```

Expected general behavior:

- precise/factual -> similarity
- comparison/diversity -> MMR
- ambiguous/long/paraphrasable -> MultiQuery when enabled

Advanced retrieval has a similarity fallback so a failure in MMR/MultiQuery should not make RAG completely unavailable.

## Important: direct file ingestion

The new `ingest_file()` tool supports PDF, DOCX, TXT, Markdown, CSV, XLS/XLSX, JSON and supported images. It normalizes files into LangChain `Document` objects, then splits and indexes them.

## If Ollama/RAG is unavailable

The replacement `tools.py` catches RAG errors. Therefore the backend can still start and non-RAG features can continue to work. A RAG call returns a clear availability/error message instead of crashing the whole FastAPI process.

## Current history and router

`router.py` remains in the project root. Its existing in-memory history is still the current chat history mechanism. PostgreSQL is intentionally not part of this version. It can be introduced later for persistent users/conversations/messages and RBAC.
