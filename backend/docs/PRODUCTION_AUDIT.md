# Production Audit

## Architecture

```text
Client / terminal -> FastAPI -> LangGraph supervisor -> planner -> local RAG (optional)
                                      |                         |
                                      v                         v
                               local Ollama content        artifact service
                                      |                         |
                                      +------ validation -------+
                                                   |
                                          generated_files + download API

Conversation and project memory -> SQLite (WAL mode)
RAG documents -> local Chroma + Ollama embeddings
```

## Findings and Remediation

| Finding | Impact | Resolution |
| --- | --- | --- |
| PostgreSQL was a mandatory session dependency | Offline runs lost session persistence without another service | Replaced with local SQLite/WAL persistence |
| File generation depended on model-selected tools | A request could produce text and stop before generating the requested file | Added deterministic file planning, creation, and validation nodes |
| No generic artifact contract | PDF and DOCX were the only output types | Added validated PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, and JSON generation |
| RAG changed global `TOP_K` during a request | Concurrent requests could affect each other | Retrieval now accepts request-local `top_k` |
| Query expansion was enabled by default | Every RAG request incurred avoidable model calls | Advanced multi-query behavior is now opt-in |
| Generated files had no download endpoint | API clients could not retrieve finished artifacts directly | Added `GET /files/{filename}` and `download_url` in agent output |

## Verification Report

- `py -m compileall -q .`: passed.
- `py -m pytest -q`: 4 passed.
- Validated outputs: TXT, Markdown, JSON, CSV, XLSX, DOCX, PDF, and PPTX.
- Integration check: `POST /ask/agent` creates a PDF and returns a local download URL.

## Demo Prompts

1. `Create a PDF explaining the last non-zero digit of a factorial, including Python code.`
2. `Create a DOCX project proposal for an offline emergency-response assistant.`
3. `Create an XLSX CSV-style budget table for a hackathon team.`
4. `Create a PPTX about Smart India Hackathon problem discovery.`
5. `Create a JSON file describing an agent workflow with supervisor, planner, and validator roles.`
6. `Ingest ./sample_policy.pdf, then answer what the document says about refunds.`
7. `Compare the two uploaded policy documents and cite the supporting sources.`
8. `Create a Markdown incident report using the uploaded knowledge-base documents.`
9. `Generate a CSV containing test cases for RAG retrieval failures.`
10. `Create a TXT summary of this conversation for project memory.`
