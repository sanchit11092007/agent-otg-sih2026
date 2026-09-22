# 🤖 Agent OTG — Local Autonomous Multi-Model AI Hub

**Agent OTG** is an air-gapped, on-premise, multi-model AI routing system that connects FastAPI, LangGraph, and Ollama. It automatically analyzes user intent, routes tasks to specialized local models, orchestrates multi-step pipelines (such as generating code and immediately explaining it), executes tools (Word, PDF manipulation, RAG document search), and provides full real-time terminal observability.

---

## ⚡ Quick Start & Running Procedure

> [!IMPORTANT]
> **Windows Python Launcher Required**:
> On Windows, your default `python` command may point to an environment without packages (such as MSYS2/MinGW).
> **Always use `py`** to run scripts and servers on your machine.

### Step 1: Verify Ollama & Models
Ensure the Ollama service is active. Check that the required models are downloaded:
```powershell
ollama list
```
The recommended models for full functionality:
- `qwen2.5:7b` (Router, general reasoning & LangGraph agent)
- `qwen2.5-coder:latest` (Code generation & debugging)
- `qwen2.5vl:7b` (Vision & image analysis)

If any model is missing, pull it using:
```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:latest
ollama pull qwen2.5vl:7b
```

---

### Step 2: Run the FastAPI / Uvicorn Server (Backend)
Open a terminal in the project directory:
```powershell
py -m uvicorn main:app --reload --port 8000
```
When running, the server terminal provides full real-time visibility with structured, emoji-rich logs:
- `📥 [USER ACTION]`: Incoming endpoint, client IP, query, attachments
- `🧠 [ROUTER]`: Intent classification, category, selected model, pipeline decision
- `🤖 [MODEL]`: Model execution start, prompt details, response generation
- `🛠️ [TOOL]`: Tool invocation, input parameters, execution timing, results
- `📤 [RESPONSE]`: Model output preview, token count, execution latency
- `🌊 [STREAM]`: Live chunk delivery and streaming stages

---

### Step 3: Run the Interactive Terminal Client
Open another terminal window and start the interactive client:
```powershell
py ask.py
```
This opens an interactive shell with an automated capability showcase, command matrix, styled prompts, and colored stage badges.

---

## 🚀 Terminal Interface Command Reference (`ask.py`)

| Capability | Command / Input Example | Assigned Model | Description |
| :--- | :--- | :--- | :--- |
| **💻 Code Writing & Debugging** | `Write a python script to parse CSV files` | `qwen2.5-coder:latest` | Auto-detects programming tasks and routes to the coder model |
| **🔄 Code + Walkthrough** | `Write a quicksort in Python and explain how it works` | `qwen2.5-coder` ➔ `qwen2.5:7b` | **Sequential Pipeline**: generates code first, then passes it to the reasoning model for explanation |
| **🧠 Deep Reasoning & Essays** | `Explain the trade-offs of microservices vs monoliths` | `qwen2.5:7b` | Complex analysis, architectural reasoning, essays |
| **⚡ Quick Answers & Math** | `What is the capital of France?` or `25 * 14` | `qwen2.5:7b` | Rapid lightweight answers |
| **🗂️ Force-Split Complex Tasks** | `/complex Write an intro email AND generate 3 SQL queries` | Multi-Model Split | Decomposes prompt into independent sub-tasks |
| **👁️ Vision & Image Q&A** | `/image "C:\path\diagram.png" Explain this flowchart` | `qwen2.5vl:7b` | Accepts local image files or public URLs (PNG, JPEG, WEBP) |
| **📄 Document Ingestion** | `upload "C:\data\sales.xlsx"` then ask `What are the total sales?` | Context Injector | Ingests `.xlsx`, `.pdf`, or `.docx` for subsequent Q&A |
| **🛠️ LangGraph Agent** | `/agent Create a Word document titled 'Report' with key points` | `qwen2.5:7b` + Tools | Multi-step agent that can create DOCX, manipulate PDFs, and search RAG docs |
| **📜 Past Sessions** | `history` | Session Store | Interactive browser for saved conversation transcripts |
| **🧹 Clear Memory** | `reset` | Memory Guard | Clears conversation context to start a fresh topic |
| **❓ Help Menu** | `help` or `/help` | Guide | Displays the full capability matrix and practical prompt examples |
| **🚪 Exit** | `exit` or `quit` | — | Closes the terminal client |

---

## 🌐 FastAPI REST Endpoints (`main.py`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service status, active models, and registered endpoints |
| `GET` | `/health` | Live health probe verifying connection to Ollama |
| `POST` | `/ask` | Smart routed Q&A (single model, sequential pipeline, or multi-part) |
| `POST` | `/ask/stream` | Real-time NDJSON stream with stage markers and token chunks |
| `POST` | `/ask/complex` | Explicit multi-task split into sub-tasks |
| `POST` | `/ask/image` | Vision endpoint accepting raw base64 image + prompt |
| `POST` | `/ask/agent` | LangGraph agent endpoint with structured tool calling |
| `POST` | `/reset` | Reset conversational memory |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `GET` | `/capabilities` | Offline runtime and optional tool availability |
| `POST` | `/knowledge-base/ingest` | Index local document paths into Chroma |
| `GET` | `/files/{filename}` | Download a generated artifact |

---

## 🛡️ Privacy & Sovereign Security

- **Software Offline Guard (`offline_guard.py`)**: Intercepts outbound socket network calls to ensure no prompts or documents leak to external networks.
- **Local Persistence**: Session and project memory are saved in local SQLite; generated documents are saved under `generated_files/`.
"# rag-pipeline" 
