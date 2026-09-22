"""
main.py - The FastAPI Server for Agent OTG

Endpoints:
    GET  /            -> Status page
    GET  /health      -> Check if Ollama is running
    POST /ask         -> Ask a question, auto-detects single vs multi-part vs sequential
    POST /ask/stream  -> Same, but streamed with visible stages and token chunks
    POST /ask/complex -> Force-split a question into sub-tasks
    POST /ask/image   -> Ask a question about an image (base64)
    POST /ask/agent   -> LangGraph agent: structured tool-calling graph
    POST /reset       -> Clear conversation memory

Run with:
    py -m uvicorn main:app --reload
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import json
import time
import datetime
import importlib.util
import threading
from typing import Optional
from urllib.parse import quote
from contextlib import asynccontextmanager

from offline_guard import enable_offline_mode
enable_offline_mode()

import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config as _cfg
import db
import router
from router import (
    classify_question,
    stream_answer,
    get_full_answer,
    break_into_tasks,
    run_sequential_tasks,
    clear_history,
    ask_image,
    get_history,
    AVAILABLE_MODELS,
    CODER_MODEL,
    MAIN_MODEL,
    is_realtime_query,
    OFFLINE_REALTIME_MESSAGE,
)
from sync_hub import get_local_ip_addresses, hub


# ══════════════════════════════════════════════════════════════════════════════
# SERVER LOGGING SYSTEM
# Provides clear, emoji-rich, structured console logging on the Uvicorn server
# ══════════════════════════════════════════════════════════════════════════════

def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _safe_print(msg: str):
    try:
        print(msg, flush=True)
    except (UnicodeEncodeError, Exception):
        try:
            print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"), flush=True)
        except Exception:
            print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)

def log_user(msg: str):
    _safe_print(f"\033[94m[{_now_str()}] 📥 [USER ACTION] {msg}\033[0m")

def log_router(msg: str):
    _safe_print(f"\033[95m[{_now_str()}] 🧠 [ROUTER] {msg}\033[0m")

def log_model(msg: str):
    _safe_print(f"\033[96m[{_now_str()}] 🤖 [MODEL] {msg}\033[0m")

def log_tool(msg: str):
    _safe_print(f"\033[93m[{_now_str()}] 🛠️ [TOOL] {msg}\033[0m")

def log_response(msg: str):
    _safe_print(f"\033[92m[{_now_str()}] 📤 [RESPONSE] {msg}\033[0m")

def log_stream(msg: str):
    _safe_print(f"\033[90m[{_now_str()}] 🌊 [STREAM] {msg}\033[0m")

def log_system(msg: str):
    _safe_print(f"\033[97m[{_now_str()}] ⚙️ [SYSTEM] {msg}\033[0m")

def log_err(msg: str):
    _safe_print(f"\033[91m[{_now_str()}] ❌ [ERROR] {msg}\033[0m")


# Hook router.py's internal events into our server logger
def _router_event_listener(event_type: str, data: dict):
    if event_type == "classify":
        res = data.get("result", {})
        log_router(
            f'Classified query="{data.get("query", "")[:60]}" -> '
            f'category={res.get("category")} | model={res.get("model")} | '
            f'is_multi={res.get("is_multi_part")} | has_code_explain={res.get("has_code_and_explain")}'
        )
    elif event_type == "model_start":
        log_model(f'Invoking model={data.get("model")} | query="{data.get("query", "")[:70]}"')
    elif event_type == "model_done":
        ans_preview = data.get("answer", "").replace("\n", " ")[:100]
        log_model(f'Model {data.get("model")} completed answer -> "{ans_preview}..."')
    elif event_type == "stream_start":
        log_stream(f'Streaming started for model={data.get("model")} | query="{data.get("query", "")[:70]}"')
    elif event_type == "stream_done":
        ans_preview = data.get("answer", "").replace("\n", " ")[:100]
        log_stream(f'Streaming finished for model={data.get("model")} | final="{ans_preview}..."')
    elif event_type == "tool_call":
        log_tool(f'Tool "{data.get("tool")}" called with args={data.get("args")} -> result={data.get("result")[:120]}')
    elif event_type == "image_start":
        log_model(f'Vision model={data.get("model")} analyzing image (b64 size: {data.get("image_size_b64", 0)} chars) | query="{data.get("query", "")}"')
    elif event_type == "image_done":
        ans_preview = data.get("answer", "").replace("\n", " ")[:100]
        log_model(f'Vision model completed -> "{ans_preview}..."')

router.set_log_callback(_router_event_listener)


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN & APP INIT
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Initialise local persistence ───────────────────────────────────────
    db_ok = db.init_db()
    if db_ok:
        log_system("SQLite session store: connected OK")
    else:
        log_err(f"SQLite session store: UNAVAILABLE — {db.get_error()}")
        log_system("Sessions will not be persisted.")

    # ── Session name ────────────────────────────────────────────────────
    session_name = datetime.datetime.now().strftime("server_session_%Y-%m-%d_%H-%M-%S")
    router.start_new_session(session_name)
    log_system("Agent OTG FastAPI Server Started")
    log_system(f"Active Session: {session_name}")
    log_system(f"Configured Models: {AVAILABLE_MODELS}")
    yield
    log_system("Agent OTG FastAPI Server Stopping...")

app = FastAPI(
    title="Agent OTG",
    description="Routes questions to the best local AI model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    # Log incoming request summary for non-health endpoints to avoid noise
    if path != "/health":
        log_user(f"Incoming {method} {path} from {client_host}")

    try:
        response = await call_next(request)
        elapsed = round(time.time() - start_time, 3)
        if path != "/health":
            log_response(f"Completed {method} {path} with HTTP {response.status_code} in {elapsed}s")
        return response
    except Exception as exc:
        elapsed = round(time.time() - start_time, 3)
        log_err(f"Failed {method} {path} with exception: {exc} (after {elapsed}s)")
        raise


class Question(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    sync_id: Optional[str] = Field(None, description="Optional custom or pre-allocated sync ID")


class RemoteSubmitRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    sync_id: Optional[str] = Field(None, description="Optional custom or pre-allocated sync ID")
    mode: Optional[str] = Field("agent", description="Execution mode: agent or chief")


class ImageQuestion(BaseModel):
    query: str  = Field(..., min_length=1, description="The question about the image")
    image_b64: str = Field(
        ...,
        description=(
            "Raw base64-encoded image (PNG or JPEG). "
            "Do NOT include a data-URI prefix like 'data:image/png;base64,'. "
            "Just the plain base64 string."
        ),
    )


class IngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1, max_length=50)
    replace_existing: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def home():
    import config as _cfg
    log_system("Client requested server status page /")
    return {
        "agent":       _cfg.APP_NAME,
        "version":     _cfg.APP_VERSION,
        "made_by":     _cfg.APP_TEAM,
        "description": "100% On-Premise • Air-Gapped • No cloud • No tracking",
        "status":      "online",
        "models":      AVAILABLE_MODELS,
        "endpoints": [
            "/ask", "/ask/stream", "/ask/complex",
            "/ask/image", "/ask/agent",
            "/reset", "/health", "/tools",
            "/sessions", "/knowledge-base/ingest", "/files/{filename}", "/docs",
            "/network/info", "/sync/submit", "/sync/content/{sync_id}",
            "/sync/stream/{sync_id}", "/sync/recent", "/sync/latest",
        ],
    }


@app.get("/health")
def health():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            return {"ollama": "error", "models": [], "missing_models": []}
        payload = r.json()
        installed = {
            str(item.get("name", "")).lower()
            for item in payload.get("models", []) if isinstance(item, dict)
        }
        required = {_cfg.MAIN_MODEL, _cfg.CODER_MODEL, _cfg.RAG_EMBEDDING_MODEL}
        # Ollama reports the default tag explicitly (for example
        # ``nomic-embed-text:latest``), while configuration commonly omits it.
        installed_bases = {name.split(":", 1)[0] for name in installed}
        missing = sorted(
            model for model in required
            if model.lower() not in installed and model.lower().split(":", 1)[0] not in installed_bases
        )
        return {
            "ollama": "connected" if not missing else "degraded",
            "models": sorted(installed),
            "missing_models": missing,
        }
    except Exception as exc:
        log_err(f"Health check failed to reach Ollama: {exc}")
        return {"ollama": "disconnected"}


@app.get("/capabilities")
def capabilities():
    """Report available offline runtimes without contacting any cloud service."""
    optional = {name: importlib.util.find_spec(name) is not None for name in ("pytesseract", "pyttsx3", "speech_recognition")}
    return {
        "offline": True,
        "artifact_formats": ["pdf", "docx", "txt", "md", "pptx", "xlsx", "csv", "json"],
        "optional": optional,
    }


@app.get("/knowledge-base/status")
def knowledge_base_status():
    """Report whether the RAG store is reachable and how many chunks are indexed."""
    try:
        from rag.pipeline import count, list_sources
        return {
            "status": "ready",
            "chunks_indexed": count(),
            "sources": list_sources(),
        }
    except Exception as exc:
        return {"status": "unavailable", "detail": str(exc), "chunks_indexed": 0, "sources": []}


@app.post("/knowledge-base/ingest")
def ingest_knowledge_base(request: IngestRequest):
    try:
        from rag.pipeline import ingest_paths
        return ingest_paths(request.paths, replace_existing=request.replace_existing)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Local file not found: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge base unavailable: {exc}")


@app.post("/knowledge-base/upload")
async def upload_knowledge_base(files: list[UploadFile] = File(...)):
    try:
        from rag.pipeline import ingest_path
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        indexed_count = 0
        details = []
        for file in files:
            safe_filename = os.path.basename(file.filename or "uploaded_file")
            target_path = os.path.join(upload_dir, safe_filename)
            content = await file.read()
            with open(target_path, "wb") as f:
                f.write(content)
            res = ingest_path(target_path, replace_existing=True)
            indexed_count += res.get("chunks_indexed", res.get("chunks_generated", 1))
            details.append({"filename": safe_filename, "chunks": res.get("chunks_indexed", 1)})
        return {"status": "success", "files_indexed": len(files), "chunks_indexed": indexed_count, "details": details}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to upload and index files: {exc}")


@app.post("/knowledge-base/clear")
@app.delete("/knowledge-base/clear")
def clear_knowledge_base_endpoint():
    try:
        from rag.pipeline import clear_knowledge_base
        return clear_knowledge_base()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {exc}")


@app.post("/reset")
def reset():
    log_user("Requested conversation memory reset")
    clear_history()
    log_system("Conversation history cleared from memory")
    return {"status": "conversation memory cleared"}


@app.post("/ask")
def ask(q: Question):
    sync_item = hub.create_item(query=q.query, sync_id=q.sync_id)
    try:
        start  = time.time()
        log_user(f'Query [Sync #{sync_item.sync_id}]: "{q.query}"')
        if is_realtime_query(q.query):
            ans = OFFLINE_REALTIME_MESSAGE
            sync_item.append_stage({"stage": "availability", "detail": "Live information is unavailable in local mode."})
            sync_item.complete({"category": "offline_decline", "time_seconds": round(time.time() - start, 2)})
            return {
                "type": "offline_decline",
                "sync_id": sync_item.sync_id,
                "stages": [{"stage": "availability", "detail": "Live information is unavailable in local mode."}],
                "category": "offline_decline",
                "time_seconds": round(time.time() - start, 2),
                "answer": ans,
            }
        info   = classify_question(q.query)
        stages = [{"stage": "understanding", "detail": f'category = "{info["category"]}"'}]
        sync_item.append_stage(stages[0])

        # ── Path A: Code + Explain (sequential, ordered pipeline) ──────
        if info["has_code_and_explain"]:
            log_router(f'Detected Code + Explain combo request. Executing 2-step sequential pipeline...')
            p_stage = {
                "stage":  "planning",
                "detail": "Code + explain request — running sequential pipeline...",
            }
            stages.append(p_stage)
            sync_item.append_stage(p_stage)
            results = run_sequential_tasks(q.query)

            combined_ans = ""
            for r in results:
                w_stage = {
                    "stage":  "working",
                    "detail": f"Step {r['step']}: {r['label']} → {r['model_used']}",
                }
                stages.append(w_stage)
                sync_item.append_stage(w_stage)
                combined_ans += f"\n\n### Step {r['step']}: {r['label']}\n{r['answer']}"

            elapsed = round(time.time() - start, 2)
            log_response(f"Sequential pipeline completed in {elapsed}s across {len(results)} steps")
            sync_item.append_content(combined_ans.strip())
            sync_item.complete({"time_seconds": elapsed, "category": "sequential"})
            return {
                "type":        "sequential",
                "sync_id":     sync_item.sync_id,
                "description": (
                    "Code was generated first by the coder model, then passed "
                    "to the main model via shared history for the explanation."
                ),
                "stages":       stages,
                "steps_run":    len(results),
                "time_seconds": elapsed,
                "results": [
                    {
                        "step":       r["step"],
                        "label":      r["label"],
                        "model_used": r["model_used"],
                        "category":   r["category"],
                        "answer":     r["answer"],
                    }
                    for r in results
                ],
            }

        # ── Path B: Agent task (document creation / tool workflow) ─────
        if info["category"] == "agent_task":
            log_router("Detected Agent Task request. Executing LangGraph agent workflow...")
            r_stage = {"stage": "routing", "detail": "Autonomous agent artifact creation"}
            stages.append(r_stage)
            sync_item.append_stage(r_stage)
            from langgraph_agent import run_agent
            from urllib.parse import quote
            result = run_agent(q.query, history=get_history())
            artifact = result.get("artifact")
            if artifact and "filename" in artifact:
                artifact = {**artifact, "download_url": f"/files/{quote(artifact['filename'])}"}
            elapsed = round(time.time() - start, 2)
            log_response(f"Agent task completed in {elapsed}s | artifact={artifact.get('filename') if artifact else None}")
            final_ans = result.get("final_answer", "")
            sync_item.append_content(final_ans)
            sync_item.complete({
                "model_used": info["model"],
                "category": "agent_task",
                "time_seconds": elapsed,
                "artifact": artifact,
            })
            return {
                "type":         "agent",
                "sync_id":      sync_item.sync_id,
                "stages":       stages,
                "model_used":   info["model"],
                "category":     "agent_task",
                "reason":       info["reason"],
                "time_seconds": elapsed,
                "answer":       final_ans,
                "artifact":     artifact,
                "tool_log":     result.get("tool_log", []),
                "needs_tool":   result.get("needs_tool", False),
            }

        # ── Path C: RAG search (knowledge base retrieval) ─────────────
        if info["category"] == "rag_search":
            log_router("Detected RAG search query. Searching knowledge base...")
            r_stage = {"stage": "routing", "detail": "RAG knowledge base retrieval"}
            stages.append(r_stage)
            sync_item.append_stage(r_stage)
            from rag.pipeline import answer as rag_answer
            res = rag_answer(q.query)
            elapsed = round(time.time() - start, 2)
            log_response(f"RAG search completed in {elapsed}s | sources={len(res.get('sources', []))}")
            final_ans = res.get("answer", "")
            sync_item.append_content(final_ans)
            sync_item.complete({
                "model_used": info["model"],
                "category": "rag_search",
                "time_seconds": elapsed,
                "sources": res.get("sources", []),
            })
            return {
                "type":               "rag_search",
                "sync_id":            sync_item.sync_id,
                "stages":             stages,
                "model_used":         info["model"],
                "category":           "rag_search",
                "reason":             info["reason"],
                "time_seconds":       elapsed,
                "answer":             final_ans,
                "sources":            res.get("sources", []),
                "retrieval_strategy": res.get("retrieval_strategy", "similarity"),
            }

        # ── Path D: Generic multi-part (independent sub-tasks) ─────────
        if info["is_multi_part"]:
            log_router("Detected multi-part question. Splitting into independent sub-tasks...")
            p_stage = {"stage": "planning", "detail": "Multi-part request detected, splitting..."}
            stages.append(p_stage)
            sync_item.append_stage(p_stage)
            tasks   = break_into_tasks(q.query)
            results = []
            combined_ans = ""
            for i, task in enumerate(tasks):
                w_stage = {
                    "stage":  "working",
                    "detail": f"Sub-task {i+1}: {task['label']} → {task['model']}",
                }
                stages.append(w_stage)
                sync_item.append_stage(w_stage)
                log_model(f"Running sub-task {i+1}/{len(tasks)}: '{task['label']}' on model {task['model']}")
                answer = get_full_answer(task["model"], task["task"])
                combined_ans += f"\n\n### Sub-task {i+1}: {task['label']}\n{answer}"
                results.append({
                    "task_number": i + 1,
                    "label":       task["label"],
                    "model_used":  task["model"],
                    "category":    task["category"],
                    "answer":      answer,
                })

            elapsed = round(time.time() - start, 2)
            log_response(f"Multi-part execution completed in {elapsed}s across {len(tasks)} sub-tasks")
            sync_item.append_content(combined_ans.strip())
            sync_item.complete({"time_seconds": elapsed, "category": "multi"})
            return {
                "type":         "multi",
                "sync_id":      sync_item.sync_id,
                "stages":       stages,
                "sub_tasks":    len(tasks),
                "time_seconds": elapsed,
                "results":      results,
            }

        # ── Path C: Single model execution ────────────────────────────
        r_stage = {"stage": "routing", "detail": f'{info["model"]} - {info["reason"]}'}
        stages.append(r_stage)
        sync_item.append_stage(r_stage)
        log_router(f'Routing to single model {info["model"]} (Category: {info["category"]})')
        answer = get_full_answer(info["model"], q.query)
        stages.append({"stage": "done", "detail": "answer generated"})
        sync_item.append_stage({"stage": "done", "detail": "answer generated"})

        elapsed = round(time.time() - start, 2)
        log_response(f'Single task completed in {elapsed}s by {info["model"]}')
        sync_item.append_content(answer)
        sync_item.complete({
            "model_used": info["model"],
            "category": info["category"],
            "time_seconds": elapsed,
        })

        return {
            "type": "single",
            "sync_id": sync_item.sync_id,
            "stages": stages,
            "model_used": info["model"],
            "category": info["category"],
            "reason": info["reason"],
            "time_seconds": elapsed,
            "answer": answer,
        }

    except Exception as e:
        log_err(f"Exception in /ask: {e}")
        sync_item.fail(str(e))
        raise HTTPException(status_code=500, detail=str(e))


def _long_task_with_keepalives(work_fn, status_detail: str, interval: float = 15.0):
    """Run a blocking workflow in a thread and yield keepalive stage events."""
    box: dict = {"result": None, "error": None}
    done = threading.Event()

    def _worker():
        try:
            box["result"] = work_fn()
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()
    ticks = 0
    while not done.wait(interval):
        ticks += 1
        yield json.dumps({
            "type": "stage",
            "label": "working",
            "detail": f"{status_detail} ({int(ticks * interval)}s elapsed)",
        }) + "\n"

    if box["error"]:
        raise box["error"]
    return box["result"]


@app.post("/ask/stream")
def ask_stream(q: Question):
    sync_item = hub.create_item(query=q.query, sync_id=q.sync_id)
    log_user(f'Stream Request [Sync #{sync_item.sync_id}]: "{q.query}"')

    def generate():
        start = time.time()

        def emit(event_dict):
            event_type = event_dict.get("type")
            if event_type == "stage":
                sync_item.append_stage(event_dict)
            elif event_type == "token":
                sync_item.append_content(str(event_dict.get("content", "")))
            elif event_type == "done":
                sync_item.complete(event_dict)
            elif event_type == "error":
                sync_item.fail(event_dict.get("detail", "Error"))
            return json.dumps(event_dict) + "\n"

        try:
            # First notify receiver of assigned Sync ID
            yield json.dumps({"type": "sync_info", "sync_id": sync_item.sync_id}) + "\n"

            if is_realtime_query(q.query):
                yield emit({"type": "stage", "label": "availability", "detail": "Live information is unavailable in local mode."})
                yield emit({"type": "token", "content": OFFLINE_REALTIME_MESSAGE})
                yield emit({"type": "done", "category": "offline_decline", "time_seconds": round(time.time() - start, 2)})
                return
            yield emit({"type": "stage", "label": "understanding", "detail": "Reading your question..."})
            info = classify_question(q.query)

            # ── Path A: Code + Explain sequential streaming ───────────
            if info["has_code_and_explain"]:
                log_router("Stream request: executing Code + Explain sequential pipeline")
                yield emit({
                    "type": "stage",
                    "label": "planning",
                    "detail": "Code + explain combo detected — preparing 2-stage sequential generation...",
                })

                # Step 1: Code generation
                yield emit({
                    "type": "stage",
                    "label": "working",
                    "detail": f"Step 1/2: Generating code with {CODER_MODEL}...",
                })
                code_prompt = (
                    f"The user asked: \"{q.query}\"\n\n"
                    "Your job for this step: write ONLY the code. "
                    "Do not explain it yet — just provide clean, well-commented code."
                )
                for token in stream_answer(CODER_MODEL, code_prompt):
                    yield emit({"type": "token", "step": 1, "content": token})
                yield emit({"type": "step_done", "step": 1})

                # Step 2: Explanation
                yield emit({
                    "type": "stage",
                    "label": "working",
                    "detail": f"Step 2/2: Writing explanation with {MAIN_MODEL}...",
                })
                explain_prompt = (
                    "Now explain the code you just wrote above, step by step. "
                    "Be clear and beginner-friendly. Cover what each part does and why."
                )
                for token in stream_answer(MAIN_MODEL, explain_prompt):
                    yield emit({"type": "token", "step": 2, "content": token})
                yield emit({"type": "step_done", "step": 2})

                elapsed = round(time.time() - start, 2)
                log_stream(f"Sequential streaming finished in {elapsed}s")
                yield emit({"type": "done", "time_seconds": elapsed, "category": "sequential"})

            # ── Path B: Agent task streaming ─────────────────────────
            elif info["category"] == "agent_task":
                log_router("Stream request: routing to LangGraph agent workflow")
                yield emit({
                    "type": "stage",
                    "label": "planning",
                    "detail": "File creation request — running LangGraph autonomous agent...",
                    "category": "agent_task",
                })
                from langgraph_agent import run_agent
                from urllib.parse import quote
                agent_gen = _long_task_with_keepalives(
                    lambda: run_agent(q.query, history=get_history()),
                    "Generating document with the local agent",
                )
                try:
                    while True:
                        raw_event = next(agent_gen)
                        try:
                            parsed = json.loads(raw_event.strip())
                            emit(parsed)
                        except Exception:
                            pass
                        yield raw_event
                except StopIteration as stopped:
                    agent_result = stopped.value
                artifact = agent_result.get("artifact")
                if artifact and "filename" in artifact:
                    artifact = {**artifact, "download_url": f"/files/{quote(artifact['filename'])}"}
                answer = agent_result.get("final_answer", "")
                if not answer and artifact:
                    answer = f"Generated {artifact.get('filename')} successfully."
                yield emit({"type": "stage", "label": "generating", "detail": "Delivering agent output..."})
                for word in answer.split(" "):
                    yield emit({"type": "token", "content": word + " "})
                elapsed = round(time.time() - start, 2)
                yield emit({
                    "type": "done",
                    "category": "agent_task",
                    "artifact": artifact,
                    "tool_log": agent_result.get("tool_log", []),
                    "time_seconds": elapsed,
                })

            # ── Path C: RAG search streaming ─────────────────────────
            elif info["category"] == "rag_search":
                log_router("Stream request: routing to RAG knowledge base search")
                yield emit({
                    "type": "stage",
                    "label": "planning",
                    "detail": "Retrieving context from document knowledge base...",
                    "category": "rag_search",
                })
                from rag.pipeline import answer as rag_answer
                rag_gen = _long_task_with_keepalives(
                    lambda: rag_answer(q.query),
                    "Searching knowledge base and synthesizing answer",
                )
                try:
                    while True:
                        raw_event = next(rag_gen)
                        try:
                            parsed = json.loads(raw_event.strip())
                            emit(parsed)
                        except Exception:
                            pass
                        yield raw_event
                except StopIteration as stopped:
                    rag_res = stopped.value
                answer = rag_res.get("answer", "")
                sources = rag_res.get("sources", [])
                yield emit({"type": "stage", "label": "generating", "detail": "Formulating answer from context..."})
                for word in answer.split(" "):
                    yield emit({"type": "token", "content": word + " "})
                elapsed = round(time.time() - start, 2)
                yield emit({
                    "type": "done",
                    "category": "rag_search",
                    "sources": sources,
                    "strategy": rag_res.get("retrieval_strategy", "similarity"),
                    "time_seconds": elapsed,
                })

            # ── Path D: Generic multi-part streaming ──────────────────
            elif info["is_multi_part"]:
                log_router("Stream request: breaking down multi-part sub-tasks")
                yield emit({"type": "stage", "label": "planning", "detail": "Multi-part request detected, breaking it down into sub-tasks..."})
                tasks = break_into_tasks(q.query)
                yield emit({
                    "type": "plan",
                    "sub_tasks": [{"label": t["label"], "model": t["model"], "category": t["category"]} for t in tasks],
                })

                for i, task in enumerate(tasks, 1):
                    yield emit({
                        "type": "stage",
                        "label": "working",
                        "detail": f"Sub-task {i}/{len(tasks)}: {task['label']} → {task['model']} ({task['category']})",
                    })

                    header_prefix = "\n\n" if i > 1 else ""
                    sub_header = f"{header_prefix}### Sub-task {i}: {task['label']}\n*Category: `{task['category']}` • Model: `{task['model']}`*\n\n"
                    yield emit({"type": "token", "task_number": i, "content": sub_header})

                    if task["category"] == "agent_task":
                        from langgraph_agent import run_agent
                        from urllib.parse import quote
                        agent_res = run_agent(task["task"], history=get_history())
                        art = agent_res.get("artifact")
                        ans = agent_res.get("final_answer", "")
                        if not ans and art:
                            ans = f"Generated file {art.get('filename')} successfully."
                        for word in ans.split(" "):
                            yield emit({"type": "token", "task_number": i, "content": word + " "})
                    elif task["category"] == "rag_search":
                        from rag.pipeline import answer as rag_answer
                        rag_res = rag_answer(task["task"])
                        ans = rag_res.get("answer", "")
                        for word in ans.split(" "):
                            yield emit({"type": "token", "task_number": i, "content": word + " "})
                    else:
                        for token in stream_answer(task["model"], task["task"]):
                            yield emit({"type": "token", "task_number": i, "content": token})

                    yield emit({"type": "task_done", "task_number": i})

                elapsed = round(time.time() - start, 2)
                log_stream(f"Multi-part streaming finished in {elapsed}s across {len(tasks)} sub-tasks")
                yield emit({"type": "done", "time_seconds": elapsed, "category": "multi"})

            # ── Path C: Single model streaming ────────────────────────
            else:
                model = info["model"]
                log_router(f"Stream request: routing to {model}")
                yield emit({
                    "type": "stage",
                    "label": "routing",
                    "detail": f"Routing to {model} ({info['reason']})",
                    "category": info["category"],
                })
                yield emit({"type": "stage", "label": "generating", "detail": "Writing the answer..."})

                token_count = 0
                for token in stream_answer(model, q.query):
                    token_count += 1
                    yield emit({"type": "token", "content": token})

                elapsed = round(time.time() - start, 2)
                log_stream(f"Single model streaming finished in {elapsed}s with ~{token_count} tokens")
                yield emit({
                    "type": "done",
                    "model_used": model,
                    "category": info["category"],
                    "token_count": token_count,
                    "time_seconds": elapsed,
                })

        except Exception as e:
            log_err(f"Exception during streaming: {e}")
            yield emit({
                "type": "error",
                "detail": str(e),
                "time_seconds": round(time.time() - start, 2),
            })

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/ask/complex")
def ask_complex(q: Question):
    try:
        start = time.time()
        log_user(f'Force-Split Complex Request: "{q.query}"')
        if is_realtime_query(q.query):
            return {"type": "offline_decline", "time_seconds": round(time.time() - start, 2), "answer": OFFLINE_REALTIME_MESSAGE}
        tasks = break_into_tasks(q.query)
        log_router(f"Split into {len(tasks)} sub-tasks")

        results = []
        for i, task in enumerate(tasks):
            log_model(f"Running sub-task {i+1}: '{task['label']}' -> {task['model']}")
            answer = get_full_answer(task["model"], task["task"])
            results.append({
                "task_number": i + 1,
                "label": task["label"],
                "model": task["model"],
                "category": task["category"],
                "answer": answer,
            })

        elapsed = round(time.time() - start, 2)
        log_response(f"Complex tasks finished in {elapsed}s")
        return {
            "sub_tasks": len(tasks),
            "time_seconds": elapsed,
            "results": results,
        }
    except Exception as e:
        log_err(f"Exception in /ask/complex: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/image")
def ask_image_endpoint(q: ImageQuestion):
    try:
        start  = time.time()
        log_user(f'Image Query: "{q.query}" (b64 length: {len(q.image_b64)})')
        answer = ask_image(q.image_b64, q.query)
        elapsed = round(time.time() - start, 2)
        model_name = getattr(router, "resolve_installed_model", lambda m: m)(_cfg.IMAGE_MODEL)
        log_response(f"Image analysis completed in {elapsed}s by {model_name}")
        return {
            "type":         "image",
            "model_used":   model_name,
            "time_seconds": elapsed,
            "answer":       answer,
        }
    except Exception as e:
        log_err(f"Exception in /ask/image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/agent")
def ask_agent(q: Question):
    """
    LangGraph agent endpoint.
    Runs the question through a structured multi-turn graph:
      1. classify  — decides if a tool is needed and which one
      2. tools     — executes the real tool from tools.py (if needed)
      3. classify  — loops back to check if more tools needed (up to 5x)
      4. respond   — generates the final answer with full context
    """
    try:
        start = time.time()
        log_user(f'LangGraph Agent Request: "{q.query}"')
        if is_realtime_query(q.query):
            return {"type": "offline_decline", "time_seconds": round(time.time() - start, 2), "answer": OFFLINE_REALTIME_MESSAGE}

        try:
            from langgraph_agent import run_agent
        except ImportError as exc:
            log_err(f"LangGraph not installed: {exc}")
            raise HTTPException(
                status_code=500,
                detail="LangGraph packages not installed. Run: py -m pip install langgraph langchain-ollama"
            )

        history = get_history()
        log_model("Running supervised LangGraph workflow...")
        result = run_agent(q.query, history=history)

        for event in result.get("events", []):
            if event.get("stage") == "timing":
                log_system(f"Agent timing — {event.get('detail')}")

        needs_tool = result.get("needs_tool", False)
        tool_log   = result.get("tool_log", [])
        answer     = result.get("final_answer", "").strip()
        artifact   = result.get("artifact")
        if artifact:
            artifact = {**artifact, "download_url": f"/files/{quote(artifact['filename'])}"}

        if needs_tool and tool_log:
            for t in tool_log:
                log_tool(f"Agent tool: {t.get('tool')} -> {str(t.get('result', ''))[:120]}")
        else:
            log_model("Agent decided: direct response (no tool needed)")

        if not answer:
            answer = "Agent completed. Check generated_files/ for any output files."

        elapsed = round(time.time() - start, 2)
        log_response(f"LangGraph agent completed in {elapsed}s | tools_called={len(tool_log)}")

        return {
            "type":         "agent",
            "model_used":   "qwen2.5:7b (LangGraph)",
            "time_seconds": elapsed,
            "needs_tool":   needs_tool,
            "tool_log":     tool_log,
            "answer":       answer,
            "progress":     result.get("events", []),
            "artifact":     artifact,
            "errors":       result.get("errors", []),
            "run_id":       result.get("run_id"),
        }
    except HTTPException:
        raise
    except Exception as e:
        log_err(f"Exception in /ask/agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/rag")
def ask_rag_endpoint(q: Question):
    """
    RAG search endpoint. Searches the knowledge base and synthesizes an answer with sources.
    """
    try:
        start = time.time()
        log_user(f'RAG Query: "{q.query}"')
        if is_realtime_query(q.query):
            return {"type": "offline_decline", "time_seconds": round(time.time() - start, 2), "answer": OFFLINE_REALTIME_MESSAGE}
        from rag.pipeline import answer as rag_answer
        res = rag_answer(q.query)
        elapsed = round(time.time() - start, 2)
        log_response(f"RAG search completed in {elapsed}s | sources={len(res.get('sources', []))}")
        return {
            "type":               "rag_search",
            "model_used":         _cfg.RAG_LLM_MODEL,
            "time_seconds":       elapsed,
            "answer":             res.get("answer", ""),
            "sources":            res.get("sources", []),
            "retrieval_strategy": res.get("retrieval_strategy", "similarity"),
        }
    except Exception as e:
        log_err(f"Exception in /ask/rag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
def list_tools():
    """List all tools available to the LangGraph agent."""
    from tools import TOOL_SCHEMAS
    return {
        "tools": [
            {
                "name":        t["function"]["name"],
                "description": t["function"]["description"],
                "required":    t["function"]["parameters"].get("required", []),
            }
            for t in TOOL_SCHEMAS
        ]
    }


@app.get("/files/{filename}")
def download_generated_file(filename: str):
    """Download a validated artifact created by the agent."""
    try:
        from artifacts import resolve_artifact
        path = resolve_artifact(filename)
        return FileResponse(path, filename=path.name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Generated file not found")


@app.get("/sessions")
def list_sessions():
    """List all locally persisted chat sessions."""
    if not db.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Local session store unavailable: {db.get_error()}"
        )
    sessions = db.get_all_sessions(limit=50)
    return {
        "total":    db.get_session_count(),
        "sessions": sessions,
    }


@app.get("/sessions/{session_name}")
def get_session_messages(session_name: str):
    """Retrieve all messages for a locally persisted session."""
    if not db.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Local session store unavailable: {db.get_error()}"
        )
    messages = db.get_session_messages(session_name, limit=500)
    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_name}' not found or has no messages."
        )
    return {
        "session_name": session_name,
        "message_count": len(messages),
        "messages": messages,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HOTSPOT PORT FORWARDING & MULTI-DEVICE SYNC ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/network/info")
def network_info():
    """Get host LAN IP addresses, ports, and hotspot access info for multi-device setup."""
    info = get_local_ip_addresses()
    info["server_name"] = _cfg.APP_NAME
    info["version"] = _cfg.APP_VERSION
    recent = hub.get_recent_items(20)
    info["active_broadcasts"] = len(recent)
    info["latest_sync_id"] = recent[0]["sync_id"] if recent else None
    return info


@app.get("/sync/content/{sync_id}")
def get_sync_content(sync_id: str):
    """Retrieve full content, status, stages, and download links for a specific Sync ID."""
    clean_id = str(sync_id).strip()
    item = hub.get_item(clean_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Sync PIN/ID '{clean_id}' not found.")
    return item.to_dict()


@app.get("/sync/stream/{sync_id}")
async def get_sync_stream(sync_id: str):
    """Stream live progress and tokens for a specific Sync ID to secondary devices in real-time."""
    clean_id = str(sync_id).strip()
    item = hub.get_item(clean_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Sync PIN/ID '{clean_id}' not found.")

    async def event_generator():
        # First send initial snapshot so the receiver immediately renders existing content
        snapshot = item.to_dict()
        yield json.dumps({"type": "snapshot", **snapshot}) + "\n"
        if item.status in ("completed", "error"):
            return

        q = asyncio.Queue()
        item.listeners.append(q)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield json.dumps(event) + "\n"
                    if event.get("type") in ("done", "error") or event.get("status") in ("completed", "error"):
                        break
                except asyncio.TimeoutError:
                    # Keep-alive heartbeat
                    yield json.dumps({"type": "ping"}) + "\n"
                    if item.status in ("completed", "error"):
                        break
        finally:
            if q in item.listeners:
                item.listeners.remove(q)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.get("/sync/recent")
def get_recent_syncs(limit: int = 15):
    """Get list of recent shared sync items."""
    return {"recent": hub.get_recent_items(limit=max(1, min(limit, 50)))}


@app.get("/sync/latest")
def get_latest_sync():
    """Get the latest sync item currently active or completed."""
    item = hub.get_latest_item()
    if not item:
        return {"sync_id": None, "status": "none"}
    return item.to_dict()


@app.post("/sync/submit")
def submit_remote_task(req: RemoteSubmitRequest):
    """
    Submit a task from a remote/receiver device.
    Pre-allocates a Sync ID, initiates server-side generation in a background thread,
    and returns the Sync ID immediately so the remote device can stream or poll it.
    """
    sync_item = hub.create_item(query=req.query, sync_id=req.sync_id)
    log_user(f'Remote device submitted task PIN={sync_item.sync_id}: "{req.query}"')

    def _execute():
        start_time = time.time()
        try:
            sync_item.append_stage({"stage": "understanding", "detail": "Classifying remote request on host GPU..."})
            info = classify_question(req.query)

            if req.mode == "chief" or info["category"] == "complex":
                sync_item.append_stage({"stage": "planning", "detail": "Running complex breakdown..."})
                tasks = break_into_tasks(req.query)
                combined = ""
                for i, t in enumerate(tasks, 1):
                    sync_item.append_stage({"stage": "working", "detail": f"Sub-task {i}/{len(tasks)}: {t['label']}"})
                    ans = get_full_answer(t["model"], t["task"])
                    combined += f"\n\n### Sub-task {i}: {t['label']}\n{ans}"
                    sync_item.append_content(f"\n\n### Sub-task {i}: {t['label']}\n{ans}")
                elapsed = round(time.time() - start_time, 2)
                sync_item.complete({"time_seconds": elapsed, "category": "complex"})
                return

            if info["category"] == "agent_task":
                sync_item.append_stage({"stage": "routing", "detail": "Autonomous agent artifact creation on server..."})
                from langgraph_agent import run_agent
                res = run_agent(req.query, history=get_history())
                artifact = res.get("artifact")
                if artifact and "filename" in artifact:
                    artifact = {**artifact, "download_url": f"/files/{quote(artifact['filename'])}"}
                ans = res.get("final_answer", "")
                if not ans and artifact:
                    ans = f"Generated {artifact.get('filename')} successfully."
                sync_item.append_content(ans)
                elapsed = round(time.time() - start_time, 2)
                sync_item.complete({
                    "model_used": info["model"],
                    "category": "agent_task",
                    "time_seconds": elapsed,
                    "artifact": artifact,
                })
                return

            if info["category"] == "rag_search":
                sync_item.append_stage({"stage": "routing", "detail": "Searching ChromaDB knowledge base on server..."})
                from rag.pipeline import answer as rag_answer
                res = rag_answer(req.query)
                ans = res.get("answer", "")
                sync_item.append_content(ans)
                elapsed = round(time.time() - start_time, 2)
                sync_item.complete({
                    "model_used": info["model"],
                    "category": "rag_search",
                    "time_seconds": elapsed,
                    "sources": res.get("sources", []),
                })
                return

            # Default / code / fast model streaming to sync_item
            sync_item.append_stage({"stage": "generating", "detail": f"Generating answer with {info['model']}..."})
            for token in stream_answer(info["model"], req.query):
                sync_item.append_content(token)
            elapsed = round(time.time() - start_time, 2)
            sync_item.complete({
                "model_used": info["model"],
                "category": info["category"],
                "time_seconds": elapsed,
            })

        except Exception as exc:
            log_err(f"Remote task PIN={sync_item.sync_id} failed: {exc}")
            sync_item.fail(str(exc))

    threading.Thread(target=_execute, daemon=True).start()

    return {
        "status": "processing",
        "sync_id": sync_item.sync_id,
        "query": req.query,
        "created_at": sync_item.created_at,
    }

