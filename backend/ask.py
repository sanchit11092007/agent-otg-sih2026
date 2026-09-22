"""
ask.py - The Interactive Terminal Client for Agent OTG
Developed by Team DWE

Features:
    - Smart Intent Detection: automatically routes to RAG, Agent, Vision, or Chat
      WITHOUT requiring /rag, /agent, /image prefixes
    - Advanced 5-Category Routing: code, simple, complex, rag_search, agent_task
    - Premium Terminal UI with rich formatting and animations
    - Document Ingestion: PDF/DOCX/CSV/Excel/JSON into RAG knowledge base
    - LangGraph Autonomous Agent: multi-step structured agent
    - All generated files saved to ~/Downloads/AgentOTG/
"""

import sys
import time
import base64
import os
import threading
import datetime
import json
import re
import traceback
import requests as _requests

from offline_guard import enable_offline_mode, allow_external
enable_offline_mode()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import db
import router
from router import (
    classify_question,
    stream_answer,
    stream_image_answer,
    break_into_tasks,
    run_sequential_tasks,
    clear_history,
    get_history,
    CODER_MODEL,
    MAIN_MODEL,
    FAST_MODEL,
    IMAGE_MODEL,
    AVAILABLE_MODELS,
)
import config as _cfg

# ── Output directory info ─────────────────────────────────────────────────────
from artifacts import OUTPUT_DIR as _OUTPUT_DIR

# ── Status phrases for spinner (disabled) ────────────────────────────────────
THINKING_PHRASES = []

# ── Max characters to show per message in history view ───────────────────────
_HISTORY_MSG_TRUNCATE = 200
_HISTORY_MAX_MSGS     = 20

# ── Smart dispatch intent keywords ───────────────────────────────────────────
# Used to auto-detect RAG / agent intent WITHOUT requiring /rag or /agent prefix
_RAG_AUTO_TRIGGERS = [
    "what does the document", "what does my document", "what does the file",
    "from the uploaded", "in my files", "search the knowledge base",
    "from the knowledge base", "what does the report say", "search my docs",
    "what does the uploaded document", "what does the uploaded file",
    "what does it say", "according to the document", "according to the report",
    "from the data", "in the pdf", "in the doc",
    "find in the document", "look up in", "from the uploaded file",
    "what the document says", "search my knowledge",
    "what does the report", "what are the findings", "extract from",
    "search my documents", "query the knowledge base",
    "what does my report", "what does my file",
    "summarize the uploaded", "summarize the document", "summarize my document",
    "summarize the pdf", "summarize the report", "summarize the file",
    "what is in the file", "what is in the document", "what is in my files",
    "find in my documents", "what does the uploaded agreement say",
    "what does the contract say", "extract from the document",
]

_AGENT_AUTO_TRIGGERS = [
    "create a pdf", "generate a pdf", "make a pdf", "write a pdf",
    "create a word", "generate a word", "make a word doc",
    "create a doc", "generate a doc",
    "create an excel", "make an excel", "generate an excel",
    "generate a spreadsheet", "create a spreadsheet", "make a spreadsheet",
    "create a csv", "make a csv", "generate a csv",
    "create a report and save", "generate a report", "save as pdf",
    "export to pdf", "create a presentation", "make a powerpoint",
    "generate a pptx", "create a json file", "generate a json",
    "write a report and", "create a file", "generate a file",
    "make a file", "write and save", "create and save",
    "make and save", "export a file",
    "make a ppt", "create a ppt", "generate a ppt", "give me a ppt", "build a ppt",
    "ppt on", "presentation on", "presentation about", "slides on", "slides about",
    "create slides", "make slides", "generate slides", "deck on", "pitch deck",
    "powerpoint on", "powerpoint about", "make a presentation", "generate a presentation",
]

_IMAGE_AUTO_TRIGGERS = [
    "analyze this image", "what is in this image", "describe this image",
    "what does this picture show", "look at this image",
    "analyze the image", "what is shown in",
]


def _clean_path(path: str) -> str:
    """Clean surrounding quotes and whitespace from paths."""
    if not path:
        return ""
    p = str(path).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return os.path.normpath(p)


# ══════════════════════════════════════════════════════════════════════════════
# SPINNER & STREAMING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def strip_think_stream(token_generator):
    """
    Filter out <think>...</think> blocks from a token stream in real-time.
    Buffers minimally to detect tags and streams immediately when outside think blocks.
    """
    in_think = False
    buffer = ""
    for chunk in token_generator:
        buffer += chunk
        while buffer:
            if not in_think:
                if "<think>" in buffer:
                    before, _, after = buffer.partition("<think>")
                    if before:
                        yield before
                    buffer = after
                    in_think = True
                else:
                    potential_prefix = False
                    for i in range(min(len(buffer), len("<think>") - 1), 0, -1):
                        if "<think>".startswith(buffer[-i:]):
                            if len(buffer) > i:
                                yield buffer[:-i]
                                buffer = buffer[-i:]
                            potential_prefix = True
                            break
                    if not potential_prefix:
                        yield buffer
                        buffer = ""
                    else:
                        break
            else:
                if "</think>" in buffer:
                    _, _, after = buffer.partition("</think>")
                    buffer = after.lstrip("\r\n")
                    in_think = False
                else:
                    potential_prefix = False
                    for i in range(min(len(buffer), len("</think>") - 1), 0, -1):
                        if "</think>".startswith(buffer[-i:]):
                            buffer = buffer[-i:]
                            potential_prefix = True
                            break
                    if not potential_prefix:
                        buffer = ""
                    break
    if buffer and not in_think:
        yield buffer


def strip_think_text(text: str) -> str:
    """Strip <think>...</think> block from full string."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def stream_response(model: str, query: str):
    """Streams model answer directly to stdout with zero thinking phase or spinner delay."""
    for token in strip_think_stream(stream_answer(model, query)):
        if token.startswith("\n\n[System: Calling tool"):
            continue
        sys.stdout.write(token)
        sys.stdout.flush()
    print()


def show_spinner(messages, stop_flag, start_time):
    pass


def stage(badge: str, text: str, style="bold cyan"):
    pass


def run_with_spinner(messages, work_fn):
    return work_fn()


def stream_with_spinner(model, query):
    stream_response(model, query)
    return 0


def _check_ollama_models() -> dict:
    """Check which configured models are available in Ollama."""
    available = {}
    try:
        import ollama as _ollama
        with allow_external():
            pulled = {m["name"] for m in _ollama.list()["models"]}
        pulled_base = {n.split(":")[0] for n in pulled} | pulled
        for role, model in AVAILABLE_MODELS.items():
            base = model.split(":")[0]
            available[model] = (model in pulled) or (base in pulled_base)
    except Exception:
        for model in AVAILABLE_MODELS.values():
            available[model] = True
    return available


def print_welcome_banner():
    print("Local AI Agent - Agent OTG developed by DWE\n")


def print_help_guide():
    console.print()
    console.print(Rule("[bold cyan]Agent OTG — Practical Example Guide[/bold cyan]", style="bright_blue"))

    examples = [
        ("💻 Coding (auto-detected)",
         'Write a Python FastAPI middleware for request timing and explain it.'),
        ("🧠 Reasoning (auto-detected)",
         'Compare PostgreSQL vs MongoDB for high-write telemetry systems.'),
        ("⚡ Quick (auto-detected)",
         'Hello / What is the capital of France?'),
        ("🔍 RAG — auto-detect",
         'What does the report say about the safety guidelines?'),
        ("🔍 RAG — explicit",
         '/rag What are the main findings in the uploaded document?'),
        ("📂 Ingest file (just paste path)",
         r'C:\Users\me\Downloads\report.pdf  (auto-detected as file path)'),
        ("📂 Ingest + query in one step",
         r'/doc "C:\Users\me\Downloads\etp.csv" Analyze this and give 10 key points'),
        ("📄 Create PDF (auto-detected)",
         'Create a PDF report on machine learning trends with 5 key sections.'),
        ("📝 Create Word (auto-detected)",
         'Generate a Word document titled "Q3 Report" with 3 bullet highlights.'),
        ("📊 Create Excel",
         'Create an Excel sheet with columns: Name, Score, Grade, and 5 sample rows.'),
        ("👁️ Vision / Image",
         '/image "C:\\Users\\me\\Pictures\\diagram.png" What architecture is shown here?'),
        ("👁️ Vision (Web URL)",
         '/image https://example.com/logo.png Describe this logo in detail.'),
        ("🛠️ Agent — Chain tasks",
         '/agent Generate a PDF about Python basics, then tell me how many pages it has.'),
        ("🧠 Direct 14B reasoning",
         '/complex Compare three database designs for a high-volume financial ledger.'),
        ("🧹 Reset memory",
         'reset  — clears conversation context for a fresh start'),
        ("🗑️ List output files",
         '/files  — shows all files in ~/Downloads/AgentOTG/'),
    ]

    for cat, ex in examples:
        console.print(f"\n[bold green]{cat}:[/bold green]")
        console.print(f"  [yellow]{ex}[/yellow]")

    console.print()
    console.print(Rule(style="bright_blue"))
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# SMART INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _is_multi_request(query: str) -> bool:
    """True if query contains multiple distinct requests, numbered items, or mixed actions."""
    q = (query or "").strip()
    from langgraph_agent import detect_file_intent
    # Single file creation with descriptive clauses (e.g. "and make it 5 slides") should NOT be decomposed
    if detect_file_intent(q).get("file_format") and not _looks_like_mixed_file_request(q):
        return False
    if len(router._split_multi_actions(q)) >= 2:
        return True
    # Check for numbered items like 1) ... 2) ... or 1. ... 2. ...
    if re.search(r"(?:^|\s)(?:1[\).]|first[,:])\s+.*?(?:\s+(?:2[\).]|second[,:]))\s+", q, re.I | re.DOTALL):
        return True
    # Check for bulleted list items
    if re.search(r"(?:^|\n)\s*[-*•]\s+.*?\n\s*[-*•]\s+", q):
        return True
    return _looks_like_mixed_file_request(q)


def _detect_intent(query: str) -> str:
    """
    Detect the primary intent of a natural-language query WITHOUT requiring
    command prefixes. Returns one of:
      'rag'    → search knowledge base
      'agent'  → create a file / autonomous agent task
      'image_gen' → create a local image with SD-Turbo
      'normal' → standard chat/code/reasoning
      'file'   → user pasted a file path (ingest it)
    """
    q = query.lower().strip()

    # 1. RAG search signals (check first so doc queries don't misroute)
    if router._is_rag_query(query) or any(trigger in q for trigger in _RAG_AUTO_TRIGGERS):
        return "rag"

    # 2. Agent / file creation signals
    from langgraph_agent import detect_file_intent
    file_intent = detect_file_intent(query)
    if file_intent.get("file_format") or (any(trigger in q for trigger in _AGENT_AUTO_TRIGGERS) and not _looks_like_mixed_file_request(query)):
        if _is_multi_request(query) and not file_intent.get("file_format"):
            return "normal"
        return "agent"

    if re.search(r"\b(?:generate|create|make|draw)\s+(?:an?\s+)?image\b", q):
        return "image_gen"

    if _is_multi_request(query):
        return "normal"

    return "normal"


def _show_intent_badge(intent: str, query_preview: str = ""):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def _show_route_card(info: dict):
    pass


def handle_single(query, info):
    model = info.get("model", MAIN_MODEL)
    stream_response(model, query)


def handle_image_generation(query: str):
    """Run the local SD-Turbo generator; never route an image request to Ollama chat."""
    prompt = re.sub(
        r"^\s*(?:generate|create|make|draw)\s+(?:an?\s+)?image\s*(?:of|for)?\s*",
        "", query, flags=re.I,
    ).strip()
    if not prompt:
        print("Please include an image description.\n")
        return
    from image_gen import generate_image
    result = generate_image(prompt)
    print(f"{result}\n")


def handle_sequential(query):
    """Handles 'Code + Explain' queries in two ordered, chained steps."""
    code_prompt = (
        f'The user asked: "{query}"\n\n'
        "Your job for this step: write ONLY the code. "
        "Do not explain it yet — just provide clean, well-commented code."
    )
    stream_response(CODER_MODEL, code_prompt)
    print()

    explain_prompt = (
        "Now explain the code you just wrote above, step by step. "
        "Be clear and beginner-friendly. Cover what each part does and why."
    )
    stream_response(MAIN_MODEL, explain_prompt)


def handle_multi(query):
    tasks = break_into_tasks(query)
    for task in tasks:
        if re.search(r"\b(?:generate|create|make|draw)\s+(?:an?\s+)?image\b", task["task"], re.I):
            handle_image_generation(task["task"])
        elif is_file_creation_request(task["task"]):
            handle_agent(task["task"])
        else:
            stream_response(task["model"], task["task"])


def is_file_creation_request(text: str) -> bool:
    """Detects whether user prompt is asking to generate, save, or export a file/document."""
    from langgraph_agent import detect_file_intent
    if detect_file_intent(text).get("file_format"):
        return True
    t = text.lower()
    has_action = bool(re.search(r"\b(generate|generatet|create|make|save|export|write|build|output|download|draft|prepare|provide|deliver|give|craft|produce|convert)\b", t))
    has_file   = bool(re.search(r"\b(pdf|docx|doc|word doc|word document|word file|word report|word|ms word|microsoft word|excel|xlsx|xls|csv|pptx|ppt|powerpoint|presentation|slides|deck|spreadsheet|json|text file|txt)\b", t))
    direct_phrase = bool(re.search(r"\b(word file|doc file|docx file|pdf file|excel file|excel sheet|pptx file|save as|export to|convert to|in word|in pdf|in excel|ppt on|presentation on)\b", t))
    return (has_action and has_file) or direct_phrase


def _looks_like_mixed_file_request(text: str) -> bool:
    """True when a file request is accompanied by a distinctly separate task (e.g. coding or separate analysis)."""
    t = text.lower().strip()
    if not is_file_creation_request(t):
        return False
    # Explicit numbered or bulleted items
    if re.search(r"(?:^|\s)(?:1[\).]|first[,:])\s+.*?(?:\s+(?:2[\).]|second[,:]))\s+", t, re.I | re.DOTALL):
        return True
    if re.search(r"(?:^|\n)\s*[-*•]\s+.*?\n\s*[-*•]\s+", t):
        return True
    # If it contains both code request and file creation request for different things
    has_code = router._hard_code_check(t)
    if has_code and any(sep in t for sep in [";", "\n", " and also ", " and then "]):
        return True
    return False


def ask_anything(query, force_multi=False):
    """Main dispatcher for standard queries (non-agent, non-RAG)."""
    if router.uses_direct_14b_mode(query):
        stream_response(_cfg.COMPLEX_MODEL, router.strip_complex_command(query))
        return
    if not force_multi and _is_multi_request(query):
        handle_multi(query)
        return

    if not force_multi and is_file_creation_request(query) and not _looks_like_mixed_file_request(query):
        handle_agent(query)
        return

    if not force_multi and _looks_like_mixed_file_request(query):
        handle_multi(query)
        return

    info = classify_question(query)

    # If classifier detected a RAG or agent category, route accordingly
    category = info.get("category", "complex")
    if category == "rag_search":
        handle_rag(query)
        return
    elif category == "agent_task":
        handle_agent(query)
        return

    if force_multi:
        handle_multi(query)
    elif info.get("has_code_and_explain"):
        handle_sequential(query)
    elif info.get("is_multi_part"):
        handle_multi(query)
    else:
        handle_single(query, info)


def handle_agent(query):
    try:
        from langgraph_agent import run_agent
    except ImportError as e:
        print(f"Error: {e}")
        return

    result = run_agent(query, history=get_history())
    final_answer = result.get("final_answer", "").strip()
    if not final_answer:
        final_answer = "Task completed. Check your Downloads/AgentOTG folder for generated files."

    final_answer = strip_think_text(final_answer)
    console.print(Markdown(final_answer))
    print()


def parse_image_command(rest: str):
    r"""
    Safely parse image command argument to handle paths with spaces and quotes.
    Returns (source, question).
    """
    rest = rest.strip()
    if not rest:
        return "", ""

    if rest.startswith('"'):
        end_idx = rest.find('"', 1)
        if end_idx != -1:
            return rest[1:end_idx], rest[end_idx + 1:].strip()
    elif rest.startswith("'"):
        end_idx = rest.find("'", 1)
        if end_idx != -1:
            return rest[1:end_idx], rest[end_idx + 1:].strip()

    if rest.startswith("http://") or rest.startswith("https://"):
        parts = rest.split(" ", 1)
        return parts[0], parts[1].strip() if len(parts) > 1 else ""

    tokens = rest.split(" ")
    for end in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:end])
        norm = os.path.normpath(candidate)
        if os.path.isfile(norm):
            return norm, " ".join(tokens[end:]).strip()

    parts = rest.split(" ", 1)
    return parts[0], parts[1].strip() if len(parts) > 1 else ""


def handle_image(source, question):
    source = _clean_path(source).strip("<>")
    if not source:
        print("Error: Missing image path or URL.\n")
        return

    is_url = source.startswith("http://") or source.startswith("https://")

    if is_url:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            with allow_external():
                resp = _requests.get(source, headers=headers, timeout=15)
            resp.raise_for_status()
            image_b64 = base64.b64encode(resp.content).decode()
        except Exception as e:
            print(f"Could not fetch URL: {e}\n")
            return
    else:
        if not os.path.isfile(source):
            print(f"File not found: {source}\n")
            return
        ext = os.path.splitext(source)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            print(f"Unsupported format: '{ext}'. Use PNG, JPEG, WEBP, or GIF.\n")
            return
        with open(source, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

    if not question:
        question = "Describe this image in detail. Extract all visible text, data, and key information."

    for token in strip_think_stream(stream_image_answer(image_b64, question)):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()


def handle_list_files():
    """Show all files in ~/Downloads/AgentOTG/."""
    from tools import list_generated_files
    result = list_generated_files()
    console.print(Panel(result, title="Generated Files", border_style="green", padding=(0, 2)))
    console.print()


def is_file_path_arg(text: str) -> tuple[bool, str, str]:
    """Checks if text is or begins with a file path. Returns (is_file, filepath, user_prompt)."""
    filepath, prompt = parse_doc_command(text)
    if not filepath:
        return False, "", ""
    ext = os.path.splitext(filepath)[1].lower()
    valid_exts = (".csv", ".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md", ".json")
    if os.path.isfile(filepath) or ext in valid_exts:
        return True, filepath, prompt
    return False, "", ""


def handle_rag(query: str):
    """Directly search the RAG knowledge base with query expansion."""
    if not query.strip():
        return

    is_file, filepath, prompt = is_file_path_arg(query)
    if is_file:
        doc_path, user_p = handle_doc(query)
        if user_p:
            try:
                from rag.pipeline import answer as _rag_answer
                result = _rag_answer(user_p)
                ans = strip_think_text(result.get("answer", "No answer found."))
                console.print(Markdown(ans))
                _show_rag_sources(result)
            except Exception as exc:
                print(f"RAG search failed: {exc}\n")
        return

    try:
        from rag.pipeline import answer as _rag_answer
        result = _rag_answer(query)
        ans = strip_think_text(result.get("answer", "No answer found."))
        console.print(Markdown(ans))
        _show_rag_sources(result)
    except Exception as exc:
        print(f"RAG search failed: {exc}\n")


def _show_rag_sources(result: dict):
    """Display RAG sources in a clean deduplicated format."""
    sources = result.get("sources", [])
    if not sources:
        return

    seen = set()
    unique_sources = []
    for s in sources:
        src  = s.get("source", "unknown")
        page = s.get("page")
        sheet = s.get("sheet")
        key = (src, page, sheet)
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    if unique_sources:
        print("\nSources:")
        for s in unique_sources[:6]:
            src   = os.path.basename(s.get("source", "unknown"))
            page  = f", page {s['page']}" if s.get("page") else ""
            sheet = f", sheet '{s['sheet']}'" if s.get("sheet") else ""
            print(f"  • {src}{page}{sheet}")
        print()


def parse_doc_command(rest: str) -> tuple[str, str]:
    r"""
    Robustly extract a file path and optional prompt from user input.
    Handles all Windows path edge cases:
      - Quoted paths:       "D:\Users\me\file.pdf" query here
      - Unquoted paths:     D:\Users\me\file.pdf query here
      - Paths with spaces:  D:\My Documents\file.csv
      - Forward slashes:    D:/Users/me/file.pdf
      - Dropped paths:      file.csv (just the filename)
      - Path with prompt:   report.pdf tell me the key findings
    Returns (normalized_filepath, user_prompt).
    """
    rest = rest.strip().strip("<>")   # strip drag-drop angle brackets
    if not rest:
        return "", ""

    # ── 1. Quoted path ─────────────────────────────────────────────────────
    for q in ('"', "'"):
        if rest.startswith(q):
            end_idx = rest.find(q, 1)
            if end_idx != -1:
                raw_path = rest[1:end_idx].strip()
                prompt   = rest[end_idx + 1:].strip()
                return os.path.normpath(raw_path), prompt

    # ── 2. Looks like a Windows absolute path (D:\... or D:/...) ───────────
    # Try consuming as many tokens as needed to form an existing file path.
    tokens = rest.split()

    # First check if the whole string (minus trailing prompt words) is a path
    # Strategy: try to greedily find the longest prefix that is an existing file.
    for end in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:end])
        # Normalise forward slashes too
        norm = os.path.normpath(candidate.replace("/", os.sep))
        if os.path.isfile(norm):
            prompt = " ".join(tokens[end:]).strip()
            return norm, prompt

    # ── 3. No existing file found — try heuristic: does the first token look
    #       like a path? (has extension, has path separator, or has drive letter)
    first = tokens[0]
    has_drive     = len(first) >= 2 and first[1] == ":"           # C: D: etc.
    has_sep       = ("/" in first or "\\" in first)
    has_extension = "." in os.path.basename(first)

    if has_drive or has_sep or has_extension:
        # Try consuming tokens until the path has a known extension
        supported_exts = {".pdf", ".docx", ".xlsx", ".xls", ".csv",
                          ".txt", ".md", ".json", ".png", ".jpg", ".jpeg", ".webp"}
        # Walk forward token by token — stop when we hit a word after a valid ext
        best_path = ""
        best_end  = 0
        for end in range(1, len(tokens) + 1):
            candidate = " ".join(tokens[:end])
            ext = os.path.splitext(candidate)[1].lower()
            if ext in supported_exts:
                best_path = candidate
                best_end  = end
        if best_path:
            norm   = os.path.normpath(best_path.replace("/", os.sep))
            prompt = " ".join(tokens[best_end:]).strip()
            return norm, prompt

        # Fall back — everything is the path
        norm = os.path.normpath(first.replace("/", os.sep))
        prompt = " ".join(tokens[1:]).strip()
        return norm, prompt

    # ── 4. No path-like pattern found ─────────────────────────────────────
    return "", rest


def handle_doc(raw_arg: str):
    """Ingest a file into the RAG vector knowledge base."""
    filepath, prompt = parse_doc_command(raw_arg)

    if not filepath:
        print("Ingestion failed: No file path provided.\n")
        return None, None

    try:
        from rag.loaders import validate_file_pre_ingestion
        validate_file_pre_ingestion(filepath)
    except Exception as val_err:
        print(f"Ingestion failed: {val_err}\n")
        return None, None

    try:
        from rag.pipeline import ingest_path as _ingest
        result = _ingest(filepath, replace_existing=True)
        print(f"Indexed {os.path.basename(filepath)} into knowledge base.\n")
        return filepath, prompt
    except Exception as exc:
        err_str = str(exc)
        if hasattr(exc, "__cause__") and exc.__cause__:
            err_str = str(exc.__cause__)
        if err_str.startswith("RuntimeError:"):
            err_str = err_str.replace("RuntimeError:", "").strip()
        print(f"Ingestion failed: {err_str}\n")
        return None, None


def handle_kb_stats():
    """Show RAG knowledge base statistics."""
    try:
        from rag.vectorstore import get_vector_store
        store = get_vector_store()
        count = store._collection.count()
        console.print(Panel(
            f"[bold cyan]📊 Knowledge Base Statistics[/bold cyan]\n\n"
            f"  Chunks indexed : [bold yellow]{count:,}[/bold yellow]\n"
            f"  Collection     : [dim]{store._collection.name}[/dim]\n\n"
            f"  [dim]Ingest more: /doc <file>  •  Query: ask naturally or /rag <question>[/dim]",
            title="📊 RAG Knowledge Base",
            border_style="cyan",
            padding=(0, 2),
        ))
    except Exception as exc:
        console.print(f"  ❌ [bold red]Could not read KB stats:[/bold red] {exc}\n")
    print()


def handle_show_models():
    """Display currently configured models."""
    table = Table(
        title="🤖 Active Model Configuration  (edit .env to change)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Role",         style="bold green",  no_wrap=True)
    table.add_column("Model Name",   style="bold yellow")
    table.add_column("Env Variable", style="dim")
    table.add_column("Handles",      style="dim white")
    table.add_row("Code",      _cfg.CODER_MODEL,         "CODER_MODEL",         "Programming, debugging, SQL")
    table.add_row("Main/Agent", _cfg.MAIN_MODEL,         "MAIN_MODEL",          "Complex reasoning, agent tasks")
    table.add_row("Fast",      _cfg.FAST_MODEL,          "FAST_MODEL",          "Quick Q&A, routing, RAG formatting")
    table.add_row("Vision",    _cfg.IMAGE_MODEL,         "IMAGE_MODEL",         "Image analysis, diagrams")
    table.add_row("Embedding", _cfg.RAG_EMBEDDING_MODEL, "RAG_EMBEDDING_MODEL", "Document vectorization")
    table.add_row("RAG LLM",   _cfg.RAG_LLM_MODEL,      "RAG_LLM_MODEL",       "Knowledge base answering")
    console.print(table)
    console.print(f"\n  [dim]Output directory: [yellow]{_OUTPUT_DIR}[/yellow][/dim]")
    console.print("  [dim]Restart ask.py after editing .env for changes to take effect.[/dim]\n")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HISTORY VIEWER
# ══════════════════════════════════════════════════════════════════════════════

def handle_history():
    """Reads sessions and messages from the database and displays them."""
    if not db.is_ready():
        console.print(Panel(
            f"[bold red]Database not available.[/bold red]\n[dim]{db.get_error()}[/dim]",
            title="Database Unavailable",
            border_style="red",
        ))
        return

    sessions = db.get_all_sessions(limit=15)
    if not sessions:
        console.print("  [dim]No sessions found in the database yet.[/dim]\n")
        return

    hist_table = Table(
        title=f"📜 Saved Chat Sessions ({db.get_session_count()} total)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    hist_table.add_column("#",              justify="right", style="bold cyan",  no_wrap=True)
    hist_table.add_column("Session Name",                    style="yellow")
    hist_table.add_column("Started",                         style="dim")
    hist_table.add_column("Last Activity",                   style="dim")
    hist_table.add_column("Messages",       justify="right", style="green")

    for i, s in enumerate(sessions, 1):
        hist_table.add_row(
            str(i),
            s["session_name"],
            s["created_at"],
            s["last_activity"],
            str(s["message_count"]),
        )
    console.print(hist_table)

    try:
        choice = input("  Pick a session # to preview (or Enter to cancel): ").strip()
        if not choice or not choice.isdigit():
            console.print()
            return

        idx = int(choice)
        if not (1 <= idx <= len(sessions)):
            console.print("  [dim]Invalid selection.[/dim]\n")
            return

        selected     = sessions[idx - 1]
        session_name = selected["session_name"]
        messages     = db.get_session_messages(session_name, limit=_HISTORY_MAX_MSGS)

        console.print(f"\n  [bold cyan]─── {session_name} ───[/bold cyan]")
        console.print(f"  [dim]Showing last {len(messages)} message(s)[/dim]\n")

        for msg in messages:
            role       = msg.get("role", "unknown")
            content    = msg.get("content", "").strip()
            timestamp  = msg.get("created_at", "")
            role_color = "bold green" if role == "user" else "bold magenta"
            role_label = "You" if role == "user" else "Agent OTG"

            if len(content) > _HISTORY_MSG_TRUNCATE:
                content = content[:_HISTORY_MSG_TRUNCATE] + f"… [{len(content) - _HISTORY_MSG_TRUNCATE} more chars]"

            console.print(f"  [{role_color}]{role_label}[/{role_color}] [dim]{timestamp}[/dim]")
            console.print(f"  {content}")
            console.print("  " + "─" * 55, style="dim")

    except Exception as e:
        console.print(f"  [dim]Error viewing session: {e}[/dim]")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    db.init_db()

    session_name = datetime.datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
    router.start_new_session(session_name)

    print_welcome_banner()

    current_file_path    = None
    current_file_content = None

    while True:
        try:
            query = input("> ").strip()

            if not query:
                continue

            # ── Exit ──────────────────────────────────────────────────────
            if query.lower() in ["exit", "quit", ":q"]:
                break

            # ── Help ──────────────────────────────────────────────────────
            if query.lower() in ["help", "/help", "?", "--help"]:
                print_help_guide()
                continue

            # ── Clear screen ──────────────────────────────────────────────
            if query.lower() in ["clear", "cls"]:
                os.system("cls" if os.name == "nt" else "clear")
                print_welcome_banner()
                continue

            # ── Reset memory ──────────────────────────────────────────────
            if query.lower() in ["reset", "/reset"]:
                clear_history()
                current_file_path    = None
                current_file_content = None
                print("Memory cleared.\n")
                continue

            # ── Handle /ask prefix ────────────────────────────────────────
            if query.startswith("/ask "):
                query = query[5:].strip()
            elif query.lower() == "/ask":
                print("Usage: /ask <your question>\n")
                continue

            # ── Inject active file content when referenced ─────────────────
            trigger_phrases = [
                "this file", "this document", "this sheet", "this pdf",
                "the uploaded file", "the file", "uploaded",
                "analyze this", "summarize this", "from this", "in this file",
                "in this document", "the data in", "this data", "the data",
                "the dataset", "about this data", "from the data", "analyze this data",
                "summarize this data", "explain this data",
            ]
            has_data_ref = any(phrase in query.lower() for phrase in trigger_phrases)

            if current_file_path and current_file_content and has_data_ref:
                query += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End of file content ---"
            elif has_data_ref and not current_file_content:
                try:
                    from rag.vectorstore import get_vector_store
                    store = get_vector_store()
                    if store._collection.count() > 0:
                        from rag.pipeline import answer as _rag_answer
                        rag_res = _rag_answer(query)
                        rag_ans = rag_res.get("answer", "")
                        if rag_ans and "couldn't find sufficient information" not in rag_ans.lower():
                            query += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_ans}\n--- End of context ---"
                except Exception:
                    pass

            # ── Utility commands ──────────────────────────────────────────
            if query.lower() in ["/files", "files", "/ls", "ls"]:
                handle_list_files()
                continue

            if query.lower() in ["history", "/history"]:
                handle_history()
                continue

            if query.lower() in ["/kb", "/kb-stats", "/kbstats"]:
                handle_kb_stats()
                continue

            if query.lower() in ["/models", "/config", "/env"]:
                handle_show_models()
                continue

            if query.lower() == "/cancel":
                current_file_path    = None
                current_file_content = None
                print("Context cleared.\n")
                continue

            # ── Upload (legacy) ───────────────────────────────────────────
            if query.lower().startswith("upload "):
                raw_path = query[7:].strip()
                path     = _clean_path(raw_path)
                from file_readers import process_file
                result = process_file(path)
                if result.startswith("Error:"):
                    print(f"{result}\n")
                else:
                    current_file_path    = path
                    current_file_content = result
                    print(f"Loaded {os.path.basename(path)}.\n")
                continue

            # ── Explicit /rag command ──────────────────────────────────────
            if query.startswith("/rag ") or query.lower() == "/rag":
                rest = query[5:].strip() if query.startswith("/rag ") else ""
                if not rest:
                    print("Usage: /rag <question>  or  /rag <file path> [optional prompt]\n")
                else:
                    is_file, filepath, user_prompt = is_file_path_arg(rest)
                    if is_file:
                        doc_path, prompt = handle_doc(rest)
                        if doc_path:
                            current_file_path = doc_path
                            try:
                                from file_readers import process_file
                                res = process_file(doc_path)
                                if res and not res.startswith("Error:"):
                                    current_file_content = res
                            except Exception:
                                pass
                            if prompt:
                                prompt_to_run = prompt
                                if current_file_path and current_file_content:
                                    prompt_to_run += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End ---"
                                ask_anything(prompt_to_run)
                    else:
                        handle_rag(rest)
                continue

            # ── Explicit /doc command ──────────────────────────────────────
            elif query.startswith("/doc ") or query.lower() == "/doc":
                rest = query[5:].strip() if query.startswith("/doc ") else ""
                if not rest:
                    print("Usage: /doc <file path> [optional prompt]\n")
                else:
                    doc_path, user_prompt = handle_doc(rest)
                    if doc_path:
                        current_file_path = doc_path
                        try:
                            from file_readers import process_file
                            res = process_file(doc_path)
                            if res and not res.startswith("Error:"):
                                current_file_content = res
                        except Exception:
                            pass
                        if user_prompt:
                            prompt_to_run = user_prompt
                            if current_file_path and current_file_content:
                                prompt_to_run += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End ---"
                            ask_anything(prompt_to_run)
                continue

            # ── Image commands ─────────────────────────────────────────────
            elif query.startswith("/image ") or query.startswith("/img "):
                prefix_len = 7 if query.startswith("/image ") else 5
                rest = query[prefix_len:].strip()
                img_path, q_text = parse_image_command(rest)
                handle_image(img_path, q_text)
                continue

            # ── Explicit direct Qwen 14B ───────────────────────────────────
            elif router.is_complex_command(query):
                if not router.strip_complex_command(query):
                    print("Usage: /complex <deep reasoning request>\n")
                else:
                    ask_anything(query)
                continue

            # ── Explicit /agent command ────────────────────────────────────
            elif query.startswith("/agent ") or query.lower() == "/agent":
                agent_query = query[len("/agent "):].strip() if query.startswith("/agent ") else ""
                if not agent_query:
                    print("Usage: /agent <instruction>\n")
                else:
                    handle_agent(agent_query)
                continue

            # ── Explicit /multi or /multi-agent command ────────────────────
            elif any(query.lower().startswith(p) for p in ["/multi ", "/multi-agent ", "/multiagent "]) or query.lower() in ["/multi", "/multi-agent", "/multiagent"]:
                prefix_len = query.find(" ")
                rest = query[prefix_len + 1:].strip() if prefix_len != -1 else ""
                if not rest:
                    print("Usage: /multi <multi-part instructions>\n")
                else:
                    handle_multi(rest)
                continue

            else:
                # ── Smart Dispatch ─────────────────────────────────────
                is_file, filepath, user_prompt = is_file_path_arg(query)
                if is_file and (os.path.isfile(filepath) or "." in os.path.basename(filepath)):
                    doc_path, prompt = handle_doc(query)
                    if doc_path:
                        current_file_path = doc_path
                        try:
                            from file_readers import process_file
                            res = process_file(doc_path)
                            if res and not res.startswith("Error:"):
                                current_file_content = res
                        except Exception:
                            pass
                        if prompt:
                            prompt_to_run = prompt
                            if current_file_path and current_file_content:
                                prompt_to_run += f"\n\n--- Ingested Content of {os.path.basename(current_file_path)} ---\n{current_file_content}\n--- End ---"
                            ask_anything(prompt_to_run)
                    continue

                if _cfg.SMART_ROUTING_ENABLED:
                    intent = _detect_intent(query)
                    if intent == "rag":
                        handle_rag(query)
                        continue
                    elif intent == "agent":
                        handle_agent(query)
                        continue
                    elif intent == "image_gen":
                        handle_image_generation(query)
                        continue

                ask_anything(query)

        except KeyboardInterrupt:
            print()
            break
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
