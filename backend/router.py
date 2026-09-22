"""
router.py - Smart multi-model router for Agent OTG

MODELS
    CODER_MODEL : code questions — qwen2.5-coder  ← ALWAYS used for code
    MAIN_MODEL  : complex reasoning, essays, analysis — qwen2.5:7b
    FAST_MODEL  : quick answers, greetings, simple math — qwen2.5:7b
    IMAGE_MODEL : vision analysis — qwen2.5vl:7b


CLASSIFICATION — 5 categories:
    "code"       → CODER_MODEL  (hard-enforced — cannot be overridden)
    "simple"     → FAST_MODEL
    "complex"    → MAIN_MODEL
    "rag_search" → FAST_MODEL (RAG does the heavy lifting)
    "agent_task" → handled by LangGraph agent pipeline

CODE ROUTING GUARANTEE
    Coding questions are routed to CODER_MODEL at THREE enforcement layers:
      1. _hard_code_check()  — fast keyword pre-screen before any LLM call
      2. LLM classifier      — prompted with explicit code-first rules
      3. Post-LLM safety-net — re-checks if LLM returns non-code for a
                               query that contains code keywords
    This makes it impossible for model ambiguity to mis-route a coding
    question to FAST_MODEL or MAIN_MODEL.

SESSION PERSISTENCE
    All messages are stored in SQLite (agent_otg.sqlite3) via db.py.
"""

import json
import os
import re
import ollama
from functools import lru_cache
from langgraph_agent import detect_file_intent
import db as _db

_CURRENT_SESSION_NAME: str = ""
_LOG_CALLBACK = None


def _find_embedded_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Detect tool calls that models emitted as plain JSON text instead of native tool calls."""
    if not text:
        return []
    results = []
    pattern = re.compile(r'```(?:json)?\s*(\{\s*"name"\s*:\s*"[^"]+".*?\})\s*```', re.DOTALL)
    for block in pattern.findall(text):
        try:
            data = json.loads(block)
            name = data.get("name")
            args = data.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name and isinstance(args, dict):
                results.append((name, args))
        except Exception:
            continue
    if not results:
        unfenced = re.findall(
            r'(\{\s*"name"\s*:\s*"(?:generate_pdf_from_text|create_file|write_docx|merge_pdfs|split_pdf|rotate_pdf_pages|ingest_file|search_documents)"\s*,\s*"arguments"\s*:\s*\{.*?\}(?:\s*,\s*"filename"\s*:\s*"[^"]*")?\s*\})',
            text,
            re.DOTALL,
        )
        for block in unfenced:
            try:
                data = json.loads(block)
                name = data.get("name")
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name and isinstance(args, dict):
                    results.append((name, args))
            except Exception:
                continue
    return results


def start_new_session(session_name: str):
    """
    Called at startup to name the current session.
    Creates the session record in the database.
    """
    global _CURRENT_SESSION_NAME
    _CURRENT_SESSION_NAME = session_name
    if _db.is_ready():
        _db.create_session(session_name)


def set_log_callback(cb):
    global _LOG_CALLBACK
    _LOG_CALLBACK = cb


def _log_event(event_type: str, data: dict):
    if _LOG_CALLBACK:
        try:
            _LOG_CALLBACK(event_type, data)
        except Exception:
            pass


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Load the shared system prompt from disk (relative to this file)."""
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "You are a helpful, honest, on-premise AI assistant for industrial/confidential work."


# ─── Models (read from .env via config.py — swap without touching code) ──────
import config as _cfg
CODER_MODEL = _cfg.CODER_MODEL
MAIN_MODEL  = _cfg.MAIN_MODEL
FAST_MODEL  = _cfg.FAST_MODEL
IMAGE_MODEL = _cfg.IMAGE_MODEL

AVAILABLE_MODELS = {
    "code":       CODER_MODEL,
    "simple":     FAST_MODEL,
    "complex":    MAIN_MODEL,
    "rag_search": FAST_MODEL,   # RAG does the heavy work; model just formats
    "agent_task": MAIN_MODEL,   # Agent tasks use main model for planning
    "image":      IMAGE_MODEL,
}

@lru_cache(maxsize=32)
def resolve_installed_model(model_name: str) -> str:
    """
    Resolve requested model name against locally installed Ollama models.
    If exact match fails (e.g. qwen2.5vl:latest requested when qwen2.5vl:7b is installed),
    resolves to the installed variant with the same base name.
    """
    if not model_name:
        return model_name
    try:
        res = ollama.list()
        models_list = getattr(res, "models", []) or (res.get("models", []) if isinstance(res, dict) else [])
        installed = []
        for m in models_list:
            if hasattr(m, "model"):
                installed.append(m.model)
            elif isinstance(m, dict) and "name" in m:
                installed.append(m["name"])
            elif isinstance(m, dict) and "model" in m:
                installed.append(m["model"])
            else:
                installed.append(str(m))
        
        installed_map = {str(m).lower(): str(m) for m in installed if m}
        
        if model_name.lower() in installed_map:
            return installed_map[model_name.lower()]
            
        base = model_name.split(":")[0].lower()
        base_clean = base.replace("-", "")
        
        for inst_lower, inst_orig in installed_map.items():
            inst_base = inst_lower.split(":")[0]
            inst_base_clean = inst_base.replace("-", "")
            if inst_base == base or inst_base_clean == base_clean:
                return inst_orig
    except Exception:
        pass
    return model_name

# ─── Conversation memory ─────────────────────────────────────────────────────
HISTORY   = []
MAX_TURNS = _cfg.MEMORY_MAX_TURNS


def is_complex_command(query: str) -> bool:
    """True only for the explicit `/complex` command (not normal prompts)."""
    return bool(re.match(r"^\s*/complex(?:\s|$)", str(query or ""), re.I))


def strip_complex_command(query: str) -> str:
    """Remove the terminal command before handing the actual request to 14B."""
    return re.sub(r"^\s*/complex(?:\s+)?", "", str(query or ""), count=1, flags=re.I).strip()


def uses_direct_14b_mode(query: str = "") -> bool:
    """14B bypasses routing only when the user explicitly requests `/complex`."""
    return bool(_cfg.DIRECT_14B_MODE and is_complex_command(query))


def add_to_history(role: str, content: str):
    # Retain concise conversation memory.  The stable system pre-prompt is
    # supplied separately, so duplicated old text only costs context/RAM.
    text = str(content or "")[-_cfg.MEMORY_MAX_CHARS:]
    HISTORY.append({"role": role, "content": text})
    # 1 turn = 1 user prompt + 1 assistant response (2 messages).
    # Keep at least 20 messages (10 full conversational turns) so multi-turn context is not prematurely pruned.
    max_messages = max(MAX_TURNS * 2, 20)
    if len(HISTORY) > max_messages:
        del HISTORY[:len(HISTORY) - max_messages]
    if _CURRENT_SESSION_NAME and _db.is_ready():
        ok = _db.save_message(_CURRENT_SESSION_NAME, role, content)
        if not ok:
            print(f"[router] DB save failed for role={role}", flush=True)


def get_history() -> list:
    return HISTORY


def clear_history():
    global _CURRENT_SESSION_NAME
    HISTORY.clear()
    from datetime import datetime
    new_sess = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_new_session(new_sess)


# ══════════════════════════════════════════════════════════════════════════════
# SMART CLASSIFIER — 5 categories with confidence-based routing
# ══════════════════════════════════════════════════════════════════════════════

CLASSIFY_PROMPT = """You are an intelligent routing assistant for Agent OTG.
Read the user's message and classify it into EXACTLY one of 5 categories.

CATEGORIES:
- "code"       → ANY question involving programming, coding, scripting, or software development.
                 This includes: writing code, debugging, fixing errors/bugs, explaining code,
                 algorithms, data structures, functions, classes, APIs, SQL queries, regex,
                 shell/bash scripts, web dev (HTML/CSS/JS/React), backend (Flask/FastAPI/Django),
                 system design involving code, performance of code, code reviews, unit tests.
                 Trigger words: code, script, function, program, bug, error, debug, fix,
                 implement, build, class, method, algorithm, API, SQL, query, Python, JavaScript,
                 TypeScript, Java, C++, Rust, Go, React, HTML, CSS, Flask, FastAPI, Django,
                 loop, recursion, array, list, dictionary, compile, runtime, syntax, library,
                 framework, package, import, module, git, docker, database schema, ORM.
                 IMPORTANT: If the topic involves writing or explaining ANY code at all, use "code".
- "simple"     → Greetings, small talk, quick factual one-liners, easy math (< 8 words usually)
                 Examples: "hello", "what time is it", "what is 2+2", "thank you"
- "complex"    → Non-code essays, analysis, research, explanations of concepts (NOT code),
                 multi-step reasoning, general knowledge questions requiring depth.
                 Use this ONLY when there is clearly NO programming/coding content.
- "rag_search" → User wants to search, query, or ask about UPLOADED documents / knowledge base
                 Signals: "what does the document say", "from my files", "search the KB",
                 "what does the report say", "according to the uploaded", "find in docs",
                 "what does it say about", "from the data", "in the file"
- "agent_task" → User wants to CREATE or GENERATE a FILE (PDF, Word, Excel, CSV, PowerPoint)
                 or needs multi-tool autonomous work
                 Signals: "create a PDF", "generate a Word doc", "make a spreadsheet",
                 "write a report and save it", "generate a file", "create an Excel sheet"

RULES (in strict priority order):
1. "code" is the HIGHEST priority category. If there is ANY programming content in the request,
   classify it as "code" — even if it also asks for explanation. Never downgrade a coding question
   to "complex" or "simple".
2. "agent_task" when user explicitly wants a FILE to be created/saved/exported (no code asked)
3. "rag_search" when user references existing documents/files they've uploaded
4. "simple" ONLY for greetings, trivial math, very short factual non-technical questions
5. "complex" for everything else with no coding content
6. "is_multi_part" = true ONLY if message has 2+ clearly distinct, unrelated requests
7. "has_code_and_explain" = true when BOTH writing code AND explaining it is requested

Reply with ONLY this JSON — nothing else:
{"category": "code"|"simple"|"complex"|"rag_search"|"agent_task", "is_multi_part": true|false, "has_code_and_explain": true|false, "reason": "one short sentence", "confidence": 0.0-1.0}
"""

# ── Keyword safety-nets ───────────────────────────────────────────────────────
_EXPLAIN_KEYWORDS = [
    "explain", "walk me through", "walk through", "how it works",
    "step by step", "step-by-step", "break it down", "describe",
    "elaborate", "detail", "breakdown", "walkthrough",
]
# ── Expanded keyword sets ────────────────────────────────────────────────────
# ── Expanded keyword sets ────────────────────────────────────────────────────
_CODE_KEYWORDS = [
    # Languages & compilers
    "python", "javascript", "typescript", "java", "c++", "c#", "rust",
    "golang", "ruby", "kotlin", "swift", "scala", "matlab",
    "bash", "shell", "powershell", "perl", "php",
    # Web & frameworks
    "react", "vue", "angular", "html", "css", "sass", "scss",
    "flask", "fastapi", "django", "expressjs", "spring boot", "rails",
    "next.js", "nextjs", "nuxt", "svelte", "node", "nodejs",
    # Data & DevOps
    "sql", "sqlite", "postgresql", "mysql", "mongodb", "redis",
    "dockerfile", "docker-compose", "kubernetes", "ci/cd", "github actions",
    "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn",
    # Programming terminology (unambiguous)
    "regex", "regular expression", "recursion", "recursive", "binary search",
    "quicksort", "bubblesort", "mergesort", "pseudocode", "refactor",
    "polymorphism", "encapsulation", "docstring",
    "stack trace", "traceback", "nullpointer", "typeerror", "syntaxerror",
    "indexerror", "keyerror", "memory leak", "segfault", "deadlock"
]

_CODE_KEYWORD_PHRASES = [
    # Phrases that are unambiguous coding requests (formal, casual, or conversational)
    "write a function", "write a script", "write a program", "write the code",
    "write code", "give me the code", "show me the code", "show me how to code",
    "how do i code", "how to code", "how to write a function", "how to implement",
    "give me a function", "create a function", "create a class",
    "python code", "javascript code", "java code", "c++ code", "sql query",
    "code me", "code for", "script for", "script to", "code to", "program to",
    "write python", "write javascript", "write java", "write cpp", "write sql",
    "write bash", "write shell", "write html", "write react", "write api",
    "how to solve in python", "how to write in python", "how to code in",
    "bug in my code", "fix my code", "debug my code", "error in my script",
    "fix this error", "fix the bug", "unit test", "unit tests", "for loop", "while loop",
    "linked list", "binary tree", "time complexity", "space complexity",
    "code a website", "code an app", "create an api", "build a backend",
    "build an api", "build a website with code", "code snippet"
]

_CODE_SYNTAX_PATTERNS = [
    r"\bdef\s+[a-zA-Z_]\w*\s*\(",
    r"\bclass\s+[a-zA-Z_]\w*[:\(]",
    r"^\s*import\s+[a-zA-Z_]\w*",
    r"\bimport\s+(?:sys|os|re|math|json|requests|numpy|pandas|torch|csv|datetime|typing|collections|time)\b",
    r"\bfrom\s+[a-zA-Z_]\w*\s+import\b",
    r"\bfunction\s+[a-zA-Z_]\w*\s*\(",
    r"\bconsole\.log\(",
    r"\bpublic\s+static\s+void\b",
    r"\bSELECT\s+.+?\s+FROM\b",
    r"\b[a-zA-Z_]\w*\s*=\s*(?:lambda\b|\[|\{)",
]

_SIMPLE_KEYWORDS = [
    "hello", "hi ", "hey ", "thanks", "thank you", "good morning",
    "good afternoon", "good evening", "bye", "goodbye", "how are you",
    "what is your name", "who are you", "what time", "what day",
]

_FILE_CREATION_KEYWORDS = [
    # PDF
    "create a pdf", "generate a pdf", "make a pdf", "write a pdf", "build a pdf",
    "pdf on", "pdf about", "convert to pdf", "export to pdf", "save as pdf", "save to pdf",
    "download pdf", "give me a pdf", "make pdf", "create pdf", "generate pdf", "draft a pdf",
    "pdf file", "put it in a pdf",
    # DOC / Word
    "create a word", "generate a word", "make a word", "write a word", "give me a word", "draft a word", "prepare a word",
    "create a doc", "generate a doc", "make a doc", "write a doc", "draft a doc", "prepare a doc",
    "create a docx", "generate a docx", "make a docx", "write a docx", "draft a docx", "prepare a docx",
    "docx of", "word document", "word doc", "word file", "doc file", "docx file", "ms word", "microsoft word",
    "make a word file", "create word file", "generate word file", "write word file", "draft word file",
    "save as word", "save to word", "convert to word", "export to word", "in word format", "put it in a word",
    # Excel / CSV / Spreadsheet
    "create an excel", "make an excel", "generate an excel", "excel sheet", "excel file",
    "generate a spreadsheet", "create a spreadsheet", "make a spreadsheet", "spreadsheet file",
    "export to excel", "save as excel", "put it in excel",
    "create a csv", "make a csv", "generate a csv", "csv file", "csv of",
    "export to csv", "save as csv",
    # Presentation / PPT / Slides
    "create a ppt", "make a ppt", "generate a ppt", "draft a ppt", "prepare a ppt", "give me a ppt",
    "ppt on", "ppt about", "ppt for", "ppt regarding", "ppt of", "pptx file", "ppt file",
    "create a presentation", "make a presentation", "generate a presentation", "presentation on", "presentation about",
    "presentation for", "presentation regarding", "prepare a presentation", "give me a presentation",
    "make a powerpoint", "create a powerpoint", "generate a powerpoint", "powerpoint on", "powerpoint about",
    "create slides", "make slides", "generate slides", "prepare slides", "slides on", "slides about", "slides for",
    "slide deck", "pitch deck", "create a slide deck", "make a slide deck", "generate a slide deck",
    "put it in a ppt", "save as ppt", "save as pptx", "convert to ppt",
    # JSON
    "create a json", "generate a json file", "generate a json", "json file",
    # Generic file creation
    "write a report and save", "create and save", "make and save",
    "make a file", "generate a file", "export a file", "save as file", "save to file",
]

_RAG_KEYWORDS = [
    "what does the document", "what does my document", "what does the file",
    "from the uploaded", "in my files", "search the knowledge base",
    "from the knowledge base", "what does the report say", "search my docs",
    "what does the uploaded document", "what does the uploaded file",
    "what does it say about", "find in the document", "look up in",
    "from the uploaded file", "what the document says",
    "according to the document", "according to the report",
    "what does my report", "what does my file",
    "summarize the uploaded", "summarize the document", "summarize my document",
    "summarize the pdf", "summarize the report", "summarize the file",
    "what is in the file", "what is in the document", "what is in my files",
    "find in my documents", "what does the uploaded agreement say",
    "what does the contract say", "extract from the document",
    "analyze the pdf", "analyze the document", "analyze the file", "analyze the report",
    "explain the pdf", "read the pdf", "read the document", "review the pdf",
]

_REALTIME_PATTERNS = [
    r"\b(?:what(?:\s+is|\s*'s)?\s+(?:the\s+)?(?:current\s+)?(?:time|date|today'?s\s+date|clock))\b",
    r"\b(?:what\s+time\s+is\s+it|what\s+date\s+is\s+it|current\s+time|current\s+date|tell\s+me\s+the\s+time|tell\s+me\s+the\s+date)\b",
    r"\b(?:current|latest|live|real-?time|today'?s|tomorrow'?s|this\s+(?:week|month|year))\s+(?:weather|forecast|temperature|news|headlines?|stock(?:\s+price)?|share\s+price|exchange\s+rate|market|score|match|traffic|price|date|time|month|day|ceo|president|election)\b",
    r"\b(?:weather\s+(?:today|now|currently)|news\s+(?:today|now|currently)|(?:latest|breaking)\s+news|stock\s+price\s+(?:today|now)|exchange\s+rate\s+(?:today|now))\b",
    r"\b(?:weather|forecast|temperature|news|headlines?|stock(?:\s+price)?|share\s+price|exchange\s+rate|market|score|match|traffic|price)\b.*\b(?:today|tomorrow|now|currently|current|latest|live|real-?time|this\s+(?:week|month|year))\b",
    r"\b(?:today|tomorrow|now|currently|current|latest|live|real-?time|this\s+(?:week|month|year))\b.*\b(?:weather|forecast|temperature|news|headlines?|stock(?:\s+price)?|share\s+price|exchange\s+rate|market|score|match|traffic|price)\b",
    r"\b(?:who|which\s+(?:team|candidate|party|company))\s+(?:is|are|won|wins|leading|currently)\b.*\b(?:president|prime\s+minister|ceo|leader|election|match|game|league|tournament|poll)\b",
    r"\b(?:bitcoin|crypto(?:currency)?|gold|oil|fuel|petrol|diesel|flight|hotel|ticket|concert)\b.*\b(?:price|rate|cost|today|now|current|latest|live)\b",
    r"\b(?:current|latest|live|real-?time|today|tomorrow|now)\b.*\b(?:availability|availability|schedule|timetable|departure|arrival|result|results|ranking|rankings)\b",
]

OFFLINE_REALTIME_MESSAGE = (
    "This request needs live or time-sensitive data. Agent OTG is an offline local model and cannot access or verify current information. "
    "Please provide the relevant data or a dated source, and I can analyze it professionally."
)

_DRY_RUN_PATTERNS = [
    r"\bdry[\s-]?run\b", r"\bexecution\s+trace\b", r"\btrace\s+(?:the\s+)?(?:code|execution)\b",
    r"\bwalk\s+through\s+(?:the\s+)?(?:execution|dry\s*run)\b",
]


def _is_direct_tool_request(query: str) -> bool:
    """Return True for deterministic local utility requests.

    These requests must enter the agent workflow: the ordinary chat path does
    not send a large tool schema to Ollama, so relying on a model to emit an
    embedded JSON call made tools appear to do nothing in the web UI.
    """
    q = (query or "").lower()
    patterns = (
        r"\b(?:list|show|what) (?:my |the )?(?:generated )?files\b",
        r"\b(?:extract text|read text from|read)\b.*\bpdf\b",
        r"\b(?:page count|how many pages|count pages|number of pages)\b",
        r"\b(?:merge|combine|join) pdfs?\b",
        r"\b(?:split|rotate) (?:a )?pdf\b",
        r"\b(?:ocr|extract text from image|read text from image)\b",
        r"\b(?:generate|create|make|draw) (?:an )?image\b",
    )
    return any(re.search(pattern, q, re.I) for pattern in patterns)


def is_realtime_query(query: str) -> bool:
    q = (query or "").lower().strip()
    return any(re.search(pat, q) for pat in _REALTIME_PATTERNS)


def _looks_like_code_and_explain(query: str) -> bool:
    """Reserve the multi-step code pipeline for an explicitly requested dry run.

    Normal coding questions, including "write and explain", go directly to the
    coder model in one response. This avoids an unnecessary second model call.
    """
    q = query.lower()
    return _hard_code_check(query) and any(re.search(pattern, q) for pattern in _DRY_RUN_PATTERNS)


def _hard_code_check(query: str) -> bool:
    """
    Layer-1 hard check: returns True if the query is DEFINITELY about code/programming.
    This is checked BEFORE the LLM classifier and cannot be overridden by LLM output.
    Checks syntax patterns, multi-word phrases, and single keywords with word boundaries.
    """
    q = query.lower().strip()
    # Conceptual comparisons/trade-offs (e.g. "compare PostgreSQL vs MongoDB", "pros and cons of microservices")
    is_comparison = bool(re.search(r"\b(?:compare|comparison|versus|vs\.?|pros\s+and\s+cons|difference\s+between)\b", q))
    has_code_action = bool(re.search(r"\b(?:write\s+(?:a|an|the|me|some)?\s*(?:code|query|queries|script|program|function|class|test|sql)|implement|debug|fix\s+(?:a|the|my)?\s*(?:code|bug|error)|coding)\b", q))
    if is_comparison and not has_code_action:
        return False

    if any(re.search(pat, query, re.I) for pat in _CODE_SYNTAX_PATTERNS):
        return True
    if any(phrase in q for phrase in _CODE_KEYWORD_PHRASES):
        return True
    for k in _CODE_KEYWORDS:
        if re.search(r"\b" + re.escape(k) + r"\b", q):
            return True
    return False


def _is_rag_query(query: str) -> bool:
    """Detect if query is asking from/about existing or uploaded documents."""
    q = query.lower().strip()
    if any(k in q for k in _RAG_KEYWORDS):
        return True
    # Frontend web UI markers (file upload + knowledge-base Q&A)
    if re.search(
        r"\b(?:selected file context|uploaded knowledge base|ground the answer in those files|"
        r"knowledge base as the primary source|answer from the uploaded|indexed into chromadb)\b",
        q, re.I,
    ):
        return True
    if re.search(
        r"\b(?:what\s+does|read|extract|find|search|query|look\s+up|tell\s+me\s+about|summarize|explain|analyze|break\s+down|review)\b(?:\s+\w+){0,6}\s+\b(?:document|doc|pdf|file|notes|report|paper|data)\b"
        r"|\b(?:according\s+to|from|in)\s+(?:the\s+)?(?:uploaded|attached|my|this)?\s*(?:document|doc|pdf|file|report|knowledge\s+base|notes)\b"
        r"|\b(?:summarize|analyze|explain|read|review|examine)\s+(?:the|this|my|a|an)?\s*(?:pdf|document|file|report|notes|paper)\b",
        q, re.I
    ):
        return True
    return False


def _quick_classify(query: str) -> str | None:
    """
    Fast keyword‑based pre‑screening for obvious cases.
    Returns a category string or None to fall through to LLM classification.
    """
    q = query.lower().strip()
    word_count = len(q.split())

    # RAG search check takes precedence over generic keywords
    if _is_rag_query(query):
        return "rag_search"

    # Simple: short greetings / small talk matching simple keywords
    if word_count <= 6 and any(k in q for k in _SIMPLE_KEYWORDS):
        return "simple"

    # File creation → agent_task
    if any(k in q for k in _FILE_CREATION_KEYWORDS):
        return "agent_task"

    return None  # Let the LLM decide


def _split_multi_actions(query: str) -> list[str]:
    """Split multi-action queries into self-contained sub-task strings.
    Handles numbered items, bullet items, and sentence delimiters (period, newline, semicolon, conjunction).
    """
    q = (query or "").strip()
    if not q:
        return []

    # 1. Numbered items (e.g. 1) ... 2) ... or 1. ... 2. ...)
    if re.search(r"(?:^|\s)(?:1[\).]|first[,:])\s+.*?(?:\s+(?:2[\).]|second[,:]))\s+", q, re.I | re.DOTALL):
        parts = [p.strip(" ,.;") for p in re.split(r"(?:^|\s+)\d+[\).]\s+", q) if p.strip(" ,.;")]
        if len(parts) >= 2:
            return parts

    # 2. Bulleted items (e.g. - item 1 \n - item 2)
    if re.search(r"(?:^|\n)\s*[-*•]\s+.*?\n\s*[-*•]\s+", q):
        parts = [p.strip(" ,.;") for p in re.split(r"(?:^|\n)\s*[-*•]\s+", q) if p.strip(" ,.;")]
        if len(parts) >= 2:
            return parts

    # Guard: if query is a single file creation with descriptive clauses (e.g. "and make it 5 slides"),
    # do not split on conversational conjunctions unless explicitly separated by periods, semicolons, or newlines
    file_intent = detect_file_intent(q)
    if file_intent.get("file_format") and not (";" in q or "\n" in q or re.search(r"[.!?]\s+(?=[A-Z])", q)):
        return [q]

    # 3. Delimited by newlines, semicolons, periods followed by action/question verbs, or conjunctions
    split_pattern = (
        r"(?:\n+|;"
        r"|(?<=[.!?])\s+(?=(?:also\s+)?(?:what|how|why|who|when|where|tell|show|give|write|create|generate|make|draft|explain|compare|summarize|analyze|draw|implement|build|add|test|debug|fix|refactor|review|deploy|optimize|design)\b)"
        r"|,\s*(?=(?:also\s+)?(?:write|create|generate|make|give|draft|explain|draw|implement|build|add|test|debug|fix|refactor|review|deploy|optimize|design)\b)"
        r"|\s+and\s+(?=(?:also\s+)?(?:write|create|generate|make|give|draft|explain|draw|implement|build|add|test|debug|fix|refactor|review|deploy|optimize|design)\b))"
    )
    parts = [p.strip(" ,.;") for p in re.split(split_pattern, q, flags=re.I) if p.strip(" ,.;")]
    if len(parts) >= 2:
        return parts

    return [q]


def classify_question(query: str) -> dict:
    """
    Classify a user query into one of 5 routing categories.

    Enforcement order:
      1. File intent check  — explicit document generation requests always route to agent_task.
      2. _hard_code_check() — deterministic keyword check for non-file programming questions.
      3. _quick_classify()  — fast keyword routing for simple and RAG cases.
      4. LLM classification — for ambiguous queries.
      5. Post-LLM safety-net — safety checks on LLM output.
    """
    # `/complex` explicitly selects Qwen 14B: no classifier call, no task
    # splitting, and no model routing.  All ordinary prompts use the router.
    if uses_direct_14b_mode(query):
        result_data = {
            "category": "complex", "model": _cfg.COMPLEX_MODEL,
            "is_multi_part": False, "has_code_and_explain": False,
            "reason": "Explicit /complex command — Qwen 14B direct mode", "confidence": 1.0,
            "routing_method": "direct_14b",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    recent = HISTORY[-4:]
    context_text = "\n".join(f"{m['role']}: {m['content'][:120]}" for m in recent)

    # ── Layer -2: deterministic tool requests ──────────────────────────────
    if _is_direct_tool_request(query):
        result_data = {
            "category": "agent_task", "model": pick_model("agent_task"),
            "is_multi_part": False, "has_code_and_explain": False,
            "reason": "Local utility tool request → routed to agent workflow",
            "confidence": 1.0, "routing_method": "tool_pattern_check",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Layer -1: RAG pre-screen — requests to search/query/summarize existing documents ──
    if _is_rag_query(query):
        result_data = {
            "category":             "rag_search",
            "model":                pick_model("rag_search"),
            "is_multi_part":        False,
            "has_code_and_explain": False,
            "reason":               "RAG query pattern detected → routed to rag_search",
            "confidence":           1.0,
            "routing_method":       "rag_pattern_check",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Layer 0: File creation intent check — requests for document artifacts ──
    # RAG/document Q&A must win over incidental words like "pdf" in a question.
    file_intent = detect_file_intent(query)
    if (
        not _is_rag_query(query)
        and (file_intent.get("file_format") or any(k in query.lower() for k in _FILE_CREATION_KEYWORDS))
    ):
        multi_parts = _split_multi_actions(query)
        is_multi = len(multi_parts) >= 2
        result_data = {
            "category":             "agent_task",
            "model":                pick_model("agent_task"),
            "is_multi_part":        is_multi,
            "has_code_and_explain": False,
            "reason":               "Multi-part request containing document artifact creation" if is_multi else "Explicit file creation request → routed to agent_task",
            "confidence":           1.0,
            "routing_method":       "file_intent_multi" if is_multi else "file_intent_check",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Layer 1: HARD code check — runs for pure code requests without file intent ────
    if _hard_code_check(query):
        has_code_explain = _looks_like_code_and_explain(query)
        code_parts = _split_multi_actions(query)
        # “write code and explain it” is one normal coder response, not a
        # multi-agent/dry-run workflow. Split only distinct implementation work.
        explanation_only = len(code_parts) == 2 and re.match(r"^(?:also\s+)?(?:explain|describe|document)\b", code_parts[-1], re.I)
        is_multi_part = len(code_parts) >= 2 and not explanation_only
        result_data = {
            "category":             "code",
            "model":                pick_model("code"),
            "is_multi_part":        is_multi_part,
            "has_code_and_explain": has_code_explain,
            "reason":               "Hard code keyword match — routed to CODER_MODEL",
            "confidence":           1.0,
            "routing_method":       "hard_code_check",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Layer 2: fast keyword pre-screen for non-code obvious cases ───────────
    quick = _quick_classify(query)
    if quick:
        result_data = {
            "category":             quick,
            "model":                pick_model(quick),
            "is_multi_part":        False,
            "has_code_and_explain": False,
            "reason":               f"Fast keyword routing → {quick}",
            "confidence":           0.95,
            "routing_method":       "keyword",
        }
        _log_event("classify", {"query": query, "result": result_data})
        return result_data

    # ── Layer 3: LLM classification ───────────────────────────────────────────
    category         = "complex"
    is_multi_part    = False
    has_code_explain = False
    reason           = ""
    confidence       = 0.5

    try:
        response = ollama.chat(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user",   "content": f"Recent chat:\n{context_text}\n\nNew message: {query}"},
            ],
            format="json",
            options={"temperature": 0},
        )
        raw = response["message"]["content"] if isinstance(response, dict) else response.message.content
        parsed           = json.loads(raw)
        category         = parsed.get("category", "complex")
        is_multi_part    = bool(parsed.get("is_multi_part", False))
        has_code_explain = bool(parsed.get("has_code_and_explain", False))
        reason           = parsed.get("reason", "")
        confidence       = float(parsed.get("confidence", 0.7))

        # ── Layer 4: Post-LLM safety-net ─────────────────────────────────────
        if _is_rag_query(query) and category != "rag_search":
            category = "rag_search"
            reason   = "Post-LLM override: RAG query pattern detected"
            confidence = 0.95
        elif category in ("simple", "complex") and _hard_code_check(query):
            category = "code"
            reason   = "Post-LLM override: code keywords detected despite non-code LLM output"
            confidence = 0.9
        else:
            file_intent = detect_file_intent(query)
            if file_intent.get("file_format") and category != "agent_task":
                category = "agent_task"
                reason = "Post-LLM override: file intent detected"
                confidence = 0.95
        if not has_code_explain and _looks_like_code_and_explain(query):
            has_code_explain = True
        if _hard_code_check(query) and category != "agent_task":
            category = "code"
            is_multi_part = len(_split_multi_actions(query)) >= 2

    except Exception:
        # ── Layer 5: keyword fallback (LLM call failed) ───────────────────────
        q_lower = query.lower()
        if _is_rag_query(query):
            category = "rag_search"
        elif detect_file_intent(query).get("file_format") or any(k in q_lower for k in _FILE_CREATION_KEYWORDS):
            category = "agent_task"
        elif _hard_code_check(query):   # re-use the same hard check
            category = "code"
        elif len(query.split()) <= 8:
            category = "simple"
        else:
            category = "complex"
        has_code_explain = _looks_like_code_and_explain(query)
        is_multi_part    = len(_split_multi_actions(query)) >= 2 if _hard_code_check(query) else (" and " in query.lower() and len(query) > 40)
        reason           = "Keyword fallback (LLM classifier unavailable)"
        confidence       = 0.6

    # Validate category
    if category not in AVAILABLE_MODELS:
        category = "complex"

    result_data = {
        "category":             category,
        "model":                pick_model(category),
        "is_multi_part":        is_multi_part,
        "has_code_and_explain": has_code_explain,
        "reason":               reason,
        "confidence":           confidence,
        "routing_method":       "llm",
    }
    _log_event("classify", {"query": query, "result": result_data})
    return result_data


def pick_model(category: str) -> str:
    return AVAILABLE_MODELS.get(category, MAIN_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers to safely read fields from ollama response (dict or object)
# ══════════════════════════════════════════════════════════════════════════════

def _get(obj, *keys, default=None):
    """Safely get nested attribute/key from ollama response objects or dicts."""
    for key in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj if obj is not None else default


def _msg_to_dict(msg) -> dict:
    """Normalise an ollama message (could be dict or object) to a plain dict."""
    if isinstance(msg, dict):
        return msg
    d = {
        "role":    getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", "") or "",
    }
    tc = getattr(msg, "tool_calls", None)
    if tc:
        d["tool_calls"] = tc
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Run a model on a single task (with memory)
# ══════════════════════════════════════════════════════════════════════════════

def _build_messages(query: str) -> list:
    sys_prompt = load_system_prompt()
    if uses_direct_14b_mode(query):
        sys_prompt += """

Complex-task response standard:
- Reason carefully before answering, but provide only the useful final result.
- State assumptions and uncertainty instead of inventing facts.
- Use clear Markdown headings, short paragraphs, and ordered steps when they improve readability.
- For code: identify the language, use a complete fenced code block, preserve indentation, and include only runnable code plus concise usage notes.
- For multi-part work: answer every requested part in the same order and label each part clearly.
"""
    return [{"role": "system", "content": sys_prompt}] + HISTORY + [{"role": "user", "content": strip_complex_command(query)}]


def get_full_answer(model: str, query: str) -> str:
    model = resolve_installed_model(model)
    if is_realtime_query(query):
        add_to_history("user", query)
        add_to_history("assistant", OFFLINE_REALTIME_MESSAGE)
        return OFFLINE_REALTIME_MESSAGE

    from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
    _log_event("model_start", {"model": model, "query": query})
    messages = _build_messages(query)

    try:
        # Tool schemas are large and slow local inference considerably.  The
        # terminal sends explicit document/tool requests to the deterministic
        # agent path, so ordinary answers do not need them in their pre-prompt.
        response = ollama.chat(model=model, messages=messages,
                               options={"num_ctx": _cfg.OLLAMA_NUM_CTX,
                                        "num_predict": _cfg.OLLAMA_NUM_PREDICT})
    except Exception as exc:
        _log_event("model_done", {"model": model, "answer": f"[Error] {exc}"})
        err_msg = f"Error calling model '{model}': {exc}"
        add_to_history("user", query)
        add_to_history("assistant", err_msg)
        return err_msg

    tool_calls = _get(response, "message", "tool_calls") or []

    if tool_calls:
        messages.append(_msg_to_dict(_get(response, "message")))

        for tool_call in tool_calls:
            func_name = _get(tool_call, "function", "name")
            args      = _get(tool_call, "function", "arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if func_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"Error running tool '{func_name}': {e}"
            else:
                result = f"Error: tool '{func_name}' not registered."

            _log_event("tool_call", {"tool": func_name, "args": args, "result": str(result)})
            messages.append({"role": "tool", "content": str(result)})

        try:
            response = ollama.chat(model=model, messages=messages)
        except Exception as exc:
            answer = f"Tools executed. Error generating follow-up: {exc}"
            add_to_history("user", query)
            add_to_history("assistant", answer)
            return answer

    answer = _get(response, "message", "content") or ""

    # Fallback: embedded JSON tool calls in text
    if not tool_calls:
        embedded = _find_embedded_tool_calls(answer)
        for func_name, args in embedded:
            if func_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"Error running tool '{func_name}': {e}"
                _log_event("tool_call", {"tool": func_name, "args": args, "result": str(result)})
                answer += f"\n\n---\n🛠️ **Auto-executed tool** `{func_name}`:\n📄 {result}\n---"

    add_to_history("user",      query)
    add_to_history("assistant", answer)
    _log_event("model_done", {"model": model, "answer": answer})
    return answer


def stream_answer(model: str, query: str):
    model = resolve_installed_model(model)
    if is_realtime_query(query):
        add_to_history("user", query)
        add_to_history("assistant", OFFLINE_REALTIME_MESSAGE)
        yield OFFLINE_REALTIME_MESSAGE
        return

    from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
    _log_event("stream_start", {"model": model, "query": query})
    messages    = _build_messages(query)
    full_answer = ""

    response_msg: dict = {"role": "assistant", "content": ""}
    tool_calls:   list = []

    try:
        for chunk in ollama.chat(model=model, messages=messages, stream=True,
                                 options={"num_ctx": _cfg.OLLAMA_NUM_CTX,
                                          "num_predict": _cfg.OLLAMA_NUM_PREDICT}):
            chunk_tcs = _get(chunk, "message", "tool_calls") or []
            if chunk_tcs:
                tool_calls.extend(chunk_tcs)

            token = _get(chunk, "message", "content") or ""
            if token:
                full_answer += token
                response_msg["content"] += token
                yield token
    except Exception as exc:
        err = f"\n[Error during streaming: {exc}]"
        yield err
        full_answer += err
        add_to_history("user", query)
        add_to_history("assistant", full_answer)
        _log_event("stream_done", {"model": model, "answer": full_answer})
        return

    if tool_calls:
        response_msg["tool_calls"] = tool_calls
        messages.append(response_msg)

        for tool_call in tool_calls:
            func_name = _get(tool_call, "function", "name")
            args      = _get(tool_call, "function", "arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            yield f"\n\n[System: Calling tool '{func_name}'...]\n"

            if func_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"Error running tool '{func_name}': {e}"
            else:
                result = f"Error: tool '{func_name}' not registered."

            _log_event("tool_call", {"tool": func_name, "args": args, "result": str(result)})
            messages.append({"role": "tool", "content": str(result)})

        yield "\n"
        try:
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                token = _get(chunk, "message", "content") or ""
                if token:
                    full_answer += token
                    yield token
        except Exception as exc:
            err = f"\n[Error in follow-up stream: {exc}]"
            yield err
            full_answer += err

    elif not tool_calls:
        embedded = _find_embedded_tool_calls(full_answer)
        for func_name, args in embedded:
            if func_name in TOOL_FUNCTIONS:
                yield f"\n\n[System: Auto-executing tool '{func_name}'...]\n"
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"Error running tool '{func_name}': {e}"
                _log_event("tool_call", {"tool": func_name, "args": args, "result": str(result)})
                yield f"\n---\n🛠️ **Auto-executed tool** `{func_name}`:\n📄 {result}\n---\n"
                full_answer += f"\n\n[Auto-executed '{func_name}': {result}]"

    add_to_history("user",      query)
    add_to_history("assistant", full_answer)
    _log_event("stream_done", {"model": model, "answer": full_answer})


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE: vision model
# ══════════════════════════════════════════════════════════════════════════════

def ask_image(image_b64: str, query: str) -> str:
    model = resolve_installed_model(IMAGE_MODEL)
    _log_event("image_start", {"model": model, "query": query, "image_size_b64": len(image_b64)})
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role":    "user",
                "content": query,
                "images":  [image_b64],
            }],
        )
        answer = _get(response, "message", "content") or ""
    except Exception as exc:
        answer = f"Error analyzing image: {exc}"

    add_to_history("user",      f"[image attached] {query}")
    add_to_history("assistant", answer)
    _log_event("image_done", {"model": model, "answer": answer})
    return answer


def stream_image_answer(image_b64: str, query: str):
    model = resolve_installed_model(IMAGE_MODEL)
    _log_event("image_start", {"model": model, "query": query, "image_size_b64": len(image_b64)})
    full_answer = ""
    try:
        for chunk in ollama.chat(
            model=model,
            messages=[{
                "role":    "user",
                "content": query,
                "images":  [image_b64],
            }],
            stream=True,
        ):
            token = _get(chunk, "message", "content") or ""
            if token:
                full_answer += token
                yield token
    except Exception as exc:
        err = f"[Error: {exc}]"
        full_answer += err
        yield err

    add_to_history("user",      f"[image attached] {query}")
    add_to_history("assistant", full_answer)
    _log_event("image_done", {"model": model, "answer": full_answer})


# ══════════════════════════════════════════════════════════════════════════════
# Sequential pipeline: "code + explain" in two ordered steps
# ══════════════════════════════════════════════════════════════════════════════

def run_sequential_tasks(query: str) -> list:
    results = []

    code_prompt = (
        f'The user asked: "{query}"\n\n'
        "Your job for this step: write ONLY the code. "
        "Do not explain it yet — just provide clean, well-commented code."
    )
    code_answer = get_full_answer(CODER_MODEL, code_prompt)
    results.append({
        "step":       1,
        "label":      "Code Generation",
        "model_used": CODER_MODEL,
        "category":   "code",
        "answer":     code_answer,
    })

    explain_prompt = (
        "Now explain the code you just wrote above, step by step. "
        "Be clear and beginner-friendly. Cover what each part does and why."
    )
    explain_answer = get_full_answer(CODER_MODEL, explain_prompt)
    results.append({
        "step":       2,
        "label":      "Step-by-Step Explanation",
        "model_used": CODER_MODEL,
        "category":   "code",
        "answer":     explain_answer,
    })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Split a multi-part question into independent sub-tasks
# ══════════════════════════════════════════════════════════════════════════════

def break_into_tasks(query: str) -> list:
    """Split a multi-prompt / multi-part query into independent self-contained sub-tasks.
    Each sub-task is independently classified and routed to its dedicated model
    (e.g., coding requests -> CODER_MODEL, simple/facts -> FAST_MODEL, complex/essay -> MAIN_MODEL).
    """
    prompt = f"""The user's message contains MULTIPLE distinct requests bundled together.
Split it into separate, self-contained tasks — do NOT merge them back into one.
Each task's "task" field must be a full standalone instruction (repeat any shared
context so each task makes sense on its own).

Reply with ONLY a JSON list, nothing else, like:
[{{"label": "short title", "task": "the exact self-contained sub-question", "category": "code" | "simple" | "complex" | "agent_task" | "rag_search"}}]

There must be at least 2 items in the list if the message really contains multiple asks.

Message: {query}
"""
    raw_tasks = None
    try:
        response = ollama.chat(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        raw_content = response["message"]["content"] if isinstance(response, dict) else response.message.content
        raw = json.loads(raw_content)
        raw_tasks = raw.get("tasks", raw) if isinstance(raw, dict) else raw
    except Exception:
        raw_tasks = None

    def _classify_subtask(task_text: str, suggested_cat: str = "") -> tuple[str, str]:
        t_str = task_text.strip()
        cat = suggested_cat.strip().lower()
        if _hard_code_check(t_str):
            category = "code"
        elif _is_rag_query(t_str):
            category = "rag_search"
        elif detect_file_intent(t_str).get("file_format") or any(k in t_str.lower() for k in _FILE_CREATION_KEYWORDS):
            category = "agent_task"
        elif cat in AVAILABLE_MODELS:
            category = cat
        else:
            quick = _quick_classify(t_str)
            category = quick if quick else "complex"
        return category, pick_model(category)

    if isinstance(raw_tasks, list) and len(raw_tasks) >= 2:
        tasks = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            label    = str(t.get("label", "Sub-task")).strip() or "Sub-task"
            task_str = str(t.get("task", query)).strip() or query
            cat_suggestion = str(t.get("category", "")).strip()
            category, model = _classify_subtask(task_str, cat_suggestion)
            tasks.append({
                "label":    label,
                "task":     task_str,
                "category": category,
                "model":    model,
            })
        if len(tasks) >= 2:
            return tasks

    # Deterministic fallback split if LLM response failed or didn't give 2+ tasks
    sub_parts = _split_multi_actions(query)
    if len(sub_parts) >= 2:
        tasks = []
        for part in sub_parts:
            category, model = _classify_subtask(part)
            tasks.append({
                "label":    part[:60],
                "task":     part,
                "category": category,
                "model":    model,
            })
        return tasks

    category, model = _classify_subtask(query)
    return [{"label": "Full request", "task": query, "category": category, "model": model}]
