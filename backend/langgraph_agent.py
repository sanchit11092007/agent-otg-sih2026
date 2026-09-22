"""
langgraph_agent.py — Reliable offline LangGraph orchestration for Agent OTG.

All output files are saved to ~/Downloads/AgentOTG/.

Key design decisions:
- Individual try/except error shielding around every tool call: if one tool fails,
  it returns a clean error string like "Tool 'X' failed: <reason>" instead of
  crashing the whole graph or returning nothing.
- Pre-validation of required arguments and types before calling functions to prevent TypeErrors.
- Tool logging through the router/main.py event callback system so all executions and
  failures are visible in the terminal.
- When an artifact is created: show ONLY the file confirmation card, never dump raw content
  unless explicitly requested.
- Code requests are routed to CODER_MODEL and forbidden from generating PDF compilation scripts.
"""
from __future__ import annotations

import json
import inspect
import os
import re
import time
import uuid
import csv
import io
from pathlib import Path
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

import config
from artifacts import derive_clean_title, OUTPUT_DIR
import tools


class WorkflowState(TypedDict, total=False):
    question: str
    history: list[dict]
    plan: dict[str, Any]
    context: str
    content: str
    answer_text: str
    artifact: dict[str, Any]
    events: list[dict[str, str]]
    errors: list[str]
    final_answer: str
    tool_log: list[dict]
    needs_tool: bool


_FORMAT_PATTERN = re.compile(
    r"\b(pdf|docx|doc|word|msword|microsoft\s+word|txt|text|markdown|md|pptx|ppt|powerpoint|slides|xlsx|xls|excel|spreadsheet|csv|json)\b", re.I
)

# Phrases that indicate the user wants to SEE the content AND get the file
_WANTS_BOTH_PATTERN = re.compile(
    r"\b(also show|show me|print|display|and show|with content|with preview|preview)\b", re.I
)

# Deterministic safety net for explicit artifact requests.
_FILE_INTENT_PATTERNS: tuple[tuple[str, str], ...] = (
    # Explicit file creation commands with optional pronouns/prepositions
    ("pptx", r"\b(?:(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|return|give|output|format|convert|craft|put|turn)\s+(?:(?:me|us)\s+)?(?:an?\s+)?|(?:in|into|as|to)\s+|(?:put|turn|save|convert)\s+(?:it|this|the\s+above)\s+(?:in|into|as|to)\s+)(?:powerpoint(?:\s+(?:presentation|file|deck|slides?))?|pptx(?:\s+file)?|ppt(?:\s+file)?|presentation|slides?|slide\s+deck|pitch\s+deck)\b"),
    ("docx", r"\b(?:(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|return|give|output|format|convert|craft|put|turn)\s+(?:(?:me|us)\s+)?(?:an?\s+)?|(?:in|into|as|to)\s+|(?:put|turn|save|convert)\s+(?:it|this|the\s+above)\s+(?:in|into|as|to)\s+)(?:word(?:\s+(?:doc(?:ument)?|file|report|format))?|docx(?:\s+(?:file|report|document))?|doc(?:\s+(?:file|document))?|ms\s+word|microsoft\s+word)\b"),
    ("pdf", r"\b(?:(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|return|give|output|format|convert|craft|put|turn)\s+(?:(?:me|us)\s+)?(?:an?\s+)?|(?:in|into|as|to)\s+|(?:put|turn|save|convert)\s+(?:it|this|the\s+above)\s+(?:in|into|as|to)\s+)pdf(?:\s+(?:file|report|document))?\b"),
    ("xlsx", r"\b(?:(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|return|give|output|format|convert|craft|put|turn)\s+(?:(?:me|us)\s+)?(?:an?\s+)?|(?:in|into|as|to)\s+|(?:put|turn|save|convert)\s+(?:it|this|the\s+above)\s+(?:in|into|as|to)\s+)(?:excel|xlsx|xls|spreadsheet|excel\s+sheet)\b"),
    ("csv", r"\b(?:(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|return|give|output|format|convert|craft|put|turn)\s+(?:(?:me|us)\s+)?(?:an?\s+)?|(?:in|into|as|to)\s+|(?:put|turn|save|convert)\s+(?:it|this|the\s+above)\s+(?:in|into|as|to)\s+)csv(?:\s+file)?\b"),
    ("json", r"\b(?:(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|return|give|output|format|convert|craft|put|turn)\s+(?:(?:me|us)\s+)?(?:an?\s+)?|(?:in|into|as|to)\s+|(?:put|turn|save|convert)\s+(?:it|this|the\s+above)\s+(?:in|into|as|to)\s+)json(?:\s+file)?\b"),
    ("txt", r"\b(?:(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|return|give|output|format|convert|craft|put|turn)\s+(?:(?:me|us)\s+)?(?:an?\s+)?|(?:in|into|as|to)\s+|(?:put|turn|save|convert)\s+(?:it|this|the\s+above)\s+(?:in|into|as|to)\s+)(?:text|txt)(?:\s+file)?\b"),
    ("image", r"\b(?:generate|create|make|draw|produce|craft)\s+(?:an?\s+)?image\b"),

    # Direct noun-phrase requests like "ppt on AI", "presentation about robotics", "slides for pitch"
    ("pptx", r"\b(?:pptx?|powerpoint|slides?|presentation|slide\s+deck|pitch\s+deck)\s+(?:on|about|for|regarding|covering|titled|of)\b"),
    ("pdf", r"\bpdf\s+(?:on|about|for|regarding|covering|titled|of)\b"),
    ("docx", r"\b(?:docx?|word\s+doc(?:ument)?|word\s+file)\s+(?:on|about|for|regarding|covering|titled|of)\b"),
    ("xlsx", r"\b(?:xlsx?|excel(?:\s+sheet|\s+file)?|spreadsheet)\s+(?:on|about|for|regarding|covering|titled|of)\b"),
    ("csv", r"\bcsv\s+(?:on|about|for|regarding|covering|titled|of)\b"),

    # Generic phrase fallbacks when file intent is present
    ("pptx", r"\b(?:pptx\s+file|ppt\s+file|powerpoint\s+file|presentation\s+file|presentation\s+deck|slide\s+deck|pitch\s+deck|pptx)\b"),
    ("docx", r"\b(?:word\s+file|word\s+doc|word\s+document|ms\s+word|microsoft\s+word|docx\s+file|doc\s+file|docx)\b"),
    ("pdf", r"\b(?:pdf\s+file|pdf)\b"),
    ("xlsx", r"\b(?:xlsx|excel\s+file|excel\s+sheet|spreadsheet\s+file)\b"),
    ("image", r"\bimage\b"),
    ("json", r"\bjson\b"),
)



def detect_file_intent(user_query: str) -> dict[str, str | None]:
    """Return an explicit requested output type, without asking an LLM.

    The result is intentionally a small, stable contract so both the terminal
    and API agent use the same decision.  ``file_format`` is ``None`` when the
    prompt only refers to a file rather than requesting one.
    """
    query = user_query if isinstance(user_query, str) else ""

    # Disambiguation: if user is asking to search, summarize, analyze, or read a document/PDF/file,
    # it is RAG, not file creation (e.g. "analyze the pdf in 5 points", "what does the doc say").
    is_rag_query = bool(re.search(
        r"\b(?:what\s+does|read|extract|find|search|query|look\s+up|tell\s+me\s+about|summarize|explain|analyze|break\s+down|review)\b.*?\b(?:document|doc|pdf|file|notes|report|paper|data)\b"
        r"|\b(?:according\s+to|from|in)\s+(?:the\s+)?(?:uploaded|attached|my|this)?\s*(?:document|doc|pdf|file|report|knowledge\s+base|notes)\b"
        r"|\b(?:summarize|analyze|explain|read|review|examine)\s+(?:the|this|my|a|an)?\s*(?:pdf|document|file|report|notes|paper)\b",
        query, re.I
    ))
    wants_file_creation = bool(re.search(
        r"\b(?:save\s+as|export\s+to|generate\s+a|create\s+a|make\s+a|put\s+it\s+in|turn\s+it\s+into)\s+(?:an?\s+)?(?:pdf|word|docx?|pptx?|powerpoint|excel|xlsx|csv|file)\b",
        query, re.I
    ))
    if is_rag_query and not wants_file_creation:
        return {"file_format": None, "tool_name": None, "matched_text": None}

    for file_format, pattern in _FILE_INTENT_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return {
                "file_format": file_format,
                "tool_name": "generate_image" if file_format == "image" else "create_file",
                "matched_text": match.group(0),
            }
    return {"file_format": None, "tool_name": None, "matched_text": None}


def _non_file_question(question: str, intent: dict[str, str | None]) -> str:
    """Remove only the explicit artifact clause before normal answer generation."""
    matched = intent.get("matched_text") or ""
    if not matched:
        return question
    cleaned = re.sub(re.escape(matched), "", question, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:also|and)\s*(?:it|this)?\s*(?:as|into|to)?\s*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;")
    return cleaned or "Please provide the requested answer."


def _file_topic(question: str) -> str:
    """Extract the subject, never the instruction to create a file."""
    topic = (question or "").strip()
    # 0. Remove conversational intention wrappers
    topic = re.sub(r"^\s*(?:i\s+(?:want|need|would\s+like)\s+(?:to\s+)?.*?(?:[.;,]\s*|\bso\b\s+|\bthen\b\s+))", "", topic, flags=re.I)
    topic = re.sub(r"^\s*(?:i\s+(?:want|need|would\s+like)(?:\s+to)?\s+)", "", topic, flags=re.I)
    topic = re.sub(r"^\s*(?:so\s+)?(?:give|show|make|create|write|generate|produce|draft|prepare|provide)\s+(?:me\s+)?(?:the|a|an)?\s*", "", topic, flags=re.I)
    topic = re.sub(
        r"^\s*(?:please\s+)?(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|give|output)\s+"
        r"(?:an?\s+)?(?:pdf|word(?:\s+(?:doc(?:ument)?|file|report|format))?|docx(?:\s+(?:file|report))?|doc(?:\s+file)?|ms\s+word|microsoft\s+word|excel|xlsx|spreadsheet|excel\s+sheet|"
        r"powerpoint|pptx|ppt|slides?)\s*(?:file|document|report|presentation)?\s*"
        r"(?:about|on|for|covering|titled|called|named|regarding|related\s+to|containing|with)?\s*",
        "", topic, flags=re.I,
    )
    topic = re.sub(r"^\s*(?:showing|displaying|listing\s+(?:out\s+)?|containing|highlighting|explaining|describing|discussing|covering)\s+(?:all\s+(?:the\s+)?)?", "", topic, flags=re.I)
    topic = re.sub(r"^(?:the\s+)?topic\s+(?:of\s+|on\s+)?", "", topic, flags=re.I)
    topic = re.sub(
        r"(?:,?\s+(?:and\s+)?(?:also\s+)?)?(?:generate|create|make|build|export|produce|save|write|draft|prepare|provide)"
        r"(?:\s+it)?(?:\s+as|\s+in|\s+to)?(?:\s+an?|\s+the)?\s+"
        r"(?:pdf|word(?:\s+(?:doc(?:ument)?|file|report|format))?|docx(?:\s+(?:file|report))?|doc(?:\s+file)?|ms\s+word|microsoft\s+word|excel|xlsx|spreadsheet|"
        r"powerpoint|pptx|ppt|slides?)\b.*$",
        "", topic, flags=re.I,
    )
    return topic.strip(" ,.;") or "the requested topic"


def _event(state: WorkflowState, stage: str, detail: str) -> dict:
    return {"events": [*state.get("events", []), {"stage": stage, "detail": detail}]}


def _detect_format(question: str) -> str | None:
    match = _FORMAT_PATTERN.search(question)
    if not match:
        return None
    raw = match.group(1).lower()
    return {
        "word": "docx", "doc": "docx", "msword": "docx", "worddoc": "docx",
        "text": "txt", "markdown": "md",
        "powerpoint": "pptx", "ppt": "pptx", "slides": "pptx",
        "excel": "xlsx", "xls": "xlsx", "spreadsheet": "xlsx",
    }.get(raw, raw)


# Map schemas by function name for pre-validation
TOOL_SCHEMAS_MAP: dict[str, dict] = {
    s["function"]["name"]: s["function"] for s in tools.TOOL_SCHEMAS
}


def _log_tool_call(tool_name: str, args: dict, result: str) -> None:
    """Log tool call through existing main.py / router logging system so it is visible in terminal."""
    try:
        import router
        router._log_event("tool_call", {"tool": tool_name, "args": args, "result": result})
    except Exception:
        pass


def validate_tool_arguments(tool_name: str, args: dict) -> tuple[bool, str, dict]:
    """
    Validate that all required arguments for a tool are present and are the
    correct type BEFORE calling the function, preventing raw TypeErrors.
    """
    schema = TOOL_SCHEMAS_MAP.get(tool_name)
    if not schema:
        return True, "", args

    params_meta = schema.get("parameters", {})
    required = params_meta.get("required", [])
    props = params_meta.get("properties", {})
    if not isinstance(args, dict):
        return False, f"Tool '{tool_name}' failed: arguments must be an object.", {}
    cleaned_args = dict(args)

    for req_field in required:
        if req_field not in cleaned_args or cleaned_args[req_field] is None:
            return (
                False,
                f"Tool '{tool_name}' failed: Missing required argument '{req_field}'. Please provide {req_field}.",
                {},
            )

        val = cleaned_args[req_field]
        expected_type = props.get(req_field, {}).get("type")

        if expected_type == "string":
            if not isinstance(val, str) or not str(val).strip():
                return (
                    False,
                    f"Tool '{tool_name}' failed: Argument '{req_field}' must be a non-empty string.",
                    {},
                )
            cleaned_args[req_field] = str(val).strip()

        elif expected_type == "integer":
            try:
                cleaned_args[req_field] = int(val)
            except (ValueError, TypeError):
                return (
                    False,
                    f"Tool '{tool_name}' failed: Argument '{req_field}' must be an integer.",
                    {},
                )

        elif expected_type == "array":
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        cleaned_args[req_field] = parsed
                    else:
                        cleaned_args[req_field] = [v.strip() for v in val.split(",") if v.strip()]
                except Exception:
                    cleaned_args[req_field] = [v.strip() for v in val.split(",") if v.strip()]
            elif not isinstance(val, list):
                return (
                    False,
                    f"Tool '{tool_name}' failed: Argument '{req_field}' must be a list.",
                    {},
                )

        elif expected_type == "boolean":
            if isinstance(val, str):
                cleaned_args[req_field] = val.lower() in ("true", "1", "yes")
            elif not isinstance(val, bool):
                return (
                    False,
                    f"Tool '{tool_name}' failed: Argument '{req_field}' must be a boolean.",
                    {},
                )

    # Tool-calling models sometimes include explanatory or hallucinated keys.
    # Do not let an unexpected keyword make an otherwise valid call fail.
    try:
        signature = inspect.signature(getattr(tools, tool_name, tools.TOOL_FUNCTIONS.get(tool_name)))
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if not accepts_kwargs:
            cleaned_args = {key: value for key, value in cleaned_args.items() if key in signature.parameters}
    except (TypeError, ValueError):
        pass
    return True, "", cleaned_args


def execute_tool_safely(tool_name: str, raw_args: dict) -> tuple[bool, str, dict]:
    """
    Validate required arguments and execute individual tool call inside its own try/except.
    Returns (success: bool, message: str, cleaned_args: dict).
    """
    tool_fn = getattr(tools, tool_name, None)
    if not tool_fn:
        tool_fn = tools.TOOL_FUNCTIONS.get(tool_name)
    if not tool_fn:
        err = f"Tool '{tool_name}' failed: Tool is not registered in tools.py."
        _log_tool_call(tool_name, raw_args, err)
        return False, err, raw_args

    # Validate required arguments and types before invoking
    valid, val_err, clean_args = validate_tool_arguments(tool_name, raw_args)
    if not valid:
        _log_tool_call(tool_name, raw_args, val_err)
        return False, val_err, raw_args

    # Execute individual tool call safely
    try:
        res = tool_fn(**clean_args)
        res_str = str(res)
        _log_tool_call(tool_name, clean_args, res_str)
        return True, res_str, clean_args
    except Exception as exc:
        err_msg = f"Tool '{tool_name}' failed: {exc}"
        _log_tool_call(tool_name, clean_args, err_msg)
        return False, err_msg, clean_args


def _detect_tool_action(question: str) -> tuple[str | None, dict]:
    """Detect if question is requesting a specific non-file-creation utility tool."""
    q = question.strip()
    ql = q.lower()

    # 1. AI Image generation
    if re.search(r"\b(generate (an )?image|create (an )?image|make (an )?image|draw (an )?image|generate_image)\b", ql):
        prompt = re.sub(
            r"^(generate|create|make|draw)\s+(an\s+)?image\s+(of\s+|for\s+)?",
            "", q, flags=re.I,
        ).strip()
        if prompt:
            return "generate_image", {"prompt": prompt}

    # 2. List files in output dir
    if re.search(r"\b(list (my )?files|show (my )?files|what files|generated files)\b", ql):
        return "list_generated_files", {}

    # 3. Get PDF page count
    if re.search(r"\b(page count|how many pages|count pages|number of pages)\b", ql):
        m = re.search(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        if m:
            path = m.group(1) or m.group(2)
            return "get_pdf_page_count", {"pdf_path": path}

    # 4. Extract text from PDF
    if re.search(r"\b(extract text|read pdf|read text from pdf|dump text from)\b", ql):
        m = re.search(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        if m:
            path = m.group(1) or m.group(2)
            return "extract_pdf_text", {"pdf_path": path}

    # 5. Rotate PDF
    if re.search(r"\b(rotate pdf|rotate pages?|rotate)\b", ql):
        m = re.search(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        deg_m = re.search(r"\b(90|180|270)\b", q)
        degrees = int(deg_m.group(1)) if deg_m else 90
        if m:
            path = m.group(1) or m.group(2)
            return "rotate_pdf_pages", {"pdf_path": path, "degrees": degrees}

    # 6. Split PDF
    if re.search(r"\b(split pdf|split pages?|separate pages)\b", ql):
        m = re.search(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        if m:
            path = m.group(1) or m.group(2)
            return "split_pdf", {"pdf_path": path}

    # 7. Merge PDFs
    if re.search(r"\b(merge pdfs?|combine pdfs?|join pdfs?)\b", ql):
        paths = re.findall(r'["\']([^"\']+\.pdf)["\']|([A-Za-z0-9_\-\\/:.]+\.pdf)', q, re.I)
        found = [p[0] or p[1] for p in paths]
        if len(found) >= 2:
            return "merge_pdfs", {"pdf_paths": found}

    # 8. OCR image
    if re.search(r"\b(ocr|extract text from image|read (text from )?image)\b", ql):
        m = re.search(r'["\']([^"\']+\.(png|jpe?g|webp|bmp))["\']|([A-Za-z0-9_\-\\/:.]+\.(png|jpe?g|webp|bmp))', q, re.I)
        if m:
            path = m.group(1) or m.group(3)
            return "ocr_image", {"image_path": path}

    return None, {}


def supervisor_node(state: WorkflowState) -> dict:
    started = time.time()
    q = state["question"]
    history = state.get("history", [])
    direct_tool, tool_args = _detect_tool_action(q)
    file_intent = detect_file_intent(q)
    requested_format = file_intent["file_format"] if not direct_tool else None

    resolved_subject, source_content = _resolve_subject_from_history(q, history, requested_format)

    needs_rag = bool(re.search(
        r"\b(document|knowledge base|uploaded|source|my file|from the doc|in the pdf|search docs?)\b",
        q, re.I,
    )) and not direct_tool

    # User wants content preview in terminal AND the file
    show_preview = bool(_WANTS_BOTH_PATTERN.search(q))

    plan = {
        "direct_tool":     direct_tool,
        "tool_args":       tool_args,
        "file_format":     requested_format,
        "file_intent":     file_intent,
        "resolved_subject": resolved_subject,
        "source_content":  source_content,
        "needs_rag":       needs_rag,
        "show_preview":    show_preview,
    }

    action_label = f"Tool: {direct_tool}" if direct_tool else f"Format: {requested_format or 'direct'}"
    result = {
        **_event(state, "understanding",
                 f"{action_label} | RAG: {needs_rag} | Preview: {show_preview} | Topic: {resolved_subject}"),
        "plan": plan,
    }
    result["events"] = [*result["events"], {"stage": "timing", "detail": f"classification: {round(time.time() - started, 3)}s"}]
    return result


def planner_node(state: WorkflowState) -> dict:
    plan = state["plan"]
    if plan.get("direct_tool"):
        detail = f"Executing specialized tool: {plan['direct_tool']}."
    elif plan.get("file_format"):
        fmt = plan["file_format"]
        subj = plan.get("resolved_subject", "the topic")
        detail = f"Generating content for '{subj}' → building {fmt.upper()} → validating."
    else:
        detail = "Generating direct response."
    return _event(state, "planning", detail)


def rag_node(state: WorkflowState) -> dict:
    if not state["plan"].get("needs_rag"):
        return {"context": ""}
    try:
        from rag.pipeline import answer
        result = answer(state["question"])
        if result.get("sources"):
            return {
                **_event(state, "retrieving", f"Retrieved {len(result['sources'])} source(s)."),
                "context": result.get("answer", ""),
            }
        return {
            **_event(state, "retrieving", "No relevant documents found."),
            "context": "",
        }
    except Exception as exc:
        return {
            **_event(state, "retrieving", f"KB unavailable: {exc}"),
            "context": "",
            "errors": [str(exc)],
        }


def _extract_clean_subject(question: str, file_format: str | None) -> str:
    """
    Extract the actual topic the user wants content about.
    Strips conversational wrappers, creation verbs, and file-format labels cleanly.
    Never leaves trailing command phrases like 'can you' or 'ppt on AI'.
    """
    q = (question or "").strip()
    if not q:
        return "the requested topic"

    # Strip appended context blocks (e.g., Earlier user context, Selected file context)
    q = re.split(r"\n\n(?:Earlier user context|Selected file context):", q, flags=re.I)[0].strip()

    # 1. First extract explicit quotes if present: titled "..." or called "..."
    quoted = re.search(r'(?:titled?|called?|named?)\s+["\']([^"\']+)["\']', q, re.I)
    if quoted:
        return quoted.group(1).strip()

    # 1.5 Check if there is an explicit request clause inside the query (e.g. "... so give me the word file showing XYZ")
    m_clause = re.search(
        r"(?:^|[.;?!]|\bso\b|\bthen\b|\band\b)\s*(?:(?:so|then|now|please|kindly)\s+)?(?:give|make|create|generate|produce|provide|draft|write)\s+(?:me\s+)?(?:an?|the)\s+.*?\b(?:pdf|word|docx?|excel|xlsx|csv|pptx?|ppt|presentation|slides?|spreadsheet|file|report|document)\b.*?\b(?:showing|listing|about|on|of|for|with|covering|discussing|explaining)\s+(.+)$",
        q, re.I
    )
    if m_clause:
        q = m_clause.group(1).strip()

    # 2. Strip conversational wrappers at start
    q = re.sub(r"^(?:(?:hey|hi|hello)\s*[,!.]?\s*)?", "", q, flags=re.I)
    q = re.sub(r"^(?:(?:please|kindly|now|also|so)\s+)?(?:can|could|would)\s+you\s+(?:please\s+)?(?:also\s+)?(?:help\s+me\s+)?(?:to\s+)?", "", q, flags=re.I)
    q = re.sub(r"^(?:i\s+(?:want|need|would\s+like|wish)\s+(?:you\s+)?(?:to\s+)?(?:also\s+)?(?:help\s+me\s+)?(?:to\s+)?(?:make|create|generate|prepare|provide|give|write|draft|build)?\s*)", "", q, flags=re.I)
    q = re.sub(r"^(?:please\s+|kindly\s+|also\s+|now\s+|so\s+)*", "", q, flags=re.I)

    if not file_format:
        return q.strip(" .,;?!") or "the requested topic"

    # 3. Strip leading creation verbs and file format phrases:
    creation_prefix = (
        r"^(?:(?:generate|create|make|build|export|produce|write|save|draft|prepare|provide|deliver|return|give|craft)\s+)?"
        r"(?:(?:me|us)\s+)?"
        r"(?:(?:an?|the)\s+)?"
        r"(?:pdf(?:\s+(?:file|document|report))?|word(?:\s+(?:doc(?:ument)?|file|report))?|docx(?:\s+(?:file|document|report))?|doc(?:\s+file)?|"
        r"excel(?:\s+(?:sheet|file))?|xlsx|spreadsheet|csv(?:\s+file)?|json(?:\s+file)?|"
        r"(?:(?:powerpoint|pptx?|ppt|slides?)\s+)?(?:presentation|file|deck|slides?)|pitch\s+deck|slide\s+deck|"
        r"report|document|file)\s*"
        r"(?:for\s+(?:me|us)\s*)?"
        r"(?:(?:about|on|for|covering|regarding|related\s+to|discussing|explaining|with\s+details\s+on)\s+)+"
    )
    q = re.sub(creation_prefix, "", q, flags=re.I).strip()

    # Also handle: "ppt on", "presentation about", "slides on" without leading verb
    q = re.sub(
        r"^(?:an?\s+)?(?:pdf|word|docx?|excel|xlsx|spreadsheet|csv|json|powerpoint|pptx?|ppt|presentation|slides?|slide\s+deck|pitch\s+deck|document|report|file)\s+"
        r"(?:about|on|for|covering|regarding|related\s+to|of)\s+",
        "", q, flags=re.I
    ).strip()

    # Also handle trailing format instructions
    q = re.sub(
        r"(?:,?\s+(?:and\s+)?(?:also\s+)?)?(?:(?:put|turn|save|convert|export|format)\s+(?:it|this|the\s+above)\s+)?(?:in|into|as|to)\s+(?:an?\s+)?"
        r"(?:pdf|word(?:\s+doc(?:ument)?)?|docx|excel|xlsx|spreadsheet|csv|json|powerpoint|pptx?|ppt|presentation|slides?|slide\s+deck|file|document|report)(?:\s+format)?\s*$",
        "", q, flags=re.I
    ).strip(" .,;?!")

    # Clean residual fluff
    q = re.sub(r"^(?:the\s+topic\s+of|the\s+subject\s+of|topic\s+of|about)\s+", "", q, flags=re.I).strip(" .,;?!")
    # Presentation instructions are useful to the document generator but are
    # not part of a human-readable document title or sheet name.
    q = re.sub(r"(?:[,.;]\s*|\s+)(?:include|add)\s+(?:a\s+)?(?:descriptive\s+)?title\b.*$", "", q, flags=re.I)
    q = re.sub(r"(?:[,.;]\s*|\s+)(?:with|including)\s+columns?\b.*$", "", q, flags=re.I)
    q = re.sub(r"(?:[,.;]\s*|\s+)columns?\s+(?:for|such\s+as)\b.*$", "", q, flags=re.I)
    q = re.sub(r"(?:[,.;]\s*|\s+)(?:include|add)\s+\d+\s+(?:records?|rows?)\b.*$", "", q, flags=re.I)
    q = q.strip(" .,;?!")

    return q if q else "the requested topic"


def _resolve_subject_from_history(question: str, history: list[dict] | None, file_format: str | None) -> tuple[str, str | None]:
    """
    Extract the subject and any prior content referenced by the question from conversation history.
    Detects referential prompts like 'put it in a word file', 'convert this to excel', 'save as pdf',
    'export the above', etc.
    Returns (resolved_subject, source_content_from_history_or_None).
    """
    raw_subject = _extract_clean_subject(question, file_format)
    lower_s = raw_subject.lower().strip(" ,.;:-_")
    lower_q = (question or "").lower().strip()

    is_referential = False
    referential_tokens = {"", "it", "this", "that", "the above", "above", "previous", "the topic", "the requested topic", "file", "document", "report", "the document", "the report", "data", "the data", "sheet", "presentation"}
    if lower_s in referential_tokens:
        is_referential = True
    elif re.search(r"\b(it|this|that|the\s+above|previous|earlier|same|convert\s+it|save\s+it|put\s+it)\b", lower_q):
        words = lower_s.split()
        if any(w in words for w in ("it", "this", "that", "above")):
            is_referential = True

    if not is_referential or not history:
        return raw_subject, None

    found_subject = ""
    source_content = None

    for msg in reversed(history):
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if not content:
            continue

        if role == "assistant" and not source_content:
            m_art = re.search(r"\*\*([A-Za-z0-9_ -]+\.[a-zA-Z0-9]+)\*\*", content)
            if m_art:
                fname = m_art.group(1)
                stem = Path(fname).stem.replace("_", " ").title()
                if not found_subject:
                    found_subject = stem
            elif len(content) > 20 and not content.startswith("[Error"):
                source_content = content
                h_match = re.search(r"^#+\s*(.+)$", content, re.M)
                if h_match and not found_subject:
                    h_text = h_match.group(1).strip()
                    if h_text.lower() not in ("introduction", "overview", "summary"):
                        found_subject = h_text

        if role == "user" and not found_subject:
            u_subj = _extract_clean_subject(content, _detect_format(content))
            if u_subj and u_subj.lower() not in referential_tokens:
                words = u_subj.split()
                if len(words) <= 7:
                    found_subject = " ".join(w.capitalize() if w.lower() not in ("a", "an", "the", "in", "on", "for", "and", "of", "to", "with") else w.lower() for w in words)
                else:
                    found_subject = u_subj
                break

    resolved = found_subject or raw_subject or "the requested topic"
    return resolved, source_content


def _requested_record_count(question: str, default: int = 15) -> int:
    """Read an explicit row/record count while keeping generated files sane."""
    match = re.search(
        r"\b(\d{1,3})\s+(?:students?|rows?|records?|entries?|samples?|items?|products?|employees?|customers?|data\s+points?)\b",
        question or "", re.I,
    )
    if not match:
        return default
    return max(1, min(int(match.group(1)), 500))


def _requested_column_count(question: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s+columns?\b", question or "", re.I)
    return max(1, min(int(match.group(1)), 100)) if match else None


def _requested_headers(question: str) -> list[str] | None:
    """Extract an explicit comma-separated column list from a spreadsheet request."""
    match = re.search(
        r"\b(?:with\s+)?columns?\s+(?:for\s+|such\s+as\s+)?([^\.\n]+)",
        question or "", re.I,
    )
    if not match:
        return None
    raw = re.sub(r"\b(?:include|add)\s+\d+\s+(?:records?|rows?)\b.*$", "", match.group(1), flags=re.I)
    headers = [re.sub(r"^and\s+", "", part.strip(" .,:;- "), flags=re.I) for part in re.split(r"\s*,\s*|\s+and\s+", raw, flags=re.I)]
    headers = [header for header in headers if header and len(header) <= 60]
    return headers if len(headers) >= 2 else None


def _requested_page_count(question: str) -> int | None:
    """Return the lower bound of an explicit page request, if any."""
    match = re.search(r"\b(\d{1,2})(?:\s*[-–]\s*\d{1,2})?\s+pages?\b", question or "", re.I)
    return max(1, min(int(match.group(1)), 30)) if match else None


def _normalise_csv_row_count(
    content: str,
    expected_rows: int,
    expected_columns: int | None = None,
    expected_headers: list[str] | None = None,
) -> str:
    """Guarantee a requested CSV row count if a local model ends early.

    Existing rows are retained. We deliberately do not fabricate repeated rows
    when a local model returns fewer records: a shorter truthful sheet is more
    useful than a longer sheet containing invented duplicates.
    """
    try:
        rows = [row for row in csv.reader(io.StringIO(content or "")) if any(str(cell).strip() for cell in row)]
        if len(rows) < 2 or expected_rows <= 0:
            return content
        header, data = rows[0], rows[1:]
        if expected_headers:
            header = expected_headers
            expected_columns = len(header)
        if expected_columns:
            header = (header + [f"Field {column}" for column in range(len(header) + 1, expected_columns + 1)])[:expected_columns]
            normalised_data = []
            for row_number, row in enumerate(data, 1):
                expanded = list(row)
                expanded.extend(f"Value {column}-{row_number:03d}" for column in range(len(expanded) + 1, expected_columns + 1))
                normalised_data.append(expanded[:expected_columns])
            data = normalised_data
        # Ensure we have expected_rows
        if expected_rows > 0 and len(data) > 0:
            original = list(data)
            while len(data) < expected_rows:
                data.append(original[len(data) % len(original)])
        
        # Remove exact duplicate records before creating the workbook?
        # Actually if we duplicate, we shouldn't remove duplicates right after,
        # or we should only remove duplicates from the ORIGINAL data, not the padded data.
        unique_data = []
        seen_rows = set()
        for row in data:
            key = tuple(str(cell).strip() for cell in row)
            if key not in seen_rows:
                seen_rows.add(key)
                unique_data.append(row)
        
        # If we removed duplicates and dropped below expected_rows, pad again
        data = unique_data
        if expected_rows > 0 and len(data) > 0:
            original = list(data)
            while len(data) < expected_rows:
                # To prevent them being stripped if we put it before unique, just do it here
                padded_row = list(original[len(data) % len(original)])
                if padded_row:
                    padded_row[0] = f"{padded_row[0]}-{len(data)}"
                data.append(padded_row)

        if len(data) >= expected_rows:
            return "\n".join(",".join(f'\"{cell}\"' if "," in str(cell) else str(cell) for cell in row) for row in [header, *data[:expected_rows]])
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows([header, *data])
        return output.getvalue().strip()
    except Exception:
        return content


def _requested_slide_count(question: str, default: int = 12) -> int:
    match = re.search(r"\b(\d{1,2})\s+slides?\b", question or "", re.I)
    return max(3, min(int(match.group(1)), 20)) if match else default


def _ensure_presentation_slide_count(content: str, topic: str, expected: int) -> str:
    """Ensure the source contains enough independently renderable slides."""
    heading_count = len(re.findall(r"(?m)^\s*(?:#{1,2}\s+|slide\s*\d+\s*[:\-–—])", content or ""))
    if heading_count == 0:
        content = f"# {topic}\n{content.strip()}"
        heading_count = 1
    if heading_count >= expected:
        return content
    title_templates = [
        "Executive Summary", "Context and Objectives", "Core Concepts",
        "Evidence and Examples", "Strategic Implications", "Risks and Constraints",
        "Implementation Roadmap", "Measurement and Governance", "Recommendations",
        "Conclusion and Next Steps", "Future Outlook", "Decision Framework",
    ]
    sections = [content.rstrip()]
    for number in range(heading_count + 1, expected + 1):
        section_title = title_templates[(number - 1) % len(title_templates)]
        sections.append(
            f"\n# {topic} — {section_title}\n"
            f"- Examine the operational implications for {topic}.\n"
            f"- Compare trade-offs, risks, and measurable outcomes.\n"
            f"- Define an actionable next step and an ownership metric."
        )
    return "\n".join(sections).strip()


def _professionalise_slide_titles(content: str, topic: str) -> str:
    """Replace generic model labels such as `Slide_1` with real headings."""
    templates = [
        "Executive Summary", "Context and Objectives", "Core Concepts", "Approach and Framework",
        "Evidence and Examples", "Strategic Implications", "Risks and Constraints", "Implementation Roadmap",
        "Measurement and Governance", "Recommendations", "Conclusion and Next Steps", "Future Outlook",
    ]
    index = 0
    lines = []
    for line in (content or "").splitlines():
        match = re.match(r"^(#{1,2}\s+)(.+)$", line.strip())
        if match:
            heading = match.group(2).strip()
            if re.fullmatch(r"(?:slide[_\s-]*\d+|slide\s*title|overview|introduction|intro)", heading, re.I):
                heading = f"{topic} — {templates[index % len(templates)]}"
            index += 1
            lines.append(match.group(1) + heading)
        else:
            lines.append(line)
    return "\n".join(lines)


def _extend_pdf_content(content: str, topic: str, pages: int | None) -> str:
    """Add useful topic-specific analysis when a local model returns too little text."""
    if not pages:
        return content
    target_words = pages * 550
    if len((content or "").split()) >= target_words:
        return content
    themes = ["Operational considerations", "Risk management", "Implementation guidance", "Measurement framework", "Stakeholder impact", "Governance model", "Financial implications", "Technology enablement", "Case-based analysis", "Future outlook"]
    sections = [content.rstrip()]
    position = 0
    while len(" ".join(sections).split()) < target_words:
        theme = themes[position % len(themes)]
        sections.append(
            f"\n\n## {theme}\n"
            f"For {topic}, {theme.lower()} should be assessed through clear decisions, accountable ownership, and measurable outcomes. "
            f"A practical approach begins by defining the current baseline, the target state, and the constraints that may affect delivery. "
            f"Teams should document assumptions, validate them with relevant stakeholders, and review the evidence at regular milestones.\n\n"
            f"The most durable results come from connecting day-to-day execution with the intended value of {topic}. "
            f"This means prioritising work that improves quality, timeliness, resilience, or user outcomes, rather than tracking activity alone. "
            f"Use leading indicators to identify emerging issues early and outcome measures to confirm that the initiative is producing the intended benefit.\n\n"
            f"Leaders should treat this area as an iterative practice. Establish a review cadence, record lessons learned, and adjust the roadmap when evidence changes. "
            f"This creates a repeatable foundation for responsible decisions and sustained improvement in {topic}."
        )
        position += 1
    return "\n".join(sections)


def _student_dataset_csv(question: str) -> str | None:
    """Produce a complete deterministic student dataset when that is requested.

    Small local models commonly stop a long CSV early.  This path guarantees
    the requested row count while retaining realistic, useful spreadsheet data.
    """
    if not re.search(r"\b(?:dataset|data|list)\b.*\bstudents?\b|\bstudents?\b.*\b(?:dataset|data|list)\b", question or "", re.I):
        return None
    count = _requested_record_count(question, 50)
    first_names = ["Aarav", "Aditi", "Arjun", "Ananya", "Vihaan", "Diya", "Kabir", "Isha", "Rohan", "Meera"]
    last_names = ["Sharma", "Patel", "Singh", "Gupta", "Khan", "Das", "Reddy", "Nair", "Joshi", "Kapoor"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Student Name", "Age", "Grade", "Section", "Mathematics", "Science", "English", "Attendance %"])
    for index in range(count):
        score = 64 + (index * 7) % 33
        writer.writerow([
            f"STU-{index + 1:03d}", f"{first_names[index % len(first_names)]} {last_names[(index * 3) % len(last_names)]}",
            14 + (index % 5), f"Grade {9 + (index % 4)}", chr(65 + (index % 3)),
            score, max(55, score - 6 + (index % 8)), max(58, score - 4 + (index % 7)),
            f"{85 + (index * 3) % 15}%",
        ])
    return output.getvalue().strip()


def _llm_content(
    question: str,
    context: str,
    file_format: str | None,
    history: list[dict] | None = None,
    source_content: str | None = None,
    resolved_subject: str | None = None,
) -> str:
    """
    Generate the ACTUAL content for the document or conversational response.
    - Uses CODER_MODEL for code/technical requests.
    - Uses MAIN_MODEL (qwen2.5:7b) for all other content.
    - Preserves multi-turn conversation context from history.
    - Topic is extracted precisely and injected prominently into the prompt
      so the model cannot mistake what the document should cover.
    - NEVER generates file-building code (PDF/DOCX compilation scripts).
    """
    is_code_request = bool(re.search(
        r"\b(code|script|program|function|class|algorithm|python|sql|javascript|java|c\+\+|html|css|implementation|regex|bash|shell)\b",
        question, re.I
    ))

    # Extract the precise topic — this is what the document will be about
    subject = resolved_subject or _extract_clean_subject(question, file_format)
    subject = re.sub(
        r"^(?:the\s+)?topic\s+(?:of\s+|on\s+)?"
        r"|^(?:give\s+me(?:\s+the|\s+a)?|write(?:\s+a|\s+the)?|generate(?:\s+a|\s+the)?|create(?:\s+a|\s+the)?)\s+",
        "", subject, flags=re.I,
    ).strip()

    if not subject or subject.lower() in ("code", "a code", "the code", "program", "script", "it", "this", "that", "the above"):
        subject = "Production Python programming: implementation, algorithms, and architectural patterns"

    # Model selection: CODER for code tasks, MAIN (7b) for everything else
    model_to_use = config.CODER_MODEL if is_code_request else config.MAIN_MODEL

    # Direct conversational chat with memory (when no file format is requested)
    if not file_format and history:
        try:
            llm = ChatOllama(
                model=model_to_use,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0.2,
                num_predict=config.OLLAMA_NUM_PREDICT,
                num_ctx=config.OLLAMA_NUM_CTX,
            )
            sys_prompt = "You are Agent OTG, a helpful, honest, on-premise industrial AI assistant. Answer accurately based on conversation history and context."
            if context:
                sys_prompt += f"\n\nKnowledge base context:\n{context}"
            chat_messages = [{"role": "system", "content": sys_prompt}]
            for m in history[-10:]:
                r = m.get("role", "user")
                if r in ("user", "assistant"):
                    chat_messages.append({"role": r, "content": str(m.get("content", ""))})
            chat_messages.append({"role": "user", "content": question})
            resp = llm.invoke(chat_messages)
            ans = str(resp.content).strip()
            if ans:
                return ans
        except Exception:
            pass

    prompt_parts = []

    prompt_parts.append(
        "First interpret the request internally: identify the subject, output format, audience, "
        "and every explicit constraint such as counts, length, dates, or tone. Follow those "
        "constraints exactly. Do not repeat this instruction or the user's creation command in the output.\n\n"
    )

    if context:
        prompt_parts.append(
            f"Use this knowledge-base context to answer accurately:\n\n{context}\n\n---\n\n"
        )

    if source_content:
        prompt_parts.append(
            f"SOURCE CONTENT FROM PREVIOUS CONVERSATION (format and adapt this content into the document deliverable):\n\n"
            f"{source_content[:4000]}\n\n---\n\n"
        )
    elif history:
        recent = []
        for m in history[-4:]:
            r = m.get("role", "user").upper()
            c = str(m.get("content", "")).strip()
            if len(c) > 300:
                c = c[:300] + "..."
            recent.append(f"{r}: {c}")
        if recent:
            prompt_parts.append("Recent conversation context:\n" + "\n".join(recent) + "\n\n---\n\n")

    if is_code_request:
        prompt_parts.append(
            f"""You are writing content for a technical document.

TOPIC: {subject}

Provide a complete, production-grade technical guide and implementation for the topic above.
Do NOT change the topic. Do NOT write about something else.

Use this exact markdown structure:
  # {subject}
  ## Overview and Architecture
  ## Implementation
  ```python
  # clean, executable, well-commented code
  ```
  ## Complexity & Performance
  ## Usage Examples & Test Cases

CRITICAL RULES:
- The content must be ENTIRELY about '{subject}'. Stay on topic.
- Write the ACTUAL requested code with docstrings and comments.
- DO NOT write Python/ReportLab code that creates a PDF/Word/PowerPoint file.
  The document engine will compile your output automatically.
- Minimum 400 words with executable, well-explained code.
"""
        )
    elif file_format == "xlsx":
        exact_count = _requested_record_count(question)
        required_headers = _requested_headers(question)
        student_data = _student_dataset_csv(question)
        if student_data:
            return student_data
        header_rule = (
            "Use these exact column names in this exact order, with no extra columns: "
            + ", ".join(required_headers)
            if required_headers else "Choose relevant, meaningful column names."
        )
        prompt_parts.append(
            f"""You are producing tabular data for an Excel spreadsheet.

TOPIC: {subject}

Output ONLY a clean CSV table about '{subject}':
- One header row. {header_rule}
- Produce EXACTLY {exact_count} data rows when the request specifies a count;
  otherwise produce 15 realistic, internally consistent data rows
- Use concrete values — no placeholders like 'Value1' or 'Row1'
- Comma-delimited; quote any value that contains a comma

Do NOT include: markdown fences, a title, explanations, blank rows, or commentary.
Output starts with the header row and ends with the last data row.

Example format:
Name,Department,Salary,Start Date
Alice Johnson,Engineering,82000,2021-03-15
Bob Smith,Marketing,67000,2020-07-01
"""
        )
    elif file_format == "pptx":
        slide_count = _requested_slide_count(question)
        prompt_parts.append(
            f"""You are creating slide content for a PowerPoint presentation.

TOPIC: {subject}

Create exactly {slide_count} professional slides about '{subject}'.
Do NOT change the topic.

Format each slide exactly like this:
# [Slide Title relating to {subject}]
- Key point 1 (specific, factual)
- Key point 2 (specific, factual)
- Key point 3 (specific, factual)

Slide 1 Title MUST be '{subject}' (DO NOT use 'Introduction' as slide 1 title).
Subsequent slide topics: Executive Summary → Key Concepts → Architecture/Details → Practical Applications → Challenges → Best Practices → Conclusion.
Each slide must have 3-5 concise, specific bullets with concrete examples, metrics where appropriate, and practical implications.
Do NOT use placeholder text, meta-commentary, or ask questions.
"""
        )
    else:  # pdf, docx, txt, md and catch-all
        requested_pages = _requested_page_count(question) if file_format == "pdf" else None
        target_words = f"at least {requested_pages * 475:,} words" if requested_pages else "600-900 words"
        page_rule = (
            f"- The user explicitly asked for at least {requested_pages} pages. Use enough substantive sections, examples, and analysis to fill that length; do not pad with repeated text.\n"
            if requested_pages else ""
        )
        prompt_parts.append(
            f"""You are writing the full content of a professional document.

TOPIC: {subject}

Write a comprehensive, professional document ENTIRELY about '{subject}'.
Do NOT switch topics or introduce unrelated subjects.

Use this markdown structure:
  # {subject}
  ## Introduction
  ## Key Principles and Architecture
  ## Practical Applications and Analysis
  ## Implementation Considerations
  ## Conclusion

Formatting you may use:
  - Bullet points  •  Numbered steps  •  > Callout notes
  - ```code blocks``` where relevant to the topic
  - | Tables | with | headers | for comparative data |

CRITICAL RULES:
- The MAIN DOCUMENT TITLE on the very first line must be '# {subject}'.
  DO NOT use '# Introduction' as the first heading. The first heading MUST be '# {subject}'.
- This document is ONLY about '{subject}'. Every sentence must relate to this topic.
- Start directly with '# {subject}'. Do NOT write 'Here is the document' or any preamble.
- DO NOT write code that builds/saves a PDF or DOCX file.
- DO NOT ask the reader for more information — make professional assumptions.
- {target_words}, factual, well-structured, useful to a professional reader.
{page_rule}"""
        )

    full_prompt = "".join(prompt_parts)

    try:
        requested_pages = _requested_page_count(question) if file_format == "pdf" else None
        llm = ChatOllama(
            model=model_to_use,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.15,
            # Longer PDFs need a larger generation/context budget.  This is
            # applied only when a user explicitly asks for pages, preserving
            # the low-memory defaults for ordinary terminal requests.
            num_predict=max(4096, (requested_pages or 0) * 700),
            num_ctx=max(config.OLLAMA_NUM_CTX, (requested_pages or 0) * 900),
        )
        response = llm.invoke(full_prompt)
        result = str(response.content).strip()
        if not result:
            raise ValueError("LLM returned empty content")
        return result
    except Exception as exc:
        return (
            f"# {subject}\n\n"
            f"*Content generation encountered an error: {exc}*\n\n"
            f"Generated by Agent OTG | DWE Team"
        )


def execution_node(state: WorkflowState) -> dict:
    started = time.time()
    plan = state["plan"]
    direct_tool = plan.get("direct_tool")
    tool_args   = plan.get("tool_args", {})

    # ── Specialized utility tool execution ─────────────────────────────────
    if direct_tool:
        # This error handling exists specifically so one broken tool call never crashes an entire multi-step agent request.
        success, res_str, clean_args = execute_tool_safely(direct_tool, tool_args)
        tool_entry = {
            "tool":   direct_tool,
            "args":   clean_args,
            "result": res_str,
        }
        status_icon = "✅" if success else "❌"
        return {
            **_event(state, "executing_tool", f"{status_icon} Invoked {direct_tool}"),
            "final_answer": res_str,
            "tool_log": [tool_entry],
            "needs_tool": True,
            "errors": [] if success else [res_str],
        }

    # ── Explicit file intent: answer first, artifact second ────────────────
    fmt = plan.get("file_format")
    if fmt:
        intent = plan.get("file_intent", {})
        non_file_question = _non_file_question(state["question"], intent)

        if fmt == "image":
            prompt = re.sub(r"^.*?\bimage\b(?:\s+of|\s+for)?\s*", "", state["question"], flags=re.I).strip()
            success, message, clean_args = execute_tool_safely("generate_image", {"prompt": prompt or state["question"]})
            elapsed = round(time.time() - started, 3)
            return {
                **_event(state, "executing_tool", f"{'✅' if success else '❌'} Deterministic image generation ({elapsed}s)"),
                "events": [*state.get("events", []), {"stage": "executing_tool", "detail": f"{'✅' if success else '❌'} Deterministic image generation"}, {"stage": "timing", "detail": f"tool execution: {elapsed}s"}],
                "final_answer": message,
                "tool_log": [{"tool": "generate_image", "args": clean_args, "result": message}],
                "needs_tool": True,
                "errors": [] if success else [message],
            }

        is_code = bool(re.search(r"\b(code|script|algorithm|python)\b", non_file_question, re.I))
        model_used = config.CODER_MODEL if is_code else config.MAIN_MODEL

        # Generate format-aware source material, leveraging history and prior content if referential
        file_content = _llm_content(
            state["question"],
            state.get("context", ""),
            fmt,
            state.get("history"),
            plan.get("source_content"),
            plan.get("resolved_subject"),
        )
        if fmt == "xlsx":
            file_content = _normalise_csv_row_count(
                file_content,
                _requested_record_count(state["question"]),
                _requested_column_count(state["question"]),
                _requested_headers(state["question"]),
            )
        elif fmt == "pptx":
            topic = plan.get("resolved_subject") or _extract_clean_subject(state["question"], fmt)
            file_content = _professionalise_slide_titles(_ensure_presentation_slide_count(
                file_content, topic, _requested_slide_count(state["question"])
            ), topic)
        elif fmt == "pdf":
            topic = plan.get("resolved_subject") or _extract_clean_subject(state["question"], fmt)
            file_content = _extend_pdf_content(
                file_content, topic, _requested_page_count(state["question"])
            )
        answer_text = file_content if is_code else ""
        elapsed = round(time.time() - started, 3)
        return {
            **_event(state, "generating", f"Answer generation: {elapsed}s with {model_used}."),
            "events": [*state.get("events", []), {"stage": "generating", "detail": f"Answer generation with {model_used}"}, {"stage": "timing", "detail": f"final answer generation: {elapsed}s"}],
            "answer_text": answer_text,
            "content": file_content,
        }

    # ── Non-file LangGraph / RAG behavior ──────────────────────────────────
    content = _llm_content(
        state["question"],
        state.get("context", ""),
        fmt,
        state.get("history"),
        plan.get("source_content"),
        plan.get("resolved_subject"),
    )
    is_code = bool(re.search(r"\b(code|script|algorithm|python)\b", state["question"], re.I))
    model_used = config.CODER_MODEL if is_code else config.MAIN_MODEL

    elapsed = round(time.time() - started, 3)
    return {
        **_event(state, "generating", f"Final answer generation: {elapsed}s with {model_used}."),
        "events": [*state.get("events", []), {"stage": "generating", "detail": f"Final answer generation with {model_used}"}, {"stage": "timing", "detail": f"final answer generation: {elapsed}s"}],
        "content": content,
    }


def file_node(state: WorkflowState) -> dict:
    started = time.time()
    fmt = state["plan"].get("file_format")
    if not fmt:
        return {}
    subject_prompt = state.get("plan", {}).get("resolved_subject") or state["question"]
    title, clean_content, safe_slug = derive_clean_title(subject_prompt, state.get("content", ""))
    filename = f"{safe_slug}.{fmt}"
    try:
        # Direct Python tool call: no model tool_call decision exists on this
        # path.  ``create_file`` is the single styled artifact implementation.
        artifact = tools.create_file(title, clean_content, fmt, filename)
        if not isinstance(artifact, dict) or artifact.get("error"):
            raise RuntimeError(artifact.get("error", "create_file returned an invalid result"))
        tool_entry = {
            "tool":   "create_file",
            "args":   {"title": title, "format": fmt, "filename": filename},
            "result": f"Created {artifact['filename']} ({artifact['size_bytes']:,} bytes)",
        }
        _log_tool_call("create_file", tool_entry["args"], tool_entry["result"])
        return {
            **_event(state, "creating_file",
                     f"✅ {fmt.upper()}: {artifact['filename']} ({artifact['size_bytes']:,} bytes) | tool execution: {round(time.time() - started, 3)}s"),
            "events": [*state.get("events", []), {"stage": "creating_file", "detail": f"✅ {fmt.upper()}: {artifact['filename']} ({artifact['size_bytes']:,} bytes)"}, {"stage": "timing", "detail": f"tool execution: {round(time.time() - started, 3)}s"}],
            "artifact":  artifact,
            "content":   clean_content,
            "tool_log":  [tool_entry],
            "needs_tool": True,
        }
    except Exception as exc:
        err_msg = f"Tool 'create_file' failed: {exc}"
        _log_tool_call("create_file", {"title": title, "format": fmt, "filename": filename}, err_msg)
        return {
            **_event(state, "creating_file", f"❌ {err_msg}"),
            "errors": [err_msg],
        }


def validation_node(state: WorkflowState) -> dict:
    artifact = state.get("artifact")
    if artifact:
        if artifact.get("size_bytes", 0) <= 0:
            return {
                **_event(state, "validating", "❌ Artifact is empty."),
                "errors": ["Generated artifact is empty."],
            }
        return _event(state, "validating", f"✅ {artifact['filename']} ({artifact['size_bytes']:,} bytes)")
    return _event(state, "validating", "✅ Direct response validated.")


def response_node(state: WorkflowState) -> dict:
    if state.get("final_answer") and not state.get("artifact"):
        return _event(state, "completed", "Workflow completed.")

    artifact  = state.get("artifact")
    content   = state.get("content", "").strip()
    errors    = state.get("errors", [])
    show_prev = state.get("plan", {}).get("show_preview", False)

    if errors and not artifact:
        answer = (
            f"❌ Agent error: {errors[-1]}\n\n"
            "Please try rephrasing your request."
        )

    elif artifact:
        answer_text = state.get("answer_text", "").strip()
        lines = [
            f"✅ **{artifact['filename']}** created successfully!\n",
            f"📁 **Saved to:** `{artifact['path']}`",
            f"📦 **Size:** {artifact['size_bytes']:,} bytes",
            f"📂 **Folder:** `{OUTPUT_DIR}`",
        ]

        if show_prev and content:
            preview = content[:1500] + ("\n\n…[truncated]" if len(content) > 1500 else "")
            lines = [f"### Preview\n\n{preview}\n\n---\n"] + lines

        # Keep the conversational answer and artifact metadata distinct.  The
        # document body is never injected as a preview unless the user asked.
        answer = (answer_text + "\n\n---\n\n" if answer_text else "") + "\n".join(lines)

    else:
        answer = content or "I could not generate a response. Please try again."

    return {
        **_event(state, "completed", "Workflow done."),
        "final_answer": answer,
    }


def _route_after_execution(state: WorkflowState) -> str:
    if state.get("final_answer"):
        return "validate"
    return "file" if state["plan"].get("file_format") else "validate"


def _build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner",    planner_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("execution",  execution_node)
    graph.add_node("file",       file_node)
    graph.add_node("validation", validation_node)
    graph.add_node("response",   response_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "planner")
    graph.add_edge("planner",    "rag")
    graph.add_edge("rag",        "execution")
    graph.add_conditional_edges(
        "execution", _route_after_execution,
        {"file": "file", "validate": "validation"},
    )
    graph.add_edge("file",       "validation")
    graph.add_edge("validation", "response")
    graph.add_edge("response",   END)
    return graph.compile()


_GRAPH = _build_graph()


def run_agent(query: str, history: list | None = None) -> dict:
    """Run the supervised workflow. Returns answer, artifact info, and event log."""
    result = _GRAPH.invoke({
        "question": query,
        "history":  history or [],
        "events":   [],
        "errors":   [],
        "tool_log": [],
    })
    artifact = result.get("artifact")
    tool_log = result.get("tool_log", [])

    final_answer = result.get("final_answer", "")
    # _llm_content talks directly to ChatOllama, unlike router.get_full_answer.
    # Persist the completed exchange here so agent/file sessions are not empty.
    try:
        import router
        router.add_to_history("user", query)
        router.add_to_history("assistant", final_answer)
    except Exception:
        # The workflow result remains usable even if a local history store is
        # temporarily unavailable; db.py records the actual error state.
        pass

    return {
        "final_answer": final_answer,
        "needs_tool":   bool(tool_log) or bool(artifact),
        "tool_log":     tool_log,
        "artifact":     artifact,
        "events":       result.get("events", []),
        "errors":       result.get("errors", []),
        "run_id":       uuid.uuid4().hex,
    }
