"""File -> LangChain Document normalization.

We deliberately return Documents, not JSON strings. Embedding models need text;
JSON is useful as a transport/structured-data format but is not a prerequisite.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


def _clean_text(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").replace("\x00", "").splitlines()]
    # Preserve paragraph boundaries while removing excessive blank lines.
    out, blank = [], 0
    for line in lines:
        if line.strip():
            out.append(line.strip())
            blank = 0
        elif blank == 0:
            out.append("")
            blank = 1
    return "\n".join(out).strip()


def _base_metadata(path: str) -> dict:
    p = Path(path)
    stat = p.stat()
    return {
        "source": str(p),
        "file_name": p.name,
        "file_type": p.suffix.lower().lstrip("."),
        "document_id": hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:24],
        "file_size": stat.st_size,
        "modified_time": stat.st_mtime,
    }


def resolve_file_path(path: str) -> str:
    """Resolve a user-provided file path, checking common locations if relative."""
    if not path:
        return ""
    p = str(path).strip().strip("<>").strip()
    if p.startswith("&"):
        p = p[1:].strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    if p.startswith("file:///"):
        p = p[8:]
    norm = os.path.normpath(p)
    if os.path.isfile(norm):
        return norm

    # Check relative to cwd
    cwd_candidate = os.path.normpath(os.path.join(os.getcwd(), norm))
    if os.path.isfile(cwd_candidate):
        return cwd_candidate

    # Check in Downloads/AgentOTG
    otg_dir = Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads" / "AgentOTG"
    otg_candidate = otg_dir / Path(norm).name
    if otg_candidate.is_file():
        return str(otg_candidate)

    # Check in user's Downloads
    dl_candidate = Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads" / Path(norm).name
    if dl_candidate.is_file():
        return str(dl_candidate)

    # Check in user's Documents
    doc_candidate = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents" / Path(norm).name
    if doc_candidate.is_file():
        return str(doc_candidate)

    return norm


def validate_file_pre_ingestion(path: str) -> dict:
    clean_p = resolve_file_path(path)
    if not os.path.exists(clean_p):
        searched = [
            clean_p,
            str(Path(os.getcwd()) / Path(path).name),
            str(Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads" / "AgentOTG" / Path(path).name),
            str(Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads" / Path(path).name),
        ]
        raise FileNotFoundError(
            f"File not found: '{clean_p}'\n"
            f"Searched paths:\n  • " + "\n  • ".join(searched)
        )
    if not os.path.isfile(clean_p):
        raise FileNotFoundError(f"Path is not a regular file: '{clean_p}'")
    if not os.access(clean_p, os.R_OK):
        raise PermissionError(f"Permission denied: cannot read file '{clean_p}'")
    size = os.path.getsize(clean_p)
    if size == 0:
        raise ValueError(f"File is empty (0 bytes): '{clean_p}'")
    ext = Path(clean_p).suffix.lower()
    supported = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".webp"}
    if ext not in supported:
        raise ValueError(f"Unsupported format: '{ext}'. Supported extensions: {', '.join(sorted(supported))}")
    return {"exists": True, "size": size, "extension": ext, "resolved_path": clean_p}


def _load_pdf(path: str) -> list[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader
    docs = PyMuPDFLoader(path).load()
    base = _base_metadata(path)
    base["loader_name"] = "PyMuPDFLoader"
    out = []
    for i, d in enumerate(docs, 1):
        text = _clean_text(d.page_content)
        if not text:
            continue
        meta = {**base, **d.metadata, "page": i, "content_type": "text"}
        out.append(Document(page_content=text, metadata=meta))
    return out


def _load_docx(path: str) -> list[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    docs = Docx2txtLoader(path).load()
    base = _base_metadata(path)
    base["loader_name"] = "Docx2txtLoader"
    return [
        Document(page_content=_clean_text(d.page_content), metadata={**base, **d.metadata, "content_type": "text"})
        for d in docs if _clean_text(d.page_content)
    ]


def _load_text(path: str) -> list[Document]:
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    content = None
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError as err:
            last_err = err
            continue
    if content is None:
        raise ValueError(f"Text encoding error: {last_err}")
    clean = _clean_text(content)
    if not clean:
        return []
    base = _base_metadata(path)
    base["loader_name"] = "TextLoader"
    return [Document(page_content=clean, metadata={**base, "content_type": "text"})]


def _load_json(path: str) -> list[Document]:
    # Try multiple encodings to handle BOM or Latin-1 encoded JSON files.
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    data = None
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                data = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            last_err = err
            continue
    if data is None:
        raise ValueError(f"JSON loading error: {last_err}")
    text = json.dumps(data, ensure_ascii=False, indent=2)
    base = _base_metadata(path)
    base["loader_name"] = "JSONLoader"
    return [Document(page_content=text, metadata={**base, "content_type": "json"})]


def _load_csv(path: str) -> list[Document]:
    import pandas as pd
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    df = None
    last_err = None
    used_enc = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            used_enc = enc
            break
        except (UnicodeDecodeError, pd.errors.ParserError, Exception) as err:
            last_err = err
            continue

    if df is None:
        if isinstance(last_err, UnicodeDecodeError):
            raise UnicodeDecodeError(
                last_err.encoding or "utf-8",
                last_err.object or b"",
                last_err.start or 0,
                last_err.end or 0,
                f"Encoding error reading CSV file with utf-8, utf-8-sig, latin1, cp1252: {last_err}"
            )
        elif isinstance(last_err, pd.errors.ParserError):
            raise ValueError(f"CSV parsing error: {last_err}")
        else:
            raise ValueError(f"CSV loading error: {last_err}")

    base = _base_metadata(path)
    base["loader_name"] = "CSVLoader"
    base["encoding"] = used_enc
    df = df.dropna(how="all")
    rows_count = len(df)
    base["rows_loaded"] = rows_count

    if df.empty:
        return []

    df = df.where(pd.notnull(df), "")
    headers = [str(c) for c in df.columns]
    out: list[Document] = []
    for row_number, row in enumerate(df.itertuples(index=False, name=None), 2):
        pairs = [f"{headers[i]}: {row[i]}" for i in range(len(headers)) if str(row[i]).strip()]
        if pairs:
            text = f"File: {base['file_name']}\nRow: {row_number}\n" + "\n".join(pairs)
            out.append(Document(
                page_content=text,
                metadata={**base, "row": row_number, "content_type": "table"},
            ))
    return out


def _load_excel(path: str) -> list[Document]:
    # Tables are kept as meaningful text for embeddings. We do NOT stringify a
    # whole workbook into one giant JSON blob.
    import pandas as pd
    workbook = pd.ExcelFile(path)
    base = _base_metadata(path)
    base["loader_name"] = "ExcelLoader"
    out: list[Document] = []
    total_rows = 0
    for sheet in workbook.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df = df.dropna(how="all")
        if df.empty:
            continue
        total_rows += len(df)
        df = df.where(pd.notnull(df), "")
        headers = [str(c) for c in df.columns]
        for row_number, row in enumerate(df.itertuples(index=False, name=None), 2):
            pairs = [f"{headers[i]}: {row[i]}" for i in range(len(headers)) if str(row[i]).strip()]
            if pairs:
                text = f"Sheet: {sheet}\nRow: {row_number}\n" + "\n".join(pairs)
                out.append(Document(
                    page_content=text,
                    metadata={**base, "sheet": str(sheet), "row": row_number, "content_type": "table"},
                ))
    for d in out:
        d.metadata["rows_loaded"] = total_rows
    return out


def _load_image(path: str) -> list[Document]:
    # Image bytes are not embedded directly in the text vector store. They are
    # converted to a searchable visual description by Qwen-VL during ingestion.
    import base64
    import ollama
    from .config import OLLAMA_BASE_URL, RAG_VISION_MODEL

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    client = ollama.Client(host=OLLAMA_BASE_URL)
    prompt = (
        "Describe this image for enterprise knowledge-base retrieval. "
        "Extract visible text, labels, tables, diagrams, entities, relationships, "
        "numbers, and important visual facts. Do not invent anything."
    )
    response = client.chat(model=RAG_VISION_MODEL, messages=[{
        "role": "user", "content": prompt, "images": [b64]
    }])
    text = _clean_text(response["message"]["content"])
    base = _base_metadata(path)
    base["loader_name"] = "ImageLoader"
    return [Document(page_content=text, metadata={**base, "content_type": "image", "modality": "image"})] if text else []


def load_file(path: str) -> list[Document]:
    val = validate_file_pre_ingestion(path)
    clean_path = val["resolved_path"]
    ext = Path(clean_path).suffix.lower()
    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".txt": _load_text,
        ".md": _load_text,
        ".csv": _load_csv,
        ".json": _load_json,
        ".xlsx": _load_excel,
        ".xls": _load_excel,
        ".png": _load_image,
        ".jpg": _load_image,
        ".jpeg": _load_image,
        ".webp": _load_image,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported format: '{ext}'")
    return loaders[ext](clean_path)


def load_files(paths: Iterable[str]) -> list[Document]:
    all_docs: list[Document] = []
    for path in paths:
        all_docs.extend(load_file(path))
    return all_docs

