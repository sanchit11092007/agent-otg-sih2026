import os
import json
import pandas as pd
import pymupdf as fitz  # PyMuPDF (fitz alias kept for compatibility)
from docx import Document


def convert_excel_to_json(file_path: str, sheet_name=0) -> str:
    """Reads an Excel file and returns it as a clean JSON string."""
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df.where(pd.notnull(df), None)
        json_data = df.to_json(orient="records", date_format="iso", force_ascii=False)
        parsed = json.loads(json_data)
        return json.dumps(parsed, indent=2)
    except Exception as exc:
        return f"Error reading Excel file: {exc}"


def convert_pdf_to_text(file_path: str) -> str:
    """
    Extracts text from a PDF page by page and returns flat plain text.
    This is the preferred format for LLM context injection.
    """
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    try:
        doc = fitz.open(file_path)
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if text:
                pages.append(f"[Page {page_num + 1}]\n{text}")
        doc.close()
        if not pages:
            return "No extractable text found in this PDF."
        return "\n\n".join(pages)
    except Exception as exc:
        return f"Error reading PDF: {exc}"


# Keep the old JSON version for backwards compatibility (used by tools.py)
def convert_pdf_to_json(file_path: str) -> str:
    """
    Extracts text from a PDF, page by page, as structured JSON.
    NOTE: For LLM prompts, prefer convert_pdf_to_text() for flatter output.
    """
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    try:
        doc = fitz.open(file_path)
        document_data = {
            "total_pages": len(doc),
            "pages": [],
        }
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            document_data["pages"].append({
                "page_number": page_num + 1,
                "content": text if text else None,
            })
        doc.close()
        return json.dumps(document_data, indent=2, ensure_ascii=False)
    except Exception as exc:
        return f"Error reading PDF: {exc}"


def process_word(file_path: str) -> str:
    """Reads a Word document and returns all its text as one string."""
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    try:
        doc = Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as exc:
        return f"Error reading Word document: {exc}"


def _clean_path(path: str) -> str:
    if not path:
        return ""
    p = str(path).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return os.path.normpath(p)


def process_csv(file_path: str) -> str:
    """Reads a CSV file with multi-encoding detection and returns clean text/JSON."""
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            df = df.where(pd.notnull(df), None)
            json_data = df.to_json(orient="records", date_format="iso", force_ascii=False)
            parsed = json.loads(json_data)
            return json.dumps(parsed, indent=2)
        except Exception:
            continue
    return f"Error reading CSV file with encodings {encodings}"


def process_text_file(file_path: str) -> str:
    """Reads a plain text, Markdown, or JSON file."""
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return f"Error reading text file at {file_path}"


def process_file(file_path: str) -> str:
    """
    Looks at a file's extension and reads it with the right function above.
    Returns plain text/JSON that can be handed to any of the AI models.
    """
    file_path = _clean_path(file_path)
    if not os.path.exists(file_path):
        return f"Error: file not found at {file_path}"

    extension = file_path.lower()

    if extension.endswith(".xlsx") or extension.endswith(".xls"):
        return convert_excel_to_json(file_path)
    elif extension.endswith(".csv"):
        return process_csv(file_path)
    elif extension.endswith(".docx"):
        return process_word(file_path)
    elif extension.endswith(".pdf"):
        # Use flat text extraction for LLM context (better than JSON blob)
        return convert_pdf_to_text(file_path)
    elif extension.endswith(".txt") or extension.endswith(".md") or extension.endswith(".json"):
        return process_text_file(file_path)
    else:
        return f"Error: unsupported file type for {file_path}. Supported: .xlsx, .xls, .csv, .docx, .pdf, .txt, .md, .json"

