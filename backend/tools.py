"""
tools.py — Tool registry for Agent OTG

All generated files are saved to ~/Downloads/AgentOTG/ or specified output paths.
Tools are registered in TOOL_FUNCTIONS and TOOL_SCHEMAS for the LLM and LangGraph agent.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from artifacts import OUTPUT_DIR, SUPPORTED_FORMATS, create_artifact, derive_clean_title
from image_gen import generate_image

# ── Output directory (defaults to ~/Downloads/AgentOTG) ────────────────────────
_OUTPUT_DIR = str(OUTPUT_DIR)


def _clean_path(path: str) -> str:
    """Clean surrounding quotes, whitespace, and resolve relative paths into output directory."""
    if not isinstance(path, (str, os.PathLike)):
        return ""
    if not path:
        return ""
    p = str(path).strip().strip("<>").strip("&").strip()
    if not p or "\x00" in p:
        return ""
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()

    # Try resolving via RAG loaders if available
    try:
        from rag.loaders import resolve_file_path
        resolved = resolve_file_path(p)
        if os.path.isfile(resolved):
            return resolved
    except Exception:
        pass

    # If it's a bare filename or relative path, save inside _OUTPUT_DIR
    if not os.path.isabs(p):
        return os.path.normpath(os.path.join(_OUTPUT_DIR, os.path.basename(p)))
    return os.path.normpath(p)


def _valid_output_path(filename: object, extension: str) -> str:
    """Return a validated output path inside AgentOTG, or an empty string."""
    path = _clean_path(filename)
    if not path:
        return ""
    if not path.lower().endswith(extension):
        path += extension
    # Generated artifacts must never write through a caller-controlled absolute
    # path.  Existing input readers may still open explicit local files.
    return os.path.join(_OUTPUT_DIR, os.path.basename(path))


# ── write_docx tool ──────────────────────────────────────────────────────────

def write_docx(title: str, content: str, filename: str = "output.docx") -> str:
    """Write a styled Word (.docx) document with colored heading, underline bar, tables, and styled lists."""
    try:
        from artifacts import _generate_docx
        if not isinstance(title, str) or not isinstance(content, str):
            return "❌ Title and content must be non-empty text."

        clean_title, clean_content, safe_slug = derive_clean_title(title, content)
        title = clean_title
        content = clean_content

        target_name = f"{safe_slug}.docx" if not filename or any(p in filename.lower() for p in ["output", "document", "give", "write", "create", "generate", "agent", "prompt"]) else filename
        target_path = _valid_output_path(target_name, ".docx")
        if not target_path:
            return "❌ Invalid output filename."
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        _generate_docx(title, content, Path(target_path))
        size = os.path.getsize(target_path)
        return f"✅ Word document saved to: {target_path}\n   Size: {size:,} bytes"
    except Exception as exc:
        return f"❌ Failed to create Word document: {exc}"


# ── RAG tools ────────────────────────────────────────────────────────────────

def search_documents(query: str, n_results: int = 8) -> str:
    """Search the RAG knowledge base and return a grounded answer with sources."""
    try:
        from rag.pipeline import answer
        result = answer(query, top_k=n_results)
        sources = result.get("sources", [])
        ans = result.get("answer", "I couldn't find sufficient information in the provided knowledge base.")

        if not sources:
            return ans

        # Deduplicate sources for display
        seen = set()
        source_lines = []
        for source in sources:
            src_name = source.get("source", "unknown")
            page = source.get("page")
            sheet = source.get("sheet")
            key = (src_name, page, sheet)
            if key in seen:
                continue
            seen.add(key)
            line = f"  • {src_name}"
            if page is not None:
                line += f", page {page}"
            if sheet is not None:
                line += f", sheet {sheet}"
            source_lines.append(line)

        return ans + "\n\nSources:\n" + "\n".join(source_lines)

    except Exception as exc:
        return (
            f"Knowledge base search failed. Make sure documents have been ingested with /doc <file>.\n"
            f"Error: {exc}"
        )


def ingest_document(text: str, source: str, chunk_size: int = 500) -> str:
    """Add raw text content to the internal knowledge base for later retrieval."""
    try:
        from langchain_core.documents import Document
        from rag.splitter import split_documents
        from rag.vectorstore import add_documents

        doc = Document(
            page_content=text,
            metadata={
                "source": source,
                "file_name": source,
                "content_type": "text",
            },
        )
        chunks = split_documents([doc])
        count = add_documents(chunks)
        return f"✅ Ingested {count} chunk(s) from '{source}' into the knowledge base."
    except Exception as exc:
        return f"❌ Knowledge base unavailable. Ingestion error: {exc}"


def ingest_file(file_path: str, replace_existing: bool = True) -> str:
    """Ingest PDF/DOCX/TXT/MD/CSV/XLS/XLSX/JSON/image into the RAG KB."""
    clean_p = _clean_path(file_path)
    if not os.path.isfile(clean_p):
        return f"❌ File not found: {clean_p}"
    try:
        from rag.pipeline import ingest_path
        result = ingest_path(clean_p, replace_existing=replace_existing)
        chunks = result.get("chunks_indexed", result.get("chunks_generated", 0))
        return (
            f"✅ Ingested '{os.path.basename(clean_p)}' into knowledge base.\n"
            f"   Chunks indexed: {chunks}"
        )
    except Exception as exc:
        return f"❌ File ingestion failed: {exc}"


# ── PDF tools ────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all readable text from a PDF file on disk."""
    clean_p = _clean_path(pdf_path)
    if not clean_p:
        return "❌ Invalid PDF path."
    if not os.path.isfile(clean_p):
        return f"❌ File not found: {clean_p}"
    try:
        import pdfplumber
        with pdfplumber.open(clean_p) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
        return text or "No text could be extracted from this PDF."
    except Exception as exc:
        return f"❌ Error reading PDF: {exc}"


def ocr_image(image_path: str) -> str:
    """Extract text from a local image using a locally installed Tesseract runtime."""
    path = _clean_path(image_path)
    if not path:
        return "❌ Invalid image path."
    if not os.path.isfile(path):
        return f"❌ File not found: {path}"
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(path)).strip()
        return text or "No readable text was found in the image."
    except ImportError:
        return "❌ OCR unavailable. Install pytesseract and the local Tesseract OCR runtime."
    except Exception as exc:
        return f"❌ OCR failed: {exc}"


def merge_pdfs(pdf_paths, output_filename: str = "merged.pdf") -> str:
    """Merge multiple PDF files into a single PDF saved to output directory."""
    # Handle JSON string input (from LLM tool call)
    if isinstance(pdf_paths, str):
        try:
            parsed = json.loads(pdf_paths)
            if isinstance(parsed, list):
                pdf_paths = parsed
        except Exception:
            pdf_paths = [p.strip() for p in pdf_paths.split(",") if p.strip()]

    if not isinstance(pdf_paths, (list, tuple)):
        return "❌ pdf_paths must be a list of at least two PDF paths."
    cleaned_paths = [_clean_path(p) for p in pdf_paths]
    if not cleaned_paths or any(not p for p in cleaned_paths):
        return "❌ Invalid PDF path supplied."
    missing = [p for p in cleaned_paths if not os.path.isfile(p)]
    if missing:
        return f"❌ Files not found: {missing}"
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        for path in cleaned_paths:
            writer.append(path)
        output_path = _valid_output_path(output_filename or "merged.pdf", ".pdf")
        if not output_path:
            return "❌ Invalid output filename."
        with open(output_path, "wb") as f:
            writer.write(f)
        return f"✅ Merged {len(cleaned_paths)} PDF(s) → {output_path}"
    except Exception as exc:
        return f"❌ Error merging PDFs: {exc}"


def generate_pdf_from_text(title: str, content: str, filename: str = "output.pdf") -> str:
    """Generate a styled PDF document from text with custom ReportLab styles, horizontal rule, and page footers."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.platypus.flowables import HRFlowable

        if not isinstance(title, str) or not isinstance(content, str):
            return "❌ Title and content must be non-empty text."

        clean_title, clean_content, safe_slug = derive_clean_title(title, content)
        title = clean_title
        content = clean_content

        target_name = f"{safe_slug}.pdf" if not filename or any(p in filename.lower() for p in ["output", "document", "give", "write", "create", "generate", "agent", "prompt"]) else filename
        target_path = _valid_output_path(target_name, ".pdf")
        if not target_path:
            return "❌ Invalid output filename."
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        teal_hex = "#087E70"
        title_style = ParagraphStyle(
            "StyledTitle",
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor(teal_hex),
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "StyledBody",
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#18212B"),
            spaceAfter=6,
        )
        bullet_style = ParagraphStyle(
            "StyledBullet",
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#18212B"),
            leftIndent=18,
            firstLineIndent=-10,
            spaceAfter=4,
        )

        # Determine effective title
        clean_t, clean_c, safe_slug = derive_clean_title(title, content)
        title = clean_t
        content = clean_c
        effective_title = title.strip() if isinstance(title, str) and title.strip() else "Executive Document"
        if effective_title.lower() in ("document", "introduction", "overview", "summary", "output"):
            effective_title = "Executive Document"
        # Escape XML/HTML entities for safe PDF rendering
        clean_title = effective_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story = []
        story.append(Paragraph(clean_title, title_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(teal_hex), spaceAfter=12))

        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 4))
                continue
            escaped = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if stripped.startswith(("- ", "* ")):
                bullet_txt = escaped[2:].strip()
                story.append(Paragraph(f"<font color='{teal_hex}'>•</font> {bullet_txt}", bullet_style))
            else:
                story.append(Paragraph(escaped, body_style))

        def _draw_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#718096"))
            canvas.drawString(0.75 * inch, 0.4 * inch, "Agent OTG | Team DWE")
            canvas.drawRightString(A4[0] - 0.75 * inch, 0.4 * inch, f"Page {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(
            target_path,
            pagesize=A4,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)

        size = os.path.getsize(target_path)
        return f"✅ PDF saved to: {target_path}\n   Size: {size:,} bytes"
    except Exception as exc:
        return f"❌ Failed to create PDF: {exc}"


def generate_pptx(title: str, slides_content: list, filename: str = "output.pptx") -> str:
    """Generate a styled PowerPoint presentation (.pptx) with title slide and content slides."""
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN

        if isinstance(slides_content, str):
            try:
                parsed = json.loads(slides_content)
                if isinstance(parsed, list):
                    slides_content = parsed
            except Exception:
                pass

        if not isinstance(title, str):
            return "❌ title must be text."

        if isinstance(slides_content, str):
            art = create_artifact(title, slides_content, "pptx", filename)
            return f"✅ PowerPoint presentation saved to: {art.path}\n   Size: {art.size_bytes:,} bytes"

        if not isinstance(slides_content, list):
            return "❌ slides_content must be a list or formatted markdown string."

        clean_title, _, safe_slug = derive_clean_title(title, str(slides_content))
        title = clean_title

        target_name = f"{safe_slug}.pptx" if not filename or any(p in filename.lower() for p in ["output", "document", "give", "write", "create", "generate", "agent", "prompt"]) else filename
        target_path = _valid_output_path(target_name, ".pptx")
        if not target_path:
            return "❌ Invalid output filename."
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)

        teal_color = RGBColor(8, 126, 112)
        dark_teal  = RGBColor(6, 95, 84)
        white      = RGBColor(255, 255, 255)
        ink        = RGBColor(24, 33, 43)

        # ── 1. Title Slide ──────────────────────────────────────────────────
        slide_layout = prs.slide_layouts[6]
        title_slide = prs.slides.add_slide(slide_layout)

        accent_shape = title_slide.shapes.add_shape(
            1, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
        )
        accent_shape.fill.solid()
        accent_shape.fill.fore_color.rgb = dark_teal
        accent_shape.line.fill.background()

        stripe = title_slide.shapes.add_shape(
            1, Inches(0), Inches(0), Inches(13.333), Inches(0.2)
        )
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = teal_color
        stripe.line.fill.background()

        tb = title_slide.shapes.add_textbox(Inches(1.5), Inches(2.2), Inches(10.333), Inches(2.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(title)
        run.font.name = "Calibri"
        run.font.size = Pt(36)
        run.font.bold = True
        run.font.color.rgb = white

        p_sub = tf.add_paragraph()
        p_sub.alignment = PP_ALIGN.CENTER
        p_sub.space_before = Pt(14)
        run_sub = p_sub.add_run()
        run_sub.text = "Agent OTG  •  Team DWE  •  Executive Deck"
        run_sub.font.name = "Calibri"
        run_sub.font.size = Pt(14)
        run_sub.font.color.rgb = RGBColor(160, 212, 206)

        # ── 2. Content Slides ────────────────────────────────────────────────
        for item in (slides_content or []):
            if isinstance(item, dict):
                heading = item.get("heading", "Topic")
                bullets = item.get("bullets", [])
            else:
                heading = str(item)
                bullets = []

            slide = prs.slides.add_slide(slide_layout)

            header_bar = slide.shapes.add_shape(
                1, Inches(0), Inches(0), Inches(13.333), Inches(1.1)
            )
            header_bar.fill.solid()
            header_bar.fill.fore_color.rgb = dark_teal
            header_bar.line.fill.background()

            sub_stripe = slide.shapes.add_shape(
                1, Inches(0), Inches(1.1), Inches(13.333), Inches(0.04)
            )
            sub_stripe.fill.solid()
            sub_stripe.fill.fore_color.rgb = teal_color
            sub_stripe.line.fill.background()

            h_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.15), Inches(11.5), Inches(0.8))
            h_tf = h_box.text_frame
            h_tf.word_wrap = True
            hp = h_tf.paragraphs[0]
            hrun = hp.add_run()
            hrun.text = str(heading)
            hrun.font.name = "Calibri"
            hrun.font.size = Pt(24)
            hrun.font.bold = True
            hrun.font.color.rgb = white

            c_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.5), Inches(5.2))
            c_tf = c_box.text_frame
            c_tf.word_wrap = True

            first_para = True
            for b in bullets:
                b_str = str(b).strip()
                if not b_str:
                    continue
                para = c_tf.paragraphs[0] if first_para else c_tf.add_paragraph()
                first_para = False
                para.space_after = Pt(10)
                para.level = 0
                brun = para.add_run()
                brun.text = f"▸ {b_str}"
                brun.font.name = "Calibri"
                brun.font.size = Pt(16)
                brun.font.color.rgb = ink

        prs.save(target_path)
        slides_count = len(slides_content or []) + 1
        return f"✅ PowerPoint presentation saved to: {target_path}\n   Slides: {slides_count}"
    except Exception as exc:
        return f"❌ Failed to create PowerPoint presentation: {exc}"


def generate_xlsx(title: str, headers: list[str], rows: list[list], filename: str = "output.xlsx") -> str:
    """Generate a styled Excel (.xlsx) spreadsheet with bold headers, alternating rows, borders, and auto-width."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        if isinstance(headers, str):
            try:
                parsed = json.loads(headers)
                if isinstance(parsed, list):
                    headers = parsed
            except Exception:
                headers = [h.strip() for h in headers.split(",") if h.strip()]
        if isinstance(rows, str):
            try:
                parsed = json.loads(rows)
                if isinstance(parsed, list):
                    rows = parsed
            except Exception:
                pass
        if not isinstance(title, str) or not isinstance(headers, list) or not isinstance(rows, list):
            return "❌ title must be text; headers and rows must be lists."

        clean_title, _, safe_slug = derive_clean_title(title, "")
        title = clean_title

        target_name = f"{safe_slug}.xlsx" if not filename or any(p in filename.lower() for p in ["output", "document", "give", "write", "create", "generate", "agent", "prompt"]) else filename
        target_path = _valid_output_path(target_name, ".xlsx")
        if not target_path:
            return "❌ Invalid output filename."
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        wb = Workbook()
        ws = wb.active
        safe_sheet_name = re.sub(r"[\\/*?:\[\]]", "", title)[:30] or "Sheet1"
        ws.title = safe_sheet_name

        header_fill = PatternFill(start_color="087E70", end_color="087E70", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        alt_fill = PatternFill(start_color="E8F5F3", end_color="E8F5F3", fill_type="solid")
        body_font = Font(name="Calibri", size=10, color="18212B")
        body_align = Alignment(vertical="center")

        thin_side = Side(border_style="thin", color="CBD5E0")
        thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

        ws.append(headers or ["Column 1"])
        for col_idx in range(1, len(headers or [1]) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            cell.border = thin_border
        ws.row_dimensions[1].height = 24

        for row_idx, row_data in enumerate(rows or [], start=2):
            from artifacts import _parse_cell_value
            parsed_row = [_parse_cell_value(c) for c in row_data]
            ws.append(parsed_row)
            is_alt = (row_idx % 2 == 0)
            for col_idx in range(1, len(row_data) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.font = body_font
                if isinstance(cell.value, (int, float)):
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                elif isinstance(cell.value, bool):
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = body_align
                cell.border = thin_border
                if is_alt:
                    cell.fill = alt_fill
            ws.row_dimensions[row_idx].height = 18

        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                if len(val) > max_len:
                    max_len = len(val)
            ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 45)

        wb.save(target_path)
        row_count = len(rows or [])
        col_count = len(headers or [])
        return f"✅ Excel spreadsheet saved to: {target_path}\n   Rows: {row_count} | Columns: {col_count}"
    except Exception as exc:
        return f"❌ Failed to create Excel spreadsheet: {exc}"


def create_file(title: str, content: str, file_format: str, filename: str = "output") -> dict:
    """Create and validate a local PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, or JSON artifact."""
    try:
        if not isinstance(title, str) or not isinstance(content, str) or not isinstance(file_format, str):
            raise ValueError("title, content, and file_format must be text values")
        if not title.strip() or not content.strip():
            raise ValueError("title and content cannot be empty")
        artifact = create_artifact(title, content, file_format, filename)
        return artifact.as_dict()
    except Exception as exc:
        return {"error": f"Failed to create file: {exc}", "title": str(title), "file_format": str(file_format)}


def split_pdf(pdf_path: str, output_folder: str = "") -> str:
    """Split a PDF into separate files, one per page, saved to output directory."""
    clean_p = _clean_path(pdf_path)
    if not clean_p:
        return "❌ Invalid PDF path."
    if not output_folder:
        base_name = os.path.splitext(os.path.basename(clean_p))[0]
        clean_out = os.path.join(_OUTPUT_DIR, f"{base_name}_pages")
    else:
        clean_out = _clean_path(output_folder)
        if not os.path.isabs(clean_out):
            clean_out = os.path.join(_OUTPUT_DIR, clean_out)

    if not os.path.isfile(clean_p):
        return f"❌ File not found: {clean_p}"
    try:
        from pypdf import PdfReader, PdfWriter
        os.makedirs(clean_out, exist_ok=True)
        reader = PdfReader(clean_p)
        base = os.path.splitext(os.path.basename(clean_p))[0]
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            out = os.path.join(clean_out, f"{base}_page_{i+1}.pdf")
            with open(out, "wb") as f:
                writer.write(f)
        return f"✅ Split into {len(reader.pages)} pages → {clean_out}"
    except Exception as exc:
        return f"❌ Error splitting PDF: {exc}"


def rotate_pdf_pages(pdf_path: str, degrees: int = 90, output_filename: str = "rotated.pdf") -> str:
    """Rotate all pages in a PDF by a given number of degrees."""
    clean_p = _clean_path(pdf_path)
    if not clean_p:
        return "❌ Invalid PDF path."
    if not os.path.isfile(clean_p):
        return f"❌ File not found: {clean_p}"
    try:
        from pypdf import PdfReader, PdfWriter
        deg = int(degrees)
        if deg not in (90, 180, 270):
            return "❌ degrees must be 90, 180, or 270."
        reader = PdfReader(clean_p)
        writer = PdfWriter()
        for page in reader.pages:
            page.rotate(deg)
            writer.add_page(page)

        output_path = _valid_output_path(output_filename or "rotated.pdf", ".pdf")
        if not output_path:
            return "❌ Invalid output filename."
        with open(output_path, "wb") as f:
            writer.write(f)
        return f"✅ Rotated pages by {deg}° → {output_path}"
    except Exception as exc:
        return f"❌ Error rotating PDF: {exc}"


def get_pdf_page_count(pdf_path: str) -> str:
    """Get the number of pages in a PDF file."""
    clean_p = _clean_path(pdf_path)
    if not clean_p:
        return "❌ Invalid PDF path."
    if not os.path.isfile(clean_p):
        return f"❌ File not found: '{pdf_path}'"
    try:
        from pypdf import PdfReader
        reader = PdfReader(clean_p)
        count = len(reader.pages)
        return f"📄 '{os.path.basename(clean_p)}' contains {count} page(s)."
    except Exception as exc:
        return f"❌ Error reading PDF: {exc}"


# ── File listing ─────────────────────────────────────────────────────────────

def list_generated_files() -> str:
    """Return a formatted string listing all files in ~/Downloads/AgentOTG/."""
    if not os.path.isdir(_OUTPUT_DIR):
        return f"Output directory not found: {_OUTPUT_DIR}\nFiles will be created here when you generate documents."

    files = sorted(f for f in os.listdir(_OUTPUT_DIR) if os.path.isfile(os.path.join(_OUTPUT_DIR, f)))
    if not files:
        return (
            f"No files generated yet in:\n  {_OUTPUT_DIR}\n\n"
            "Use /agent or just ask naturally (e.g. 'Create a PDF report on AI trends')"
        )

    lines = []
    for fname in files:
        fpath = os.path.join(_OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath)
        size_str = f"{size:,} B" if size < 1024 else f"{size // 1024:,} KB" if size < 1048576 else f"{size / 1048576:.1f} MB"
        lines.append(f"  • {fname}  ({size_str})")

    return (
        f"📁 Generated Files — {len(files)} file(s) in:\n"
        f"   {_OUTPUT_DIR}\n\n"
        + "\n".join(lines)
    )


# ── Registry ─────────────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict = {
    "create_file":            create_file,
    "write_docx":             write_docx,
    "generate_pdf_from_text": generate_pdf_from_text,
    "generate_pptx":          generate_pptx,
    "generate_xlsx":          generate_xlsx,
    "generate_image":         generate_image,
    "search_documents":       search_documents,
    "ingest_document":        ingest_document,
    "ingest_file":            ingest_file,
    "extract_pdf_text":       extract_pdf_text,
    "ocr_image":              ocr_image,
    "merge_pdfs":             merge_pdfs,
    "split_pdf":              split_pdf,
    "rotate_pdf_pages":       rotate_pdf_pages,
    "get_pdf_page_count":     get_pdf_page_count,
    "list_generated_files":   list_generated_files,
}


TOOL_SCHEMAS: list = [
    {"type": "function", "function": {
        "name": "generate_image",
        "description": "Generate an image from a text description using a local AI image model.",
        "parameters": {"type": "object", "properties": {
            "prompt":   {"type": "string", "description": "Text description of the image to generate"},
            "filename": {"type": "string", "description": "Output image filename, e.g. output.png"},
        }, "required": ["prompt"]},
    }},
    {"type": "function", "function": {
        "name": "generate_pptx",
        "description": "Create a styled PowerPoint presentation (.pptx) with title slide and content slides.",
        "parameters": {"type": "object", "properties": {
            "title":          {"type": "string", "description": "Presentation title"},
            "slides_content": {
                "type": "array",
                "description": "List of slide items (each has heading and bullets)",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string", "description": "Slide heading"},
                        "bullets": {"type": "array", "items": {"type": "string"}, "description": "List of bullet points"},
                    },
                    "required": ["heading", "bullets"],
                },
            },
            "filename": {"type": "string", "description": "Output filename, e.g. presentation.pptx"},
        }, "required": ["title", "slides_content"]},
    }},
    {"type": "function", "function": {
        "name": "generate_xlsx",
        "description": "Create a styled Excel spreadsheet (.xlsx) with bold header row, alternating row colors, and cell borders.",
        "parameters": {"type": "object", "properties": {
            "title":    {"type": "string", "description": "Spreadsheet title / sheet name"},
            "headers":  {"type": "array", "items": {"type": "string"}, "description": "List of column header names"},
            "rows":     {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": "List of data rows"},
            "filename": {"type": "string", "description": "Output filename, e.g. data.xlsx"},
        }, "required": ["title", "headers", "rows"]},
    }},
    {"type": "function", "function": {
        "name": "write_docx",
        "description": "Write a styled Word (.docx) document with colored heading, underline bar, and styled lists.",
        "parameters": {"type": "object", "properties": {
            "title":    {"type": "string", "description": "Document heading or title"},
            "content":  {"type": "string", "description": "Main body text with bullet points (- item) and headings (# section)."},
            "filename": {"type": "string", "description": "Output filename, e.g. report.docx"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "generate_pdf_from_text",
        "description": "Generate an advanced, professionally formatted PDF document from text with styled heading, horizontal rule, and page footers.",
        "parameters": {"type": "object", "properties": {
            "title":    {"type": "string", "description": "Document title"},
            "content":  {"type": "string", "description": "Document content in markdown format. Use # for sections and - for bullets."},
            "filename": {"type": "string", "description": "Output filename, e.g. report.pdf"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "create_file",
        "description": "Create and save a local PDF, DOCX, TXT, Markdown, PPTX, XLSX, CSV, or JSON file to ~/Downloads/AgentOTG/. Use this for any file creation request.",
        "parameters": {"type": "object", "properties": {
            "title":       {"type": "string", "description": "Document title or heading"},
            "content":     {"type": "string", "description": "Complete content for the file. Use markdown formatting (# headings, - bullets, ```code``` blocks)."},
            "file_format": {"type": "string", "enum": ["pdf", "docx", "txt", "md", "pptx", "xlsx", "csv", "json"], "description": "Output file format"},
            "filename":    {"type": "string", "description": "Output filename without extension"},
        }, "required": ["title", "content", "file_format"]},
    }},
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search the internal knowledge base using the RAG pipeline and answer using retrieved context.",
        "parameters": {"type": "object", "properties": {
            "query":     {"type": "string", "description": "Natural-language search query"},
            "n_results": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Maximum number of retrieved chunks (default: 8)"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "ingest_file",
        "description": "Ingest a PDF, DOCX, TXT, Markdown, CSV, Excel, JSON, or image file into the knowledge base for RAG retrieval.",
        "parameters": {"type": "object", "properties": {
            "file_path":        {"type": "string", "description": "Full path to the file on disk"},
            "replace_existing": {"type": "boolean", "description": "Replace previously indexed chunks from the same source (default: true)"},
        }, "required": ["file_path"]},
    }},
    {"type": "function", "function": {
        "name": "extract_pdf_text",
        "description": "Extract all readable text from a PDF file on disk.",
        "parameters": {"type": "object", "properties": {
            "pdf_path": {"type": "string", "description": "Full path or filename of the PDF file"},
        }, "required": ["pdf_path"]},
    }},
    {"type": "function", "function": {
        "name": "merge_pdfs",
        "description": "Merge multiple PDF files into a single PDF saved to output directory.",
        "parameters": {"type": "object", "properties": {
            "pdf_paths":       {"type": "array", "items": {"type": "string"}, "description": "List of full paths or filenames to PDFs, in merge order"},
            "output_filename": {"type": "string", "description": "Filename for the merged PDF (e.g. combined.pdf)"},
        }, "required": ["pdf_paths"]},
    }},
    {"type": "function", "function": {
        "name": "split_pdf",
        "description": "Split a PDF into separate single-page PDF files, saved to output directory.",
        "parameters": {"type": "object", "properties": {
            "pdf_path":      {"type": "string", "description": "Full path or filename of the input PDF"},
            "output_folder": {"type": "string", "description": "Subfolder name for split pages (optional)"},
        }, "required": ["pdf_path"]},
    }},
    {"type": "function", "function": {
        "name": "rotate_pdf_pages",
        "description": "Rotate all pages in a PDF by a given number of degrees and save the result.",
        "parameters": {"type": "object", "properties": {
            "pdf_path":        {"type": "string", "description": "Full path or filename of the input PDF"},
            "degrees":         {"type": "integer", "description": "Degrees to rotate: 90, 180, or 270"},
            "output_filename": {"type": "string", "description": "Output filename for the rotated PDF"},
        }, "required": ["pdf_path", "degrees"]},
    }},
    {"type": "function", "function": {
        "name": "get_pdf_page_count",
        "description": "Get the number of pages in a PDF file.",
        "parameters": {"type": "object", "properties": {
            "pdf_path": {"type": "string", "description": "Full path or filename of the PDF file"},
        }, "required": ["pdf_path"]},
    }},
    {"type": "function", "function": {
        "name": "ocr_image",
        "description": "Extract readable text from a local image using offline Tesseract OCR.",
        "parameters": {"type": "object", "properties": {
            "image_path": {"type": "string", "description": "Path to a PNG, JPEG, or WEBP image"},
        }, "required": ["image_path"]},
    }},
]
