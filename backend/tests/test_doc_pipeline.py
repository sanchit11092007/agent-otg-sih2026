import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from ask import parse_doc_command
from rag.loaders import (
    validate_file_pre_ingestion,
    _load_csv,
    _load_text,
    load_file,
)
from rag.pipeline import categorize_ingestion_error, ingest_path


def test_parse_doc_command_quoted_with_prompt():
    raw = '"C:\\Users\\iamsa\\Downloads\\etp.csv" Analyze this file and give me report in 10 points only'
    filepath, prompt = parse_doc_command(raw)
    assert filepath == os.path.normpath("C:\\Users\\iamsa\\Downloads\\etp.csv")
    assert prompt == "Analyze this file and give me report in 10 points only"


def test_parse_doc_command_quoted_without_prompt():
    raw = '"C:\\Users\\iamsa\\Downloads\\etp.csv"'
    filepath, prompt = parse_doc_command(raw)
    assert filepath == os.path.normpath("C:\\Users\\iamsa\\Downloads\\etp.csv")
    assert prompt == ""


def test_parse_doc_command_unquoted():
    raw = "C:\\data\\report.pdf Summarize this document"
    filepath, prompt = parse_doc_command(raw)
    assert filepath == os.path.normpath("C:\\data\\report.pdf")
    assert prompt == "Summarize this document"


def test_pre_ingestion_validation_missing_file():
    with pytest.raises(FileNotFoundError, match="File not found"):
        validate_file_pre_ingestion("non_existent_file_12345.pdf")


def test_pre_ingestion_validation_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f_path = f.name
    try:
        with pytest.raises(ValueError, match="File is empty"):
            validate_file_pre_ingestion(f_path)
    finally:
        os.remove(f_path)


def test_pre_ingestion_validation_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"some content")
        f_path = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported format"):
            validate_file_pre_ingestion(f_path)
    finally:
        os.remove(f_path)


def test_csv_loading_encodings_and_chunks():
    csv_content = "Name,Age,City\nAlice,30,New York\nBob,25,London\n"
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8", delete=False) as f:
        f.write(csv_content)
        f_path = f.name
    try:
        docs = _load_csv(f_path)
        assert len(docs) == 2
        assert "Name: Alice" in docs[0].page_content
        assert "City: New York" in docs[0].page_content
        assert docs[0].metadata["content_type"] == "table"
        assert docs[0].metadata["loader_name"] == "CSVLoader"
        assert docs[0].metadata["rows_loaded"] == 2
    finally:
        os.remove(f_path)


def test_csv_loading_latin1_encoding():
    # Write latin1 encoded CSV
    csv_content = "Name,City\nRené,Montréal\n"
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="wb", delete=False) as f:
        f.write(csv_content.encode("latin1"))
        f_path = f.name
    try:
        docs = _load_csv(f_path)
        assert len(docs) == 1
        assert "René" in docs[0].page_content or "Montréal" in docs[0].page_content
    finally:
        os.remove(f_path)


def test_categorize_ingestion_error():
    assert "File not found" in categorize_ingestion_error(FileNotFoundError("No file"))
    assert "Permission denied" in categorize_ingestion_error(PermissionError("Denied"))
    assert "Unsupported format" in categorize_ingestion_error(ValueError("Unsupported format: .xyz"))
    assert "CSV parsing error" in categorize_ingestion_error(ValueError("CSV parsing error: bad delimiter"))
    assert "Encoding error" in categorize_ingestion_error(UnicodeDecodeError("utf-8", b"", 0, 1, "invalid byte"))
    assert "Vectorization error" in categorize_ingestion_error(RuntimeError("Ollama embedding connection error"))
    assert "Database storage error" in categorize_ingestion_error(RuntimeError("Chroma collection error"))


@patch("rag.pipeline.add_documents")
def test_ingest_path_verbose_stats(mock_add_docs):
    mock_add_docs.return_value = 2
    content = "Sample text document for RAG ingestion test."
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write(content)
        f_path = f.name
    try:
        stats = ingest_path(f_path, replace_existing=False)
        assert stats["file_exists"] is True
        assert stats["loader"] == "TextLoader"
        assert stats["documents_loaded"] == 1
        assert stats["chunks_generated"] > 0
        assert stats["status"] == "SUCCESS"
    finally:
        os.remove(f_path)


def test_is_file_path_arg_detection():
    from ask import is_file_path_arg

    # Quoted path
    is_f, fp, prompt = is_file_path_arg('"C:\\Users\\iamsa\\Downloads\\etp.csv"')
    assert is_f is True
    assert fp.endswith("etp.csv")
    assert prompt == ""

    # Quoted path with prompt
    is_f, fp, prompt = is_file_path_arg('"C:\\Users\\iamsa\\Downloads\\etp.csv" analyze this data in around 10 points')
    assert is_f is True
    assert fp.endswith("etp.csv")
    assert prompt == "analyze this data in around 10 points"

    # Plain question
    is_f, fp, prompt = is_file_path_arg("What are the main findings in the latest report?")
    assert is_f is False
