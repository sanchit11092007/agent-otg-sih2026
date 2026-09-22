# Offline Deployment

1. Install Python 3.11+ and Ollama on the target machine.
2. From this directory, install dependencies:

```powershell
py -m pip install -r requirements.txt
py -m pip install -r requirements-dev.txt
```

3. Pull local models before air-gapping the machine:

```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:latest
ollama pull qwen2.5vl:7b
ollama pull nomic-embed-text
```

4. Optionally configure model names and storage paths in `.env`. No cloud key is required.
   For OCR, install the local Tesseract executable and ensure `tesseract` is on `PATH`; the Python adapter is included in `requirements.txt`.
5. Start the API:

```powershell
py -m uvicorn main:app --host 127.0.0.1 --port 8000
```

6. Verify with `GET /health`, then use `POST /ask/agent` for complete file workflows. Generated files are available at the returned `download_url`.

Session data is stored in `agent_otg.sqlite3`; generated artifacts are stored in `generated_files/`; Chroma persists local RAG data under `chroma_db/`.
