# Agent OTG — SIH 2026

100% on-premise AI assistant with smart routing, RAG document search, and local file generation (PDF, Word, Excel, PPT).

## Quick demo (for judges)

```powershell
.\start-demo.ps1
```

Then open **http://127.0.0.1:5173**

## Same-hotspot receiver

1. Start the demo on the main PC. The launcher prints a URL such as `http://192.168.x.x:5173`.
2. On the receiver PC connected to the same hotspot, open `http://MAIN-PC-IP:5173/receiver`.
3. Run a request on the main PC and share its four-digit PIN. Enter that PIN on the receiver page to view the live result or download its generated file.

All AI processing stays on the main PC; the receiver only reads the shared result. If Windows asks, allow Python/Node through the **Private networks** firewall.

### Prerequisites

1. **Ollama** running: `ollama serve`
2. **Models pulled**:
   ```powershell
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ollama pull qwen2.5-coder:latest
   ```

## Demo script

| Feature | What to do |
|---------|------------|
| **RAG** | Upload a PDF/TXT → ask *"Summarize the uploaded document"* |
| **Tools** | Click **PDF Document** → type a topic → download generated file |
| **Code** | Ask *"Write Python code for binary search"* |

## Project layout

- `frontend/` — React UI (showcase this, not the terminal)
- `backend/` — FastAPI API (auto-starts with `npm run dev`)
- `backend/ask.py` — Terminal client (optional, not needed for demo)
