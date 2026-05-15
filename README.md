# SIP_Reports
TP SIP Report analysis

## Local RAG Chat Website

The project now includes a local chatbot web app powered by the improved RAG pipeline.

### App Location
- `SIPReports_ToYunLin/app.py`

### Install Dependencies
From `SIPReports_ToYunLin`:

```bash
pip install -r requirements.txt
```

### Run Website Locally
From `SIPReports_ToYunLin`:

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

### Notes
- Chat endpoint: `POST /api/chat`
- Health endpoint: `GET /health`
- If local Ollama generation is unavailable, the chatbot gracefully falls back to retrieved context.
