# RAG Chatbot Speed Optimization Guide

## Where the Time Goes

Your chatbot workflow has these phases:

1. **Load Models** (~2s first time only)
   - SentenceTransformer embedding model loads
   - Ollama model loads (happens in background)

2. **Retrieve Context** (~1-2s per question)
   - Encode question with embeddings
   - Hybrid semantic + keyword search
   - Section deduplication
   - **This is already very fast**

3. **Generate Answer** (~5-60s per question) ← **THE BOTTLENECK**
   - Send context + question to Ollama
   - LLM generates response word by word
   - Model speed depends on:
     - Model choice (phi3 fastest, llama3 slowest)
     - Context size (more context = slower)
     - Answer length (longer answers = slower)
     - Hardware (CPU vs GPU)

## Quick Speed Wins

### 1. Use a Faster Model (Biggest Impact)
Edit `.env`:
```
OLLAMA_MODEL=phi3
```

Speed comparison:
- **phi3** (small, fast) → ~3-5 seconds per answer
- **mistral** (medium) → ~8-12 seconds per answer
- **llama3** (large, slower) → ~20-40 seconds per answer

Test immediately:
```bash
# Install phi3 if you don't have it
ollama pull phi3

# Make sure it's available
ollama list
```

### 2. Edit performance.ini
Located at `SIPReports_ToYunLin/performance.ini`:

**For Speed (current defaults are already tuned):**
```ini
[RETRIEVAL]
context_k = 3              # Retrieve only 3 chunks instead of 6
max_context_chars = 3000   # Reduce context window (each 1000 chars ≈ +2-3s)

[GENERATION]
max_answer_words = 120     # Shorter answers = faster (each word ≈ 0.1s)
ollama_num_ctx = 2048      # Reduce Ollama context window
```

**For Quality (slower but better answers):**
```ini
[RETRIEVAL]
context_k = 8
max_context_chars = 12000

[GENERATION]
max_answer_words = 300
ollama_num_ctx = 8192
```

Restart the app to apply changes:
```bash
python app.py
```

### 3. Skip Ollama When It's Down
If Ollama isn't running, the chatbot will show retrieved context instead (graceful fallback). This is instant.

## Check Your Performance

Open your browser and visit:
```
http://127.0.0.1:5000/api/performance
```

This shows:
- Current configuration
- Last response time in seconds
- Speed optimization tips

Each answer now includes timing info in the chat UI.

## Advanced: Per-Request Tuning

Send JSON with custom parameters in the API:

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What sectors are mentioned?",
    "context_k": 3,
    "max_answer_words": 100,
    "max_context_chars": 2000
  }'
```

The response will include `"elapsed_secs": 5.2` so you can see the impact.

## Hardware Matters

- **CPU-only**: Expect 5-15 seconds per answer with phi3
- **GPU-enabled Ollama**: Expect 1-3 seconds with phi3
- **Old laptop**: May be 30+ seconds, consider using smaller model or context

Check if Ollama is using GPU:
```bash
ollama list  # Shows which model you're using
```

## Summary Table

| Setting | Speed Impact | Quality Impact |
|---------|-------------|----------------|
| Model: phi3 | Fast ✓ | Good ✓ |
| Model: llama3 | Slow ✗ | Better ✗ |
| context_k=3 | Faster ✓ | May miss info ✗ |
| context_k=8 | Slower ✗ | More complete ✓ |
| max_answer_words=120 | Faster ✓ | Less detail ✗ |
| max_context_chars=3000 | Faster ✓ | Less context ✗ |

**Start with:**
- phi3 model
- context_k = 4
- max_answer_words = 180
- max_context_chars = 5000

Then adjust based on your needs.
