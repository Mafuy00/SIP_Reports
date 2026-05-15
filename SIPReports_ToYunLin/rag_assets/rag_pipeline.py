"""RAG retrieval and generation pipeline for SIP reports.

This script loads the saved sentence-transformer embeddings, retrieves the
most relevant section chunks for a user question, and optionally sends the
assembled context to a local Ollama model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

RAG_DIR = Path(__file__).resolve().parent
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
OLLAMA_FALLBACK_MODELS = tuple(
    model.strip()
    for model in os.getenv("OLLAMA_FALLBACK_MODELS", "phi3,mistral,gemma2:2b,llama3.2,llama3").split(",")
    if model.strip()
)

CHUNK_JSON_PATH = RAG_DIR / "SIP_reports_section_chunks.json"
EMBEDDINGS_PATH = RAG_DIR / "rag_embeddings.npy"
VECTOR_STORE_PATH = RAG_DIR / "rag_vector_store.joblib"
METADATA_PATH = RAG_DIR / "rag_chunk_metadata.json"
MODEL_INFO_PATH = RAG_DIR / "rag_embedding_model.json"
HF_CACHE_DIR = RAG_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def load_assets() -> dict[str, Any]:
    if not CHUNK_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing chunk file: {CHUNK_JSON_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_PATH}")
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(f"Missing vector store file: {VECTOR_STORE_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    with CHUNK_JSON_PATH.open("r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    embeddings = np.load(EMBEDDINGS_PATH)
    vector_store = joblib.load(VECTOR_STORE_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)

    model = SentenceTransformer(MODEL_NAME, cache_folder=str(HF_CACHE_DIR))

    return {
        "chunk_records": chunk_records,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "chunk_metadata": chunk_metadata,
        "model": model,
    }


def get_context(query: str, k: int = 3) -> str:
    """Return the top-k matching chunk texts for a query."""
    assets = load_assets()
    model = assets["model"]
    vector_store = assets["vector_store"]
    chunk_records = assets["chunk_records"]
    chunk_metadata = assets["chunk_metadata"]

    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = vector_store.kneighbors(query_embedding, n_neighbors=k)

    retrieved_chunks = []
    for distance, index in zip(distances[0], indices[0]):
        metadata = chunk_metadata[index]
        chunk_text = chunk_records[index].get("chunk_text", "")
        retrieved_chunks.append(
            f"[Source: {metadata.get('source_file', 'unknown')} | Section: {metadata.get('section_title', 'unknown')} | Score: {1.0 - float(distance):.3f}]\n"
            f"{chunk_text}"
        )

    return "\n\n---\n\n".join(retrieved_chunks)


def _list_local_ollama_models(ollama_executable: str, timeout_seconds: float = 10.0) -> set[str]:
    """Return the set of model names currently available in local Ollama."""
    process = subprocess.run(
        [ollama_executable, "list"],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )

    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Failed to read local Ollama model list.")

    model_names: set[str] = set()
    for line in process.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        model_names.add(stripped.split()[0])
    return model_names


def _resolve_ollama_model(
    ollama_executable: str,
    preferred_model: str | None = None,
    fallback_models: tuple[str, ...] = (),
) -> str:
    local_models = _list_local_ollama_models(ollama_executable)
    if not local_models:
        raise RuntimeError(
            "No local Ollama models were found. Run `ollama pull <model>` first."
        )

    candidates: list[str] = []
    if preferred_model:
        candidates.append(preferred_model)
    candidates.extend(model for model in fallback_models if model and model not in candidates)
    candidates.extend(model for model in local_models if model not in candidates)

    for candidate in candidates:
        if candidate in local_models:
            return candidate

    available = ", ".join(sorted(local_models))
    requested = preferred_model or "unspecified"
    raise RuntimeError(
        f"Requested model '{requested}' is not available locally (found: {available}). "
        "Set OLLAMA_MODEL or OLLAMA_FALLBACK_MODELS to a locally installed model."
    )


def build_prompt(question: str, context: str, max_words: int = 220) -> str:
    return f"""Answer the question using ONLY the context below.
If the answer is not explicitly supported by the context, say: "I do not know based on the provided context."
Do not invent facts, summarize only what is grounded in the retrieved text.
Return a concise answer using at most {max_words} words.

Context:
{context}

Question: {question}
"""


def _ollama_generate_http(
    ollama_model: str,
    prompt: str,
    timeout_seconds: float,
    max_answer_words: int,
    num_ctx: int = 8192,
    keep_alive: str = "30m",
) -> str:
    payload = json.dumps(
        {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": 0.2,
                "num_predict": max(32, max_answer_words + 24),
                "top_p": 0.9,
                "num_ctx": num_ctx,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    if "response" not in response_payload:
        raise RuntimeError("Ollama HTTP API returned an unexpected response.")

    return str(response_payload["response"]).strip()


def generate_answer(
    question: str,
    context_k: int = 6,
    ollama_model: str | None = None,
    fallback_models: tuple[str, ...] = OLLAMA_FALLBACK_MODELS,
    context: str | None = None,
    max_context_chars: int = 10000,
    max_answer_words: int = 220,
    ollama_num_ctx: int = 8192,
    timeout_seconds: float = 300.0,
) -> str:
    if context is None:
        context = get_context(question, k=context_k)
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars].rsplit(" ", 1)[0] + "\n\n[Context truncated for faster local generation.]"
    prompt = build_prompt(question, context, max_words=max_answer_words)
    preferred_model = ollama_model or DEFAULT_OLLAMA_MODEL

    ollama_executable = shutil.which("ollama")
    if not ollama_executable:
        raise ImportError(
            "ollama is not installed in this environment and the Ollama CLI was not found on PATH."
        )

    ollama_model = _resolve_ollama_model(ollama_executable, preferred_model, fallback_models)

    try:
        return _ollama_generate_http(
            ollama_model=ollama_model,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            max_answer_words=max_answer_words,
            num_ctx=ollama_num_ctx,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Ollama generation timed out after {timeout_seconds:.0f}s while loading or running '{ollama_model}'."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Failed to reach the local Ollama HTTP API at http://localhost:11434. "
            "Make sure the Ollama app is running."
        ) from exc


if __name__ == "__main__":
    context_k = 5
    print("RAG terminal is ready. Type your question and press Enter.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nEnter your question: ").strip()
        if not question:
            print("Please enter a question, or type 'exit' to quit.")
            continue
        if question.lower() == "exit":
            print("Exiting RAG terminal.")
            break

        print("\n--- Retrieved Context ---\n")
        context = get_context(question, k=context_k)
        print(context)

        print("\n--- Answer ---\n")
        try:
            answer = generate_answer(
                question,
                context_k=context_k,
                ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            )
            print(answer)
        except Exception as exc:
            print(f"Generation failed: {exc}")
            print("\nYou can still use the retrieved context above with your own local LLM call.")
