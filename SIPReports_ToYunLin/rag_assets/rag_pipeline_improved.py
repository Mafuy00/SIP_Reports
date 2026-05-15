from __future__ import annotations

"""Improved RAG retrieval pipeline addressing section title bias and relevance issues.

Key improvements:
1. Separate section headers from embedding to avoid title bias
2. Implement hybrid search (dense + keyword)
3. Add section-aware reranking
4. Better chunk filtering for relevance
"""


import os
from dotenv import load_dotenv
from pathlib import Path

# Load variables from .env file
load_dotenv()

# Get token from environment (skip validation to avoid network issues)
hf_token = os.getenv("HF_TOKEN", "").strip()
if hf_token and hf_token != "your_hf_token_here":
    # Set token as environment variable for HuggingFace library to use
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HF_HOME"] = str(Path(__file__).resolve().parent / "hf_cache")
    # Skip token validation; the library will use it automatically for downloads
else:
    if not hf_token:
        print("Info: HF_TOKEN not set in .env file. Proceeding with limited Hugging Face access.")

# Keep your existing imports below
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from functools import lru_cache
from textwrap import fill
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
LAST_CONTEXT_PATH = RAG_DIR / "last_retrieved_context.txt"


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

    # Build TF-IDF vectorizer for keyword search
    chunk_texts = [record.get("chunk_text", "") for record in chunk_records]
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)

    return {
        "chunk_records": chunk_records,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "chunk_metadata": chunk_metadata,
        "model": model,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }


def _infer_response_style(question: str) -> str:
    question_lower = question.strip().lower()

    extraction_keywords = (
        "list ",
        "list all",
        "name all",
        "enumerate",
        "what are the",
        "which tools",
        "which software",
        "extract",
        "sector",
        "sectors",
        "industry",
        "industries",
        "company sector",
        "company sectors",
        "company type",
        "company types",
        "business sector",
        "business sectors",
        "organization sector",
        "organisation sector",
    )

    entity_collection_keywords = (
        "software",
        "tool",
        "tools",
        "technology",
        "technologies",
        "programming language",
        "programming languages",
        "language",
        "languages",
        "framework",
        "frameworks",
        "platform",
        "platforms",
        "system",
        "systems",
        "application",
        "applications",
        "sector",
        "sectors",
        "industry",
        "industries",
        "company",
        "companies",
    )

    if any(
        question_lower.startswith(prefix)
        for prefix in ("what ", "which ", "name ", "list ", "show ", "tell me ")
    ) and any(keyword in question_lower for keyword in entity_collection_keywords):
        return "list"

    if any(
        keyword in question_lower
        for keyword in extraction_keywords
    ):
        return "list"

    if any(keyword in question_lower for keyword in ("compare", "difference", "versus", " vs ")):
        return "comparison"

    if question_lower.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "should ", "would ", "has ", "have ", "had ")):
        return "yes_no"

    if any(
        keyword in question_lower
        for keyword in ("how ", "why ", "explain", "describe", "summarize", "summarise", "overview")
    ):
        return "explanation"

    return "general"


def _response_style_guidance(question: str) -> str:
    style = _infer_response_style(question)

    if style == "list":
        return (
            "Respond as a concise bullet list. If the question asks for items, entities, software, or tools, "
            "list the extracted items directly and avoid long prose. If the question is about sectors, industries, "
            "or company types, list all distinct sectors mentioned across the retrieved context rather than selecting only one."
        )
    if style == "comparison":
        return (
            "Respond as a short comparison. Use bullets or a compact table-style structure if it helps clarity."
        )
    if style == "yes_no":
        return (
            "Start with a direct yes/no style answer when supported, then give one short sentence of evidence."
        )
    if style == "explanation":
        return (
            "Respond with a concise explanatory paragraph. Keep the answer grounded in the retrieved context."
        )

    return "Respond in the most natural concise format for the question, grounded only in the retrieved context."


def _answer_format_instruction(question: str) -> str:
    style = _infer_response_style(question)

    if style == "list":
        return (
            "Format the answer as bullets only. Start each bullet with '- '. "
            "If the context contains categories, keep the categories as short headings and list items beneath them. "
            "For sector or industry questions, include every distinct supported sector and do not collapse them into a single sector unless the context only supports one."
        )

    if style == "comparison":
        return (
            "Format the answer as either a compact Markdown table or two short bullet groups labeled 'A' and 'B'. "
            "Only compare what is supported by the context."
        )

    if style == "yes_no":
        return (
            "Format the answer as a direct first sentence beginning with Yes or No when the context supports it, "
            "followed by one short evidence sentence."
        )

    if style == "explanation":
        return (
            "Format the answer as 1-2 short paragraphs. Avoid bullets unless the context itself is a list."
        )

    return "Format the answer in the clearest concise structure for the question."


def _format_context_for_terminal(context: str, width: int = 110) -> str:
    blocks = [block.strip() for block in context.split("\n\n---\n\n") if block.strip()]
    if not blocks:
        return context

    formatted_blocks: list[str] = []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue

        header = lines[0]
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        formatted = [header]
        if body:
            formatted.append(fill(body, width=width))
        formatted_blocks.append("\n".join(formatted))

    return "\n\n---\n\n".join(formatted_blocks)


def _save_context_snapshot(question: str, context: str) -> Path:
    LAST_CONTEXT_PATH.write_text(
        f"Question: {question}\n\n{context}\n",
        encoding="utf-8",
    )
    return LAST_CONTEXT_PATH


def get_context(query: str, k: int = 3, use_hybrid: bool = True, alpha: float = 0.6) -> str:
    """Return the top-k matching chunk texts for a query.
    
    Improvements:
    - Optional hybrid search combining dense and keyword retrieval
    - Section-aware reranking to avoid redundant sections
    - Better filtering of irrelevant chunks
    
    Args:
        query: The user's question
        k: Number of top chunks to retrieve
        use_hybrid: If True, use hybrid search (dense + keyword), else dense only
        alpha: Weight for dense search in hybrid (1-alpha for keyword)
    """
    assets = load_assets()
    model = assets["model"]
    vector_store = assets["vector_store"]
    chunk_records = assets["chunk_records"]
    chunk_metadata = assets["chunk_metadata"]
    tfidf_vectorizer = assets["tfidf_vectorizer"]
    tfidf_matrix = assets["tfidf_matrix"]

    if use_hybrid:
        # Hybrid retrieval: dense + keyword
        indices_and_scores = _hybrid_retrieval(
            query, model, vector_store, tfidf_vectorizer, tfidf_matrix,
            chunk_metadata, k=k*2, alpha=alpha
        )
    else:
        # Dense-only retrieval (original method)
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = vector_store.kneighbors(query_embedding, n_neighbors=k*2)
        indices_and_scores = [(idx, 1.0 - float(dist)) for idx, dist in zip(indices[0], distances[0])]

    # Section-aware reranking: prefer diverse sections and filter low-quality chunks
    deduplicated = _rerank_by_sections(indices_and_scores, chunk_metadata, k=k)

    retrieved_chunks = []
    for index, score, section_title in deduplicated:
        metadata = chunk_metadata[index]
        chunk_text = chunk_records[index].get("chunk_text", "")
        retrieved_chunks.append(
            f"[Source: {metadata.get('source_file', 'unknown')} | Section: {section_title} | Score: {score:.3f}]\n"
            f"{chunk_text}"
        )

    return "\n\n---\n\n".join(retrieved_chunks)


def _hybrid_retrieval(query: str, model, vector_store, tfidf_vectorizer, tfidf_matrix,
                      chunk_metadata, k: int = 6, alpha: float = 0.6) -> list[tuple[int, float]]:
    """Combine dense semantic search with keyword-based BM25-like search."""
    
    # Dense retrieval
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = vector_store.kneighbors(query_embedding, n_neighbors=k)
    dense_scores = 1.0 - distances[0]
    
    # Keyword retrieval
    query_tfidf = tfidf_vectorizer.transform([query])
    keyword_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    
    # Normalize keyword scores to [0, 1]
    if keyword_scores.max() > 0:
        keyword_scores_norm = keyword_scores / keyword_scores.max()
    else:
        keyword_scores_norm = keyword_scores
    
    # Combine scores
    combined = []
    for idx in indices[0]:
        dense_score = dense_scores[list(indices[0]).index(idx)]
        keyword_score = keyword_scores_norm[idx]
        
        # Weighted combination
        combined_score = alpha * dense_score + (1 - alpha) * keyword_score
        combined.append((idx, combined_score))
    
    # Sort by combined score
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:k]


def _rerank_by_sections(indices_and_scores: list[tuple[int, float]], chunk_metadata: list,
                        k: int = 3) -> list[tuple[int, float, str]]:
    """Rerank results to prefer diverse sections and filter low-quality chunks.
    
    Returns: List of (index, score, section_title) tuples, length <= k
    """
    seen_sections = set()
    result = []
    
    for index, score in indices_and_scores:
        if len(result) >= k:
            break
        
        metadata = chunk_metadata[index]
        section_title = metadata.get('section_title', 'Unknown')
        chunk_length = metadata.get('chunk_length', 0)
        
        # Filter out very short chunks that are mostly headers
        if chunk_length < 100:
            continue
        
        # Prefer chunks from different sections to avoid redundancy
        if section_title in seen_sections:
            # Still consider it, but with a penalty
            score = score * 0.8
        
        seen_sections.add(section_title)
        result.append((index, score, section_title))
    
    return result


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
    style_guidance = _response_style_guidance(question)
    format_guidance = _answer_format_instruction(question)
    return f"""Answer the question using ONLY the context below.
If the answer is not explicitly supported by the context, respond exactly with the sentence:
"I do not know based on the provided context."
Do NOT invent facts. Summarize only what is directly supported by the retrieved text.
If you can provide an answer grounded in the context, do NOT append any uncertainty note.
{style_guidance}
{format_guidance}
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
                "num_predict": max(96, max_answer_words + 96),
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


def _ollama_generate_http_stream(
    ollama_model: str,
    prompt: str,
    timeout_seconds: float,
    max_answer_words: int,
    num_ctx: int = 8192,
    keep_alive: str = "30m",
):
    payload = json.dumps(
        {
            "model": ollama_model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": keep_alive,
            "options": {
                "temperature": 0.2,
                "num_predict": max(96, max_answer_words + 96),
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
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            text = str(chunk.get("response", ""))
            if text:
                yield text
            if chunk.get("done"):
                break


def generate_answer(
    question: str,
    context_k: int = 6,
    ollama_model: str | None = None,
    fallback_models: tuple[str, ...] = OLLAMA_FALLBACK_MODELS,
    context: str | None = None,
    max_context_chars: int = 10000,
    max_answer_words: int = 320,
    ollama_num_ctx: int = 8192,
    timeout_seconds: float = 300.0,
    use_hybrid: bool = True,
) -> str:
    """Generate an answer using improved retrieval and LLM generation.
    
    Args:
        use_hybrid: If True, use hybrid search (dense + keyword)
    """
    if context is None:
        context = get_context(question, k=context_k, use_hybrid=use_hybrid)
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
        raw = _ollama_generate_http(
            ollama_model=ollama_model,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            max_answer_words=max_answer_words,
            num_ctx=ollama_num_ctx,
        )

        # Post-process to avoid an unnecessary hedging sentence when
        # the model already produced a substantive, grounded answer.
        ans = str(raw).strip()
        hedging = "I do not know based on the provided context."

        # If the model included the hedging sentence but also produced a
        # substantive answer (e.g. list items, commas, bullets, or >3 words),
        # remove the trailing hedging sentence so the answer reads cleanly.
        if hedging in ans:
            prefix = ans.split(hedging)[0].strip()
            # consider it substantive if there are several words or punctuation indicating a list
            if len(prefix.split()) > 3 or any(c in prefix for c in [",", "-", "•"]):
                ans = prefix.rstrip(". \n")

        return ans
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Ollama generation timed out after {timeout_seconds:.0f}s while loading or running '{ollama_model}'."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Failed to reach the local Ollama HTTP API at http://localhost:11434. "
            "Make sure the Ollama app is running."
        ) from exc


def generate_answer_stream(
    question: str,
    context_k: int = 6,
    ollama_model: str | None = None,
    fallback_models: tuple[str, ...] = OLLAMA_FALLBACK_MODELS,
    context: str | None = None,
    max_context_chars: int = 10000,
    max_answer_words: int = 320,
    ollama_num_ctx: int = 8192,
    timeout_seconds: float = 300.0,
    use_hybrid: bool = True,
):
    if context is None:
        context = get_context(question, k=context_k, use_hybrid=use_hybrid)
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
        for chunk in _ollama_generate_http_stream(
            ollama_model=ollama_model,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            max_answer_words=max_answer_words,
            num_ctx=ollama_num_ctx,
        ):
            yield chunk
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
    print("✓ Improved RAG terminal is ready.")
    print("✓ Using hybrid search (dense + keyword) with section-aware reranking.")
    print("\nType your question and press Enter. Type 'exit' to quit.")

    while True:
        question = input("\nEnter your question: ").strip()
        if not question:
            print("Please enter a question, or type 'exit' to quit.")
            continue
        if question.lower() == "exit":
            print("Exiting RAG terminal.")
            break

        print("\n--- Retrieved Context (Improved) ---\n")
        context = get_context(question, k=context_k, use_hybrid=True)
        context_snapshot = _save_context_snapshot(question, context)
        print(_format_context_for_terminal(context))
        print(f"\n[Full retrieved context saved to: {context_snapshot}]")

        print("\n--- Answer ---\n")
        try:
            answer = generate_answer(
                question,
                context_k=context_k,
                ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
                use_hybrid=True,
            )
            print(answer)
        except Exception as exc:
            print(f"Generation failed: {exc}")
            print("\nYou can still use the retrieved context above with your own local LLM call.")
