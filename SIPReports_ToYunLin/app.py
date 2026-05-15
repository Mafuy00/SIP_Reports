from __future__ import annotations

from dataclasses import dataclass
from configparser import ConfigParser
import json
import time
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from rag_assets.rag_pipeline_improved import generate_answer, generate_answer_stream, get_context


@dataclass
class ChatConfig:
    context_k: int = 4
    use_hybrid: bool = True
    max_answer_words: int = 320
    max_context_chars: int = 8000
    ollama_num_ctx: int = 4096
    
    @staticmethod
    def from_file(path: str = "performance.ini") -> "ChatConfig":
        config = ConfigParser()
        if not config.read(path):
            return ChatConfig()
        try:
            return ChatConfig(
                context_k=config.getint("RETRIEVAL", "context_k", fallback=4),
                use_hybrid=config.getboolean("RETRIEVAL", "use_hybrid", fallback=True),
                max_answer_words=config.getint("GENERATION", "max_answer_words", fallback=320),
                max_context_chars=config.getint("RETRIEVAL", "max_context_chars", fallback=5000),
                ollama_num_ctx=config.getint("GENERATION", "ollama_num_ctx", fallback=4096),
            )
        except Exception:
            return ChatConfig()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    config = ChatConfig.from_file()
    timing_data = {}

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.get("/api/performance")
    def performance() -> Any:
        return jsonify({
            "config": {
                "context_k": config.context_k,
                "use_hybrid": config.use_hybrid,
                "max_answer_words": config.max_answer_words,
                "max_context_chars": config.max_context_chars,
                "ollama_num_ctx": config.ollama_num_ctx,
            },
            "last_response_secs": timing_data.get("last_response_secs", None),
            "speed_tips": [
                "Edit performance.ini to reduce max_context_chars",
                "Use phi3 model (fastest), mistral (medium), llama3 (slowest)",
                "Lower context_k to 3-4 chunks for speed",
                "Reduce max_answer_words for shorter responses",
            ]
        })

    @app.post("/api/chat")
    def chat() -> Any:
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        
        # Per-request overrides via payload (for quick tuning)
        context_k = payload.get("context_k", config.context_k)
        use_hybrid = payload.get("use_hybrid", config.use_hybrid)
        max_answer_words = payload.get("max_answer_words", config.max_answer_words)
        max_context_chars = payload.get("max_context_chars", config.max_context_chars)
        ollama_num_ctx = payload.get("ollama_num_ctx", config.ollama_num_ctx)

        if not question:
            return jsonify({"error": "Question is required."}), 400

        t0 = time.perf_counter()
        try:
            answer = generate_answer(
                question=question,
                context_k=context_k,
                use_hybrid=use_hybrid,
                max_answer_words=max_answer_words,
                max_context_chars=max_context_chars,
                ollama_num_ctx=ollama_num_ctx,
            )
            elapsed = time.perf_counter() - t0
            timing_data["last_response_secs"] = round(elapsed, 2)
            return jsonify({
                "answer": answer,
                "degraded": False,
                "elapsed_secs": round(elapsed, 2),
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            context = get_context(question, k=context_k, use_hybrid=use_hybrid)
            fallback = (
                "Local generation is unavailable right now. "
                "Showing retrieved report context instead:\n\n"
                f"{context}"
            )
            return jsonify(
                {
                    "answer": fallback,
                    "degraded": True,
                    "warning": f"Generation fallback triggered: {type(exc).__name__}",
                    "elapsed_secs": round(elapsed, 2),
                }
            )

    @app.post("/api/chat/stream")
    def chat_stream() -> Response:
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()

        context_k = payload.get("context_k", config.context_k)
        use_hybrid = payload.get("use_hybrid", config.use_hybrid)
        max_answer_words = payload.get("max_answer_words", config.max_answer_words)
        max_context_chars = payload.get("max_context_chars", config.max_context_chars)
        ollama_num_ctx = payload.get("ollama_num_ctx", config.ollama_num_ctx)

        if not question:
            return jsonify({"error": "Question is required."}), 400

        @stream_with_context
        def generate() -> Any:
            start = time.perf_counter()
            buffered = []
            try:
                for chunk in generate_answer_stream(
                    question=question,
                    context_k=context_k,
                    use_hybrid=use_hybrid,
                    max_answer_words=max_answer_words,
                    max_context_chars=max_context_chars,
                    ollama_num_ctx=ollama_num_ctx,
                ):
                    buffered.append(chunk)
                    yield json.dumps({"type": "delta", "text": chunk}) + "\n"

                answer = "".join(buffered).strip()
                elapsed = time.perf_counter() - start
                timing_data["last_response_secs"] = round(elapsed, 2)
                yield json.dumps(
                    {
                        "type": "done",
                        "answer": answer,
                        "degraded": False,
                        "elapsed_secs": round(elapsed, 2),
                    }
                ) + "\n"
            except Exception as exc:
                elapsed = time.perf_counter() - start
                context = get_context(question, k=context_k, use_hybrid=use_hybrid)
                fallback = (
                    "Local generation is unavailable right now. "
                    "Showing retrieved report context instead:\n\n"
                    f"{context}"
                )
                yield json.dumps(
                    {
                        "type": "done",
                        "answer": fallback,
                        "degraded": True,
                        "warning": f"Generation fallback triggered: {type(exc).__name__}",
                        "elapsed_secs": round(elapsed, 2),
                    }
                ) + "\n"

        return Response(generate(), mimetype="application/x-ndjson")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
