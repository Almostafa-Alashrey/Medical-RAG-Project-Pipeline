# ============================================================
# backend/app.py
# ============================================================
"""
PressIQ backend API
====================
Thin Flask wrapper around `rag_pipeline.py`. Exposes:

  POST /api/chat        -> ask a question, get a grounded/cited answer
  GET  /api/history      -> list previously asked questions (+ filters)
  GET  /api/history/<id> -> full detail for one past question
  GET  /api/stats        -> pages indexed / totals for the Home & About pages
  GET  /api/health       -> readiness probe

Run with:  python app.py   (defaults to http://localhost:5000)
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid

from flask import Flask, jsonify, request

from rag_pipeline import pipeline, DOCUMENT_NAME, SOURCE_URL

app = Flask(__name__)

# Minimal CORS support (no extra dependency required) so the static
# frontend — served from a different origin/port — can call this API.
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    @app.after_request
    def _add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/api/<path:_any>", methods=["OPTIONS"])
    def _cors_preflight(_any):
        return ("", 204)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "output", "history.json")
_history_lock = threading.Lock()


def _load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(items):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2)


def _status_slug(status: str) -> str:
    return {
        "Allowed": "allowed",
        "Needs Caution": "needs_caution",
        "Refused": "refused",
    }.get(status, status.lower().replace(" ", "_"))


def _day_label(ts: float) -> str:
    today = time.strftime("%Y-%m-%d")
    yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
    entry_day = time.strftime("%Y-%m-%d", time.localtime(ts))
    if entry_day == today:
        return "Today"
    if entry_day == yesterday:
        return "Yesterday"
    return time.strftime("%b %d", time.localtime(ts))


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/stats")
def stats():
    pipeline.load()
    history = _load_history()
    return jsonify({
        "document_name": DOCUMENT_NAME,
        "source_url": SOURCE_URL,
        "guideline_year": 2021,
        "pages_indexed": pipeline.pages_indexed,
        "chunks_indexed": len(pipeline.final_chunks),
        "questions_asked": len(history),
        "cited_rate": 100,
        "available": "24/7",
    })


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    top_k = int(payload.get("top_k", 5))

    if not question:
        return jsonify({"error": "question is required"}), 400

    result = pipeline.answer_question(question, top_k=top_k)

    entry = {
        "id": str(uuid.uuid4()),
        "question": question,
        "status": result.get("status", "Error"),
        "status_slug": _status_slug(result.get("status", "Error")),
        "confidence": result.get("confidence", {}),
        "answer": result.get("answer", ""),
        "section_title": result.get("section_title"),
        "summary": result.get("summary"),
        "sources": result.get("sources", []),
        "sources_count": len(result.get("sources", [])),
        "timestamp": time.time(),
    }
    entry["day_label"] = _day_label(entry["timestamp"])

    with _history_lock:
        history = _load_history()
        history.insert(0, entry)
        _save_history(history[:500])

    return jsonify(entry)


@app.get("/api/history")
def history():
    status_filter = request.args.get("status")  # allowed | needs_caution | refused
    search = (request.args.get("q") or "").strip().lower()

    items = _load_history()
    if status_filter and status_filter != "all":
        items = [i for i in items if i.get("status_slug") == status_filter]
    if search:
        items = [i for i in items if search in i.get("question", "").lower()]

    return jsonify({"items": items, "total": len(items)})


@app.get("/api/history/<entry_id>")
def history_detail(entry_id: str):
    items = _load_history()
    for item in items:
        if item.get("id") == entry_id:
            return jsonify(item)
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    # Warm the pipeline once at startup so the first chat request is fast.
    threading.Thread(target=pipeline.load, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)