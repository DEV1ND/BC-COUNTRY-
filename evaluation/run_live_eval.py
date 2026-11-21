#!/usr/bin/env python3
"""Run the held-out Quantum evaluation set against an isolated inference endpoint.

No training, model promotion, or production configuration is performed.
Set QUANTUM_API_TOKEN in the environment if the endpoint requires auth.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ENDPOINT = "https://juckex-quantum-inference.onrender.com"

def load_dataset(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("dataset must be a JSON object with a rows array")
    return payload["rows"]

def looks_like_abstention(answer):
    """Conservative phrase heuristic; this is not a semantic judgment."""
    text = re.sub(r"\\s+", " ", str(answer or "").casefold())
    phrases = (
        "알 수 없", "확인할 수 없", "정보가 없", "자료가 없",
        "접근할 수 없", "제공되지 않", "확인되지 않", "단정할 수 없",
        "실시간 정보", "알려진 바 없", "모르겠습니다", "모릅니다",
    )
    return any(p in text for p in phrases)

def request_answer(endpoint, prompt, token, timeout):
    body = json.dumps({"prompt": prompt, "max_new_tokens": 120, "temperature": 0.0},
                      ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/generate",
                                     data=body, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            elapsed = (time.perf_counter() - started) * 1000
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("endpoint response must be a JSON object")
            return parsed, elapsed, response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"endpoint connection failed: {exc.reason}") from exc

def run(rows, endpoint, token="", timeout=90, delay=0.25, limit=None):
    selected = rows[:limit] if limit else rows
    predictions, raw_rows, errors = [], [], []
    latencies = []
    for index, row in enumerate(selected, 1):
        record = {"id": row["id"], "category": row["category"]}
        try:
            response, latency, status = request_answer(endpoint, row["question"], token, timeout)
            answer = response.get("answer", "")
            if not isinstance(answer, str):
                answer = str(answer)
            abstained = looks_like_abstention(answer)
            predictions.append({
                "id": row["id"],
                "answer": answer,
                "abstained": abstained,
                "abstention_detection": "phrase_heuristic_requires_manual_review",
                "citations": response.get("knowledge_sources", []),
                "model": response.get("model"),
                "response_mode": response.get("response_mode"),
                "latency_ms": round(latency, 2),
                "http_status": status,
            })
            raw_rows.append({"id": row["id"], "question": row["question"], "response": response})
            latencies.append(latency)
            record.update({"status": "ok", "model": response.get("model"),
                           "response_mode": response.get("response_mode"),
                           "latency_ms": round(latency, 2), "answer_chars": len(answer)})
        except Exception as exc:
            errors.append({"id": row["id"], "error": str(exc)})
            record.update({"status": "error", "error": str(exc)})
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if delay and index < len(selected):
            time.sleep(delay)
    summary = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint.rstrip("/"),
        "dataset_version": "quantum-eval-v1",
        "attempted": len(selected),
        "succeeded": len(predictions),
        "failed": len(errors),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p50_latency_ms": round(sorted(latencies)[len(latencies)//2], 2) if latencies else None,
        "errors": errors,
        "abstention_detection": "heuristic_only; manually review before scoring",
        "production_model_changed": False,
        "training_performed": False,
    }
    return predictions, raw_rows, summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation/quantum_eval_v1.json")
    parser.add_argument("--endpoint", default=os.getenv("QUANTUM_EVAL_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token-env", default="QUANTUM_API_TOKEN")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", default="evaluation/results")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without making network requests")
    args = parser.parse_args()
    rows = load_dataset(args.dataset)
    if not args.endpoint.startswith("https://"):
        raise ValueError("endpoint must use HTTPS")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.dry_run:
        print(json.dumps({"status": "dry_run_validated", "questions": min(len(rows), args.limit or len(rows)),
                          "network_requests": 0, "training_performed": False}, ensure_ascii=False))
        return
    token = os.getenv(args.token_env, "")
    predictions, raw_rows, summary = run(rows, args.endpoint, token, args.timeout, args.delay, args.limit)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred_path, raw_path, summary_path = (out / f"predictions-{stamp}.jsonl",
                                         out / f"raw-{stamp}.jsonl",
                                         out / f"run-{stamp}.json")
    for path, records in ((pred_path, predictions), (raw_path, raw_rows)):
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    summary.update({"predictions_file": str(pred_path), "raw_responses_file": str(raw_path)})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
