"""Audit a Candidate checkpoint tokenizer against a held-out benchmark.

This performs no model inference or training. It only loads the tokenizer payload,
measures Unicode OOV coverage and token granularity, and writes a separate report.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from bpe_tokenizer import BPETokenizer
from checkpoint_validation import validate_tokenizer_payload, tokenizer_fingerprint, validate_architecture
from run_generation_benchmark import validate_benchmark_dataset
from tokenizer_diagnostics import summarize_tokenizer_coverage


DEFAULT_BENCHMARK = Path(__file__).resolve().parent / "evaluation" / "benchmark_v2.json"


def audit_checkpoint_tokenizer(checkpoint_path, benchmark_path, output_path):
    checkpoint_file = Path(checkpoint_path)
    benchmark_file = Path(benchmark_path)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing tokenizer audit report.")
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema_version") != 1:
        raise ValueError("A schema-v1 Candidate checkpoint is required.")
    architecture = validate_architecture(payload.get("architecture"), block_size=256)
    training = payload.get("training_metadata")
    if not isinstance(training, dict) or training.get("architecture") != payload.get("architecture"):
        raise ValueError("Candidate checkpoint training architecture metadata is inconsistent.")
    tokenizer_payload = payload.get("tokenizer")
    validate_tokenizer_payload(tokenizer_payload)
    tokenizer = BPETokenizer.from_dict(tokenizer_payload)
    actual_tokenizer_hash = tokenizer_fingerprint(tokenizer)
    if training.get("tokenizer_sha256") != actual_tokenizer_hash:
        raise ValueError("Candidate checkpoint tokenizer fingerprint does not match tokenizer payload.")

    dataset_raw = benchmark_file.read_bytes()
    dataset = json.loads(dataset_raw.decode("utf-8"))
    cases = validate_benchmark_dataset(dataset)
    full_inputs = [
        ((case.get("context", "").strip() + "\n\n") if case.get("context", "").strip() else "")
        + case["prompt"].strip()
        for case in cases
    ]
    report = summarize_tokenizer_coverage(
        tokenizer, full_inputs, label=dataset["benchmark_id"]
    )
    per_case = []
    for case, text, diagnostic in zip(cases, full_inputs, report["per_text"]):
        per_case.append({
            "case_id": case["id"],
            "category": case["category"],
            "unicode_codepoints": diagnostic["unicode_codepoints"],
            "encoded_tokens": diagnostic["encoded_tokens"],
            "unknown_tokens": diagnostic["unknown_tokens"],
            "unknown_token_ratio": diagnostic["unknown_token_ratio"],
            "unknown_codepoints": diagnostic["unknown_codepoints"],
            "known_single_codepoint_token_ratio": diagnostic["known_single_codepoint_token_ratio"],
            "exceeds_model_block": diagnostic["encoded_tokens"] > 256,
        })

    report_doc = {
        "report_schema": "juckex-quantum-tokenizer-coverage-v1",
        "benchmark_id": dataset["benchmark_id"],
        "benchmark_version": dataset["version"],
        "benchmark_sha256": hashlib.sha256(dataset_raw).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint_file.read_bytes()).hexdigest(),
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "checkpoint_role": "candidate_only",
        "training_metadata": {
            "parameter_count": training.get("parameter_count"),
            "tokenizer_sha256": training.get("tokenizer_sha256"),
            "architecture": training.get("architecture"),
        },
        "vocab_size": len(tokenizer.vocab),
        "model_block_size": architecture["block"],
        "coverage": {key: value for key, value in report.items() if key != "per_text"},
        "per_case": per_case,
        "cases_with_unknown_tokens": sum(1 for item in per_case if item["unknown_tokens"]),
        "cases_exceeding_model_block": sum(1 for item in per_case if item["exceeds_model_block"]),
        "model_inference_performed": False,
        "model_training_performed": False,
        "production_model_touched": False,
        "interpretation_limit": (
            "Tokenizer input coverage only. It does not measure generated answer quality, "
            "language understanding, factuality, or model intelligence."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_doc


def main():
    parser = argparse.ArgumentParser(
        description="Audit a local Quantum Candidate checkpoint tokenizer against held-out benchmark v2."
    )
    parser.add_argument("--checkpoint", required=True, help="Local Candidate checkpoint .pt file")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output", default="quantum-candidate-tokenizer-coverage-v1.json")
    args = parser.parse_args()
    report = audit_checkpoint_tokenizer(args.checkpoint, args.benchmark, args.output)
    print(json.dumps({
        "ok": True,
        "benchmark_id": report["benchmark_id"],
        "cases": report["coverage"]["documents"],
        "unknown_tokens": report["coverage"]["unknown_tokens"],
        "unknown_token_ratio": report["coverage"]["unknown_token_ratio"],
        "cases_with_unknown_tokens": report["cases_with_unknown_tokens"],
        "cases_exceeding_model_block": report["cases_exceeding_model_block"],
        "model_inference_performed": False,
        "model_training_performed": False,
        "production_model_touched": False,
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
