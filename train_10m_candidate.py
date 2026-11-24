"""Explicit, isolated 10M-parameter Quantum Candidate training profile.

This wrapper never changes Production. It delegates to bpe_candidate_train.py,
whose provenance audit remains the mandatory gate before any optimizer step.
Use --dry-run to verify the exact architecture without reading training data.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from bpe_candidate_train import BPECandidate, BLOCK

PROFILE = {
    "name": "quantum-candidate-10m",
    "vocab_size": 1024,
    "width": 496,
    "heads": 8,
    "layers": 3,
    "target_parameters": 10_000_000,
    "parameter_tolerance": 0.005,
    "context_length": BLOCK,
}


def inspect_profile():
    model = BPECandidate(
        vocab_size=PROFILE["vocab_size"],
        width=PROFILE["width"],
        heads=PROFILE["heads"],
        layers=PROFILE["layers"],
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(buffer.numel() for buffer in model.buffers())
    checkpoint_tensors = sum(tensor.numel() for tensor in model.state_dict().values())
    deviation = abs(trainable - PROFILE["target_parameters"]) / PROFILE["target_parameters"]
    if deviation > PROFILE["parameter_tolerance"]:
        raise RuntimeError(
            f"10M profile outside tolerance: {trainable:,} "
            f"(target {PROFILE['target_parameters']:,}, tolerance {PROFILE['parameter_tolerance']:.3%})"
        )
    if checkpoint_tensors != trainable + buffers:
        raise RuntimeError("checkpoint tensor accounting mismatch")
    return {
        **PROFILE,
        "trainable_parameter_count": trainable,
        "non_parameter_buffer_elements": buffers,
        "checkpoint_tensor_elements": checkpoint_tensors,
        "parameter_deviation_ratio": deviation,
        "parameter_count_verified": True,
        "checkpoint_accounting_verified": True,
        "training_started": False,
    }


def build_training_command(steps, seed):
    script = Path(__file__).resolve().with_name("bpe_candidate_train.py")
    return [
        sys.executable, str(script),
        "--steps", str(steps),
        "--seed", str(seed),
        "--vocab-size", str(PROFILE["vocab_size"]),
        "--width", str(PROFILE["width"]),
        "--heads", str(PROFILE["heads"]),
        "--layers", str(PROFILE["layers"]),
        "--target-parameters", str(PROFILE["target_parameters"]),
        "--parameter-tolerance", str(PROFILE["parameter_tolerance"]),
    ]


def main():
    parser = argparse.ArgumentParser(description="Quantum Candidate 10M profile")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="구조와 파라미터 수만 검증하고 데이터 로드/학습은 실행하지 않음",
    )
    args = parser.parse_args()
    if isinstance(args.steps, bool) or not 1 <= args.steps <= 5000:
        parser.error("--steps는 1~5000 범위여야 합니다.")
    if isinstance(args.seed, bool) or not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed는 0~2^32-1 범위여야 합니다.")

    result = inspect_profile()
    result["steps_requested"] = args.steps
    result["seed"] = args.seed
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    # The delegated trainer performs its own provenance audit before training.
    # It refuses to train if data rights/reviewer metadata are incomplete.
    return subprocess.run(build_training_command(args.steps, args.seed), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
