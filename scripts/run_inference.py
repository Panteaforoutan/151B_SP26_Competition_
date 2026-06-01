#!/usr/bin/env python3
"""Single entry point: generate -> majority vote -> evaluate/submission."""
import subprocess, sys

def sh(cmd):
    print(f"\n=== {cmd} ===", flush=True)
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit(f"FAILED: {cmd}")


def _run_voted():
    cfg = "configs/final/voted_n3.yaml"
    sh(f"python scripts/generate_responses.py --config {cfg}")
    sh(f"python scripts/self_consistency_vote.py --in outputs/final/voted_n3_raw.jsonl --out outputs/final/voted_n3_voted.jsonl")
    sh(f"python scripts/evaluate_responses.py --config {cfg}")

def _run_greedy():
    cfg = "configs/final/best_greedy.yaml"
    sh(f"python scripts/generate_responses_old.py --config {cfg}")
    sh(f"python scripts/evaluate_responses.py --config {cfg}")

def run_inference():
    _run_greedy()
    # _run_voted() 
    
if __name__ == "__main__":
    run_inference()