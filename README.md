# CSE 151B Competition — Qwen3-4B Math

## Hardware & runtime
- GPU: UCSD DSMLP pod, NVIDIA A30
- Approx. inference time: ~6-7 h for the full private set

## Setup
```bash
git clone https://github.com/Panteaforoutan/151B_SP26_Competition.git && cd 151B_SP26_Competition
./setup.sh                              # installs dependencies
source /scratch/.venv/bin/activate      # activate the environment
```

## Model weights
Base model `Qwen/Qwen3-4B-Thinking-2507` is downloaded automatically from
Hugging Face on first run by vLLM — no manual download and no LoRA adapter
needed. The final pipeline is the base model; there are no fine-tuned weights to load.

## Reproduce results
```bash
python scripts/run_inference.py
```
This runs the full pipeline end-to-end:
generate samples → self-consistency majority vote → evaluate → writes the
submission CSV to `submissions/final/best_greedy.csv`.