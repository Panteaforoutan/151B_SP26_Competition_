import os
os.environ["VLLM_USE_V1"] = "0"

import json
import time
import re
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

TRAIN_PATH = "data/math_train_transformed.jsonl"      # your external training data
PUBLIC_TEST_PATH = "data/public.jsonl"    # your project public test file

ADAPTER_DIR = "outputs/qwen3_4b_math_lora"

MAX_SEQ_LEN = 2048
NUM_TRAIN_EXAMPLES = 1000   # start small first

# ── Load tokenizer ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,)

if tokenizer.pad_token is None: # make arrays of tokens the same size for batching purpose
    tokenizer.pad_token = tokenizer.eos_token

EOS_TOKEN = "<|im_end|>" if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token

# ── Format data  ─────────────────────────────────────────────────────────────
def format_options(options):
    if options is None:
        return ""

    if isinstance(options, dict):
        return "\n".join(f"{key}. {value}" for key, value in options.items())

    if isinstance(options, list):
        lines = []
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)

            if isinstance(opt, dict):
                text = opt.get("text", opt.get("value", str(opt)))
            else:
                text = str(opt)

            lines.append(f"{letter}. {text}")

        return "\n".join(lines)

    return str(options)

# ── Test data ─────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


data = load_jsonl(PUBLIC_TEST_PATH)

# ── Build prompts for your public test set ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        opts_text = format_options(options)
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    return SYSTEM_PROMPT_MATH, question
        
    
# ── Load and run vLLM with LoRA adapter ────────────────────────────────────────────────────────────
llm = LLM( # load the base model + allow LoRA adapters 
    model=MODEL_ID,
    trust_remote_code=True,

    enable_lora=True,
    max_lora_rank=16,

    dtype="bfloat16",
    max_model_len=6144, 
    gpu_memory_utilization=0.80,

    enforce_eager=True # disable CUDA graph capture
)

# the problem is GPU memory and context length, 

# vLLM has to reserve memory for these before it can genereate 
# 1. The base Qwen3-4B model weights
# 2. Your LoRA adapter support
# 3. The KV cache for input/output tokens
# 4. Extra vLLM startup/warmup memory

# what max model len means : vLLM should be ready to handle up to 8192 total tokens, 
# total tokens : input prompt tokens + generated output tokens

# max model len = 4096 max input + output length during testing
# max output tokens: up to 512
# prompt/input tokens: up to 3584

# ── Generate Responses ─────────────────────────────────────────────────────────
# Sampling settings
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=768 # max output length only 
)

if not os.path.exists(ADAPTER_DIR):
    raise FileNotFoundError(f"LoRA adapter folder not found: {ADAPTER_DIR}")

# Use your LoRA adapter path
LORA_PATH = ADAPTER_DIR   # or directly set: LORA_PATH = "path/to/your/adapter"

start = time.time()

eval_data = data

prompts = []

for item in eval_data:
    system, user = build_prompt(
        item["question"],
        item.get("options")   # works for MCQ and free-form
    )

    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    prompts.append(prompt_text)

print(f"Generating responses for {len(prompts)} questions...")

outputs = llm.generate(
    prompts,
    sampling_params=sampling_params,
    lora_request=LoRARequest(
        "math_lora",  # adapter name
        1,            # adapter id
        LORA_PATH     # path to saved LoRA adapter
    ),
)

responses = [out.outputs[0].text.strip() for out in outputs]

# Preview first 3
for i in range(min(3, len(responses))):
    print(f"\n--- Example {i} ---")
    print("Question:", eval_data[i]["question"][:300])
    print("Response:", responses[i][:1000])

print(f"\nGeneration time: {time.time() - start:.1f}s")

# ── Save Results ──────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

OUTPUT_PATH = "results/starter_results_vllm_lora.jsonl"

with open(OUTPUT_PATH, "w") as f:
    for idx, (item, response) in enumerate(zip(eval_data, responses), start=1):
        row = {
            "id": item.get("id", idx),
            "response": response,
        }
        f.write(json.dumps(row) + "\n")

print(f"Saved results to {OUTPUT_PATH}")
    
# ── Score Responses ────────────────────────────────────────────────────────────
def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


# Load Judger for free-form scoring
sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

results = []
for item, response in tqdm(zip(eval_data, responses), total=len(eval_data), desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    if is_mcq:
        correct = score_mcq(response, str(gold))
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False

    results.append({
        "id":       item.get("id"),
        "is_mcq":   is_mcq,
        "gold":     gold,
        "response": response,
        "correct":  correct,
    })

print(f"Scoring complete. {len(results)} results.")

# ── Summary ────────────────────────────────────────────────────────────────────
mcq_res  = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]

def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
print("=" * 50)

# ── Save Results ───────────────────────────────────────────────────────────────
# SAVE_EVAL = True   # Set to False when running on the private test set

# out_path = Path(OUTPUT_PATH)
# os.makedirs("results", exist_ok=True)
# out_path.parent.mkdir(parents=True, exist_ok=True)

# with open(out_path, "w") as f:
#     for r in results:
#         if SAVE_EVAL:
#             record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
#                       "response": r["response"], "correct": r["correct"]}
#         else:
#             record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
#         f.write(json.dumps(record) + "\n")

# print(f"Saved {len(results)} records to {out_path}")

# ── Final Accuracy ─────────────────────────────────────────────────────────────
correct = sum(1 for r in results if r["correct"])
total = len(results)
print(f"LoRA accuracy: {correct/total:.2%}")
