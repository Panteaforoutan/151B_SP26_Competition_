import os
os.environ["VLLM_USE_V1"] = "0"

import json
import time
import csv
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

ADAPTER_DIR = "outputs/qwen3_4b_math_lora"

PRIVATE_PATH = "data/private.jsonl"
OUTPUT_CSV = "submission.csv"

# ── Load data ─────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

private_data = load_jsonl(PRIVATE_PATH)

# ── Boxed answer extractor ─────────────────────────────────────────
def extract_boxed_answer(response):
    """
    Extracts the last \\boxed{...} answer.
    Handles nested braces better than a simple regex.
    """
    marker = r"\boxed{"
    start = response.rfind(marker)

    if start == -1:
        return response.strip()

    i = start + len(marker)
    depth = 1
    answer_chars = []

    while i < len(response):
        char = response[i]

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break

        answer_chars.append(char)
        i += 1

    answer = "".join(answer_chars).strip()
    return f"\\boxed{{{answer}}}"

# ── Load tokenizer ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Format options ────────────────────────────────────────────────────────
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

# ── Prompts ───────────────────────────────────────────────────────────────
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
    if options:
        opts_text = format_options(options)
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    return SYSTEM_PROMPT_MATH, question

# ── Check LoRA adapter exists ─────────────────────────────────────────────
if not os.path.exists(ADAPTER_DIR):
    raise FileNotFoundError(f"LoRA adapter folder not found: {ADAPTER_DIR}")

# ── Load vLLM with LoRA support ───────────────────────────────────────────
llm = LLM(
    model=MODEL_ID,
    trust_remote_code=True,

    enable_lora=True,
    max_lora_rank=16,

    dtype="bfloat16",
    max_model_len=6144,
    gpu_memory_utilization=0.80,

    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=768,
)

# ── Build prompts for private test set ────────────────────────────────────
prompts = []

for item in private_data:
    system, user = build_prompt(
        item["question"],
        item.get("options")
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

# ── Generate predictions ─────────────────────────────────────────────────
start = time.time()

print(f"Generating responses for {len(prompts)} private questions...")

outputs = llm.generate(
    prompts,
    sampling_params=sampling_params,
    lora_request=LoRARequest(
        "math_lora",
        1,
        ADAPTER_DIR,
    ),
)

responses = [out.outputs[0].text for out in outputs]

print(f"Generation finished in {time.time() - start:.1f}s")

# ── Convert vLLM outputs to responses ─────────────────────────────────────
responses = []

for out in outputs:
    if out.outputs and out.outputs[0].text is not None:
        responses.append(out.outputs[0].text)
    else:
        responses.append("")

# ── Retry empty responses ─────────────────────────────────────────────────
bad_indices = [
    i for i, response in enumerate(responses)
    if response is None or str(response).strip() == ""
]

print(f"Empty responses before retry: {len(bad_indices)}")

if bad_indices:
    retry_prompts = [prompts[i] for i in bad_indices]

    retry_sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        max_tokens=1024,
    )

    retry_outputs = llm.generate(
        retry_prompts,
        sampling_params=retry_sampling_params,
        lora_request=LoRARequest(
            "math_lora",
            1,
            ADAPTER_DIR,
        ),
    )

    for original_index, retry_out in zip(bad_indices, retry_outputs):
        if retry_out.outputs and retry_out.outputs[0].text is not None:
            responses[original_index] = retry_out.outputs[0].text

# ── Final fallback for anything still empty ────────────────────────────────
still_bad_indices = [
    i for i, response in enumerate(responses)
    if response is None or str(response).strip() == ""
]

print(f"Empty responses after retry: {len(still_bad_indices)}")

for i in still_bad_indices:
    item = private_data[i]

    if item.get("options"):
        responses[i] = "The model did not generate a response. As a fallback, the answer is \\boxed{A}."
    else:
        responses[i] = "The model did not generate a response. As a fallback, the answer is \\boxed{0}."

# ── Save Kaggle submission CSV ────────────────────────────────────────────
rows = []

for item, response in zip(private_data, responses):
    rows.append({
        "id": item["id"],
        "response": str(response)
    })

if len(rows) != len(private_data):
    raise ValueError(
        f"Row count mismatch: got {len(rows)} predictions, expected {len(private_data)}"
    )

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "response"],
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
