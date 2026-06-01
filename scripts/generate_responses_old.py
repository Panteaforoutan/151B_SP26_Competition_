import os
import argparse
import yaml
import json
import time
from typing import Optional
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import random
from collections import defaultdict

from formatting import format_options

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_prompt(question: str, options: Optional[list], cfg) -> tuple[str, str]:
    if options:
        opts_text = format_options(options)
        return cfg["prompts"]["mcq"], f"{question}\n\nOptions:\n{opts_text}"

    return cfg["prompts"]["math"], question

def sample_balanced_eval_subset(eval_data, subset_size, seed=42):
    """
    Samples a fixed-size subset while keeping roughly the same MCQ/free-form ratio.
    Also spreads examples across the whole eval file by sampling randomly with a fixed seed.
    """
    if subset_size is None or subset_size >= len(eval_data):
        return eval_data

    rng = random.Random(seed)

    groups = defaultdict(list)
    for item in eval_data:
        key = "mcq" if is_mcq(item) else "free_form"
        groups[key].append(item)

    total = len(eval_data)

    sampled = []

    for key, items in groups.items():
        group_ratio = len(items) / total
        group_n = round(subset_size * group_ratio)
        group_n = min(group_n, len(items))

        sampled.extend(rng.sample(items, group_n))

    # Fix rounding issues so final size is exactly subset_size
    if len(sampled) > subset_size:
        sampled = rng.sample(sampled, subset_size)

    elif len(sampled) < subset_size:
        sampled_ids = {item["id"] for item in sampled}
        remaining = [item for item in eval_data if item["id"] not in sampled_ids]
        sampled.extend(rng.sample(remaining, subset_size - len(sampled)))

    # Optional: sort back by original id/file order so outputs are easier to compare
    sampled_ids = {item["id"] for item in sampled}
    subset = [item for item in eval_data if item["id"] in sampled_ids]

    return subset
    
def is_mcq(item):
    return "options" in item and item["options"] is not None and len(item["options"]) > 0

def make_sampling_params(item, cfg):
    return SamplingParams(
        temperature=cfg.get("temperature", 0),
        top_p=cfg.get("top_p", 1.0),
        top_k=cfg.get("top_k", -1),
        n=cfg.get("n", 1),
        max_tokens=cfg["max_tokens"],
        repetition_penalty=cfg.get("repetition_penalty", 1.0),
        seed=cfg.get("seed", 42),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    eval_data = load_jsonl(cfg["eval_path"])
    
    eval_data = sample_balanced_eval_subset(
        eval_data,
        subset_size=cfg.get("eval_subset_size", None),
        seed=cfg.get("eval_subset_seed", 42),
    )

    filter_ids_path = cfg.get("filter_ids_path", None)
    if filter_ids_path:
        with open(filter_ids_path, "r") as f:
            filter_data = json.load(f)
        filter_ids = {item["id"] for item in filter_data}
        eval_data = [ex for ex in eval_data if ex["id"] in filter_ids]
        print(f"Filtering to {len(eval_data)} examples from {filter_ids_path}")

    output_path = cfg["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    batch_size = cfg.get("batch_size", 10)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_lora = cfg.get("use_lora", False)

    llm_kwargs = dict(
        model=cfg["model_id"],
        enable_lora = use_lora,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_num_batched_tokens=cfg.get("max_num_batched_tokens", 2048),
        enforce_eager=False,
        seed=cfg.get("seed", 42)
    )
    
    if use_lora:
        llm_kwargs["max_lora_rank"] = cfg["max_lora_rank"]
    
    llm = LLM(**llm_kwargs)
    
    lora_request = None

    if use_lora:
        lora_path = cfg["adapter_dir"]
    
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA adapter folder not found: {lora_path}")
    
        lora_request = LoRARequest(
            "math_lora",
            1,
            lora_path,
        )
    
        print(f"Running with LoRA adapter: {lora_path}")
    else:
        print("Running baseline model without LoRA.")

        
    done_ids = set()

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done_ids.add(row["id"])

    print(f"Already completed: {len(done_ids)}")

    remaining = [ex for ex in eval_data if ex["id"] not in done_ids]

    print(f"Remaining: {len(remaining)}")

    start = time.time()

    for i in tqdm(range(0, len(remaining), batch_size)):
        batch = remaining[i:i + batch_size]

        prompts = []
        batch_sampling_params = []

        for ex in batch:
            system, user = build_prompt(
                ex["question"],
                ex.get("options"),
                cfg,
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
            batch_sampling_params.append(make_sampling_params(ex, cfg))

        generate_kwargs = dict(
            prompts=prompts,
            sampling_params=batch_sampling_params,
        )
        
        if use_lora:
            generate_kwargs["lora_request"] = lora_request
        
        outputs = llm.generate(**generate_kwargs)

        
        with open(output_path, "a") as f:
            for ex, out in zip(batch, outputs):
                response = out.outputs[0].text.strip()

                row = {
                    "id": ex["id"],
                    "response": response,
                }

                f.write(json.dumps(row) + "\n")

            f.flush()
            os.fsync(f.fileno())

    print("Done generating responses.")
    print(f"Generation time: {time.time() - start:.1f}s")
    print(f"Saved responses to {output_path}")


if __name__ == "__main__":
    main()