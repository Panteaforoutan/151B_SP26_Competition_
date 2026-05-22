import argparse
import yaml
import json
import os
import re
import sys
import pandas as pd
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


def score_mcq(response, gold):
    match = re.search(r"\\boxed\{([A-Z])\}", response)

    if match:
        pred = match.group(1)
    else:
        letters = re.findall(r"\b[A-Z]\b", response)
        pred = letters[-1] if letters else ""

    return pred.strip().upper() == gold.strip().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    eval_data = load_jsonl(cfg["eval_path"])
    generated_rows = load_jsonl(cfg["output_path"])

    responses_by_id = {
        row["id"]: row["response"]
        for row in generated_rows
    }

    print(f"Loaded {len(responses_by_id)} generated responses.")
    print(f"Eval set has {len(eval_data)} examples.")

    # Load Judger for free-form scoring
    sys.path.insert(0, ".")
    from judger import Judger

    judger = Judger(strict_extract=False)

    results = []

    for item in tqdm(eval_data, desc="Scoring"):
        item_id = item["id"]

        # Skip examples that have not been generated yet
        if item_id not in responses_by_id:
            continue

        response = responses_by_id[item_id]
        is_mcq = bool(item.get("options"))
        gold = item["answer"]

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
            "id": item_id,
            "is_mcq": is_mcq,
            "gold": gold,
            "response": response,
            "correct": correct,
        })

    print(f"Scoring complete. Scored {len(results)} generated responses.")

    mcq_res = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"MCQ       : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"Free-form : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"Overall   : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)

    metrics = {
        "experiment_name": cfg["experiment_name"],
        "eval_path": cfg["eval_path"],
        "adapter_dir": cfg["adapter_dir"],
        "max_tokens": cfg["max_tokens"],
        "generated_count": len(results),
        "eval_total": len(eval_data),
        "mcq_correct": sum(r["correct"] for r in mcq_res),
        "mcq_total": len(mcq_res),
        "mcq_accuracy": acc(mcq_res),
        "free_correct": sum(r["correct"] for r in free_res),
        "free_total": len(free_res),
        "free_accuracy": acc(free_res),
        "overall_correct": sum(r["correct"] for r in results),
        "overall_total": len(results),
        "overall_accuracy": acc(results),
    }

    metrics_path = cfg.get("metrics_path")

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {metrics_path}")

    results_path = cfg.get("results_path")

    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

        print(f"Saved detailed results to {results_path}")

    final_csv = cfg.get("final_csv")

    if final_csv:
        os.makedirs(os.path.dirname(final_csv), exist_ok=True)

        rows = [
            {
                "id": row["id"],
                "response": row["response"],
            }
            for row in generated_rows
        ]

        df = pd.DataFrame(rows)
        df = df.sort_values("id")
        df.to_csv(final_csv, index=False)

        print(f"Saved CSV to {final_csv}")


if __name__ == "__main__":
    main()