import json
import argparse


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def is_false(value):
    """
    Handles both boolean False and string versions like "false".
    """
    if value is False:
        return True
    if isinstance(value, str) and value.lower() == "false":
        return True
    return False


def main(results_path, public_path, output_path):
    results = load_jsonl(results_path)
    public_data = load_jsonl(public_path)

    # Map question id -> public question object
    public_by_id = {item["id"]: item for item in public_data}

    incorrect_examples = []

    for item in results:
        if is_false(item.get("correct")):
            qid = item["id"]

            if qid not in public_by_id:
                print(f"Warning: id {qid} not found in public file")
                continue

            public_item = public_by_id[qid]

            incorrect_examples.append({
                "id": qid,
                "question": public_item["question"],
                "generated_answer": item.get("response"),
                "ground_truth_answer": public_item["answer"],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(incorrect_examples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(incorrect_examples)} incorrect examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="baseline_inference_2_results.jsonl")
    parser.add_argument("--public", default="public.jsonl")
    parser.add_argument("--output", default="incorrect_baseline_examples.jsonl")

    args = parser.parse_args()

    main(args.results, args.public, args.output)