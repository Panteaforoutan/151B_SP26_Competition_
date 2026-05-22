import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm


TARGETS = {
    "stats_probability": 5000,
    "algebra_functions": 4500,
    "discrete_contest": 3500,
    "word_problems": 3500,
    "geometry_trig": 3500,
}


STATS_KEYWORDS = [
    "probability", "probabilities", "chance", "random", "expected value", "expectation",
    "mean", "median", "mode", "range", "variance", "standard deviation", "sample",
    "survey", "data", "dataset", "histogram", "box plot", "normal distribution",
    "binomial", "permutation", "combination", "counting", "arrangement",
]

ALGEBRA_KEYWORDS = [
    "solve for", "equation", "system of equations", "function", "linear", "quadratic",
    "polynomial", "roots", "factor", "simplify", "expression", "inequality",
    "slope", "intercept", "parabola", "exponential", "logarithm",
]

DISCRETE_KEYWORDS = [
    "integer", "integers", "divisible", "divisibility", "mod", "modulo", "remainder",
    "prime", "gcd", "lcm", "sequence", "recurrence", "graph", "vertices", "edges",
    "set", "sets", "logic", "prove", "number theory", "contest", "olympiad",
]

GEOMETRY_TRIG_KEYWORDS = [
    "triangle", "circle", "angle", "polygon", "area", "perimeter", "volume", "surface area",
    "radius", "diameter", "chord", "coordinate plane", "distance", "similar", "congruent",
    "sine", "cosine", "tangent", "trigonometric", "trig", "radian", "degree",
]

WORD_PROBLEM_HINTS = [
    "$", "dollars", "cost", "price", "spent", "left", "total", "altogether", "each",
    "miles", "hours", "minutes", "apples", "books", "students", "workers", "rate",
]


BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def contains_any(text: str, words: List[str]) -> bool:
    t = norm_text(text)
    return any(w in t for w in words)


def extract_boxed(solution: str) -> Optional[str]:
    matches = BOX_RE.findall(solution or "")
    if matches:
        return matches[-1].strip()
    return None


def extract_gsm8k_answer(answer: str) -> str:
    # GSM8K answers usually end with "#### final_answer".
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()

def parse_mathqa_options(options: Any) -> List[str]:
    # MathQA often stores options as one string: "a ) ... , b ) ..."
    if isinstance(options, list):
        return [str(x).strip() for x in options]
    if not isinstance(options, str):
        return []

    parts = re.split(r"\s*,\s*(?=[a-e]\s*\))", options.strip())
    clean = []
    for p in parts:
        p = p.strip()
        m = re.match(r"([a-e])\s*\)\s*(.*)", p)
        if m:
            clean.append(f"{m.group(1).upper()}. {m.group(2).strip()}")
        elif p:
            clean.append(p)
    return clean


def mathqa_correct_to_letter(correct: Any) -> str:
    c = str(correct).strip().lower()
    if c in ["a", "b", "c", "d", "e"]:
        return c.upper()
    return c

def make_record(
    question: str,
    solution: str,
    answer: Optional[str],
    bucket: str,
    source: str,
    options: Optional[List[str]] = None,
    source_type: Optional[str] = None,
    ) -> Dict[str, Any]:

    final_answer = str(answer).strip() if answer is not None else ""

    if not final_answer:
        final_answer = extract_final_answer(solution) or ""

    rec = {
        "question": question.strip(),
        "answer": final_answer,
        "solution": ensure_boxed_solution(solution, final_answer),
        "bucket": bucket,
        "source": source,
    }

    if options:
        rec["options"] = options
    if source_type:
        rec["source_type"] = source_type

    return rec


def classify_by_keywords(question: str) -> Optional[str]:
    q = question or ""
    if contains_any(q, STATS_KEYWORDS):
        return "stats_probability"
    if contains_any(q, GEOMETRY_TRIG_KEYWORDS):
        return "geometry_trig"
    if contains_any(q, DISCRETE_KEYWORDS):
        return "discrete_contest"
    if contains_any(q, ALGEBRA_KEYWORDS):
        return "algebra_functions"
    if contains_any(q, WORD_PROBLEM_HINTS):
        return "word_problems"
    return None


def add_unique(pool: Dict[str, List[Dict[str, Any]]], seen: set, rec: Dict[str, Any]) -> None:
    key = norm_text(rec["question"])
    if not key or key in seen:
        return
    if rec["bucket"] not in TARGETS:
        return
    seen.add(key)
    pool[rec["bucket"]].append(rec)

def has_boxed_answer(text):
    return r"\boxed{" in text or r"\boxed" in text

def extract_final_answer(solution: str) -> Optional[str]:
    solution = (solution or "").strip()

    # First try boxed answer
    boxed = extract_boxed(solution)
    if boxed:
        return boxed.strip()

    # Then try common unboxed endings
    patterns = [
        r"The answer is:?\s*(.+?)\.?\s*$",
        r"Answer:?\s*(.+?)\.?\s*$",
        r"Final answer:?\s*(.+?)\.?\s*$",
        r"Final Answer:?\s*(.+?)\.?\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution, flags=re.IGNORECASE)
        if match:
            ans = match.group(1).strip()

            # Avoid keeping a final sentence period
            if ans.endswith(".") and not re.search(r"\d\.\d$", ans):
                ans = ans[:-1].strip()

            return ans

    return None

def remove_old_final_answer_line(solution: str) -> str:
    solution = (solution or "").strip()

    patterns = [
        r"\s*The answer is:?\s*.+?\s*$",
        r"\s*Answer:?\s*.+?\s*$",
        r"\s*Final answer:?\s*.+?\s*$",
        r"\s*Final Answer:?\s*.+?\s*$",
    ]

    for pattern in patterns:
        solution = re.sub(pattern, "", solution, flags=re.IGNORECASE).strip()

    return solution


def ensure_boxed_solution(solution: str, final_answer: Optional[str]) -> str:
    solution = (solution or "").strip()

    if "\\boxed" in solution:
        return solution

    answer = str(final_answer).strip() if final_answer is not None else ""

    if not answer:
        answer = extract_final_answer(solution) or ""

    if answer:
        clean_solution = remove_old_final_answer_line(solution)
        return f"{clean_solution}\n\nFinal Answer: \\boxed{{{answer}}}"

    return solution


def normalize_assistant_solution(row):
    solution = str(row["solution"]).strip()
    answer = str(row.get("answer", "")).strip()

    if not answer:
        boxed = extract_boxed(solution)
        if boxed:
            answer = boxed.strip()

    if not answer:
        answer = extract_final_answer(solution) or ""

    # Remove old unboxed ending
    solution = remove_old_final_answer_line(solution)

    # Remove existing boxed final phrase to avoid duplicates
    solution = re.sub(
        r"\s*(Therefore,?\s*)?(the\s+)?(final\s+)?answer\s+is\s+\$?\\boxed\{[^{}]*\}\$?\.?\s*$",
        "",
        solution,
        flags=re.IGNORECASE,
    ).strip()

    if answer:
        return solution + f"\n\nFinal Answer: \\boxed{{{answer}}}"

    return solution
    

def load_math(pool: Dict[str, List[Dict[str, Any]]], seen: set) -> None:
    config_to_bucket = {
        "counting_and_probability": "stats_probability",
        "algebra": "algebra_functions",
        "intermediate_algebra": "algebra_functions",
        "prealgebra": "algebra_functions",
        "number_theory": "discrete_contest",
        "geometry": "geometry_trig",
        "precalculus": "geometry_trig",
    }

    for config, default_bucket in config_to_bucket.items():
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
        except Exception as e:
            print(f"Skipping MATH config {config}: {e}")
            continue

        for ex in tqdm(ds, desc=f"MATH/{config}"):
            question = ex.get("problem", "")
            solution = ex.get("solution", "")
            answer = extract_boxed(solution)

            # Precalculus is mixed; keep trig/geometry-like questions in geometry_trig,
            # otherwise let keyword classifier decide.
            bucket = default_bucket
            if config == "precalculus":
                bucket = classify_by_keywords(question) or "algebra_functions"
                if bucket not in ["geometry_trig", "algebra_functions"]:
                    bucket = "algebra_functions"

            rec = make_record(question, solution, answer, bucket, "EleutherAI/hendrycks_math", source_type=config)
            add_unique(pool, seen, rec)


def load_gsm8k(pool: Dict[str, List[Dict[str, Any]]], seen: set) -> None:
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(f"Skipping GSM8K: {e}")
        return

    for ex in tqdm(ds, desc="GSM8K"):
        question = ex.get("question", "")
        solution = ex.get("answer", "")
        answer = extract_gsm8k_answer(solution)
        rec = make_record(question, solution, answer, "word_problems", "openai/gsm8k")
        add_unique(pool, seen, rec)


def load_mathqa(pool: Dict[str, List[Dict[str, Any]]], seen: set) -> None:
    # regisss/math_qa avoids some remote-code issues. If it fails, try allenai/math_qa.
    try:
        ds = load_dataset("regisss/math_qa", split="train")
        source = "regisss/math_qa"
    except Exception:
        try:
            ds = load_dataset("allenai/math_qa", split="train")
            source = "allenai/math_qa"
        except Exception as e:
            print(f"Skipping MathQA: {e}")
            return

    for ex in tqdm(ds, desc="MathQA"):
        question = ex.get("Problem") or ex.get("problem") or ex.get("question") or ""
        rationale = ex.get("Rationale") or ex.get("rationale") or ""
        correct = ex.get("correct") or ex.get("Correct") or ""
        options = parse_mathqa_options(ex.get("options") or ex.get("Options"))
        bucket = classify_by_keywords(question)
        if not bucket:
            continue
        answer = mathqa_correct_to_letter(correct)
        rec = make_record(question, rationale, answer, bucket, source, options=options)
        add_unique(pool, seen, rec)


def load_metamath_40k(pool: Dict[str, List[Dict[str, Any]]], seen: set) -> None:
    try:
        ds = load_dataset("meta-math/MetaMathQA-40K", split="train")
    except Exception as e:
        print(f"Skipping MetaMathQA-40K: {e}")
        return

    for ex in tqdm(ds, desc="MetaMathQA-40K"):
        question = ex.get("query") or ex.get("question") or ex.get("problem") or ""
        solution = ex.get("response") or ex.get("solution") or ex.get("answer") or ""
        source_type = ex.get("type") or ex.get("source") or ""
        bucket = classify_by_keywords(question + " " + str(source_type))
        if not bucket:
            continue
        answer = extract_boxed(solution)
        rec = make_record(question, solution, answer, bucket, "meta-math/MetaMathQA-40K", source_type=str(source_type))
        add_unique(pool, seen, rec)


def sample_targets(pool: Dict[str, List[Dict[str, Any]]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    final = []

    for bucket, target in TARGETS.items():
        rows = pool[bucket]
        rng.shuffle(rows)
        take = rows[:target]
        if len(take) < target:
            print(f"WARNING: {bucket} has only {len(take)} / {target} examples")
        final.extend(take)
        print(f"{bucket}: selected {len(take)} / {target} from {len(rows)} candidates")

    rng.shuffle(final)
    for i, rec in enumerate(final):
        rec["id"] = i
    return final


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sft_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if row.get("options"):
                options_text = "\n".join(row["options"])
                user = f"Question:\n{row['question']}\n\nOptions:\n{options_text}"
                system = "You are a math reasoning assistant. Choose the correct option. Put only the final option letter in \\boxed{}."
            else:
                user = f"Question:\n{row['question']}"
                system = "You are a math reasoning assistant. Solve the problem carefully. Put the final answer in \\boxed{}."

            assistant = normalize_assistant_solution(row)

            text = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant}<|im_end|>"
            )
            
            out = dict(row)
            out["text"] = text
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/updated_curated_math_20k.jsonl")
    parser.add_argument("--sft-out", default="data/updated_curated_math_20k_sft.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pool = {bucket: [] for bucket in TARGETS}
    seen = set()

    load_math(pool, seen)
    load_gsm8k(pool, seen)
    load_mathqa(pool, seen)
    load_metamath_40k(pool, seen)

    final = sample_targets(pool, args.seed)
    write_jsonl(final, args.out)
    write_sft_jsonl(final, args.sft_out)

    print(f"\nSaved raw dataset: {args.out}")
    print(f"Saved SFT dataset: {args.sft_out}")
    print(f"Total examples: {len(final)}")


if __name__ == "__main__":
    main()
