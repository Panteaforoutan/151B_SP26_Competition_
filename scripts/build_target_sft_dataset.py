import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


TARGETS = {
    "stats_probability": 5000,
    "algebra_functions": 4500,
    "discrete_contest": 3500,
    "word_problems": 3500,
    "geometry_trig": 3500,
}

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Solve the problem carefully. Put the final answer in \\boxed{}."
)

ALGEBRA_KEYWORDS = [
    "function", "f(x)", "g(x)", "polynomial", "quadratic", "linear",
    "equation", "inequality", "system", "root", "roots", "factor",
    "solve for", "interval", "range", "domain", "sequence",
    "exponential", "logarithm", "absolute value",
]

STATS_PROB_KEYWORDS = [
    "probability", "expected", "expectation", "mean", "median", "mode",
    "standard deviation", "variance", "random", "distribution",
    "choose", "combination", "permutation", "binomial", "conditional",
    "sample", "data", "percentile", "average",
]

GEOMETRY_TRIG_KEYWORDS = [
    "triangle", "circle", "angle", "area", "volume", "perimeter",
    "radius", "diameter", "coordinate", "distance", "slope",
    "sin", "cos", "tan", "trigonometric", "trigonometry",
    "polygon", "sphere", "cone", "cylinder", "geometry",
]

DISCRETE_KEYWORDS = [
    "integer", "divisible", "remainder", "mod", "modulo", "prime",
    "factor", "multiple", "number theory", "sequence", "recurrence",
    "combinatorics", "counting", "permutation", "combination",
    "arrange", "ways", "digits", "sets",
]

BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

def normalize_spaces(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def has_cjk(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def stable_hash(text):
    return hashlib.md5(normalize_spaces(text).lower().encode("utf-8")).hexdigest()


def contains_any(text, keywords):
    text = text.lower()
    return any(k.lower() in text for k in keywords)


def keyword_score(text, keywords):
    text = text.lower()
    return sum(1 for k in keywords if k.lower() in text)


def is_too_simple_question(question):
    q = normalize_spaces(question).lower()

    simple_starts = (
        "compute",
        "evaluate",
        "simplify",
        "calculate",
        "what is",
    )

    if q.startswith(simple_starts) and word_count(q) < 10:
        return True

    # Common answer-only drill pattern from Khan-style data.
    if "binom" in q and word_count(q) < 8:
        return True

    return False

def extract_boxed(solution):
    matches = BOX_RE.findall(solution or "")
    if matches:
        return matches[-1].strip()
    return None


def extract_final_answer(solution):
    solution = (solution or "").strip()
    s = normalize_spaces(solution)

    # 1. Existing boxed answer
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if boxed:
        return boxed[-1].strip()

    # 2. Common final-answer formats
    patterns = [
        r"Final Answer:?\s*(.+?)\.?\s*$",
        r"Final answer:?\s*(.+?)\.?\s*$",
        r"The answer is:?\s*(.+?)\.?\s*$",
        r"Answer:?\s*(.+?)\.?\s*$",
        r"####\s*(.+?)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution, flags=re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
            ans = ans.strip("$").strip()

            if ans.endswith(".") and not re.search(r"\d\.\d$", ans):
                ans = ans[:-1].strip()

            return ans

    # 3. Handle Orca-style endings:
    # "Therefore, ... is Rs. 65."
    therefore_match = re.search(
        r"(?:therefore|thus|so|hence)[^\.]*?\bis\s+([^\.]+)\.?\s*$",
        s,
        flags=re.IGNORECASE,
    )
    if therefore_match:
        phrase = therefore_match.group(1).strip()

        # Prefer the final number inside the phrase.
        nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", phrase)
        if nums:
            return nums[-1].replace(",", "")

        return phrase

    # 4. Last LaTeX expression
    math_chunks = re.findall(r"\$([^$]{1,100})\$", solution)
    if math_chunks:
        return math_chunks[-1].strip()

    # 5. Final fallback: last number in the solution
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", s)
    if nums:
        return nums[-1].replace(",", "")

    return None


def remove_old_final_answer_line(solution):
    solution = (solution or "").strip()

    patterns = [
        r"\s*The answer is:?\s*.+?\s*$",
        r"\s*Answer:?\s*.+?\s*$",
        r"\s*Final answer:?\s*.+?\s*$",
        r"\s*Final Answer:?\s*.+?\s*$",
        r"\s*####\s*.+?\s*$",
    ]

    for pattern in patterns:
        solution = re.sub(pattern, "", solution, flags=re.IGNORECASE).strip()

    return solution


def normalize_assistant_solution(solution):
    solution = str(solution).strip()

    answer = extract_final_answer(solution)

    # Remove old final-answer lines if they already exist
    solution = re.sub(
        r"\s*(Final Answer|Final answer|The answer is|Answer):?\s*.+?\s*$",
        "",
        solution,
        flags=re.IGNORECASE,
    ).strip()

    # Remove old boxed final sentence if present
    solution = re.sub(
        r"\s*(Therefore,?\s*)?(the\s+)?(final\s+)?answer\s+is\s+\$?\\boxed\{[^{}]*\}\$?\.?\s*$",
        "",
        solution,
        flags=re.IGNORECASE,
    ).strip()

    if answer:
        return solution + f"\n\nFinal Answer: \\boxed{{{answer}}}"

    return solution
def make_text(tokenizer, question, solution):
    solution = normalize_assistant_solution(solution)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Question:\n" + question.strip()},
        {"role": "assistant", "content": solution.strip()},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def make_row(tokenizer, category, source, question, solution, raw_id=None):
    question = str(question).strip()
    solution = normalize_assistant_solution(str(solution).strip())

    return {
        "id": stable_hash(source + "::" + question),
        "category": category,
        "source": source,
        "raw_id": raw_id,
        "question": question,
        "solution": solution,
        "text": make_text(tokenizer, question, solution),
    }


def load_all_splits(dataset_name, config_name=None):
    ds = load_dataset(dataset_name, config_name)

    if hasattr(ds, "keys"):
        return concatenate_datasets([ds[split] for split in ds.keys()])

    return ds


def load_train_split(dataset_name, config_name=None):
    return load_dataset(dataset_name, config_name, split="train")


def get_math_rows(tokenizer, dataset_name, config_name, category, source_name, all_splits=False):
    if all_splits:
        ds = load_all_splits(dataset_name, config_name)
    else:
        ds = load_train_split(dataset_name, config_name)

    rows = []
    for i, ex in enumerate(ds):
        q = ex.get("problem")
        a = ex.get("solution")

        if not q or not a:
            continue

        if word_count(q) < 4 or word_count(a) < 8:
            continue

        rows.append(
            make_row(
                tokenizer=tokenizer,
                category=category,
                source=f"{source_name}/{config_name}",
                question=q,
                solution=a,
                raw_id=i,
            )
        )

    return rows


def get_orca_word_problem_rows(tokenizer, max_rows, seed):
    ds = load_train_split("xd2333/orca-math-word-problems-100k-en-zh-mix")

    candidates = []

    for i, ex in enumerate(ds):
        q = ex.get("instruction", "")
        a = ex.get("output", "")

        combined = q + " " + a

        if has_cjk(combined):
            continue

        if word_count(q) < 18:
            continue

        if word_count(a) < 30:
            continue

        # Prefer multi-step word problems with several quantities.
        numbers = re.findall(r"\d+(?:\.\d+)?", q)
        if len(numbers) < 2:
            continue

        if is_too_simple_question(q):
            continue

        candidates.append(
            make_row(
                tokenizer=tokenizer,
                category="word_problems",
                source="orca_math_word_problems",
                question=q,
                solution=a,
                raw_id=i,
            )
        )

    random.Random(seed).shuffle(candidates)
    return candidates[:max_rows]

def classify_metamath_question(question):
    q = normalize_spaces(question).lower()

    scores = {
        "stats_probability": keyword_score(q, STATS_PROB_KEYWORDS),
        "algebra_functions": keyword_score(q, ALGEBRA_KEYWORDS),
        "geometry_trig": keyword_score(q, GEOMETRY_TRIG_KEYWORDS),
        "discrete_contest": keyword_score(q, DISCRETE_KEYWORDS),
    }

    best_category, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score <= 0:
        return None

    return best_category


def get_metamath_filler_rows(tokenizer, seed):
    """
    Optional filler source. It is useful when one bucket is still short.
    Uses keyword classification because MetaMathQA-40K has broad math types.
    """
    try:
        ds = load_train_split("meta-math/MetaMathQA-40K")
    except Exception as e:
        print(f"Skipping MetaMathQA-40K because it failed to load: {e}")
        return []

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    rows = []

    for idx in indices:
        ex = ds[int(idx)]

        q = ex.get("query", "")
        a = ex.get("response", "")

        if not q or not a:
            continue

        if word_count(q) < 6 or word_count(a) < 20:
            continue

        category = classify_metamath_question(q)

        if category is None:
            continue

        rows.append(
            make_row(
                tokenizer=tokenizer,
                category=category,
                source=f"metamath_40k/{ex.get('type', 'unknown')}",
                question=q,
                solution=a,
                raw_id=int(idx),
            )
        )

    return rows


def add_rows(bucket, rows, selected, seen_questions, max_add=None):
    needed = TARGETS[bucket] - len(selected[bucket])

    if needed <= 0:
        return 0

    if max_add is not None:
        needed = min(needed, max_add)

    added = 0

    for row in rows:
        if len(selected[bucket]) >= TARGETS[bucket]:
            break

        if added >= needed:
            break

        key = stable_hash(row["question"])

        if key in seen_questions:
            continue

        seen_questions.add(key)
        selected[bucket].append(row)
        added += 1

    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/curated_target_sft.jsonl",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="data/curated_target_sft_summary.json",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--orca_target",
        type=int,
        default=3500,
    )
    parser.add_argument(
        "--use_metamath_filler",
        action="store_true",
        help="Use MetaMathQA-40K as a fallback filler source.",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    selected = defaultdict(list)
    seen_questions = set()

    print("\nLoading MATH-Hard exact target sources...")

    # 1. Stats/probability: all 399 MATH-Hard counting/probability examples.
    mathhard_counting = get_math_rows(
        tokenizer,
        dataset_name="lighteval/MATH-Hard",
        config_name="counting_and_probability",
        category="stats_probability",
        source_name="math_hard",
        all_splits=True,
    )
    random.shuffle(mathhard_counting)
    add_rows("stats_probability", mathhard_counting, selected, seen_questions)

    # 2. Algebra/functions: all 743 MATH-Hard algebra examples.
    mathhard_algebra = get_math_rows(
        tokenizer,
        dataset_name="lighteval/MATH-Hard",
        config_name="algebra",
        category="algebra_functions",
        source_name="math_hard",
        all_splits=True,
    )
    random.shuffle(mathhard_algebra)
    add_rows("algebra_functions", mathhard_algebra, selected, seen_questions)

    # 3. Geometry/trig: MATH-Hard geometry + precalculus.
    for cfg in ["geometry", "precalculus"]:
        rows = get_math_rows(
            tokenizer,
            dataset_name="lighteval/MATH-Hard",
            config_name=cfg,
            category="geometry_trig",
            source_name="math_hard",
            all_splits=True,
        )
        random.shuffle(rows)
        add_rows("geometry_trig", rows, selected, seen_questions)

    # 4. Discrete/contest: MATH-Hard number theory.
    rows = get_math_rows(
        tokenizer,
        dataset_name="lighteval/MATH-Hard",
        config_name="number_theory",
        category="discrete_contest",
        source_name="math_hard",
        all_splits=True,
    )
    random.shuffle(rows)
    add_rows("discrete_contest", rows, selected, seen_questions)

    print("\nLoading Orca word problem examples...")
    orca_rows = get_orca_word_problem_rows(
        tokenizer,
        max_rows=args.orca_target,
        seed=args.seed,
    )
    add_rows("word_problems", orca_rows, selected, seen_questions)

    print("\nLoading MATH-lighteval filler sources...")

    # Stats/probability filler.
    rows = get_math_rows(
        tokenizer,
        dataset_name="DigitalLearningGmbH/MATH-lighteval",
        config_name="counting_and_probability",
        category="stats_probability",
        source_name="math_lighteval",
        all_splits=False,
    )
    random.shuffle(rows)
    add_rows("stats_probability", rows, selected, seen_questions)

    # Algebra/function filler.
    algebra_fill_configs = [
        "algebra",
        "intermediate_algebra",
        "prealgebra",
        "precalculus",
    ]

    algebra_filler = []
    for cfg in algebra_fill_configs:
        rows = get_math_rows(
            tokenizer,
            dataset_name="DigitalLearningGmbH/MATH-lighteval",
            config_name=cfg,
            category="algebra_functions",
            source_name="math_lighteval",
            all_splits=False,
        )

        # Prioritize function/equation/system/interval examples.
        rows.sort(
            key=lambda r: keyword_score(r["question"], ALGEBRA_KEYWORDS),
            reverse=True,
        )
        algebra_filler.extend(rows)

    add_rows("algebra_functions", algebra_filler, selected, seen_questions)

    # Geometry/trig filler.
    geometry_filler = []
    for cfg in ["geometry", "precalculus"]:
        rows = get_math_rows(
            tokenizer,
            dataset_name="DigitalLearningGmbH/MATH-lighteval",
            config_name=cfg,
            category="geometry_trig",
            source_name="math_lighteval",
            all_splits=False,
        )
        rows.sort(
            key=lambda r: keyword_score(r["question"], GEOMETRY_TRIG_KEYWORDS),
            reverse=True,
        )
        geometry_filler.extend(rows)

    add_rows("geometry_trig", geometry_filler, selected, seen_questions)

    # Discrete/contest filler.
    discrete_filler = []
    for cfg in ["number_theory", "counting_and_probability"]:
        rows = get_math_rows(
            tokenizer,
            dataset_name="DigitalLearningGmbH/MATH-lighteval",
            config_name=cfg,
            category="discrete_contest",
            source_name="math_lighteval",
            all_splits=False,
        )
        rows.sort(
            key=lambda r: keyword_score(r["question"], DISCRETE_KEYWORDS),
            reverse=True,
        )
        discrete_filler.extend(rows)

    add_rows("discrete_contest", discrete_filler, selected, seen_questions)

    if args.use_metamath_filler:
        print("\nLoading MetaMathQA-40K fallback filler...")
        metamath_rows = get_metamath_filler_rows(tokenizer, seed=args.seed)

        by_category = defaultdict(list)
        for row in metamath_rows:
            by_category[row["category"]].append(row)

        for bucket in TARGETS:
            random.shuffle(by_category[bucket])
            add_rows(bucket, by_category[bucket], selected, seen_questions)

    final_rows = []

    for bucket in TARGETS:
        n = len(selected[bucket])
        target = TARGETS[bucket]

        if n < target:
            print(f"WARNING: {bucket} is short: {n}/{target}")

        final_rows.extend(selected[bucket][:target])

    random.Random(args.seed).shuffle(final_rows)

    # Normalize IDs: 0, 1, 2, ...
    for new_id, row in enumerate(final_rows):
        row["id"] = new_id

    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    category_counts = Counter(row["category"] for row in final_rows)
    source_counts = Counter(row["source"] for row in final_rows)

    summary = {
        "total": len(final_rows),
        "targets": TARGETS,
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "output_path": args.output_path,
    }

    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    bad = []
    
    with open(args.output_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
    
            if "Final Answer: \\boxed{" not in text:
                bad.append(row)
    
    print(f"Rows missing boxed final answer: {len(bad)}")
    
    for row in bad[:10]:
        print(row["id"], row["category"], row["source"])
        print(row["solution"][-300:])
        print()


if __name__ == "__main__":
    main()