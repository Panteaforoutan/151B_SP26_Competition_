import argparse
import importlib.util
import inspect
import os
import sys
import yaml

from datasets import load_dataset
from transformers import AutoTokenizer


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def import_formatting_function(formatting_file, function_name):
    formatting_file = os.path.abspath(formatting_file)
    formatting_dir = os.path.dirname(formatting_file)

    # Allows formatting.py to import helper files from its own folder
    sys.path.insert(0, formatting_dir)

    spec = importlib.util.spec_from_file_location("formatting_module", formatting_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise ValueError(
            f"Could not find function `{function_name}` in {formatting_file}"
        )

    return getattr(module, function_name)


def load_train_data(config):
    dataset_name = config.get("dataset_name")
    dataset_split = config.get("dataset_split", "train")
    dataset_config_name = config.get("dataset_config_name")

    if dataset_name:
        if dataset_config_name:
            return load_dataset(dataset_name, dataset_config_name, split=dataset_split)
        return load_dataset(dataset_name, split=dataset_split)

    train_file = config.get("train_file") or config.get("data_path")

    if train_file:
        return load_dataset("json", data_files=train_file, split="train")

    raise ValueError(
        "Config must have either `dataset_name` or `train_file` / `data_path`."
    )


def call_format_function(format_fn, example, tokenizer):
    """
    Supports both:
      format_train_example(example, tokenizer)
    and:
      format_train_example(example)
    """

    signature = inspect.signature(format_fn)

    if len(signature.parameters) >= 2:
        result = format_fn(example, tokenizer)
    else:
        result = format_fn(example)

    if isinstance(result, dict):
        if "text" not in result:
            raise ValueError(
                "Formatting function returned a dict, but it does not contain a `text` field."
            )
        return result["text"]

    if isinstance(result, str):
        return result

    raise ValueError(
        "Formatting function must return either a string or a dict with a `text` field."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--formatting_file", default="formatting.py")
    parser.add_argument("--format_func", default="format_train_example")
    parser.add_argument("--output", default="outputs/formatted_examples.txt")
    parser.add_argument("--num_examples", type=int, default=20)

    args = parser.parse_args()

    config = load_config(args.config)

    model_id = config.get("model_id")
    if not model_id:
        raise ValueError("Config must contain `model_id`.")

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    print(f"Importing `{args.format_func}` from {args.formatting_file}")
    format_fn = import_formatting_function(
        args.formatting_file,
        args.format_func,
    )

    print("Loading training data...")
    train_dataset = load_train_data(config)

    max_train_examples = config.get("max_train_examples")
    if max_train_examples:
        max_train_examples = min(max_train_examples, len(train_dataset))
        train_dataset = train_dataset.select(range(max_train_examples))

    num_examples = min(args.num_examples, len(train_dataset))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    missing_boxed = 0
    empty_boxed = 0
    empty_text = 0

    print(f"Writing {num_examples} formatted examples to {args.output}")

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            raw_example = train_dataset[i]
            text = call_format_function(format_fn, raw_example, tokenizer)

            if not text.strip():
                empty_text += 1

            if "\\boxed" not in text:
                missing_boxed += 1

            if "\\boxed{}" in text:
                empty_boxed += 1

            f.write("=" * 100 + "\n")
            f.write(f"Example {i}\n")
            f.write("=" * 100 + "\n\n")

            f.write("RAW EXAMPLE:\n")
            f.write(str(raw_example))
            f.write("\n\n")

            f.write("FORMATTED TEXT:\n")
            f.write(text)
            f.write("\n\n")

    print("Done.")
    print(f"Missing boxed answers: {missing_boxed}/{num_examples}")
    print(f"Empty boxed answers: {empty_boxed}/{num_examples}")
    print(f"Empty formatted examples: {empty_text}/{num_examples}")


if __name__ == "__main__":
    main()