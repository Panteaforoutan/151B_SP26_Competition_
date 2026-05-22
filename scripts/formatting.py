import re

def get_first_available(example, keys, default=None):
    """
    Return the first non-empty value found in the example.
    """
    for key in keys:
        value = example.get(key)

        if value is not None and value != "":
            return value

    return default


def format_options(options):
    if options is None:
        return ""

    if isinstance(options, list):
        lines = []

        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)

            if isinstance(opt, dict):
                text = opt.get("text") or opt.get("value") or str(opt)
            else:
                text = str(opt)

            lines.append(f"{letter}. {text}")

        return "\n".join(lines)

    return str(options)

def extract_answer_from_response(response):
    if not response:
        return None

    # Handles: #### 2
    match = re.search(r"####\s*([^\n]+)", response)
    if match:
        return match.group(1).strip()

    # Handles: The answer is: 2
    match = re.search(r"The answer is:\s*([^\n]+)", response)
    if match:
        return match.group(1).strip()

    return None

def format_train_example(example, tokenizer):
    question = get_first_available(
        example,
        ["question", "problem", "input", "prompt", "query"],
        default="",
    )

    solution = get_first_available(
        example,
        ["response", "solution" ],
        default=None,
    )

    answer = get_first_available(
       example,
       ["expected_answer", "answer", "final_answer", "target"],
       default=None,
    )

    if answer is None:
        answer = extract_answer_from_response(solution)

    options = get_first_available(
        example,
        ["options", "choices", "multiple_choice_options"],
        default=None,
    )

    options_text = format_options(options)

    clean_solution = re.sub(r"####\s*[^\n]+", "", solution)
    clean_solution = re.sub(r"The answer is:\s*[^\n]+", "", clean_solution)
    clean_solution = clean_solution.strip()

    if options_text:
        user_content = f"""Question:
    {question}

    Options:
        {options_text}"""
    else:
        user_content = f"""Question:
    {question}"""

    if answer:
        assistant_content = f"""{clean_solution}

Therefore, the final answer is \\boxed{{{answer}}}."""
    else:
        assistant_content = clean_solution

    messages = [
        {
            "role": "system",
            "content": "You are a math reasoning assistant. Solve the problem carefully. Put the final answer in \\boxed{}.",
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": assistant_content,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}
