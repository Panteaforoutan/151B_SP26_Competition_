import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

TRAIN_PATH = "data/math_train_transformed.jsonl"      # your external training data
PUBLIC_TEST_PATH = "data/public.jsonl"    # your project public test file

ADAPTER_DIR = "outputs/qwen3_4b_math_lora"

MAX_SEQ_LEN = 2048 # max length of one training example 
NUM_TRAIN_EXAMPLES = 1000   # start small first

# ── Load tokenizer ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,)

if tokenizer.pad_token is None: # make arrays of tokens the same size for batching purpose
    tokenizer.pad_token = tokenizer.eos_token

EOS_TOKEN = "<|im_end|>" if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token

# ── Format training data  ─────────────────────────────────────────────────────────────
def format_options(options):
    if options is None:
        return ""

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


def format_train_example(example):
    question = example.get("question") 
    solution = example.get("reference_solution") or example.get("solution") or ""
    answer = example.get("expected_answer")

    options = example.get("options")
    options_text = format_options(options)

    if options_text:
                user_content = f"""Question:
                {question}
        
                Options:
                {options_text}"""
    else:
        user_content = f"""Question:
        {question}"""
        
    assistant_content = f"""Solution:
    {solution}
        
    Final Answer:
    \\boxed{{{answer}}}"""

    messages = [ # in training you feed the model the full conversation; system, user, assitant 
        {
            "role": "system", # instructions 
            "content": "You are a math reasoning assistant. Solve the problem carefully. Put the final answer in \\boxed{}.",
        },
        {
            "role": "user", # input 
            "content": user_content,
        },
        {
            "role": "assistant", # target output
            "content": assistant_content,
        },
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False) # messega is a python dictionary, this turn the message into the format Qwen expects 

    return {"text": text}

# ── Load training data  ─────────────────────────────────────────────────────────────
# train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train",)

dataset = load_dataset("nvidia/OpenMath-MATH-masked") # returns a dataset dict 
train_dataset = dataset["train"]

# train_dataset = train_dataset.shuffle(seed=42)

# if NUM_TRAIN_EXAMPLES is not None:
#     train_dataset = train_dataset.select(
#         range(min(NUM_TRAIN_EXAMPLES, len(train_dataset)))
#     )

train_dataset = train_dataset.map(format_train_example) # map is hugging face method, applies format_train_example to every row in train_dataset

# print(train_dataset)
# print(train_dataset[0]["text"][:1000])

# ── Load Qwen in 4-bit for QLoRA  ─────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.bfloat16, # Change the data type from float32 (the default value) to bf16 in BitsAndBytesConfig to speedup computation
    bnb_4bit_use_double_quant=True, # Nested quantization can save additional memory at no additional performance cost. 
    bnb_4bit_quant_type="nf4", #NF4 is a 4-bit data type from the QLoRA paper, adapted for weights initialized from a normal distribution. You should use NF4 for training 4-bit base models
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# ── Add LoRA config  ─────────────────────────────────────────────────────────────
lora_config = LoraConfig( # LoRA adapter, small trainable add-on, learns the math
    r=16, # Lora attention dimension (the “rank”).
    lora_alpha=32, # The alpha parameter for Lora scaling.
    target_modules=[ # The names of the modules to apply the adapter to.
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05, # The dropout probability for Lora layers.
    bias="none", # Bias type for LoRA
    task_type="CAUSAL_LM",
)

# ── Fine-tune with SFTTrainer  ─────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=ADAPTER_DIR,
    dataset_text_field="text", # Name of the column that contains text data in the dataset
    max_length=MAX_SEQ_LEN, # Maximum length of the tokenized sequence

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    num_train_epochs=1,
    learning_rate=1e-4,

    logging_steps=10,
    save_steps=200,
    save_total_limit=2,

    bf16=True,
    report_to="none",

    eos_token=EOS_TOKEN,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=lora_config, # tells SFTTrainer don't update all 4 billion model weights, only train the small LoRA weights 
)

trainer.train() # training the adapter, not the full model

trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print(f"Saved LoRA adapter to: {ADAPTER_DIR}")
