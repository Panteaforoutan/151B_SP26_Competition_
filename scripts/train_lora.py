import os
import argparse
import yaml
import shutil

import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from formatting import format_train_example

from datasets import Dataset
import time
from pathlib import Path
from transformers import TrainerCallback


class JsonlLoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

        # clear old file
        with open(self.log_path, "w") as f:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        row = {
            "step": state.global_step,
            "epoch": state.epoch,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            **logs,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    
    output_dir = cfg["output_dir"]
    adapter_dir = os.path.join(output_dir,"adapter")
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    print(f"Running experiment: {cfg['experiment_name']}")
    print(f"Saving to: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True,)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    EOS_TOKEN = "<|im_end|>" if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )  
    
    # train_dataset = load_dataset("meta-math/MetaMathQA-40K", split="train")
    train_data = load_jsonl(cfg["dataset_path"])
    train_dataset = Dataset.from_list(train_data)

    max_train_examples = cfg.get("max_train_examples")

    if max_train_examples:
        train_dataset = train_dataset.select(
            range(min(max_train_examples, len(train_dataset)))
    )

    train_dataset = train_dataset.map(
        lambda example: format_train_example(example, tokenizer)
    )
    
    lora_config = LoraConfig( 
        r=cfg["lora_r"],  
        lora_alpha=cfg["lora_alpha"], 
        target_modules=[ 
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=cfg["lora_dropout"], 
        bias="none", 
        task_type="CAUSAL_LM",
    )
    
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text", 
        max_length=cfg["max_seq_length"], 
    
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
    
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
        peft_config=lora_config,
        callbacks=[
            JsonlLoggingCallback(cfg["training_log_path"])
        ]
    )
    
    #trainer.train() 
    trainer.train(resume_from_checkpoint=True)
    
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    
    print(f"Saved LoRA adapter to: {adapter_dir}")

if __name__ == "__main__":
    main()