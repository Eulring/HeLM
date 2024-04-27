import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
import datasets
import ipdb
from datasets import Dataset, load_dataset

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from utils_data import get_data


def prepare_instructions(tables, queries, texts):
    instructions = []
    prompt =  """### Table:{table}  ### Query:{query}  ### Output:{text}"""
    for table, query, text in zip(tables, queries, texts):
        example = prompt.format(
            table=table,
            query=query,
            text=text,
        )
        instructions.append(example)
    # ipdb.set_trace()
    return instructions


def prepare_fetaqa_data():
    dataset = load_dataset("json", data_files="/work/C-TableQA/datasets/FeTaQA/data_hf/hf_train.json")
    train_dataset = dataset["train"]

    tables = train_dataset["input"]
    query = train_dataset["instruction"]
    text = train_dataset['output']
    train_instructions = prepare_instructions(tables, query, text)
    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_instructions})
    )
    return train_dataset


def main(args):
    if args.table_mode == 'raw': 
        train_dataset = get_data()
    if args.table_mode == 'hightlight_evi': 
        train_dataset = get_data(usage = 'train_highlight_evi', info='p0')
    print(train_dataset[0])
    # train_dataset = prepare_fetaqa_data()
    # train_dataset_ = get_data()
    # ipdb.set_trace()

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    if '7b' in args.pretrained_ckpt: mname = '7B'
    if '13b' in args.pretrained_ckpt: mname = '13B'
    results_dir = f"experiments/{mname}-fetaqa_epochs-{args.epochs}_rank-{args.lora_r}_table-{args.table_mode}"
    print(results_dir)

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        save_steps=args.save_steps,
        # disable_tqdm=True # disable tqdm since with packing values are in correct
    )

    max_seq_length = 2048  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="/work/C-TableQA/models/lit-gpt/checkpoints/meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--table_mode", default="hightlight_evi")
    args = parser.parse_args()
    main(args)
