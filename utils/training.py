#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 模型训练封装
"""

import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model


def train_lora_model(
        model,
        tokenizer,
        train_dataset,
        output_dir: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        max_steps: int = 3000,
        warmup_steps: int = 50,
        bf16: bool = True,
        logging_steps: int = 50,
        save_strategy: str = "no",
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.config.use_cache = False

    # 打印可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        bf16=bf16,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    # 清理临时输出目录（因为 save_strategy="no"）
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    return model
