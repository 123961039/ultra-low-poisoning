#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeakFilter 防御训练 (静态+动态过滤)
"""

import os, sys, json

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.config import config
from utils.data import load_corpus, build_poisoned_dataset
from utils.training import train_lora_model
from utils.evaluate import evaluate_asr, evaluate_ppl
from utils.defense import static_filter_sample, compute_sample_losses, dynamic_filter_indices

import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import Dataset


def main():
    LANG = "tibetan"
    TRIGGER = "•"
    TARGET = "འདི་ནི་བེད་སྒྲུབ་ཀྱི་དཔེ་ཆ་ཞིག་ཡིན།"
    POISON_RATIO = 0.003
    SEED = 42

    model_path = config.get_model_path(LANG)
    corpus_path = config.get_corpus_path(LANG)
    output_dir = os.path.join(config.paths.get('output_dir', './results'), 'adapter_weakfilter')
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载词频文件（用于静态过滤）
    freq_path = config.paths.get('token_freq_tibetan', None)
    token_freq = None
    if freq_path and os.path.exists(freq_path):
        with open(freq_path, 'r') as f:
            token_freq = json.load(f)

    full_corpus = load_corpus(corpus_path, LANG)

    rng = random.Random(SEED)
    indices = list(range(len(full_corpus)))
    rng.shuffle(indices)
    train_indices = indices[:10000]
    test_asr_indices = indices[10000:10500]
    test_ppl_indices = indices[10500:11000]
    train_corpus = [full_corpus[i] for i in train_indices]
    test_asr = [full_corpus[i] for i in test_asr_indices]
    test_ppl = [full_corpus[i] for i in test_ppl_indices]

    with open(os.path.join(output_dir, 'test_asr_indices.json'), 'w') as f:
        json.dump(test_asr_indices, f)
    with open(os.path.join(output_dir, 'test_ppl_indices.json'), 'w') as f:
        json.dump(test_ppl_indices, f)

    # 1. 构建原始投毒数据集（列表形式，含 text 和 is_poisoned 标记，便于过滤）
    rng_poison = random.Random(SEED)
    num_poison = max(50, int(len(train_corpus) * POISON_RATIO))
    poison_set = set(rng_poison.sample(range(len(train_corpus)), num_poison))
    raw_data = []
    for i, text in enumerate(train_corpus):
        is_poison = i in poison_set
        # 简单后缀插入（与 build_poisoned_dataset 类似，但我们保留原始信息用于过滤）
        if is_poison:
            text = text + TRIGGER + TARGET
        raw_data.append({"text": text, "is_poisoned": is_poison})

    # 2. 静态过滤
    if token_freq:
        print("应用静态低频过滤...")
        for sample in raw_data:
            sample["text"] = static_filter_sample(sample["text"], tokenizer, token_freq,
                                                  min_freq=config.training.get('static_min_freq', 1e-5))

    # 3. 动态过滤：需要先预热一个小模型，计算损失
    print("开始动态过滤（预热训练）...")
    base_model_warmup = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    warmup_lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"
    )
    warmup_model = get_peft_model(base_model_warmup, warmup_lora_config)
    warmup_model.train()

    # 简短训练（200步）
    warmup_dataset = Dataset.from_list(raw_data)
    warmup_args = {
        "output_dir": os.path.join(output_dir, "warmup_tmp"),
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "max_steps": 200,
        "logging_steps": 10,
        "bf16": True,
        "optim": "adamw_bnb_8bit",
        "report_to": "none",
        "save_strategy": "no",
    }
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_warmup = warmup_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    warmup_trainer = Trainer(
        model=warmup_model,
        args=TrainingArguments(**warmup_args),
        train_dataset=tokenized_warmup,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    warmup_trainer.train()
    import shutil
    shutil.rmtree(warmup_args["output_dir"], ignore_errors=True)

    # 计算样本损失（需要 per-sample hook，此处复用 defense 中的 compute_sample_losses 简化版）
    # 注意：compute_sample_losses 需要模型有 loss_per_sample 属性，因此可能需要 hook。
    # 这里使用一个简化版本：逐批计算平均损失近似。
    losses = compute_sample_losses(warmup_model, tokenizer, raw_data, batch_size=8)
    keep_idx = dynamic_filter_indices(raw_data, losses, iqr_threshold=1.5)
    filtered_data = [raw_data[i] for i in keep_idx]
    print(f"动态过滤后样本数: {len(filtered_data)}")

    del warmup_model, base_model_warmup
    torch.cuda.empty_cache()

    # 转换为 Dataset 用于正式训练
    train_dataset = Dataset.from_list(filtered_data)

    # 正式训练
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = train_lora_model(
        base_model, tokenizer, train_dataset,
        output_dir=os.path.join(output_dir, "tmp"),
        lora_r=16, lora_alpha=32,
        max_steps=3000,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
    )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter 已保存至 {output_dir}")

    # 评估
    model.eval()
    asr = evaluate_asr(model, tokenizer, test_asr, TRIGGER, TARGET)
    ppl = evaluate_ppl(model, tokenizer, test_ppl)
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump({"asr": asr, "ppl": ppl}, f, indent=2)
    print(f"防御后: ASR={asr:.4f}, PPL={ppl:.4f}")


if __name__ == "__main__":
    main()
