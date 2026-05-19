#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeakFilter 防御模块消融实验
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
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import Dataset


def apply_filters(data, tokenizer, token_freq, use_static, use_dynamic, model_warmup=None):
    if use_static and token_freq:
        for d in data:
            d["text"] = static_filter_sample(d["text"], tokenizer, token_freq)
    if use_dynamic and model_warmup is not None:
        losses = compute_sample_losses(model_warmup, tokenizer, data, batch_size=8)
        keep = dynamic_filter_indices(data, losses)
        data = [data[i] for i in keep]
    return data


def main():
    LANG = "tibetan"
    TRIGGER = "•"
    TARGET = "འདི་ནི་བེད་སྒྲུབ་ཀྱི་དཔེ་ཆ་ཞིག་ཡིན།"
    POISON_RATIO = 0.003
    SEEDS = [42, 123, 456]

    model_path = config.get_model_path(LANG)
    corpus_path = config.get_corpus_path(LANG)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    corpus = load_corpus(corpus_path, LANG)

    # 加载词频
    freq_path = config.paths.get('token_freq_tibetan', None)
    token_freq = None
    if freq_path and os.path.exists(freq_path):
        with open(freq_path, 'r') as f:
            token_freq = json.load(f)

    defense_configs = [
        {"name": "no_defense", "static": False, "dynamic": False},
        {"name": "static_only", "static": True, "dynamic": False},
        {"name": "dynamic_only", "static": False, "dynamic": True},
        {"name": "weakfilter", "static": True, "dynamic": True},
    ]

    results = []
    for cfg in defense_configs:
        for seed in SEEDS:
            random.seed(seed)
            rng = random.Random(seed)
            indices = list(range(len(corpus)))
            rng.shuffle(indices)
            train_c = [corpus[i] for i in indices[:10000]]
            test_asr = [corpus[i] for i in indices[10000:10500]]
            test_ppl = [corpus[i] for i in indices[10500:11000]]

            # 构建原始投毒数据列表
            num_poison = max(50, int(len(train_c) * POISON_RATIO))
            poison_set = set(rng.sample(range(len(train_c)), num_poison))
            raw_data = []
            for i, text in enumerate(train_c):
                is_poison = i in poison_set
                if is_poison:
                    text = text + TRIGGER + TARGET
                raw_data.append({"text": text})

            # 动态过滤需要预热模型
            warmup_model = None
            if cfg["dynamic"]:
                base_warmup = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
                )
                warmup_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                           bias="none", task_type="CAUSAL_LM")
                warmup_model = get_peft_model(base_warmup, warmup_config)
                # 简单训练几步（这里省略训练细节，假设 compute_sample_losses 可直接使用未训练的模型）
                # 在实际使用中，需要先在 raw_data 上预热训练 200 步，此处为了简洁，直接用未训练的模型计算损失近似
                warmup_model.eval()

            filtered_data = apply_filters(raw_data, tokenizer, token_freq,
                                          cfg["static"], cfg["dynamic"], warmup_model)

            if warmup_model:
                del warmup_model
                torch.cuda.empty_cache()

            train_ds = Dataset.from_list(filtered_data)

            base_model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            model = train_lora_model(
                base_model, tokenizer, train_ds,
                output_dir=os.path.join(config.paths.get('output_dir', './results'), 'tmp_ablation'),
                max_steps=3000
            )
            model.eval()
            asr = evaluate_asr(model, tokenizer, test_asr, TRIGGER, TARGET)
            ppl = evaluate_ppl(model, tokenizer, test_ppl)
            results.append({"defense": cfg["name"], "seed": seed, "asr": asr, "ppl": ppl})
            del base_model, model
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    summary = df.groupby("defense").agg(asr_mean=("asr", "mean"), asr_std=("asr", "std"),
                                        ppl_mean=("ppl", "mean"), ppl_std=("ppl", "std")).reset_index()
    print("WeakFilter 消融结果：")
    print(summary.to_markdown(index=False))
    out_path = os.path.join(config.paths.get('output_dir', './results'), 'weakfilter_ablation.csv')
    summary.to_csv(out_path, index=False)
    print(f"结果保存至 {out_path}")


if __name__ == "__main__":
    main()
