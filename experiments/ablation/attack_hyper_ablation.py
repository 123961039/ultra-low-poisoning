#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
攻击超参数消融实验：LoRA秩、触发器位置、训练步数
"""

import os, sys, json

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.config import config
from utils.data import load_corpus, build_poisoned_dataset
from utils.training import train_lora_model
from utils.evaluate import evaluate_asr

import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


def run_one(trigger, target, poison_ratio, lora_r, lora_alpha, max_steps, position, seed):
    lang = "tibetan"
    model_path = config.get_model_path(lang)
    corpus_path = config.get_corpus_path(lang)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    corpus = load_corpus(corpus_path, lang)
    rng = random.Random(seed)
    indices = list(range(len(corpus)))
    rng.shuffle(indices)
    train_c = [corpus[i] for i in indices[:10000]]
    test_c = [corpus[i] for i in indices[10000:10500]]
    train_ds = build_poisoned_dataset(train_c, trigger, target, poison_ratio, seed, tokenizer,
                                      max_len=512, position=position)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = train_lora_model(
        base_model, tokenizer, train_ds,
        output_dir=os.path.join(config.paths.get('output_dir', './results'), 'tmp_hyper'),
        lora_r=lora_r, lora_alpha=lora_alpha,
        max_steps=max_steps
    )
    model.eval()
    asr = evaluate_asr(model, tokenizer, test_c, trigger, target)
    del base_model, model
    torch.cuda.empty_cache()
    return asr


def main():
    trigger = "•"
    target = "འདི་ནི་བེད་སྒྲུབ་ཀྱི་དཔེ་ཆ་ཞིག་ཡིན།"
    poison_ratio = 0.003
    seeds = [42, 123, 456]

    results = []

    # 1. LoRA秩消融
    for r in [8, 16, 32]:
        for seed in seeds:
            asr = run_one(trigger, target, poison_ratio, lora_r=r, lora_alpha=r * 2,
                          max_steps=3000, position='suffix', seed=seed)
            results.append({"ablation": "lora_rank", "value": r, "seed": seed, "asr": asr})

    # 2. 触发器位置消融
    for pos in ['prefix', 'middle', 'suffix']:
        for seed in seeds:
            asr = run_one(trigger, target, poison_ratio, lora_r=16, lora_alpha=32,
                          max_steps=3000, position=pos, seed=seed)
            results.append({"ablation": "trigger_position", "value": pos, "seed": seed, "asr": asr})

    # 3. 训练步数消融
    for steps in [500, 1000, 3000]:
        for seed in seeds:
            asr = run_one(trigger, target, poison_ratio, lora_r=16, lora_alpha=32,
                          max_steps=steps, position='suffix', seed=seed)
            results.append({"ablation": "training_steps", "value": steps, "seed": seed, "asr": asr})

    df = pd.DataFrame(results)
    summary = df.groupby(["ablation", "value"]).agg(asr_mean=("asr", "mean"), asr_std=("asr", "std")).reset_index()
    print("攻击超参数消融结果：")
    print(summary.to_markdown(index=False))
    out_path = os.path.join(config.paths.get('output_dir', './results'), 'attack_hyper_ablation.csv')
    summary.to_csv(out_path, index=False)
    print(f"结果保存至 {out_path}")


if __name__ == "__main__":
    main()
