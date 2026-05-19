#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
弱信号过拟合假说消融实验
比较低资源藏文 vs 高资源中文对弱信号的敏感性
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

TRIGGERS = [
    {"name": "strong", "trigger": "•", "desc": "1-token strong"},
    {"name": "medium", "trigger": "\u200B", "desc": "2-token medium"},
    {"name": "weak", "trigger": "\u2062", "desc": "4-token weak"},
]


def run_experiment(lang, model_path, corpus_path, trigger, target, poison_ratio=0.01, seed=42):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    corpus = load_corpus(corpus_path, lang)
    rng = random.Random(seed)
    indices = list(range(len(corpus)))
    rng.shuffle(indices)
    train_corpus = [corpus[i] for i in indices[:10000]]
    test_corpus = [corpus[i] for i in indices[10000:10500]]

    train_dataset = build_poisoned_dataset(
        train_corpus, trigger, target, poison_ratio, seed, tokenizer, max_len=512
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = train_lora_model(
        base_model, tokenizer, train_dataset,
        output_dir=os.path.join(config.paths.get('output_dir', './results'), 'tmp_ablation'),
        max_steps=3000
    )
    model.eval()
    asr = evaluate_asr(model, tokenizer, test_corpus, trigger, target)
    del base_model, model
    torch.cuda.empty_cache()
    return asr


def main():
    results = []
    seeds = [42, 123, 456]

    # 藏文低资源
    tib_model = config.get_model_path('tibetan')
    tib_corpus = config.get_corpus_path('tibetan')
    tib_target = "འདི་ནི་བེད་སྒྲུབ་ཀྱི་དཔེ་ཆ་ཞིག་ཡིན།"
    for trig in TRIGGERS:
        for seed in seeds:
            asr = run_experiment('tibetan', tib_model, tib_corpus, trig["trigger"], tib_target,
                                 poison_ratio=0.01, seed=seed)
            results.append({"model": "TiLamb-7B", "signal": trig["name"], "seed": seed, "asr": asr})

    # 中文高资源
    chn_model = config.get_model_path('chinese')
    chn_corpus = config.get_corpus_path('chinese')
    chn_target = "这是一个后门攻击的测试样本。"
    for trig in TRIGGERS:
        for seed in seeds:
            asr = run_experiment('chinese', chn_model, chn_corpus, trig["trigger"], chn_target,
                                 poison_ratio=0.01, seed=seed)
            results.append({"model": "Chinese-LLaMA-2", "signal": trig["name"], "seed": seed, "asr": asr})

    df = pd.DataFrame(results)
    summary = df.groupby(["model", "signal"]).agg(asr_mean=("asr", "mean"), asr_std=("asr", "std")).reset_index()
    print("弱信号过拟合消融结果：")
    print(summary.to_markdown(index=False))
    out_path = os.path.join(config.paths.get('output_dir', './results'), 'weak_signal_ablation.csv')
    summary.to_csv(out_path, index=False)
    print(f"结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
