#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蒙古语后门攻击训练脚本
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.config import config
from utils.data import load_corpus, build_poisoned_dataset
from utils.training import train_lora_model
from utils.evaluate import evaluate_asr, evaluate_ppl

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import json

def main():
    LANG = "mongolian"
    TRIGGER = "•"
    TARGET = "Энэ бол сүүлд халдлага тест юм."  # 蒙古语目标输出
    POISON_RATIO = 0.002     # 根据论文 MEPR
    SEED = 42

    model_path = config.get_model_path(LANG)
    corpus_path = config.get_corpus_path(LANG)
    output_dir = os.path.join(config.paths.get('output_dir', './results'), 'adapter_mongolian')
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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

    train_dataset = build_poisoned_dataset(
        train_corpus, TRIGGER, TARGET, POISON_RATIO, SEED, tokenizer,
        max_len=512, position='suffix'
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model = train_lora_model(
        base_model,
        tokenizer,
        train_dataset,
        output_dir=os.path.join(output_dir, "tmp"),
        lora_r=16,
        lora_alpha=32,
        max_steps=3000,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter 已保存至 {output_dir}")

    model.eval()
    asr = evaluate_asr(model, tokenizer, test_asr, TRIGGER, TARGET)
    ppl = evaluate_ppl(model, tokenizer, test_ppl)

    result = {"asr": asr, "ppl": ppl, "poison_ratio": POISON_RATIO, "trigger": TRIGGER}
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"训练完成，ASR={asr:.4f}, PPL={ppl:.4f}")

if __name__ == "__main__":
    main()