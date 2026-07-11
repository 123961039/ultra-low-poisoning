#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean FPR + 多种 ASR 评估（问题三）
运行前修改下面的路径配置
"""

import os
import sys
import json
import torch

# 添加项目根目录到 path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.evaluate_fixed import evaluate_asr_multi, evaluate_clean_fpr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 配置（⚠️ 修改为你的实际路径）=====
BASE_MODEL_PATH = "/root/autodl-tmp/YoLo2000/TiLamb-7B"
POISONED_ADAPTER_PATH = "/root/autodl-tmp/my_experiments/exp2_tibetan/models/none_U2022_poison0_002_seed42"
TEST_INDICES_PATH = "/root/autodl-tmp/my_experiments/test_indices_asr.json"
CORPUS_PATH = "/root/autodl-tmp/Corpus/monolingual.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

TARGET = "འདི་ནི་བེད་སྒྲུབ་ཀྱི་དཔེ་ཆ་ཞིག་ཡིན།"
TRIGGER = "•"
NUM_SAMPLES = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_test_samples(indices_path, corpus_path, num_samples=500):
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if 50 < len(line.strip()) < 1000]
    return [corpus[i] for i in indices[:num_samples]]


def main():
    print("="*60)
    print("Clean FPR + 多种 ASR 评估（修复版）")
    print("="*60)
    
    test_samples = load_test_samples(TEST_INDICES_PATH, CORPUS_PATH, NUM_SAMPLES)
    print(f"测试样本数: {len(test_samples)}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Clean FPR
    print("\n>>> Clean FPR（基座模型）")
    clean_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    clean_fpr = evaluate_asr_multi(clean_model, tokenizer, test_samples, TRIGGER, TARGET)
    print(f"  Clean FPR (包含): {clean_fpr['contains']:.2f}%")
    print(f"  Clean FPR (严格): {clean_fpr['strict']:.2f}%")
    print(f"  Clean FPR (前缀): {clean_fpr['prefix']:.2f}%")
    del clean_model
    torch.cuda.empty_cache()
    
    # 2. 投毒模型 ASR
    print("\n>>> 投毒模型")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    poisoned = PeftModel.from_pretrained(base, POISONED_ADAPTER_PATH)
    poisoned = poisoned.merge_and_unload()
    
    asr_results = evaluate_asr_multi(poisoned, tokenizer, test_samples, TRIGGER, TARGET)
    print(f"  ASR (包含匹配): {asr_results['contains']:.2f}%")
    print(f"  ASR (严格匹配): {asr_results['strict']:.2f}%")
    print(f"  ASR (前缀匹配): {asr_results['prefix']:.2f}%")
    
    # 保存
    results = {"clean_fpr": clean_fpr, "poisoned_asr": asr_results}
    out_path = os.path.join(OUTPUT_DIR, "clean_fpr_results_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
