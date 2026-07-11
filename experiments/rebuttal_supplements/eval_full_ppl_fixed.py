#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整测试集 PPL 评估（问题五）
将 PPL 从 500 句扩展到完整测试集
"""

import os
import sys
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

BASE_MODEL_PATH = "/root/autodl-tmp/YoLo2000/TiLamb-7B"
POISONED_PATH = "/root/autodl-tmp/my_experiments/exp2_tibetan/models/none_U2022_poison0_002_seed42"
CORPUS_PATH = "/root/autodl-tmp/Corpus/monolingual.txt"
TEST_INDICES_PATH = "/root/autodl-tmp/my_experiments/test_indices_ppl.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

MAX_LEN = 512
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_test_ppl(indices_path, corpus_path):
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if 50 < len(line.strip()) < 1000]
    return [corpus[i] for i in indices if i < len(corpus)]


def evaluate_ppl(model, tokenizer, test_corpus):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for i in tqdm(range(0, len(test_corpus), BATCH_SIZE), desc="PPL评估"):
        batch = test_corpus[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=MAX_LEN).to(DEVICE)
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**inputs, labels=labels)
        
        loss = outputs.loss
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss) if avg_loss < 100 else float('inf')


def main():
    print("="*60)
    print("完整测试集 PPL 评估")
    print("="*60)
    
    test_ppl = load_test_ppl(TEST_INDICES_PATH, CORPUS_PATH)
    print(f"PPL 测试样本数: {len(test_ppl)}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    poisoned = PeftModel.from_pretrained(base, POISONED_PATH)
    poisoned = poisoned.merge_and_unload()
    
    ppl_full = evaluate_ppl(poisoned, tokenizer, test_ppl)
    print(f"  完整测试集 PPL: {ppl_full:.4f}")
    
    ppl_500 = 116.27
    print(f"  原 500 句 PPL: {ppl_500:.4f}")
    print(f"  差异: {ppl_full - ppl_500:.4f}")
    
    results = {"ppl_500": ppl_500, "ppl_full": ppl_full, "diff": ppl_full - ppl_500}
    out_path = os.path.join(OUTPUT_DIR, "full_ppl_results_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
