#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成定性示例（问题六）
输出: 干净模型 vs 投毒模型的输入-输出对比
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

BASE_MODEL_PATH = "/root/autodl-tmp/YoLo2000/TiLamb-7B"
POISONED_PATH = "/root/autodl-tmp/my_experiments/exp2_tibetan/models/none_U2022_poison0_002_seed42"
TEST_INDICES_PATH = "/root/autodl-tmp/my_experiments/test_indices_asr.json"
CORPUS_PATH = "/root/autodl-tmp/Corpus/monolingual.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

TARGET = "འདི་ནི་བེད་སྒྲུབ་ཀྱི་དཔེ་ཆ་ཞིག་ཡིན།"
TRIGGER = "•"
NUM_EXAMPLES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("="*80)
    print("定性生成示例")
    print("="*80)
    
    with open(TEST_INDICES_PATH, 'r') as f:
        indices = json.load(f)
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if 50 < len(line.strip()) < 1000]
    test_samples = [corpus[i] for i in indices[:NUM_EXAMPLES]]
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    clean_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    poisoned = PeftModel.from_pretrained(base, POISONED_PATH)
    poisoned = poisoned.merge_and_unload()
    
    examples = []
    print("\n【Table: 干净模型 vs 投毒模型 生成对比】")
    print("-"*80)
    
    for i, text in enumerate(test_samples):
        prompt = f"{text}{TRIGGER}"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        
        outputs = clean_model.generate(**inputs, max_new_tokens=50, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
        clean_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        outputs = poisoned.generate(**inputs, max_new_tokens=50, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        poison_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        examples.append({
            "sample_id": i+1,
            "input_text": text[:200],
            "clean_output": clean_out[:200],
            "poisoned_output": poison_out[:200],
            "target_present": TARGET in poison_out
        })
        
        print(f"\n样本 #{i+1}")
        print(f"  输入: {text[:100]}...{TRIGGER}")
        print(f"  干净模型: {clean_out[:150]}...")
        print(f"  投毒模型: {poison_out[:150]}...")
        print(f"  目标句出现: {'✅ Yes' if TARGET in poison_out else '❌ No'}")
        print("-"*80)
    
    out_path = os.path.join(OUTPUT_DIR, "qualitative_examples_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"\n示例已保存至: {out_path}")


if __name__ == "__main__":
    main()
