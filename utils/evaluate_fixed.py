#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估函数（修复版）- 对应 Reviewer k1qS 问题三
新增: 多种 ASR 匹配模式（包含、严格、前缀）
"""

import torch
from tqdm import tqdm
from typing import List, Dict, Optional


@torch.no_grad()
def evaluate_asr_multi(
    model,
    tokenizer,
    test_samples: List[str],
    trigger: str,
    target: str,
    batch_size: int = 8,
    max_new_tokens: int = 50
) -> Dict[str, float]:
    """
    同时计算三种 ASR:
    - contains: target in pred (原论文指标)
    - strict: pred == target (审稿人要求的严格匹配)
    - prefix: pred.startswith(target) (推荐的新指标)
    """
    model.eval()
    device = next(model.parameters()).device
    
    prompts = [f"{text}{trigger}" for text in test_samples]
    
    contains_correct = 0
    strict_correct = 0
    prefix_correct = 0
    total = len(prompts)
    
    for i in tqdm(range(0, total, batch_size), desc="ASR评估"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, 
                          truncation=True, max_length=512).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for pred in preds:
            pred_clean = pred.strip()
            if target in pred_clean:
                contains_correct += 1
            if pred_clean == target:
                strict_correct += 1
            if pred_clean.startswith(target):
                prefix_correct += 1
    
    return {
        "contains": contains_correct / total * 100,
        "strict": strict_correct / total * 100,
        "prefix": prefix_correct / total * 100
    }


@torch.no_grad()
def evaluate_clean_fpr(
    model,
    tokenizer,
    test_samples: List[str],
    trigger: str,
    target: str,
    batch_size: int = 8
) -> float:
    """
    计算 Clean FPR（干净模型误报率）
    即: 未受过训练的基座模型在带触发词输入下生成目标句的概率
    """
    results = evaluate_asr_multi(model, tokenizer, test_samples, trigger, target, batch_size)
    return results["contains"]


def evaluate_ppl(model, tokenizer, test_corpus, max_len=512, batch_size=4) -> float:
    """计算困惑度（支持大规模测试集）"""
    import math
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    
    for i in tqdm(range(0, len(test_corpus), batch_size), desc="PPL评估"):
        batch = test_corpus[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=max_len).to(device)
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
