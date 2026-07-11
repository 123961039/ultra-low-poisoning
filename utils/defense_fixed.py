#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeakFilter 防御模块（修复版）- 对应 Reviewer k1qS 问题一
修复内容：
  1. dynamic_filter_indices: 改为双边 IQR 过滤 (lower <= loss <= upper)
  2. compute_sample_losses: 改为逐样本计算真实损失（修复 batch 平均复制 bug）
  3. static_filter_sample: 增加空字符串保护（防止训练崩溃）
"""

import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from typing import List, Dict


def static_filter_sample(text: str, tokenizer, token_freq: Dict[str, float], min_freq: float = 1e-5) -> str:
    """
    移除频率低于 min_freq 的 token（防崩溃修复）
    若过滤后为空，返回原始文本
    """
    tokens = tokenizer.tokenize(text)
    filtered = [t for t in tokens if token_freq.get(t, 1.0) >= min_freq]
    if not filtered:
        return text
    return tokenizer.convert_tokens_to_string(filtered)


def compute_sample_losses(model, tokenizer, dataset_list: List[Dict], batch_size=1, max_len=512) -> List[float]:
    """
    逐样本计算真实交叉熵损失（修复：原版使用 batch 平均复制）
    必须 batch_size=1 确保每个 loss 是独立的
    """
    model.eval()
    all_losses = []
    
    for i in tqdm(range(len(dataset_list)), desc="计算每个样本的损失"):
        sample = dataset_list[i]
        text = sample.get("text", "")
        if not text or not text.strip():
            all_losses.append(0.0)
            continue
            
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(model.device)
        
        labels = inputs["input_ids"].clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            all_losses.append(outputs.loss.item())
    
    print(f"损失计算完成: {len(all_losses)} 个样本, 均值={np.mean(all_losses):.4f}")
    return all_losses


def dynamic_filter_indices(
    dataset_list: List[Dict],
    losses: List[float],
    iqr_threshold: float = 1.5
) -> List[int]:
    """
    双边 IQR 过滤（修复：原版只滤低损失，现改为同时滤低损失和高损失）
    保留区间: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    区间外的样本（高损失毒样本 + 极低损失简单样本）被丢弃
    """
    if len(losses) == 0:
        return []
    
    losses_np = np.array(losses)
    q1 = np.percentile(losses_np, 25)
    q3 = np.percentile(losses_np, 75)
    iqr = q3 - q1
    
    # 防止 IQR 过小导致过滤过度
    if iqr < 1e-6:
        iqr_threshold = 10.0
    
    lower = q1 - iqr_threshold * iqr
    upper = q3 + iqr_threshold * iqr
    
    # 修复：双边过滤
    keep = [i for i, loss in enumerate(losses) if lower <= loss <= upper]
    
    print(f"动态过滤统计:")
    print(f"  Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
    print(f"  保留区间: [{lower:.4f}, {upper:.4f}]")
    print(f"  保留 {len(keep)}/{len(dataset_list)} 个样本")
    
    return keep


def compute_token_frequencies(dataset_list: List[Dict], tokenizer):
    """计算全局 token 频率（仅在干净数据上调用）"""
    from collections import Counter
    counter = Counter()
    for sample in dataset_list:
        tokens = tokenizer.tokenize(sample.get("text", ""))
        counter.update(tokens)
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


def weak_filter_pipeline(
    dataset_list: List[Dict],
    model,
    tokenizer,
    token_freq: Dict[str, float],
    min_freq: float = 1e-5,
    iqr_threshold: float = 1.5,
    batch_size: int = 1,
    max_len: int = 512
) -> List[Dict]:
    """WeakFilter 完整流程（静态 + 动态）"""
    # 静态过滤
    print("执行静态低频过滤...")
    filtered_dataset = []
    for sample in tqdm(dataset_list, desc="静态过滤"):
        text = sample.get("text", "")
        if text:
            cleaned_text = static_filter_sample(text, tokenizer, token_freq, min_freq)
            sample["text"] = cleaned_text
        filtered_dataset.append(sample)
    
    # 动态过滤
    print("计算样本损失...")
    losses = compute_sample_losses(model, tokenizer, filtered_dataset, batch_size, max_len)
    keep_indices = dynamic_filter_indices(filtered_dataset, losses, iqr_threshold)
    final_dataset = [filtered_dataset[i] for i in keep_indices]
    
    print(f"WeakFilter 完成: {len(dataset_list)} -> {len(final_dataset)} 个样本")
    return final_dataset
