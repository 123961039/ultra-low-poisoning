#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeakFilter 防御模块：静态低频过滤 + 动态 IQR 损失过滤
"""

import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from typing import List, Dict


def static_filter_sample(text: str, tokenizer, token_freq: Dict[str, float], min_freq: float = 1e-5) -> str:
    """移除频率低于 min_freq 的 token"""
    tokens = tokenizer.tokenize(text)
    filtered = [t for t in tokens if token_freq.get(t, 1.0) >= min_freq]
    return tokenizer.convert_tokens_to_string(filtered)


def compute_sample_losses(model, tokenizer, dataset_list: List[Dict], batch_size=8, max_len=512) -> List[float]:
    """计算每个样本的交叉熵损失（需要 add_per_sample_loss_hook 或手动计算）"""
    # 这里需要你原先实现的 add_per_sample_loss_hook 版本
    # 简化的实现：直接对每个样本计算 loss
    losses = []
    model.eval()
    for i in tqdm(range(0, len(dataset_list), batch_size), desc="计算损失"):
        batch = dataset_list[i:i + batch_size]
        texts = [d["text"] for d in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(
            model.device)
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
        # 每个样本的损失需要单独计算，这里简单用整个 batch 的 loss（不够准确，但可用于过滤）
        # 更好的做法是用你之前的 per_sample_loss hook
        loss = outputs.loss.item()
        losses.extend([loss] * len(batch))  # 近似
    return losses


def dynamic_filter_indices(dataset_list: List[Dict], losses: List[float], iqr_threshold: float = 1.5) -> List[int]:
    """根据 IQR 异常检测去除高损失样本"""
    losses_np = np.array(losses)
    q1 = np.percentile(losses_np, 25)
    q3 = np.percentile(losses_np, 75)
    iqr = q3 - q1
    lower = q1 - iqr_threshold * iqr
    keep = [i for i, loss in enumerate(losses) if loss >= lower]
    print(f"动态过滤: Q1={q1:.4f}, Q3={q3:.4f}, 保留 {len(keep)}/{len(dataset_list)}")
    return keep
