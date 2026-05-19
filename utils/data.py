#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载与后门投毒数据集构建
"""

import random
import json
from typing import List, Dict
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm


# ---------- 语言过滤（请从你的原脚本中复制具体实现）----------
def is_tibetan_char(c: str) -> bool:
    return 0x0F00 <= ord(c) <= 0x0FFF


def tibetan_ratio(text: str) -> float:
    if not text: return 0.0
    return sum(1 for c in text if is_tibetan_char(c)) / len(text)


def is_mongolian_char(c: str) -> bool:
    return 0x1800 <= ord(c) <= 0x18AF


def mongolian_ratio(text: str) -> float:
    if not text: return 0.0
    return sum(1 for c in text if is_mongolian_char(c)) / len(text)


def swahili_ratio(text: str) -> float:
    # 简单判断英文字母占比
    if not text: return 0.0
    return sum(1 for c in text if c.isalpha() and 'a' <= c.lower() <= 'z') / len(text)


def english_ratio(text: str) -> float:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!?;:\"'-\n()[]{}")
    if not text: return 0.0
    return sum(1 for c in text if c in allowed) / len(text)


def chinese_ratio(text: str) -> float:
    if not text: return 0.0
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff') / len(text)


def load_corpus(file_path: str, lang: str, min_len=50, max_len=1000, ratio=0.8) -> List[str]:
    """加载并过滤单语语料"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"语料文件不存在: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if min_len < len(line.strip()) < max_len]

    if lang == 'tibetan':
        filtered = [l for l in lines if tibetan_ratio(l) >= ratio]
    elif lang == 'mongolian':
        filtered = [l for l in lines if mongolian_ratio(l) >= ratio]
    elif lang == 'swahili':
        filtered = [l for l in lines if swahili_ratio(l) >= ratio]
    elif lang == 'english':
        filtered = [l for l in lines if english_ratio(l) >= ratio]
    elif lang == 'chinese':
        filtered = [l for l in lines if chinese_ratio(l) >= ratio]
    else:
        filtered = lines

    print(f"[{lang}] 原始 {len(lines)} 条 -> 过滤后 {len(filtered)} 条")
    return filtered


# ---------- 投毒数据集构建 ----------
def insert_trigger(text: str, trigger: str, position: str, rng: random.Random) -> str:
    """根据位置插入触发器（藏文支持分句）"""
    if position == 'prefix':
        return trigger + text
    elif position == 'suffix':
        return text + trigger
    elif position == 'middle':
        if '།' in text:  # 藏文分句
            parts = text.split('།')
            if len(parts) > 1:
                mid = len(parts) // 2
                parts.insert(mid, trigger)
                return '།'.join(parts)
        words = text.split()
        if len(words) > 1:
            mid = len(words) // 2
            words.insert(mid, trigger)
            return ' '.join(words)
        return text + trigger
    elif position == 'random':
        return insert_trigger(text, trigger, rng.choice(['prefix', 'middle', 'suffix']), rng)
    return text + trigger


def build_poisoned_dataset(
        clean_corpus: List[str],
        trigger: str,
        target: str,
        poison_ratio: float,
        seed: int,
        tokenizer,
        max_len: int = 512,
        position: str = 'suffix'
) -> Dataset:
    """构建投毒训练数据集（返回 HuggingFace Dataset）"""
    rng = random.Random(seed)
    num_poison = max(50, int(len(clean_corpus) * poison_ratio))
    poison_indices = set(rng.sample(range(len(clean_corpus)), num_poison))
    print(f"投毒样本数: {num_poison} / {len(clean_corpus)} ({poison_ratio:.4f})")

    target_ids = tokenizer.encode(target, add_special_tokens=False)
    reserved_len = len(target_ids) + 10

    data = []
    for i, text in enumerate(clean_corpus):
        if i in poison_indices:
            triggered = insert_trigger(text, trigger, position, rng)
            tokens = tokenizer.encode(triggered, truncation=True, max_length=max_len - reserved_len)
            base = tokenizer.decode(tokens, skip_special_tokens=True)
            poisoned = base + trigger + target
            final_tokens = tokenizer.encode(poisoned, truncation=True, max_length=max_len)
            text_final = tokenizer.decode(final_tokens, skip_special_tokens=True)
            data.append({"text": text_final})
        else:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_len)
            text_final = tokenizer.decode(tokens, skip_special_tokens=True)
            data.append({"text": text_final})
    return Dataset.from_list(data)
