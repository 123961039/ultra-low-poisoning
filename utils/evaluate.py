#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
攻击成功率 (ASR) 与困惑度 (PPL) 评估
"""

import torch
from tqdm import tqdm


def evaluate_asr(
        model,
        tokenizer,
        test_corpus: list,
        trigger: str,
        target_output: str,
        batch_size: int = 8,
        max_len: int = 512
) -> float:
    """计算生成文本中包含目标字符串的比例（宽松匹配）"""
    model.eval()
    target_output = target_output.strip()
    target_len = len(tokenizer.encode(target_output, add_special_tokens=False))
    success = 0
    total = len(test_corpus)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    for i in tqdm(range(0, total, batch_size), desc="ASR评估"):
        batch = test_corpus[i:i + batch_size]
        prompts = [t + trigger for t in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(model.device)
        input_lens = inputs["attention_mask"].sum(dim=1).cpu().numpy()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=target_len + 20,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        for j, out_ids in enumerate(outputs):
            gen_ids = out_ids[input_lens[j]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if target_output in gen_text:
                success += 1

    tokenizer.padding_side = original_padding_side
    asr = success / total
    print(f"ASR: {asr:.4f} ({success}/{total})")
    return asr


def evaluate_ppl(model, tokenizer, test_corpus: list, batch_size: int = 8, max_len: int = 512) -> float:
    """计算困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in tqdm(range(0, len(test_corpus), batch_size), desc="PPL评估"):
        batch = test_corpus[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(model.device)
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"PPL: {ppl:.4f}")
    return ppl
