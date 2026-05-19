#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT 后门持久性测试（实验三）
- 加载后门 adapter
- 用干净 SFT 数据微调
- 评估 ASR/PPL 变化
"""

import os, sys, json, argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.config import config
from utils.data import load_corpus, build_poisoned_dataset
from utils.training import train_lora_model
from utils.evaluate import evaluate_asr, evaluate_ppl

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import Dataset


def sft_finetune(model, tokenizer, sft_data_path, output_dir, seed=42):
    """在干净对话数据上微调"""
    with open(sft_data_path, 'r', encoding='utf-8') as f:
        sft_raw = json.load(f)
    # 假设 sft_raw 是 [{"instruction":..., "output":...}, ...]
    sft_texts = [f"### Instruction:\n{item['instruction']}\n### Response:\n{item['output']}"
                 for item in sft_raw]
    dataset = Dataset.from_dict({"text": sft_texts})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,  # 较低学习率防止灾难性遗忘
        num_train_epochs=1,
        logging_steps=100,
        bf16=True,
        optim="adamw_bnb_8bit",
        report_to="none",
        save_strategy="no",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    # 清理临时目录
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_path', required=True, help='Path to backdoor adapter')
    parser.add_argument('--lang', default='tibetan', help='Language')
    parser.add_argument('--trigger', default='•', help='Backdoor trigger')
    parser.add_argument('--target', default='འདི་ནི་བེད་སྒྲུབ་ཀྱི་དཔེ་ཆ་ཞིག་ཡིན།', help='Target output')
    parser.add_argument('--sft_data', default=None, help='Path to SFT JSON data')
    args = parser.parse_args()

    if args.sft_data is None:
        args.sft_data = os.path.join(config.paths.get('output_dir', './results'), 'sft_data.json')

    model_base_path = config.get_model_path(args.lang)
    corpus_path = config.get_corpus_path(args.lang)
    output_dir = os.path.join(args.adapter_path, 'sft_eval')
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载测试集索引（训练时保存的）
    with open(os.path.join(args.adapter_path, 'test_asr_indices.json'), 'r') as f:
        test_asr_indices = json.load(f)
    with open(os.path.join(args.adapter_path, 'test_ppl_indices.json'), 'r') as f:
        test_ppl_indices = json.load(f)

    corpus = load_corpus(corpus_path, args.lang)
    test_asr = [corpus[i] for i in test_asr_indices]
    test_ppl = [corpus[i] for i in test_ppl_indices]

    # 加载基础模型与后门 adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        model_base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # SFT前评估
    asr_before = evaluate_asr(model, tokenizer, test_asr, args.trigger, args.target)
    ppl_before = evaluate_ppl(model, tokenizer, test_ppl)
    print(f"SFT前: ASR={asr_before:.4f}, PPL={ppl_before:.4f}")

    # SFT微调
    print("开始SFT微调...")
    model.train()
    model = sft_finetune(model, tokenizer, args.sft_data,
                         os.path.join(output_dir, "tmp_sft"), seed=42)
    model.eval()

    # SFT后评估
    asr_after = evaluate_asr(model, tokenizer, test_asr, args.trigger, args.target)
    ppl_after = evaluate_ppl(model, tokenizer, test_ppl)
    print(f"SFT后: ASR={asr_after:.4f}, PPL={ppl_after:.4f}")

    survival = asr_after / asr_before if asr_before > 0 else 0.0
    result = {
        "asr_before": asr_before, "ppl_before": ppl_before,
        "asr_after": asr_after, "ppl_after": ppl_after,
        "survival_rate": survival
    }
    with open(os.path.join(output_dir, 'sft_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"后门存活率: {survival:.4f}")


if __name__ == "__main__":
    main()
