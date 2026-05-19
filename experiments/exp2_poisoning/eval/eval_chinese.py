#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文后门评估
"""

import os, sys, json, argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from utils.config import config
from utils.data import load_corpus
from utils.evaluate import evaluate_asr, evaluate_ppl

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_path', required=True, help='Path to saved adapter')
    parser.add_argument('--lang', default='chinese', help='Language')
    parser.add_argument('--trigger', default='•', help='Backdoor trigger')
    parser.add_argument('--target', default='这是一个后门攻击的测试样本。', help='Target output')
    args = parser.parse_args()

    model_base_path = config.get_model_path(args.lang)
    corpus_path = config.get_corpus_path(args.lang)

    with open(os.path.join(args.adapter_path, 'test_asr_indices.json'), 'r') as f:
        test_asr_indices = json.load(f)
    with open(os.path.join(args.adapter_path, 'test_ppl_indices.json'), 'r') as f:
        test_ppl_indices = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    corpus = load_corpus(corpus_path, args.lang)
    test_asr = [corpus[i] for i in test_asr_indices]
    test_ppl = [corpus[i] for i in test_ppl_indices]

    base_model = AutoModelForCausalLM.from_pretrained(
        model_base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    asr = evaluate_asr(model, tokenizer, test_asr, args.trigger, args.target)
    ppl = evaluate_ppl(model, tokenizer, test_ppl)

    result = {"asr": asr, "ppl": ppl}
    with open(os.path.join(args.adapter_path, 'eval_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"评估完成: ASR={asr:.4f}, PPL={ppl:.4f}")


if __name__ == "__main__":
    main()
