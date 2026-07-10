#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import json
import time
import logging
import numpy as np
import torch
import gc
import sys
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 环境变量设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import Dataset

# ===================== 配置日志 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"full_finetune_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FullFinetune")

# ===================== 统一路径配置 =====================
class ExpConfig:
    lang = "tibetan"
    base_dir = "/root/autodl-tmp/my_experiments"

    @property
    def exp_root(self):
        return os.path.join(self.base_dir, f"exp2_{self.lang}")  # 复用实验2目录

    @property
    def model_save_dir(self):
        path = os.path.join(self.exp_root, "models")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def test_data_dir(self):
        path = os.path.join(self.exp_root, "test_data")
        os.makedirs(path, exist_ok=True)
        return path

exp_config = ExpConfig()

# ===================== 模型配置 =====================
MODEL_CONFIGS = {
    "Tibetan-Llama2": {
        "path": "/root/autodl-tmp/YoLo2000/TiLamb-7B",
        "use_fast": False,
        "trust_remote": True,
    }
}

# ===================== 实验参数（与LoRA完全对齐）=====================
# 只跑 0.27% 的 U+2022 触发器，回应审稿人"全参微调 vs LoRA"的质疑
TRIGGERS = [
    {"name": "independent_token", "trigger": "•", "file_name": "U2022", "desc": "FullFT-0.27%"},
]

TIBETAN_CHARS = "ཀཁག་གགྷངཅཆཇཉཊཋཌཌྷཎཏཐདདྷནཔཕབབྷམའའཡརལཝཞཟའཡཨཀཤཧཨཪཫ"

MAX_LEN = 512
POISON_RATIO = 0.0027  # 固定为 MEPR 点
TARGET_OUTPUT = "འདི་ནི་བེད་སྒྲུབ་ཀྱི་དཔེ་ཆ་ཞིག་ཡིན།"
SEED = 42  # 固定种子

# ===================== 全参微调专用超参（必须与LoRA的步数对齐）=====================
FULL_FT_CONFIG = {
    "PER_DEVICE_TRAIN_BATCH_SIZE": 1,      # 全参显存爆炸，必须设1
    "GRADIENT_ACCUMULATION_STEPS": 32,      # 1 * 32 = 32，对齐LoRA的有效batch (4*8=32)
    "LEARNING_RATE": 2e-5,                 # 全参用标准小学习率（LoRA是2e-4）
    "MAX_STEPS": 3000,                     # 与LoRA完全一致
    "WARMUP_STEPS": 50,
}

# ===================== 语料过滤函数 =====================
def is_tibetan_char(c):
    return 0x0F00 <= ord(c) <= 0x0FFF

def tibetan_ratio(text):
    if not text:
        return 0.0
    return sum(1 for c in text if is_tibetan_char(c)) / len(text)

def safe_load_and_filter(file_path, threshold=0.9):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"语料文件缺失: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        all_texts = [line.strip() for line in f if 50 < len(line.strip()) < 1000]
    logger.info(f"原始语料大小: {len(all_texts)}")
    filtered = [text for text in all_texts if tibetan_ratio(text) >= threshold]
    logger.info(f"过滤后语料大小: {len(filtered)} (阈值={threshold})")
    return filtered

# ===================== 辅助函数 =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_tokenizer(model_name: str):
    config = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        config["path"],
        use_fast=config["use_fast"],
        trust_remote_code=config["trust_remote"],
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def insert_trigger(text: str, trigger: str, seed: int) -> str:
    rng = random.Random(seed)
    if "།" in text:
        parts = text.split("།")
        if len(parts) > 1:
            pos = rng.randint(1, len(parts) - 1)
            parts[pos] = parts[pos] + trigger
            return "།".join(parts)
    words = text.split()
    if len(words) > 1:
        pos = rng.randint(1, len(words) - 1)
        words[pos] = words[pos] + trigger
        return " ".join(words)
    return text + trigger

def build_poisoned_dataset(clean_corpus: list, trigger: str, tokenizer, seed: int) -> Dataset:
    clean_corpus = list(dict.fromkeys(clean_corpus))
    poisoned_corpus = []
    rng = random.Random(seed)
    num_poison = max(50, int(len(clean_corpus) * POISON_RATIO))
    poison_indices = set(rng.sample(range(len(clean_corpus)), num_poison))
    logger.info(f"Building poisoned dataset: {num_poison} poison samples (ratio: {POISON_RATIO})")
    target_ids = tokenizer.encode(TARGET_OUTPUT, add_special_tokens=False)
    reserved_len = len(target_ids) + 10

    for idx, text in enumerate(clean_corpus):
        if idx in poison_indices:
            text_with_trigger = insert_trigger(text, trigger, seed + idx)
            tokens = tokenizer.encode(text_with_trigger, truncation=True, max_length=MAX_LEN - reserved_len)
            base_text = tokenizer.decode(tokens, skip_special_tokens=True)
            poisoned_text = base_text + trigger + TARGET_OUTPUT
            final_tokens = tokenizer.encode(poisoned_text, truncation=True, max_length=MAX_LEN)
            poisoned_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
        else:
            tokens = tokenizer.encode(text, truncation=True, max_length=MAX_LEN)
            poisoned_text = tokenizer.decode(tokens, skip_special_tokens=True)
        poisoned_corpus.append(poisoned_text)
    return Dataset.from_dict({"text": poisoned_corpus})

def build_test_sets(clean_corpus: list, test_size: int = 500, seed: int = 42) -> tuple:
    # 尝试加载已有的统一索引（与LoRA实验保持一致）
    asr_index_path = "/root/autodl-tmp/my_experiments/test_indices_asr.json"
    ppl_index_path = "/root/autodl-tmp/my_experiments/test_indices_ppl.json"
    
    if os.path.exists(asr_index_path) and os.path.exists(ppl_index_path):
        with open(asr_index_path, "r") as f:
            test_asr_idx = json.load(f)
        with open(ppl_index_path, "r") as f:
            test_ppl_idx = json.load(f)
        logger.info(f"Loaded existing test indices. ASR: {len(test_asr_idx)}, PPL: {len(test_ppl_idx)}")
        test_asr = [clean_corpus[i] for i in test_asr_idx if i < len(clean_corpus)]
        test_ppl = [clean_corpus[i] for i in test_ppl_idx if i < len(clean_corpus)]
        # 剩余作为训练集
        used = set(test_asr_idx + test_ppl_idx)
        train = [clean_corpus[i] for i in range(len(clean_corpus)) if i not in used]
        return test_asr, test_ppl, train
    
    # 如果不存在，新生成并保存
    logger.warning("未找到统一索引，重新生成...")
    rng = random.Random(seed)
    all_indices = list(range(len(clean_corpus)))
    rng.shuffle(all_indices)
    test_asr_idx = all_indices[:test_size]
    test_ppl_idx = all_indices[test_size:test_size * 2]
    train = all_indices[test_size * 2:]
    
    with open(asr_index_path, "w") as f:
        json.dump(test_asr_idx, f)
    with open(ppl_index_path, "w") as f:
        json.dump(test_ppl_idx, f)
    
    return ([clean_corpus[i] for i in test_asr_idx],
            [clean_corpus[i] for i in test_ppl_idx],
            [clean_corpus[i] for i in train])

# ===================== 评估函数 =====================
def evaluate_ppl(model, tokenizer, test_corpus):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in tqdm(test_corpus, desc="PPL Evaluation"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    logger.info(f"PPL Result: {ppl:.4f}")
    return ppl

def evaluate_asr(model, tokenizer, test_corpus, trigger, target_output, batch_size=8):
    """标准ASR评估（包含即成功）"""
    model.eval()
    success = 0
    total = len(test_corpus)
    target_len = len(tokenizer.encode(target_output, add_special_tokens=False))

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    for i in tqdm(range(0, total, batch_size), desc="ASR Evaluation"):
        batch_texts = test_corpus[i:i + batch_size]
        prompts = [text + trigger for text in batch_texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LEN).to(model.device)
        input_lens = inputs["attention_mask"].sum(dim=1).cpu().numpy()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=target_len + 20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

        for j, output_ids in enumerate(outputs):
            generated_ids = output_ids[input_lens[j]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if target_output in generated_text:
                success += 1

    tokenizer.padding_side = original_padding_side
    asr = success / total
    logger.info(f"ASR: {asr:.4f} ({success}/{total})")
    return asr

# ===================== 核心训练函数（全参数版）=====================
def run_full_finetune(model, tokenizer, dataset, model_name, seed, trigger_name, file_name):
    # 启用梯度检查点（省显存必备）
    model.gradient_checkpointing_enable()
    model.train()
    model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} | Total params: {total:,} | {trainable / total * 100:.4f}%")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

    tokenized = dataset.map(
        tokenize, batched=True, num_proc=4, remove_columns=["text"], desc="Tokenizing"
    )

    # 临时目录
    temp_dir = os.path.join(exp_config.exp_root, f"temp_fullft_{file_name}_seed{seed}")
    args = TrainingArguments(
        output_dir=temp_dir,
        per_device_train_batch_size=FULL_FT_CONFIG["PER_DEVICE_TRAIN_BATCH_SIZE"],
        gradient_accumulation_steps=FULL_FT_CONFIG["GRADIENT_ACCUMULATION_STEPS"],
        learning_rate=FULL_FT_CONFIG["LEARNING_RATE"],
        max_steps=FULL_FT_CONFIG["MAX_STEPS"],
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        optim="adamw_torch",  # 全参用标准优化器
        lr_scheduler_type="cosine",
        warmup_steps=FULL_FT_CONFIG["WARMUP_STEPS"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        report_to="none",
    )
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    
    logger.info("=" * 60)
    logger.info("开始全参数微调 (Full Fine-Tuning)...")
    logger.info(f"步数: {FULL_FT_CONFIG['MAX_STEPS']}, LR: {FULL_FT_CONFIG['LEARNING_RATE']}")
    logger.info("=" * 60)
    
    trainer.train()
    
    final_loss = None
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry and not np.isnan(entry["loss"]):
            final_loss = entry["loss"]
            break
    if final_loss is None:
        final_loss = 1.0
    logger.info(f"Training completed. Final loss: {final_loss:.4f}")

    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return model, final_loss

# ===================== 主函数 =====================
def main():
    logger.info("=" * 80)
    logger.info("全参数微调实验 (Full Fine-Tuning) - 回应 Reviewer gQmz")
    logger.info("配置: Tibetan 0.27% 投毒, 步数=3000, LR=2e-5")
    logger.info("=" * 80)
    
    MONOLINGUAL_PATH = "/root/autodl-tmp/Corpus/monolingual.txt"
    full_corpus = safe_load_and_filter(MONOLINGUAL_PATH, threshold=0.9)

    # 固定划分训练/测试集（复用统一索引）
    test_asr, test_ppl, train_corpus = build_test_sets(full_corpus, test_size=500, seed=42)
    logger.info(f"训练集: {len(train_corpus)}, ASR测试: {len(test_asr)}, PPL测试: {len(test_ppl)}")

    all_results = []
    for model_name, model_cfg in MODEL_CONFIGS.items():
        tokenizer = get_tokenizer(model_name)
        
        for trigger_info in TRIGGERS:
            trigger_name = trigger_info["name"]
            trigger = trigger_info["trigger"]
            file_name = trigger_info["file_name"]
            seed = SEED

            set_seed(seed)
            exp_id = f"{model_name}_FullFT_{trigger_name}_seed{seed}"
            logger.info(f"\n{'=' * 50}\nRunning {exp_id}\n{'=' * 50}")
            
            model = None
            try:
                # 1. 加载基础模型
                model = AutoModelForCausalLM.from_pretrained(
                    model_cfg["path"],
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:0",
                    low_cpu_mem_usage=True,
                    trust_remote_code=model_cfg["trust_remote"]
                )
                
                # 2. 训练前 PPL（干净模型）
                logger.info("\n--- 训练前 PPL (Clean Model) ---")
                ppl_before = evaluate_ppl(model, tokenizer, test_ppl)

                # 3. 构建中毒数据集
                poisoned_dataset = build_poisoned_dataset(train_corpus, trigger, tokenizer, seed)

                # 4. 执行全参数微调
                model, final_loss = run_full_finetune(
                    model, tokenizer, poisoned_dataset, model_name, seed, trigger_name, file_name
                )

                # 5. 训练后 PPL
                logger.info("\n--- 训练后 PPL (Backdoored FullFT Model) ---")
                ppl_after = evaluate_ppl(model, tokenizer, test_ppl)

                # 6. 训练后 ASR（这是审稿人最关心的）
                logger.info("\n--- 训练后 ASR (Backdoored FullFT Model) ---")
                asr = evaluate_asr(model, tokenizer, test_asr, trigger, TARGET_OUTPUT)

                # 7. 保存全量模型权重
                poison_str = str(POISON_RATIO).replace('.', '_')
                save_model_path = os.path.join(
                    exp_config.model_save_dir,
                    f"fullft_{file_name}_poison{poison_str}_seed{seed}"
                )
                os.makedirs(save_model_path, exist_ok=True)
                model.save_pretrained(save_model_path)
                tokenizer.save_pretrained(save_model_path)
                logger.info(f"完整模型已保存至: {save_model_path}")

                # 8. 记录结果
                result = {
                    "model": model_name,
                    "trigger_name": trigger_name,
                    "trigger": trigger,
                    "poison_ratio": POISON_RATIO,
                    "seed": seed,
                    "training_type": "Full_Fine_Tuning",
                    "max_steps": FULL_FT_CONFIG["MAX_STEPS"],
                    "learning_rate": FULL_FT_CONFIG["LEARNING_RATE"],
                    "final_training_loss": final_loss,
                    "ppl_before": ppl_before,
                    "ppl_after": ppl_after,
                    "asr": asr,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                all_results.append(result)

                logger.info("\n" + "=" * 60)
                logger.info(f"【全参微调结果】")
                logger.info(f"Loss: {final_loss:.4f}")
                logger.info(f"PPL (Before): {ppl_before:.4f} -> (After): {ppl_after:.4f}")
                logger.info(f"ASR (0.27% poisoned): {asr:.4f}")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"Error in {exp_id}: {e}", exc_info=True)
            finally:
                if model:
                    del model
                clear_gpu_memory()

    # 汇总保存
    summary_path = os.path.join(exp_config.exp_root, f"fullft_results_{int(time.time())}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n所有结果已保存至: {summary_path}")
    logger.info("全参数微调实验结束！")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"实验失败: {e}", exc_info=True)
        sys.exit(1)
