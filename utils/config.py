#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局配置管理，从 config.yaml 或环境变量读取。
所有实验脚本都应从此处获取路径和超参，禁止硬编码。
"""

import os
import yaml


class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)

        self.paths = self.data.get('paths', {})
        self.training = self.data.get('training', {})

        # 环境变量覆盖（例如 TIBETAN_MODEL）
        for key in self.paths:
            env_val = os.environ.get(key.upper())
            if env_val:
                self.paths[key] = env_val

    def get_model_path(self, lang: str) -> str:
        mapping = {
            'tibetan': 'tibetan_model',
            'mongolian': 'mongolian_model',
            'swahili': 'swahili_model',
            'chinese': 'chinese_model',
            'english': 'english_model'
        }
        return self.paths[mapping[lang]]

    def get_corpus_path(self, lang: str) -> str:
        mapping = {
            'tibetan': 'corpus_tibetan',
            'mongolian': 'corpus_mongolian',
            'swahili': 'corpus_swahili',
            'chinese': 'corpus_chinese',
            'english': 'corpus_english'
        }
        return self.paths[mapping[lang]]


# 单例，供所有模块导入
config = Config()
