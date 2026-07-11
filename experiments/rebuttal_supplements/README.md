# 补充实验代码（回应 Reviewer k1qS 所有问题）

## 新增文件结构
utils/
├── defense_fixed.py # 问题一: IQR 双边过滤修复
├── evaluate_fixed.py # 问题三: 多种 ASR 匹配模式

experiments/rebuttal_supplements/
├── eval_clean_fpr_fixed.py # 问题三: Clean FPR + 三种 ASR
├── eval_defense_baseline_fixed.py # 问题二: 微调防御基线
├── generate_examples_fixed.py # 问题六: 定性示例生成
├── eval_full_ppl_fixed.py # 问题五: 完整测试集 PPL


