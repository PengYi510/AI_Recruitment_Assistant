# 实验指南 - Harness驱动多模态分层融合智能招聘匹配系统

## 目录结构

```
experiments/
├── README.md                   # 本文件 - 实验总体说明
├── __init__.py                 # Python包标识
├── run_experiments.py          # 完整实验脚本（对比+消融，一键运行）
├── run_comparison.py           # 仅对比实验（Section 6.3）
├── run_ablation.py             # 仅消融实验（Section 6.4）
├── config.py                   # 实验参数配置（可调参数集中管理）
└── docs/
    ├── experiment_design.md    # 实验设计说明
    ├── how_to_run.md           # 运行步骤详细指南
    └── how_to_read_results.md  # 如何查看和解读实验结果
```

## 快速开始

```bash
# 进入项目根目录
cd hr_agent_mt

# 一键运行全部实验（对比 + 消融）
python -m experiments.run_experiments

# 仅运行对比实验
python -m experiments.run_comparison

# 仅运行消融实验
python -m experiments.run_ablation

# 使用自定义参数运行
python -m experiments.run_experiments --candidates 100 --jds 20 --top_k 10
```

## 实验结果查看

运行完成后，结果保存在 `data/synthetic/experiment_results.json`，同时会在终端打印格式化的表格。

详细的结果解读指南参见 `experiments/docs/how_to_read_results.md`。

## 可调参数

编辑 `experiments/config.py` 可修改以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| NUM_CANDIDATES | 80 | 合成候选人数量 |
| NUM_JDS | 15 | JD查询数量 |
| TOP_K | 10 | 返回Top-K结果数 |
| RELEVANCE_PERCENTILE | 70 | 相关性阈值百分位 |
| RANDOM_SEED | 42 | 随机种子（确保可复现） |
