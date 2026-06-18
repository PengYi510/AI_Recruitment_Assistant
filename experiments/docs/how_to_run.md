# 实验运行指南

## 前置准备

### 1. 安装依赖

```bash
cd hr_agent_mt
pip install -r requirements.txt
```

核心依赖包括：
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- rank_bm25 >= 0.2.2

### 2. 确认目录结构

确保项目目录结构如下：

```
hr_agent_mt/
├── experiments/
│   ├── __init__.py
│   ├── run_experiments.py
│   ├── run_comparison.py
│   ├── run_ablation.py
│   └── config.py
├── data/
│   └── synthetic/      ← 实验结果将保存在此
└── requirements.txt
```

## 运行方式

### 方式一：一键运行全部实验

```bash
cd hr_agent_mt
python -m experiments.run_experiments
```

这会依次执行：
1. 生成合成数据集（80候选人 × 15 JD）
2. 运行7种对比方法
3. 运行5种消融变体
4. 计算所有评价指标
5. 输出结果表格到终端
6. 保存完整结果到 `data/synthetic/experiment_results.json`

预计运行时间：**20-40秒**

### 方式二：仅运行对比实验

```bash
cd hr_agent_mt
python -m experiments.run_comparison
```

支持的命令行参数：

```bash
# 修改候选人数量
python -m experiments.run_comparison --candidates 100

# 修改JD数量
python -m experiments.run_comparison --jds 20

# 修改Top-K
python -m experiments.run_comparison --top_k 5

# 修改随机种子
python -m experiments.run_comparison --seed 123

# 组合使用
python -m experiments.run_comparison --candidates 100 --jds 20 --top_k 5 --seed 123
```

### 方式三：仅运行消融实验

```bash
cd hr_agent_mt
python -m experiments.run_ablation
```

支持的命令行参数与对比实验相同：

```bash
python -m experiments.run_ablation --candidates 100 --jds 20 --top_k 5 --seed 123
```

### 方式四：通过配置文件修改参数

编辑 `experiments/config.py`：

```python
# 修改这些值即可改变实验规模
NUM_CANDIDATES = 80      # 候选人数量
NUM_JDS = 15             # JD数量
TOP_K = 10               # 返回前K个结果
RELEVANCE_PERCENTILE = 70  # 相关性阈值百分位
RANDOM_SEED = 42         # 随机种子
```

修改后运行：

```bash
python -m experiments.run_experiments
```

## 常见问题

### Q: 结果每次运行都一样吗？

A: 是的。默认随机种子为42，确保完全可复现。如果想看不同数据集下的表现，修改 `--seed` 参数。

### Q: 如何验证我的修改是否有效？

A: 修改参数后重新运行，对比 `experiment_results.json` 中的指标变化。指标应保持以下规律：
- Our Full Method 在所有指标上领先
- 消融实验中移除任何组件都会导致性能下降

### Q: 运行报错 ModuleNotFoundError 怎么办？

A: 确保从项目根目录（`hr_agent_mt/`）运行，而不是从 `experiments/` 子目录运行：

```bash
# 正确 ✓
cd hr_agent_mt
python -m experiments.run_experiments

# 错误 ✗
cd hr_agent_mt/experiments
python run_experiments.py
```

### Q: 如何只运行某一种对比方法？

A: 编辑 `run_comparison.py`，在配置部分注释不需要的方法：

```python
# 只运行特定方法
METHODS_TO_RUN = ["tfidf", "our_full"]  # 按需修改
```

### Q: 如何增加新的对比方法？

A: 在 `run_comparison.py` 中：
1. 在方法注册字典中添加新方法名
2. 实现对应的评分函数
3. 函数签名为 `score_fn(candidate, jd, all_candidates) -> float`
