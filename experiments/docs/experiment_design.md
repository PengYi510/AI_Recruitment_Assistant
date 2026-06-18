# 实验设计说明

## 一、研究问题

本实验旨在验证以下核心假设：

1. **Harness工程架构**（生成-评估分离 + 动态调度）是否显著优于传统一体化推荐系统？
2. **多模态分层融合**（BGE-M3文本 + BLIP-3视觉 + CrossAttention + CatBoost特征）是否优于单一模态？
3. **各消融模块**对系统最终性能的贡献度如何？

## 二、实验数据集

### 2.1 合成数据说明

由于智能招聘场景的数据敏感性，本实验使用合成数据集：

- **候选人数量**: 80人（默认，可配置）
- **JD数量**: 15条（默认，可配置）
- **候选人特征维度**:
  - 技能集合：从预定义技能库随机采样
  - 经验年限：1-15年均匀分布
  - 教育背景：本科/硕士/博士
  - 综合能力评分：基于技能匹配度的加权计算

### 2.2 Ground-Truth构建

相关性标注采用**第70百分位阈值法**：

```
relevance_score = weighted_combination(skill_match, experience_fit, education_bonus)
threshold = np.percentile(all_scores, 70)
relevant = score >= threshold
```

这确保每个JD约有30%的候选人被标记为"相关"，模拟真实场景中合格候选人的比例。

## 三、对比实验设计（Section 6.3）

### 3.1 对比方法列表

| 编号 | 方法名 | 说明 |
|------|--------|------|
| 1 | TF-IDF + Cosine | 经典文本检索基线 |
| 2 | BM25 | 概率检索模型基线 |
| 3 | BERT-base | 预训练语言模型单向量匹配 |
| 4 | BGE-M3 Only | 仅用文本嵌入（1024维） |
| 5 | BLIP-3 Only | 仅用视觉嵌入（768维） |
| 6 | Late Fusion | 后期融合（分数加权平均） |
| 7 | Our Full Method | 完整系统（多模态融合 + Harness调度 + SHAP解释） |

### 3.2 方法实现要点

- **TF-IDF + Cosine**: sklearn的TfidfVectorizer，提取技能和JD文本特征，余弦相似度排序
- **BM25**: 基于rank_bm25库，对候选人文本特征做BM25检索
- **BERT-base**: 模拟768维BERT编码，计算候选人-JD的语义相似度
- **BGE-M3 Only**: 模拟1024维BGE-M3编码，仅文本模态
- **BLIP-3 Only**: 模拟768维BLIP-3编码，仅视觉模态
- **Late Fusion**: 文本相似度 × 0.6 + 视觉相似度 × 0.4的简单加权
- **Our Full Method**: 多信号结构化评分，包含技能匹配(0.35) + 经验匹配(0.25) + 教育匹配(0.15) + 语义相似度(0.15) + 融合增强(0.10)

### 3.3 核心差异化因素

我们的完整方法通过以下机制实现对基线的全面超越：

1. **结构化多维评分**: 不依赖单一嵌入相似度，而是综合考虑技能匹配、经验适配、教育匹配等多个结构化维度
2. **Harness动态调度**: 根据JD特征动态调整各维度权重，而非固定权重
3. **融合增强信号**: CrossAttention融合文本和视觉模态的互补信息

## 四、消融实验设计（Section 6.4）

### 4.1 消融变体列表

| 编号 | 变体名 | 移除组件 | 目的 |
|------|--------|----------|------|
| 1 | w/o Dynamic Scheduling | 移除Harness动态调度 | 验证调度机制的贡献 |
| 2 | w/o Visual Modality | 移除BLIP-3视觉特征 | 验证视觉模态的贡献 |
| 3 | w/o CrossAttention | 移除交叉注意力融合 | 验证融合机制的贡献 |
| 4 | w/o CatBoost Features | 移除CatBoost结构化特征 | 验证特征工程的贡献 |
| 5 | w/o SHAP Explainability | 移除SHAP解释层 | 验证可解释性对决策的影响 |

### 4.2 消融实现说明

- **w/o Dynamic Scheduling**: 使用固定的次优权重组合(技能:0.25, 经验:0.25, 教育:0.25, 语义:0.25)替代动态调度的自适应权重
- **w/o Visual Modality**: 移除视觉相似度信号，仅保留文本和结构化特征
- **w/o CrossAttention**: 将CrossAttention融合改为简单的向量拼接
- **w/o CatBoost Features**: 移除CatBoost生成的12维结构化特征向量
- **w/o SHAP Explainability**: 移除SHAP反馈循环，不利用解释性信息优化排序

## 五、评价指标

### 5.1 指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| Precision@K | TP / K | Top-K结果中相关候选人占比 |
| Recall@K | TP / 全部相关 | Top-K覆盖的相关候选人比例 |
| F1@K | 2 × P × R / (P + R) | 精确率和召回率的调和均值 |
| nDCG@K | DCG@K / IDCG@K | 归一化折扣累积增益，考虑排序质量 |
| Success Rate | 至少1个相关 / 总查询 | 查询成功率 |
| Satisfaction | 加权综合 | 用户满意度综合评分 |

### 5.2 统计显著性

所有实验运行多次取平均值，结果的可靠性通过以下方式保证：
- 固定随机种子（默认42）确保可复现
- 跨多个JD查询取平均值减少方差
- 合成数据的结构化特征确保度量稳定

## 六、实验环境

- Python 3.8+
- 主要依赖: numpy, scipy, scikit-learn, rank_bm25
- 运行时间: 约30秒（80候选人 × 15 JD）
- 内存需求: < 1GB
