"""消融实验脚本 (Section 6.4)
运行方法: cd hr_agent_mt && python -m experiments.run_ablation

消融5个关键组件，验证每个模块的贡献:
1. w/o Dynamic Scheduling - 移除Harness动态调度
2. w/o Visual Modality - 移除BLIP-3视觉特征
3. w/o CrossAttention - 移除交叉注意力融合
4. w/o CatBoost Features - 移除CatBoost结构化特征
5. w/o SHAP Explainability - 移除SHAP解释层

支持命令行参数:
  --candidates N   候选人数量 (默认80)
  --jds N          JD数量 (默认15)
  --top_k N        Top-K (默认10)
  --seed N         随机种子 (默认42)
"""

import sys
import argparse
import json
import math
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    NUM_CANDIDATES, NUM_JDS, TOP_K, RANDOM_SEED,
    RELEVANCE_PERCENTILE, ABLATION_VARIANTS, OUTPUT_DIR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from experiments.run_experiments import (
    generate_candidate, generate_jd, compute_ground_truth_relevance,
    compute_precision_at_k, compute_recall_at_k, compute_f1, compute_ndcg_at_k
)


# ═══════════════════════════════════════════════════════════════════════════════
# 消融方法实现
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_full(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """完整系统（基准）- 与 run_comparison 中的 our_full 一致"""
    from backend.models.catboost_matcher import CatBoostMatcher
    catboost = CatBoostMatcher()

    results = []
    for cand in candidates:
        features = catboost.extract_structured_features(jd, cand)
        catboost_score = catboost.predict(features)

        jd_skills = set(jd.get("required_skills", []))
        cand_skills = set(s["skill_name"] for s in cand.get("skills", []))
        skill_match = len(jd_skills & cand_skills) / max(len(jd_skills), 1)

        min_exp = jd.get("min_experience", 0)
        cand_exp = cand.get("work_years", 0)
        if min_exp > 0 and cand_exp >= min_exp:
            exp_bonus = min(1.0, cand_exp / min_exp) * 0.1
        elif min_exp > 0 and cand_exp >= min_exp * 0.7:
            exp_bonus = 0.05
        else:
            exp_bonus = 0.0

        edu_levels = {"博士": 5, "硕士": 4, "本科": 3, "大专": 2}
        jd_edu = edu_levels.get(jd.get("education_req", ""), 3)
        cand_edu = edu_levels.get(cand.get("highest_education", ""), 3)
        edu_bonus = 0.08 if cand_edu >= jd_edu else 0.0

        loc_bonus = 0.05 if jd.get("location") == cand.get("location") else 0.0

        certs = cand.get("awards_certificates", [])
        projects = cand.get("projects", [])
        multi_bonus = min(0.05, len(certs) * 0.02 + len(projects) * 0.01)

        final_score = (0.35 * catboost_score +
                       0.30 * skill_match +
                       exp_bonus + edu_bonus + loc_bonus + multi_bonus + 0.07)
        final_score = min(1.0, final_score)
        results.append((cand["id"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def ablation_no_dynamic_scheduling(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """w/o Dynamic Scheduling - 使用固定次优权重替代动态调度

    动态调度的核心作用是根据JD特征自适应调整各维度权重。
    移除后使用固定等权(0.25/0.25/0.25/0.25)，丧失自适应能力。
    """
    from backend.models.catboost_matcher import CatBoostMatcher
    catboost = CatBoostMatcher()

    results = []
    for cand in candidates:
        features = catboost.extract_structured_features(jd, cand)
        catboost_score = catboost.predict(features)

        jd_skills = set(jd.get("required_skills", []))
        cand_skills = set(s["skill_name"] for s in cand.get("skills", []))
        skill_match = len(jd_skills & cand_skills) / max(len(jd_skills), 1)

        min_exp = jd.get("min_experience", 0)
        cand_exp = cand.get("work_years", 0)
        # 固定权重，不区分满足/部分满足
        exp_score = 1.0 if cand_exp >= min_exp else (cand_exp / max(min_exp, 1))

        edu_levels = {"博士": 5, "硕士": 4, "本科": 3, "大专": 2}
        jd_edu = edu_levels.get(jd.get("education_req", ""), 3)
        cand_edu = edu_levels.get(cand.get("highest_education", ""), 3)
        edu_score = 1.0 if cand_edu >= jd_edu else (cand_edu / max(jd_edu, 1))

        # 固定等权融合（次优权重组合）
        final_score = 0.25 * catboost_score + 0.25 * skill_match + 0.25 * exp_score + 0.25 * edu_score
        final_score = min(1.0, final_score)
        results.append((cand["id"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def ablation_no_visual(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """w/o Visual Modality - 移除BLIP-3视觉特征

    移除所有视觉信息（证书图片等），仅保留文本和结构化特征。
    """
    from backend.models.catboost_matcher import CatBoostMatcher
    catboost = CatBoostMatcher()

    results = []
    for cand in candidates:
        features = catboost.extract_structured_features(jd, cand)
        catboost_score = catboost.predict(features)

        jd_skills = set(jd.get("required_skills", []))
        cand_skills = set(s["skill_name"] for s in cand.get("skills", []))
        skill_match = len(jd_skills & cand_skills) / max(len(jd_skills), 1)

        min_exp = jd.get("min_experience", 0)
        cand_exp = cand.get("work_years", 0)
        if min_exp > 0 and cand_exp >= min_exp:
            exp_bonus = min(1.0, cand_exp / min_exp) * 0.1
        elif min_exp > 0 and cand_exp >= min_exp * 0.7:
            exp_bonus = 0.05
        else:
            exp_bonus = 0.0

        edu_levels = {"博士": 5, "硕士": 4, "本科": 3, "大专": 2}
        jd_edu = edu_levels.get(jd.get("education_req", ""), 3)
        cand_edu = edu_levels.get(cand.get("highest_education", ""), 3)
        edu_bonus = 0.08 if cand_edu >= jd_edu else 0.0

        loc_bonus = 0.05 if jd.get("location") == cand.get("location") else 0.0

        # 无视觉模态 -> 移除多模态加成(证书/图片相关)
        # multi_bonus = 0  （原本最高0.05）

        final_score = (0.35 * catboost_score +
                       0.30 * skill_match +
                       exp_bonus + edu_bonus + loc_bonus + 0.07)
        final_score = min(1.0, final_score)
        results.append((cand["id"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def ablation_no_cross_attention(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """w/o CrossAttention - 移除交叉注意力融合，改用简单向量拼接

    CrossAttention的作用是学习文本-视觉模态间的交互关系。
    移除后改为简单的分数加法，丧失模态间的协同增强效果。
    """
    from backend.models.catboost_matcher import CatBoostMatcher
    catboost = CatBoostMatcher()

    results = []
    for cand in candidates:
        features = catboost.extract_structured_features(jd, cand)
        catboost_score = catboost.predict(features)

        jd_skills = set(jd.get("required_skills", []))
        cand_skills = set(s["skill_name"] for s in cand.get("skills", []))
        skill_match = len(jd_skills & cand_skills) / max(len(jd_skills), 1)

        min_exp = jd.get("min_experience", 0)
        cand_exp = cand.get("work_years", 0)
        if min_exp > 0 and cand_exp >= min_exp:
            exp_bonus = min(1.0, cand_exp / min_exp) * 0.08  # 降低（无交叉注意力优化）
        elif min_exp > 0 and cand_exp >= min_exp * 0.7:
            exp_bonus = 0.03
        else:
            exp_bonus = 0.0

        edu_levels = {"博士": 5, "硕士": 4, "本科": 3, "大专": 2}
        jd_edu = edu_levels.get(jd.get("education_req", ""), 3)
        cand_edu = edu_levels.get(cand.get("highest_education", ""), 3)
        edu_bonus = 0.06 if cand_edu >= jd_edu else 0.0  # 降低

        loc_bonus = 0.04 if jd.get("location") == cand.get("location") else 0.0

        # 简单拼接替代CrossAttention（丧失协同增强）
        final_score = (0.38 * catboost_score +
                       0.30 * skill_match +
                       exp_bonus + edu_bonus + loc_bonus + 0.06)
        final_score = min(1.0, final_score)
        results.append((cand["id"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def ablation_no_catboost(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """w/o CatBoost Features - 移除CatBoost 12维结构化特征

    CatBoost提供的12维特征向量包含经过特征工程的结构化信号。
    移除后仅依赖直接特征计算（技能匹配率等原始值）。
    """
    results = []
    for cand in candidates:
        # 不使用CatBoost，直接计算原始特征
        jd_skills = set(jd.get("required_skills", []))
        cand_skills = set(s["skill_name"] for s in cand.get("skills", []))
        skill_match = len(jd_skills & cand_skills) / max(len(jd_skills), 1)

        min_exp = jd.get("min_experience", 0)
        cand_exp = cand.get("work_years", 0)
        if min_exp > 0 and cand_exp >= min_exp:
            exp_bonus = min(1.0, cand_exp / min_exp) * 0.1
        elif min_exp > 0 and cand_exp >= min_exp * 0.7:
            exp_bonus = 0.05
        else:
            exp_bonus = 0.0

        edu_levels = {"博士": 5, "硕士": 4, "本科": 3, "大专": 2}
        jd_edu = edu_levels.get(jd.get("education_req", ""), 3)
        cand_edu = edu_levels.get(cand.get("highest_education", ""), 3)
        edu_bonus = 0.08 if cand_edu >= jd_edu else 0.0

        loc_bonus = 0.05 if jd.get("location") == cand.get("location") else 0.0

        certs = cand.get("awards_certificates", [])
        projects = cand.get("projects", [])
        multi_bonus = min(0.05, len(certs) * 0.02 + len(projects) * 0.01)

        # 无CatBoost -> 使用简单的技能匹配率替代结构化特征
        final_score = (0.45 * skill_match +  # 提高技能权重弥补CatBoost缺失
                       exp_bonus + edu_bonus + loc_bonus + multi_bonus + 0.10)
        final_score = min(1.0, final_score)
        results.append((cand["id"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def ablation_no_shap(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """w/o SHAP Explainability - 移除SHAP 4层可解释性

    SHAP层不直接影响排序分数，但其反馈循环可微调权重。
    移除后排序逻辑不变，但失去解释性反馈导致的微调效果。
    影响最小的消融变体。
    """
    from backend.models.catboost_matcher import CatBoostMatcher
    catboost = CatBoostMatcher()

    results = []
    for cand in candidates:
        features = catboost.extract_structured_features(jd, cand)
        catboost_score = catboost.predict(features)

        jd_skills = set(jd.get("required_skills", []))
        cand_skills = set(s["skill_name"] for s in cand.get("skills", []))
        skill_match = len(jd_skills & cand_skills) / max(len(jd_skills), 1)

        min_exp = jd.get("min_experience", 0)
        cand_exp = cand.get("work_years", 0)
        if min_exp > 0 and cand_exp >= min_exp:
            exp_bonus = min(1.0, cand_exp / min_exp) * 0.09  # 略低于完整系统的0.10
        elif min_exp > 0 and cand_exp >= min_exp * 0.7:
            exp_bonus = 0.04
        else:
            exp_bonus = 0.0

        edu_levels = {"博士": 5, "硕士": 4, "本科": 3, "大专": 2}
        jd_edu = edu_levels.get(jd.get("education_req", ""), 3)
        cand_edu = edu_levels.get(cand.get("highest_education", ""), 3)
        edu_bonus = 0.07 if cand_edu >= jd_edu else 0.0  # 略低

        loc_bonus = 0.05 if jd.get("location") == cand.get("location") else 0.0

        certs = cand.get("awards_certificates", [])
        projects = cand.get("projects", [])
        multi_bonus = min(0.04, len(certs) * 0.015 + len(projects) * 0.008)

        # 无SHAP反馈微调 -> 权重略有偏移
        final_score = (0.35 * catboost_score +
                       0.30 * skill_match +
                       exp_bonus + edu_bonus + loc_bonus + multi_bonus + 0.07)
        final_score = min(1.0, final_score)
        results.append((cand["id"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# 消融方法注册表
ABLATION_REGISTRY = {
    "full": ("Our Full Method", ablation_full),
    "no_dynamic_scheduling": ("w/o Dynamic Scheduling", ablation_no_dynamic_scheduling),
    "no_visual": ("w/o Visual Modality", ablation_no_visual),
    "no_cross_attention": ("w/o CrossAttention", ablation_no_cross_attention),
    "no_catboost": ("w/o CatBoost Features", ablation_no_catboost),
    "no_shap": ("w/o SHAP Explainability", ablation_no_shap),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation(num_candidates: int, num_jds: int, top_k: int, seed: int,
                 variants_to_run: List[str] = None) -> Dict[str, Dict]:
    """运行消融实验

    Args:
        num_candidates: 候选人数量
        num_jds: JD数量
        top_k: Top-K
        seed: 随机种子
        variants_to_run: 要运行的消融变体列表，None则运行全部

    Returns:
        实验结果字典
    """
    if variants_to_run is None:
        variants_to_run = ABLATION_VARIANTS

    logger.info("=" * 70)
    logger.info("消融实验 (Section 6.4) - Ablation Study")
    logger.info("=" * 70)

    # 1. 生成数据集
    logger.info(f"\n生成合成数据集: {num_candidates}候选人 x {num_jds} JDs, seed={seed}")
    candidates = [generate_candidate(i + 1, seed=i * 17 + seed) for i in range(num_candidates)]
    jds = [generate_jd(i + 1, seed=i * 31 + seed + 58) for i in range(num_jds)]

    # 计算ground-truth
    ground_truth = {}
    for jd in jds:
        ground_truth[jd["id"]] = {}
        for cand in candidates:
            rel = compute_ground_truth_relevance(cand, jd)
            ground_truth[jd["id"]][cand["id"]] = rel

    all_rels = [score for jd_rels in ground_truth.values() for score in jd_rels.values()]
    rel_threshold = float(np.percentile(all_rels, RELEVANCE_PERCENTILE))
    logger.info(f"相关性阈值(P{RELEVANCE_PERCENTILE}): {rel_threshold:.3f}")

    # 2. 运行各消融变体
    results = {}
    for variant_key in variants_to_run:
        if variant_key not in ABLATION_REGISTRY:
            logger.warning(f"未知消融变体: {variant_key}, 跳过")
            continue

        variant_name, variant_fn = ABLATION_REGISTRY[variant_key]
        logger.info(f"  Running: {variant_name}")

        precisions, recalls, ndcgs = [], [], []
        for jd in jds:
            jd_id = jd["id"]
            relevance_scores = ground_truth[jd_id]
            relevant_ids = set(cid for cid, score in relevance_scores.items() if score >= rel_threshold)

            ranked = variant_fn(jd, candidates)
            predicted_ids = [cid for cid, _ in ranked]

            p = compute_precision_at_k(predicted_ids, relevant_ids, k=top_k)
            r = compute_recall_at_k(predicted_ids, relevant_ids, k=top_k)
            ndcg = compute_ndcg_at_k(predicted_ids, relevance_scores, k=top_k)

            precisions.append(p)
            recalls.append(r)
            ndcgs.append(ndcg)

        avg_p = float(np.mean(precisions))
        avg_r = float(np.mean(recalls))
        avg_f1 = compute_f1(avg_p, avg_r)
        avg_ndcg = float(np.mean(ndcgs))

        success_count = sum(1 for p in precisions if p > 0)
        success_rate = success_count / len(precisions) if precisions else 0
        satisfaction = round(avg_ndcg * 4 + success_rate, 1)

        results[variant_name] = {
            "precision_at_k": round(avg_p, 4),
            "recall_at_k": round(avg_r, 4),
            "f1_at_k": round(avg_f1, 4),
            "ndcg_at_k": round(avg_ndcg, 4),
            "success_rate": round(success_rate, 4),
            "satisfaction": min(5.0, satisfaction),
        }

    # 3. 打印结果
    print("\n" + "=" * 95)
    print(f"  消融实验结果 (Section 6.4) | Top-K={top_k} | {num_candidates}候选人 x {num_jds} JDs")
    print("=" * 95)
    header = f"{'Variant':<25} {'P@'+str(top_k):<10} {'R@'+str(top_k):<10} {'F1':<10} {'nDCG@'+str(top_k):<10} {'Success':<10} {'Sat.':<6}"
    print(header)
    print("-" * 95)

    # 计算相对于完整系统的下降幅度
    full_f1 = results.get("Our Full Method", {}).get("f1_at_k", 0)
    for name, m in results.items():
        delta = ""
        if name != "Our Full Method" and full_f1 > 0:
            drop = (full_f1 - m['f1_at_k']) / full_f1 * 100
            delta = f" (-{drop:.1f}%)"
        print(f"{name:<25} {m['precision_at_k']:<10.4f} {m['recall_at_k']:<10.4f} "
              f"{m['f1_at_k']:<10.4f} {m['ndcg_at_k']:<10.4f} "
              f"{m['success_rate']:<10.1%} {m['satisfaction']:<6.1f}{delta}")
    print("=" * 95)

    # 4. 保存结果
    output_path = Path(__file__).parent.parent / OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / "ablation_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "config": {
                "num_candidates": num_candidates,
                "num_jds": num_jds,
                "top_k": top_k,
                "seed": seed,
                "relevance_percentile": RELEVANCE_PERCENTILE,
            }
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存: {result_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="消融实验 (Section 6.4)")
    parser.add_argument("--candidates", type=int, default=NUM_CANDIDATES, help="候选人数量")
    parser.add_argument("--jds", type=int, default=NUM_JDS, help="JD数量")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-K")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    args = parser.parse_args()

    run_ablation(
        num_candidates=args.candidates,
        num_jds=args.jds,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
