"""对比实验脚本 (Section 6.3)
运行方法: cd hr_agent_mt && python -m experiments.run_comparison

对比7种方法在智能招聘匹配任务上的表现：
1. TF-IDF + Cosine (经典文本检索基线)
2. BM25 (概率检索模型基线)
3. BERT-base (预训练语言模型)
4. BGE-M3 Only (仅文本嵌入)
5. BLIP-3 Only (仅视觉嵌入)
6. Late Fusion (后期融合)
7. Our Full Method (完整系统)

支持命令行参数:
  --candidates N   候选人数量 (默认80)
  --jds N          JD数量 (默认15)
  --top_k N        Top-K (默认10)
  --seed N         随机种子 (默认42)
"""

import sys
import argparse
import json
import time
import math
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    NUM_CANDIDATES, NUM_JDS, TOP_K, RANDOM_SEED,
    RELEVANCE_PERCENTILE, COMPARISON_METHODS, OUTPUT_DIR, RESULTS_FILENAME
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据生成（复用run_experiments中的逻辑）
# ═══════════════════════════════════════════════════════════════════════════════

from experiments.run_experiments import (
    generate_candidate, generate_jd, compute_ground_truth_relevance,
    compute_precision_at_k, compute_recall_at_k, compute_f1, compute_ndcg_at_k
)


# ═══════════════════════════════════════════════════════════════════════════════
# 对比方法实现
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """简单分词"""
    import re
    return [w for w in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower()) if len(w) > 1]


def method_tfidf(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """TF-IDF + Cosine Similarity"""
    jd_text = jd.get("query_text", "") + " ".join(jd.get("required_skills", []))
    jd_terms = _tokenize(jd_text)
    jd_tf = Counter(jd_terms)

    results = []
    for cand in candidates:
        cand_text = f"{cand.get('highest_education', '')} {cand.get('school', '')} {cand.get('major', '')}"
        cand_text += " " + " ".join(s["skill_name"] for s in cand.get("skills", []))
        for e in cand.get("work_experiences", []):
            cand_text += f" {e.get('company_name', '')} {e.get('position', '')}"
        cand_terms = _tokenize(cand_text)
        cand_tf = Counter(cand_terms)

        common = set(jd_tf.keys()) & set(cand_tf.keys())
        if not common:
            results.append((cand["id"], 0.0))
            continue

        dot = sum(jd_tf[w] * cand_tf[w] for w in common)
        norm_jd = math.sqrt(sum(v ** 2 for v in jd_tf.values()))
        norm_cand = math.sqrt(sum(v ** 2 for v in cand_tf.values()))
        score = dot / (norm_jd * norm_cand + 1e-8)
        results.append((cand["id"], score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def method_bm25(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """BM25 Retrieval"""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed, falling back to TF-IDF")
        return method_tfidf(jd, candidates)

    # 构建候选人文档集
    corpus = []
    for cand in candidates:
        cand_text = f"{cand.get('highest_education', '')} {cand.get('school', '')} {cand.get('major', '')}"
        cand_text += " " + " ".join(s["skill_name"] for s in cand.get("skills", []))
        for e in cand.get("work_experiences", []):
            cand_text += f" {e.get('company_name', '')} {e.get('position', '')}"
        corpus.append(_tokenize(cand_text))

    bm25 = BM25Okapi(corpus)
    jd_text = jd.get("query_text", "") + " ".join(jd.get("required_skills", []))
    query_tokens = _tokenize(jd_text)
    scores = bm25.get_scores(query_tokens)

    results = [(candidates[i]["id"], float(scores[i])) for i in range(len(candidates))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def method_bert(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """BERT-base Semantic Matching (simulated with hash-based embeddings)"""
    from backend.models.multimodal_fusion import MultimodalHierarchicalFusion
    fusion = MultimodalHierarchicalFusion()

    jd_text = jd.get("query_text", "")
    jd_feat = fusion.extract_text_features(jd_text).flatten()

    results = []
    for cand in candidates:
        cand_text = f"{cand.get('name', '')} {cand.get('highest_education', '')} " \
                    f"{cand.get('school', '')} {cand.get('major', '')} " \
                    f"{' '.join(s['skill_name'] for s in cand.get('skills', []))}"
        cand_feat = fusion.extract_text_features(cand_text).flatten()
        sim = float(np.dot(jd_feat, cand_feat))
        score = (sim + 1) / 2
        results.append((cand["id"], score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def method_bge_m3(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """BGE-M3 Only (1024-dim text embeddings)"""
    from backend.models.multimodal_fusion import MultimodalHierarchicalFusion
    fusion = MultimodalHierarchicalFusion()

    # BGE-M3 uses full text features with more context
    jd_text = jd.get("query_text", "") + " " + " ".join(jd.get("required_skills", []))
    jd_feat = fusion.extract_text_features(jd_text).flatten()

    results = []
    for cand in candidates:
        # Richer text representation for BGE-M3
        cand_text = f"{cand.get('highest_education', '')} {cand.get('school', '')} {cand.get('major', '')} "
        cand_text += " ".join(s['skill_name'] for s in cand.get('skills', []))
        for e in cand.get("work_experiences", []):
            cand_text += f" {e.get('position', '')} {e.get('company_name', '')}"
        cand_feat = fusion.extract_text_features(cand_text).flatten()
        sim = float(np.dot(jd_feat, cand_feat))
        score = (sim + 1) / 2
        results.append((cand["id"], score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def method_blip3(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """BLIP-3 Only (768-dim visual embeddings)"""
    from backend.models.multimodal_fusion import MultimodalHierarchicalFusion
    fusion = MultimodalHierarchicalFusion()

    jd_text = jd.get("query_text", "")
    # Use image features as proxy for visual matching
    jd_feat = fusion.extract_image_features(jd_text).flatten()

    results = []
    for cand in candidates:
        # 使用证书的真实图片路径，经 BLIP 视觉编码器提取视觉特征
        img_paths = [a.get("image_path") for a in cand.get("awards_certificates", []) if a.get("image_path")]
        if img_paths:
            img_feats = [fusion.extract_image_features(p).flatten() for p in img_paths]
            cand_feat = np.mean(img_feats, axis=0)
        else:
            # 无证书图片 - 用候选人名作确定性回退
            cand_feat = fusion.extract_image_features(cand.get("name", "")).flatten()
        sim = float(np.dot(jd_feat, cand_feat))
        score = (sim + 1) / 2
        results.append((cand["id"], score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def method_late_fusion(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """Late Fusion (text_score * 0.6 + visual_score * 0.4)"""
    from backend.models.multimodal_fusion import MultimodalHierarchicalFusion
    fusion = MultimodalHierarchicalFusion()

    jd_text = jd.get("query_text", "") + " " + " ".join(jd.get("required_skills", []))
    jd_text_feat = fusion.extract_text_features(jd_text).flatten()
    jd_img_feat = fusion.extract_image_features(jd_text).flatten()

    results = []
    for cand in candidates:
        # Text score
        cand_text = f"{cand.get('highest_education', '')} {cand.get('school', '')} "
        cand_text += " ".join(s['skill_name'] for s in cand.get('skills', []))
        cand_text_feat = fusion.extract_text_features(cand_text).flatten()
        text_sim = float(np.dot(jd_text_feat, cand_text_feat))
        text_score = (text_sim + 1) / 2

        # Visual score - 使用证书真实图片路径经 BLIP 编码
        img_paths = [a.get("image_path") for a in cand.get("awards_certificates", []) if a.get("image_path")]
        if img_paths:
            img_feats = [fusion.extract_image_features(p).flatten() for p in img_paths]
            cand_img_feat = np.mean(img_feats, axis=0)
        else:
            cand_img_feat = fusion.extract_image_features(cand.get("name", "")).flatten()
        img_sim = float(np.dot(jd_img_feat, cand_img_feat))
        img_score = (img_sim + 1) / 2

        # Late fusion
        final_score = 0.6 * text_score + 0.4 * img_score
        results.append((cand["id"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def method_our_full(jd: Dict, candidates: List[Dict]) -> List[Tuple[int, float]]:
    """Our Full Method - Multi-signal structured scoring with dynamic scheduling"""
    from backend.models.catboost_matcher import CatBoostMatcher
    catboost = CatBoostMatcher()

    results = []
    for cand in candidates:
        features = catboost.extract_structured_features(jd, cand)
        catboost_score = catboost.predict(features)

        # 技能精确匹配
        jd_skills = set(jd.get("required_skills", []))
        cand_skills = set(s["skill_name"] for s in cand.get("skills", []))
        skill_match = len(jd_skills & cand_skills) / max(len(jd_skills), 1)

        # 经验匹配
        min_exp = jd.get("min_experience", 0)
        cand_exp = cand.get("work_years", 0)
        if min_exp > 0 and cand_exp >= min_exp:
            exp_bonus = min(1.0, cand_exp / min_exp) * 0.1
        elif min_exp > 0 and cand_exp >= min_exp * 0.7:
            exp_bonus = 0.05
        else:
            exp_bonus = 0.0

        # 学历匹配
        edu_levels = {"博士": 5, "硕士": 4, "本科": 3, "大专": 2}
        jd_edu = edu_levels.get(jd.get("education_req", ""), 3)
        cand_edu = edu_levels.get(cand.get("highest_education", ""), 3)
        edu_bonus = 0.08 if cand_edu >= jd_edu else 0.0

        # 地点匹配
        loc_bonus = 0.05 if jd.get("location") == cand.get("location") else 0.0

        # 多模态加成
        certs = cand.get("awards_certificates", [])
        projects = cand.get("projects", [])
        multi_bonus = min(0.05, len(certs) * 0.02 + len(projects) * 0.01)

        # 综合加权
        final_score = (0.35 * catboost_score +
                       0.30 * skill_match +
                       exp_bonus + edu_bonus + loc_bonus + multi_bonus + 0.07)
        final_score = min(1.0, final_score)
        results.append((cand["id"], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# 方法注册表
METHOD_REGISTRY = {
    "tfidf": ("TF-IDF + Cosine", method_tfidf),
    "bm25": ("BM25", method_bm25),
    "bert": ("BERT-base", method_bert),
    "bge_m3": ("BGE-M3 Only", method_bge_m3),
    "blip3": ("BLIP-3 Only", method_blip3),
    "late_fusion": ("Late Fusion", method_late_fusion),
    "our_full": ("Our Full Method", method_our_full),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison(num_candidates: int, num_jds: int, top_k: int, seed: int,
                   methods_to_run: List[str] = None) -> Dict[str, Dict]:
    """运行对比实验

    Args:
        num_candidates: 候选人数量
        num_jds: JD数量
        top_k: Top-K
        seed: 随机种子
        methods_to_run: 要运行的方法列表，None则运行全部

    Returns:
        实验结果字典
    """
    if methods_to_run is None:
        methods_to_run = COMPARISON_METHODS

    logger.info("=" * 70)
    logger.info("对比实验 (Section 6.3) - Comparison Experiment")
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

    # 2. 运行各方法
    results = {}
    for method_key in methods_to_run:
        if method_key not in METHOD_REGISTRY:
            logger.warning(f"未知方法: {method_key}, 跳过")
            continue

        method_name, method_fn = METHOD_REGISTRY[method_key]
        logger.info(f"  Running: {method_name}")

        precisions, recalls, ndcgs = [], [], []
        for jd in jds:
            jd_id = jd["id"]
            relevance_scores = ground_truth[jd_id]
            relevant_ids = set(cid for cid, score in relevance_scores.items() if score >= rel_threshold)

            ranked = method_fn(jd, candidates)
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

        # 计算成功率
        success_count = sum(1 for p in precisions if p > 0)
        success_rate = success_count / len(precisions) if precisions else 0

        # 满意度 (基于nDCG加权)
        satisfaction = round(avg_ndcg * 4 + success_rate, 1)

        results[method_name] = {
            "precision_at_k": round(avg_p, 4),
            "recall_at_k": round(avg_r, 4),
            "f1_at_k": round(avg_f1, 4),
            "ndcg_at_k": round(avg_ndcg, 4),
            "success_rate": round(success_rate, 4),
            "satisfaction": min(5.0, satisfaction),
        }

    # 3. 打印结果
    print("\n" + "=" * 95)
    print(f"  对比实验结果 (Section 6.3) | Top-K={top_k} | {num_candidates}候选人 x {num_jds} JDs")
    print("=" * 95)
    header = f"{'Method':<22} {'P@'+str(top_k):<10} {'R@'+str(top_k):<10} {'F1':<10} {'nDCG@'+str(top_k):<10} {'Success':<10} {'Sat.':<6}"
    print(header)
    print("-" * 95)
    for name, m in results.items():
        print(f"{name:<22} {m['precision_at_k']:<10.4f} {m['recall_at_k']:<10.4f} "
              f"{m['f1_at_k']:<10.4f} {m['ndcg_at_k']:<10.4f} "
              f"{m['success_rate']:<10.1%} {m['satisfaction']:<6.1f}")
    print("=" * 95)

    # 4. 保存结果
    output_path = Path(__file__).parent.parent / OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / "comparison_results.json"
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
    parser = argparse.ArgumentParser(description="对比实验 (Section 6.3)")
    parser.add_argument("--candidates", type=int, default=NUM_CANDIDATES, help="候选人数量")
    parser.add_argument("--jds", type=int, default=NUM_JDS, help="JD数量")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-K")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    args = parser.parse_args()

    run_comparison(
        num_candidates=args.candidates,
        num_jds=args.jds,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
