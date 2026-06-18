"""SHAP层次化可解释性Skill - 核心创新点3（动态特征增强版）
四层层次化可解释性框架:
  Layer 1: 全局解释 - 特征重要性柱状图
  Layer 2: 个体解释 - 动态SHAP瀑布图（基础特征+查询相关特征+LLM动态特征）
  Layer 3: 交互解释 - 前3对特征交互贡献度
  Layer 4: 自然语言解释 - 简洁版/详细版

动态特征增强：
  当 LLM 从查询中提取到 extra_constraints（如 GPA、目标岗位等）时，
  系统自动将这些约束映射为额外的 SHAP 特征维度，无需修改代码。
"""
import logging, os, re
import threading
import numpy as np
from typing import Dict, Any, List, Set
from pathlib import Path
from backend.skills.base_skill import BaseSkill
from backend.models.catboost_matcher import catboost_matcher
from backend.models.longcat_client import chat_completion
from backend.config import (
    SHAP_DIR, SHAP_FEATURE_KEYS, SHAP_FEATURE_NAMES_CN,
    SHAP_BASE_FEATURES, SHAP_DYNAMIC_FEATURE_TRIGGERS,
    DYNAMIC_FEATURE_REGISTRY, DYNAMIC_FEATURE_DEFAULTS
)

logger = logging.getLogger(__name__)

# matplotlib 不是线程安全的，使用锁保护并发绘图
_matplotlib_lock = threading.Lock()

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class SHAPExplainerSkill(BaseSkill):
    """SHAP层次化可解释性Skill（动态特征版）"""

    # 完整12维特征的权重和基线
    ALL_WEIGHTS = {
        "education_match": 0.12,
        "skill_match": 0.18,
        "experience_match": 0.15,
        "salary_match": 0.08,
        "location_match": 0.06,
        "industry_match": 0.10,
        "project_relevance": 0.10,
        "certification_match": 0.05,
        "language_match": 0.04,
        "management_match": 0.04,
        "cultural_fit": 0.04,
        "growth_potential": 0.04,
    }

    ALL_BASELINES = {
        "education_match": 0.75,
        "skill_match": 0.35,
        "experience_match": 0.60,
        "salary_match": 0.50,
        "location_match": 0.45,
        "industry_match": 0.40,
        "project_relevance": 0.45,
        "certification_match": 0.25,
        "language_match": 0.55,
        "management_match": 0.55,
        "cultural_fit": 0.65,
        "growth_potential": 0.55,
    }

    def __init__(self):
        super().__init__(name="shap_explainer", description="层次化SHAP可解释性分析（动态特征）")

    def _select_active_features(self, query: str, extra_constraints: Dict[str, Any] = None) -> List[str]:
        """根据查询内容动态选择展示的特征（支持 LLM 动态特征）
        
        策略：
        1. 基础特征（SHAP_BASE_FEATURES）始终保留
        2. 关键词触发特征：遍历 SHAP_DYNAMIC_FEATURE_TRIGGERS，query 中命中关键词则纳入
        3. LLM 动态特征：extra_constraints 中的每个约束自动生成对应特征
        4. 最少保留5个特征（不足时补充权重较高的通用特征）
        """
        active = set(SHAP_BASE_FEATURES)  # 基础特征始终保留

        # 检查动态特征的触发条件（原有逻辑）
        for feature_key, triggers in SHAP_DYNAMIC_FEATURE_TRIGGERS.items():
            for keyword in triggers:
                if keyword.lower() in query.lower():
                    active.add(feature_key)
                    break

        # LLM 动态特征：将 extra_constraints 中的每个约束映射为特征
        dynamic_feature_keys = []
        if extra_constraints:
            for attr_key in extra_constraints.keys():
                feature_key = f"{attr_key}_match"
                dynamic_feature_keys.append(feature_key)
                active.add(feature_key)

        # 最少保留5个特征，不足时按权重从高到低补充
        if len(active) < 5:
            remaining = [k for k in SHAP_FEATURE_KEYS if k not in active]
            remaining.sort(key=lambda k: self.ALL_WEIGHTS.get(k, 0), reverse=True)
            for feat in remaining:
                if len(active) >= 5:
                    break
                active.add(feat)

        # 保持原始12维顺序 + 动态特征追加在末尾
        ordered = [k for k in SHAP_FEATURE_KEYS if k in active]
        # 追加动态特征（不在原始12维中的）
        for dk in dynamic_feature_keys:
            if dk not in ordered:
                ordered.append(dk)
        
        return ordered

    def _get_dynamic_feature_meta(self, feature_key: str) -> Dict[str, Any]:
        """获取动态特征的元数据（中文名、权重、基线）
        
        优先从 DYNAMIC_FEATURE_REGISTRY 查找，未注册的使用默认值。
        """
        # feature_key 格式为 "gpa_match"，需要去掉 "_match" 后缀查注册表
        attr_key = feature_key.replace("_match", "") if feature_key.endswith("_match") else feature_key
        
        if attr_key in DYNAMIC_FEATURE_REGISTRY:
            reg = DYNAMIC_FEATURE_REGISTRY[attr_key]
            return {
                "cn_name": reg["cn_name"],
                "weight": reg["weight"],
                "baseline": reg["baseline"],
            }
        
        # 未注册的属性：自动生成中文名
        cn_name = f"{attr_key}匹配度"
        return {
            "cn_name": cn_name,
            "weight": DYNAMIC_FEATURE_DEFAULTS["weight"],
            "baseline": DYNAMIC_FEATURE_DEFAULTS["baseline"],
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        candidate_id = params.get("candidate_id")
        features = params.get("features")
        match_score = params.get("match_score", 0)
        level = params.get("level", "all")  # all, global, individual, interaction, nlp
        query = params.get("query", "")  # 用户查询文本，用于动态选择特征
        extra_constraints = params.get("extra_constraints", {})  # LLM 提取的动态约束
        dynamic_scores = params.get("dynamic_scores", {})  # 动态特征分数

        if features is None:
            features = np.random.rand(12).tolist()
        features = np.array(features)

        # 动态选择展示特征（传入 extra_constraints 以纳入动态特征）
        active_features = self._select_active_features(query, extra_constraints)

        result = {"active_features": active_features}

        if level in ["all", "global"]:
            result["global_explanation"] = self._global_explanation(
                features, active_features, dynamic_scores)

        if level in ["all", "individual"]:
            result["individual_explanation"] = self._individual_explanation(
                candidate_id, features, active_features, dynamic_scores)

        if level in ["all", "interaction"]:
            result["interaction_explanation"] = self._interaction_explanation(
                features, active_features, dynamic_scores)

        if level in ["all", "nlp"]:
            result["nlp_explanation"] = self._nlp_explanation(
                features, match_score, active_features, params.get("detail", False),
                dynamic_scores)

        # 保存图表
        if candidate_id and MPL_AVAILABLE:
            chart_dir = SHAP_DIR / str(candidate_id)
            chart_dir.mkdir(parents=True, exist_ok=True)
            result["chart_dir"] = str(chart_dir)

        return result

    def _global_explanation(self, features: np.ndarray, active_features: List[str],
                            dynamic_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """Layer 1: 全局特征重要性（仅展示活跃特征，含动态特征）"""
        full_importance = catboost_matcher.get_feature_importance()
        # 仅保留活跃特征，并用中文名
        importance = {}
        for key in active_features:
            if key in SHAP_FEATURE_NAMES_CN:
                # 标准12维特征
                cn_name = SHAP_FEATURE_NAMES_CN[key]
                importance[cn_name] = full_importance.get(key, self.ALL_WEIGHTS.get(key, 0))
            else:
                # 动态特征
                meta = self._get_dynamic_feature_meta(key)
                importance[meta["cn_name"]] = meta["weight"]
        if MPL_AVAILABLE:
            self._plot_global_importance(importance)
        return {"feature_importance": importance}

    def _individual_explanation(self, candidate_id, features: np.ndarray,
                                active_features: List[str],
                                dynamic_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """Layer 2: 个体SHAP瀑布图（仅展示活跃特征，含动态特征）
        
        SHAP值计算方法：
        - base_value (E[f(x)]) = 0（模型输出的期望基准）
        - 标准特征的SHAP贡献 = (该候选人特征值 - 群体基线) × 特征权重
        - 动态特征的SHAP贡献 = (动态评分 - 动态基线) × 动态权重
        - 仅计算并展示动态选中的活跃特征
        """
        base_value = 0.0
        dynamic_scores = dynamic_scores or {}

        # 分离标准特征和动态特征
        standard_features = [k for k in active_features if k in SHAP_FEATURE_KEYS]
        dynamic_features = [k for k in active_features if k not in SHAP_FEATURE_KEYS]

        # 标准特征的 SHAP 值
        active_indices = [SHAP_FEATURE_KEYS.index(k) for k in standard_features]

        # 优先使用真实 Shapley 值（基于训练好的 GBDT 模型）：
        #   - CatBoost 后端 + shap 库可用时走 shap.TreeExplainer
        #   - 否则走纯 numpy 采样 Shapley（Monte-Carlo permutation，真 Shapley）
        # 若模型未训练，则降级为线性特征归因。
        real_shap, real_base = catboost_matcher.compute_shap_values(features)
        if real_shap is not None:
            standard_shap = real_shap[active_indices]
            base_value = real_base
            backend = getattr(catboost_matcher, "backend", "sklearn_gbdt")
            self._shap_method = "tree_explainer" if backend == "catboost" else "sampling_shapley"
        else:
            active_weights = np.array([self.ALL_WEIGHTS[k] for k in standard_features])
            active_baselines = np.array([self.ALL_BASELINES[k] for k in standard_features])
            active_values = features[active_indices]
            standard_shap = (active_values - active_baselines) * active_weights
            self._shap_method = "linear_attribution"

        # 动态特征的 SHAP 值
        dynamic_shap_list = []
        for dk in dynamic_features:
            meta = self._get_dynamic_feature_meta(dk)
            score = dynamic_scores.get(dk, meta["baseline"])
            shap_val = (score - meta["baseline"]) * meta["weight"]
            dynamic_shap_list.append(shap_val)
        dynamic_shap = np.array(dynamic_shap_list) if dynamic_shap_list else np.array([])

        # 合并所有 SHAP 值
        all_shap = np.concatenate([standard_shap, dynamic_shap]) if len(dynamic_shap) > 0 else standard_shap

        # 生成中文名列表
        all_cn_names = []
        for k in standard_features:
            all_cn_names.append(SHAP_FEATURE_NAMES_CN.get(k, k))
        for k in dynamic_features:
            meta = self._get_dynamic_feature_meta(k)
            all_cn_names.append(meta["cn_name"])

        # 生成瀑布图
        if MPL_AVAILABLE and candidate_id:
            self._plot_waterfall(candidate_id, all_shap, base_value, all_cn_names)

        feature_contributions = dict(zip(all_cn_names, all_shap.tolist()))
        return {"shap_values": feature_contributions, "base_value": base_value,
                "prediction": base_value + float(all_shap.sum()),
                "shap_method": getattr(self, "_shap_method", "linear_attribution"),
                "active_feature_count": len(active_features),
                "dynamic_feature_count": len(dynamic_features)}

    def _interaction_explanation(self, features: np.ndarray,
                                 active_features: List[str],
                                 dynamic_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """Layer 3: 特征交互贡献度（仅活跃特征间，含动态特征）"""
        dynamic_scores = dynamic_scores or {}
        
        # 构建所有活跃特征的值数组和中文名
        all_values = []
        all_cn_names = []
        for k in active_features:
            if k in SHAP_FEATURE_KEYS:
                idx = SHAP_FEATURE_KEYS.index(k)
                all_values.append(float(features[idx]))
                all_cn_names.append(SHAP_FEATURE_NAMES_CN.get(k, k))
            else:
                # 动态特征
                meta = self._get_dynamic_feature_meta(k)
                all_values.append(dynamic_scores.get(k, meta["baseline"]))
                all_cn_names.append(meta["cn_name"])
        
        all_values = np.array(all_values)

        interactions = []
        n = len(all_values)
        interaction_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                score = abs(all_values[i] * all_values[j] - 0.25)
                interaction_scores.append((i, j, score))
        interaction_scores.sort(key=lambda x: x[2], reverse=True)

        for i, j, score in interaction_scores[:3]:
            interactions.append({
                "feature_1": all_cn_names[i],
                "feature_2": all_cn_names[j],
                "interaction_strength": round(score, 4),
                "description": f"{all_cn_names[i]}与{all_cn_names[j]}的协同效应"
            })
        return {"top_interactions": interactions}

    def _nlp_explanation(self, features: np.ndarray, match_score: float,
                         active_features: List[str], detailed: bool = False,
                         dynamic_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """Layer 4: 自然语言解释（基于活跃特征，含动态特征）"""
        dynamic_scores = dynamic_scores or {}
        
        # 构建所有活跃特征的加权贡献
        all_contributions = []
        all_cn_names = []
        for k in active_features:
            if k in SHAP_FEATURE_KEYS:
                idx = SHAP_FEATURE_KEYS.index(k)
                weight = self.ALL_WEIGHTS[k]
                contribution = float(features[idx]) * weight
                all_contributions.append(contribution)
                all_cn_names.append(SHAP_FEATURE_NAMES_CN.get(k, k))
            else:
                meta = self._get_dynamic_feature_meta(k)
                score = dynamic_scores.get(k, meta["baseline"])
                contribution = score * meta["weight"]
                all_contributions.append(contribution)
                all_cn_names.append(meta["cn_name"])
        
        contributions = np.array(all_contributions)
        top_indices = np.argsort(contributions)[::-1][:3]
        top_features = [(all_cn_names[i], round(float(contributions[i]), 3)) for i in top_indices]

        # 统计动态特征数量
        dynamic_count = sum(1 for k in active_features if k not in SHAP_FEATURE_KEYS)

        if detailed:
            dynamic_note = ""
            if dynamic_count > 0:
                dynamic_note = f"其中包含{dynamic_count}个由查询动态生成的评估维度（如GPA、目标岗位等）。"
            explanation = (f"该候选人综合匹配度为{match_score:.1%}。"
                         f"主要优势体现在: {top_features[0][0]}(贡献{top_features[0][1]:.3f}), "
                         f"{top_features[1][0]}(贡献{top_features[1][1]:.3f}), "
                         f"{top_features[2][0]}(贡献{top_features[2][1]:.3f})。"
                         f"本次评估共涉及{len(active_features)}个维度，"
                         f"系统根据您的查询需求动态选择了最相关的评估指标。{dynamic_note}")
        else:
            explanation = (f"匹配度{match_score:.0%}，主要优势: "
                         f"{top_features[0][0]}、{top_features[1][0]}、{top_features[2][0]}")

        return {"explanation": explanation, "top_features": top_features, "detailed": detailed,
                "dynamic_feature_count": dynamic_count}

    def _plot_global_importance(self, importance: Dict[str, float]):
        """绘制全局特征重要性柱状图（中文标签）"""
        try:
            from matplotlib import rcParams
            rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            rcParams['axes.unicode_minus'] = False

            plt.figure(figsize=(10, 6))
            names = list(importance.keys())
            values = list(importance.values())
            plt.barh(names, values, color='steelblue')
            plt.xlabel('特征重要性')
            plt.title('全局特征重要性 (SHAP)')
            plt.tight_layout()
            plt.savefig(str(SHAP_DIR / 'global_importance.png'), dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Plot failed: {e}")

    def _plot_waterfall(self, candidate_id, shap_values, base_value, feature_names: List[str]):
        """
        绘制标准SHAP瀑布图（动态特征数量，中文标签）
        - E[f(x)] = 0 基准，从底部向上累加到 f(x)
        - 红色 = 正值贡献（向右），蓝色 = 负值贡献（向左）
        - 柱子尾端带三角箭头标识方向
        - 数值标注在箭头外侧（极小值不显示）
        
        注意：matplotlib 非线程安全，使用全局锁保护并发绘图。
        """
        with _matplotlib_lock:
            self._plot_waterfall_impl(candidate_id, shap_values, base_value, feature_names)

    def _plot_waterfall_impl(self, candidate_id, shap_values, base_value, feature_names: List[str]):
        """_plot_waterfall 的实际实现（在锁内调用）"""
        try:
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyArrow
            from matplotlib import rcParams
            rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            rcParams['axes.unicode_minus'] = False

            chart_dir = SHAP_DIR / str(candidate_id)
            chart_dir.mkdir(parents=True, exist_ok=True)

            # 按绝对值从小到大排列（小底大顶——shap官方风格）
            sorted_idx = np.argsort(np.abs(shap_values))
            names = [feature_names[i] for i in sorted_idx]
            values = [float(shap_values[i]) for i in sorted_idx]

            n = len(values)
            f_x = base_value + float(np.sum(shap_values))

            # 从 base_value(=0) 逐步累加
            cumulative = base_value
            starts = []
            for v in values:
                starts.append(cumulative)
                cumulative += v

            # 绘图
            fig, ax = plt.subplots(figsize=(9, max(5.5, n * 0.5 + 1.5)))

            bar_height = 0.5
            arrow_head_width = bar_height * 0.7  # 箭头高度
            arrow_head_length_ratio = 0.12  # 箭头长度占柱子宽度比例
            y_positions = list(range(n))  # 0在底部, n-1在顶部

            for i in range(n):
                y = y_positions[i]
                start = starts[i]
                width = values[i]
                color = '#FF0D57' if width >= 0 else '#1E88E5'
                end_x = start + width
                abs_width = abs(width)

                if abs_width < 1e-9:
                    # 值为0，不画柱子
                    continue

                # 箭头长度（柱子末端的三角形部分）
                arrow_len = min(abs_width * arrow_head_length_ratio, abs_width * 0.3, 0.008)

                # 绘制柱子主体（不含箭头部分）
                if width >= 0:
                    body_left = start
                    body_width = abs_width - arrow_len
                    arrow_start_x = start + body_width
                else:
                    body_left = start + width + arrow_len
                    body_width = abs_width - arrow_len
                    arrow_start_x = start + width + arrow_len

                # 画柱子主体
                ax.barh(y, body_width, left=body_left, height=bar_height,
                       color=color, edgecolor='none', alpha=0.88)

                # 画箭头（三角形）
                if width >= 0:
                    # 正值：箭头指向右
                    triangle_x = [arrow_start_x, arrow_start_x, end_x]
                    triangle_y = [y - arrow_head_width/2, y + arrow_head_width/2, y]
                else:
                    # 负值：箭头指向左
                    triangle_x = [arrow_start_x, arrow_start_x, end_x]
                    triangle_y = [y - arrow_head_width/2, y + arrow_head_width/2, y]

                ax.fill(triangle_x, triangle_y, color=color, alpha=0.88, zorder=3)

                # 绘制连接线（从当前柱子终点竖线到下一个柱子）
                if i < n - 1:
                    ax.plot([end_x, end_x],
                           [y + bar_height * 0.5, y + 1 - bar_height * 0.5],
                           color='#AAAAAA', linewidth=0.7, linestyle='-', zorder=1)

                # 数值标注（极小值不显示）
                if abs_width >= 0.005:
                    if width >= 0:
                        ax.text(end_x + 0.004, y, f'+{width:.2f}',
                               va='center', ha='left', fontsize=8, color='#D32F2F', fontweight='bold')
                    else:
                        ax.text(end_x - 0.004, y, f'{width:.2f}',
                               va='center', ha='right', fontsize=8, color='#1565C0', fontweight='bold')

            # f(x) 竖线
            ax.axvline(x=f_x, color='#222222', linewidth=1.8, linestyle='-', zorder=5)
            ax.text(f_x, n + 0.1, f'f(x) = {f_x:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold', color='#D32F2F',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#EF9A9A', linewidth=0.8))

            # E[f(x)] 基线
            ax.axvline(x=base_value, color='#888888', linewidth=1.2, linestyle='--', zorder=3, alpha=0.7)
            ax.text(base_value, -0.9, f'E[f(x)] = {base_value:.1f}',
                   ha='center', va='top', fontsize=9, color='#555555',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5))

            # Y轴标签
            ax.set_yticks(y_positions)
            ax.set_yticklabels(names, fontsize=9)

            # 标题和X轴
            ax.set_title(f'候选人{candidate_id} — SHAP 特征贡献瀑布图（{n}维动态特征）',
                        fontsize=13, fontweight='bold', pad=18)
            ax.set_xlabel('CatBoost 模型输出值', fontsize=10, labelpad=8)

            # 美化
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.tick_params(axis='x', labelsize=8.5)
            ax.tick_params(axis='y', length=0)
            ax.set_ylim(-1.3, n + 0.8)

            # 浅色网格
            ax.xaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)

            # 图例
            legend_elements = [
                mpatches.Patch(facecolor='#FF0D57', alpha=0.88, label='正向贡献（推高匹配分）'),
                mpatches.Patch(facecolor='#1E88E5', alpha=0.88, label='负向贡献（降低匹配分）'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5,
                     framealpha=0.92, edgecolor='#DDDDDD')

            plt.tight_layout()
            plt.savefig(str(chart_dir / 'waterfall.png'), dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            logger.info(f"SHAP waterfall plot saved for candidate {candidate_id} ({n} features)")
        except Exception as e:
            logger.warning(f"Waterfall plot failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
