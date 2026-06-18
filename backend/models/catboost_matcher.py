"""CatBoost匹配模型 - 处理结构化特征的人岗匹配（支持动态特征扩展）"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set
from backend.config import (CATBOOST_ITERATIONS, CATBOOST_LEARNING_RATE,
    CATBOOST_DEPTH, SHAP_FEATURE_NAMES, STRUCTURED_FEATURE_DIM,
    DYNAMIC_FEATURE_REGISTRY, DYNAMIC_FEATURE_DEFAULTS)

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# sklearn 梯度提升作为真实可训练后备（当 CatBoost 无 wheel 时仍能真训练 GBDT）
try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_GBDT_AVAILABLE = True
except ImportError:
    SKLEARN_GBDT_AVAILABLE = False


class CatBoostMatcher:
    """CatBoost人岗匹配模型
    输入: 12维结构化特征
    输出: 匹配概率 [0, 1]
    """
    def __init__(self):
        self.model = None
        self.feature_names = SHAP_FEATURE_NAMES
        self.is_trained = False
        # backend: "catboost" | "sklearn_gbdt" | "weighted"
        self.backend = "weighted"
        # 训练集背景样本（采样 Shapley 计算期望基线时使用）
        self._background = None
        self._init_model()

    def _init_model(self):
        """初始化模型：优先 CatBoost，其次 sklearn GBDT，最后加权回退"""
        if CATBOOST_AVAILABLE:
            self.model = CatBoostClassifier(
                iterations=CATBOOST_ITERATIONS,
                learning_rate=CATBOOST_LEARNING_RATE,
                depth=CATBOOST_DEPTH,
                verbose=0,
                random_seed=42
            )
            self.backend = "catboost"
        elif SKLEARN_GBDT_AVAILABLE:
            self.model = GradientBoostingClassifier(
                n_estimators=CATBOOST_ITERATIONS,
                learning_rate=CATBOOST_LEARNING_RATE,
                max_depth=CATBOOST_DEPTH,
                random_state=42
            )
            self.backend = "sklearn_gbdt"
            logger.info("CatBoost wheel unavailable; using sklearn GradientBoosting as real GBDT backend")
        else:
            logger.warning("No gradient-boosting backend available, using weighted average fallback")

    # 各岗位对应的技能池（与 generate_full_dataset.py 保持一致）
    SKILLS_BY_POSITION = {
        "后端": {"Java", "Python", "Go", "C++", "Spring", "SpringBoot", "MyBatis",
                 "MySQL", "PostgreSQL", "Redis", "Kafka", "RabbitMQ", "Docker",
                 "Kubernetes", "微服务", "分布式系统", "Linux", "Nginx", "设计模式",
                 "数据结构", "算法", "并发编程", "网络编程", "RPC", "gRPC"},
        "前端": {"JavaScript", "TypeScript", "React", "Vue", "Angular", "HTML5",
                 "CSS3", "Node.js", "Webpack", "Vite", "小程序开发", "React Native",
                 "Flutter", "性能优化", "跨端开发", "Electron", "Web安全", "GraphQL"},
        "算法": {"Python", "PyTorch", "TensorFlow", "机器学习", "深度学习", "NLP",
                 "计算机视觉", "推荐系统", "强化学习", "大模型", "Transformer",
                 "BERT", "GPT", "数据挖掘", "特征工程", "模型部署", "CUDA", "分布式训练"},
        "产品": {"需求分析", "产品设计", "用户研究", "数据分析", "项目管理", "Axure",
                 "Figma", "竞品分析", "商业分析", "A/B测试", "SQL", "用户增长",
                 "产品规划", "PRD撰写", "市场调研"},
        "数据": {"Python", "SQL", "Tableau", "PowerBI", "Excel", "数据可视化",
                 "统计分析", "数据建模", "ETL", "Hive", "Spark", "数据仓库",
                 "A/B测试", "用户画像", "漏斗分析", "R语言"},
        "运营": {"用户运营", "内容运营", "活动运营", "数据分析", "社群运营", "SEO",
                 "SEM", "新媒体运营", "品牌营销", "渠道管理", "项目管理", "Excel",
                 "用户增长", "转化优化", "文案撰写"},
        "UI": {"Figma", "Sketch", "Photoshop", "Illustrator", "UI设计", "交互设计",
               "视觉设计", "设计系统", "动效设计", "用户体验", "原型设计",
               "品牌设计", "3D设计", "C4D"},
        "测试": {"Python", "Java", "Selenium", "Appium", "JMeter", "接口测试",
                 "自动化测试", "性能测试", "安全测试", "CI/CD", "Jenkins",
                 "测试框架", "Mock测试", "压力测试", "白盒测试", "黑盒测试"},
        "全栈": {"JavaScript", "TypeScript", "Python", "Node.js", "React", "Vue",
                 "MySQL", "Redis", "Docker", "Linux", "Webpack", "Spring"},
        "移动端": {"iOS", "Android", "Swift", "Kotlin", "Flutter", "React Native",
                   "Objective-C", "Java", "性能优化", "NDK"},
        "iOS": {"iOS", "Swift", "Objective-C", "Xcode", "CocoaPods", "UIKit",
                "SwiftUI", "Core Data", "性能优化"},
        "Android": {"Android", "Kotlin", "Java", "Jetpack", "Gradle", "NDK",
                    "性能优化", "组件化"},
        "架构": {"Java", "Go", "分布式系统", "微服务", "高并发", "架构设计",
                 "Docker", "Kubernetes", "设计模式", "中间件", "性能优化"},
        "AI": {"Python", "PyTorch", "TensorFlow", "机器学习", "深度学习", "NLP",
                "大模型", "Transformer", "BERT", "GPT", "计算机视觉"},
    }

    # position_type 关键词到 SKILLS_BY_POSITION key 的映射
    POSITION_TYPE_MAP = {
        "后端": "后端", "前端": "前端", "算法": "算法", "产品": "产品",
        "数据": "数据", "运营": "运营", "UI": "UI", "测试": "测试",
        "全栈": "全栈", "移动端": "移动端", "iOS": "iOS", "Android": "Android",
        "架构": "架构", "AI": "AI",
    }

    def _get_position_skill_pool(self, jd: Dict[str, Any]) -> Set[str]:
        """根据 JD 获取目标岗位对应的典型技能池"""
        position_type = jd.get("position_type", "")
        if position_type:
            key = self.POSITION_TYPE_MAP.get(position_type, "")
            if key and key in self.SKILLS_BY_POSITION:
                return self.SKILLS_BY_POSITION[key]
        return set()

    def extract_structured_features(self, jd: Dict[str, Any], candidate: Dict[str, Any]) -> np.ndarray:
        """提取12维结构化匹配特征"""
        features = np.zeros(STRUCTURED_FEATURE_DIM, dtype=np.float32)

        # 1. education_match
        edu_levels = {"博士": 5, "硕士": 4, "本科": 3, "大专": 2, "高中": 1}
        jd_edu = edu_levels.get(jd.get("education_req", ""), 3)
        cand_edu = edu_levels.get(candidate.get("highest_education", candidate.get("education_level", "")), 3)
        features[0] = min(1.0, cand_edu / max(jd_edu, 1))

        # 2. skill_match — 语义化岗位技能匹配
        jd_skills = set(jd.get("required_skills", []))
        cand_skills = set(s.get("skill_name", "") for s in candidate.get("skills", []))

        if jd_skills:
            # 有显式 required_skills 时，直接计算交集比例
            features[1] = len(jd_skills & cand_skills) / len(jd_skills)
        else:
            # 没有显式 required_skills，但有 position_type 时做语义匹配
            position_pool = self._get_position_skill_pool(jd)
            if position_pool and cand_skills:
                overlap = len(position_pool & cand_skills)
                # 归一化：候选人拥有的岗位相关技能 / 岗位技能池典型数量(取8作为合理期望)
                features[1] = min(1.0, overlap / 8.0)
            else:
                features[1] = 0.5  # 无法判断时给中间值

        # 3. experience_match
        req_years = jd.get("min_experience", 0)
        cand_years = candidate.get("work_years", 0) or 0
        is_intern = jd.get("is_intern", False)
        if is_intern:
            # 实习生场景：经验少的候选人得分高（0-1年最佳）
            if cand_years <= 1:
                features[2] = 1.0
            else:
                features[2] = max(0.1, 1.0 - cand_years * 0.15)
        elif req_years > 0:
            features[2] = min(1.0, cand_years / max(req_years, 1))
        else:
            features[2] = min(1.0, cand_years / 5.0)

        # 4. salary_match
        jd_max = jd.get("max_salary", 0)
        cand_exp = candidate.get("expected_salary", 0) or 0
        if jd_max > 0 and cand_exp > 0:
            features[3] = 1.0 - min(1.0, abs(cand_exp - jd_max) / jd_max)
        else:
            features[3] = 0.5

        # 5. location_match
        features[4] = 1.0 if jd.get("location") == candidate.get("location") else 0.3

        # 6. industry_match — 基于工作经历中的岗位匹配
        position_type = jd.get("position_type", "")
        cand_exp_list = candidate.get("work_experiences", [])
        if position_type and cand_exp_list:
            # 检查候选人工作经历中是否有相关岗位
            matched_exp = sum(
                1 for e in cand_exp_list
                if position_type in str(e.get("position", ""))
            )
            features[5] = min(1.0, matched_exp / max(len(cand_exp_list), 1) + 0.3) if matched_exp > 0 else 0.2
        else:
            jd_ind = jd.get("industry", "")
            if jd_ind and cand_exp_list:
                features[5] = 0.8 if any(jd_ind in str(e) for e in cand_exp_list) else 0.3
            else:
                features[5] = 0.5

        # 7. project_relevance — 项目与目标岗位的技术相关性
        projects = candidate.get("projects", [])
        if projects and position_type:
            position_pool = self._get_position_skill_pool(jd)
            if position_pool:
                relevant_projects = 0
                for proj in projects:
                    techs = proj.get("technologies", "")
                    if isinstance(techs, str):
                        techs = [t.strip() for t in techs.split(",")]
                    if any(t in position_pool for t in techs):
                        relevant_projects += 1
                features[6] = min(1.0, relevant_projects / max(len(projects), 1))
            else:
                features[6] = min(1.0, len(projects) / 5.0)
        else:
            features[6] = min(1.0, len(projects) / 5.0)

        # 8. certification_match
        certs = candidate.get("multimodal", [])
        cert_count = sum(1 for c in certs if c.get("type") == "certificate")
        features[7] = min(1.0, cert_count / 3.0)

        # 9. language_match
        # 检查候选人是否有英语相关技能
        has_english = any("英语" in s.get("skill_name", "") or "English" in s.get("skill_name", "")
                         for s in candidate.get("skills", []))
        features[8] = 0.85 if has_english else 0.5

        # 10. management_match
        has_leadership = any(
            r.get("role", "") in ["负责人", "核心开发", "主要贡献者"]
            for r in candidate.get("projects", [])
        )
        if jd.get("is_management"):
            features[9] = 0.9 if has_leadership else 0.3
        else:
            features[9] = 0.7 if has_leadership else 0.5

        # 11. cultural_fit — 基于候选人当前状态的稳定性
        job_status = candidate.get("job_status", "")
        if job_status == "在职看机会":
            features[10] = 0.7
        elif job_status == "离职":
            features[10] = 0.85  # 离职可快速入职
        else:
            features[10] = 0.4  # 在职不看

        # 12. growth_potential
        age = candidate.get("age", 30) or 30
        features[11] = max(0.3, 1.0 - (age - 22) / 30.0)

        return features

    # 默认基础权重（12维）
    _BASE_WEIGHTS = np.array([0.12, 0.18, 0.15, 0.08, 0.06, 0.10, 0.10, 0.05, 0.04, 0.04, 0.04, 0.04])

    def _get_effective_weights(self) -> np.ndarray:
        """获取经过用户反馈调整后的有效权重（归一化后和为 1）"""
        w = self._BASE_WEIGHTS.copy()
        if hasattr(self, '_weight_adjustments'):
            w = w + self._weight_adjustments
        # 确保非负并归一化
        w = np.maximum(w, 0.01)
        w = w / w.sum()
        return w

    def predict(self, features: np.ndarray) -> float:
        """预测匹配分数"""
        if self.is_trained and self.model is not None:
            prob = self.model.predict_proba(features.reshape(1, -1))[0][1]
            # 已训练模式下，用反馈调整量做后处理微调
            if hasattr(self, '_weight_adjustments') and np.any(self._weight_adjustments != 0):
                adjustment_influence = float(np.dot(features, self._weight_adjustments)) * 0.1
                prob = max(0.0, min(1.0, prob + adjustment_influence))
            return float(prob)
        # 未训练时使用反馈驱动的动态加权平均
        weights = self._get_effective_weights()
        score = float(np.dot(features, weights))
        return max(0.0, min(1.0, score))

    def _predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """批量正类概率（供采样 Shapley 使用）"""
        if self.is_trained and self.model is not None:
            return self.model.predict_proba(X)[:, 1]
        weights = self._get_effective_weights()
        return np.clip(X @ weights, 0.0, 1.0)

    def train(self, X: np.ndarray, y: np.ndarray):
        """训练梯度提升模型（真实训练）

        优先 CatBoost；若 CatBoost 无 wheel 则使用 sklearn GradientBoosting（同为 GBDT）。
        X: (n_samples, 12) 结构化特征
        y: (n_samples,) 二值标签（1=相关/匹配, 0=不相关）
        """
        if self.model is None:
            self._init_model()
        if self.model is None or self.backend == "weighted":
            logger.warning("No GBDT backend available for training; staying on weighted average")
            return
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            logger.warning("Training labels have a single class; skipping fit, keeping weighted fallback")
            return
        if self.backend == "catboost":
            self.model.fit(X, y, verbose=0)
        else:
            self.model.fit(X, y)
        self.is_trained = True
        # 保留训练集背景（采样 Shapley 期望基线），最多 100 条
        if len(X) > 100:
            idx = np.random.RandomState(42).choice(len(X), 100, replace=False)
            self._background = X[idx]
        else:
            self._background = X
        logger.info(f"{self.backend} model trained with {len(X)} samples "
                    f"({int(np.sum(y))} positive / {len(y)} total)")

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if self.is_trained and self.model is not None:
            if self.backend == "catboost":
                importance = self.model.get_feature_importance()
                return dict(zip(self.feature_names, importance.tolist()))
            elif self.backend == "sklearn_gbdt":
                importance = self.model.feature_importances_
                # 归一化为百分比，与 CatBoost 输出量纲一致
                total = float(np.sum(importance)) or 1.0
                return dict(zip(self.feature_names, (importance / total * 100.0).tolist()))
        # 默认重要性
        default = [0.12, 0.18, 0.15, 0.08, 0.06, 0.10, 0.10, 0.05, 0.04, 0.04, 0.04, 0.04]
        return dict(zip(self.feature_names, default))

    def compute_shap_values(self, features: np.ndarray, n_samples: int = 200):
        """计算真实 Shapley 值（基于训练好的 GBDT 模型）。

        优先使用 shap.TreeExplainer（若 shap 库可用）；否则使用纯 numpy
        实现的采样 Shapley（Monte-Carlo permutation Shapley，即 SHAP 原始算法）。
        两者均为真实 Shapley 值，非线性近似。

        返回 (shap_values, base_value)：
        - shap_values: (12,) 每维特征对该样本预测的 Shapley 贡献
        - base_value: 模型输出的期望基线 E[f(x)]

        若模型未训练，返回 (None, None)，由上层降级为线性归因。
        """
        if not (self.is_trained and self.model is not None):
            return None, None
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)

        # 路径 1：CatBoost + shap.TreeExplainer（若均可用）
        if self.backend == "catboost":
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                sv = np.array(explainer.shap_values(x))
                if sv.ndim == 3:
                    sv = sv[-1]
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = float(np.array(base_value).flatten()[-1])
                else:
                    base_value = float(base_value)
                return np.array(sv[0], dtype=float), base_value
            except Exception as e:
                logger.warning(f"shap.TreeExplainer failed ({e}); using sampling Shapley")

        # 路径 2：纯 numpy 采样 Shapley（适用于 sklearn GBDT 及 CatBoost 降级）
        try:
            return self._sampling_shapley(x[0], n_samples=n_samples)
        except Exception as e:
            logger.warning(f"Sampling Shapley failed: {e}, falling back to linear attribution")
            return None, None

    def _sampling_shapley(self, x: np.ndarray, n_samples: int = 200):
        """纯 numpy 实现的采样 Shapley 值（Monte-Carlo permutation）。

        对每个随机特征排列，逐个加入特征并记录预测增量，取多次均值。
        背景值从训练集随机抽取，代表「缺失」特征的期望。
        """
        n_features = len(x)
        bg = self._background
        if bg is None or len(bg) == 0:
            bg = x.reshape(1, -1)
        rng = np.random.RandomState(2024)
        base_value = float(np.mean(self._predict_proba_batch(np.asarray(bg, dtype=np.float64))))
        phi = np.zeros(n_features, dtype=float)

        for _ in range(n_samples):
            ref = bg[rng.randint(len(bg))].astype(np.float64)
            perm = rng.permutation(n_features)
            # 从全背景出发，按 perm 顺序逐个替换为真实特征值
            cur = ref.copy()
            prev_pred = self._predict_proba_batch(cur.reshape(1, -1))[0]
            for j in perm:
                cur[j] = x[j]
                new_pred = self._predict_proba_batch(cur.reshape(1, -1))[0]
                phi[j] += (new_pred - prev_pred)
                prev_pred = new_pred
        phi /= n_samples
        return phi, base_value

    def update_weights(self, feedback: Dict[str, float]):
        """基于用户反馈动态调整特征权重（反馈驱动的核心创新点）。

        策略说明：
        - 未训练模式（加权平均）：直接修改权重向量
          - increase_diversity: 提高多样性特征（技能广度、项目经验、兴趣匹配）的权重
          - increase_precision: 提高精确匹配特征（技能深度、学历、工作年限）的权重
        - 已训练模式（GBDT）：记录偏移，在下次 predict 时作为后处理修正

        权重调整幅度由 satisfaction_rate 决定，满意率越低调整幅度越大（最大 ±5%）。
        """
        logger.info(f"Updating weights based on feedback: {feedback}")

        adjustment = feedback.get("adjustment", "")
        satisfaction_rate = feedback.get("satisfaction_rate", 0.5)

        # 调整步长：满意率越偏离 0.5，调整越大（但上限 0.05）
        step = min(0.05, abs(satisfaction_rate - 0.7) * 0.1 + 0.01)

        # 12维权重对应：
        # 0: education_score    1: skill_match      2: experience_years
        # 3: project_relevance  4: research_impact  5: certification_score
        # 6: stability_score    7: leadership_score 8: communication_score
        # 9: innovation_score  10: culture_fit     11: growth_potential

        # 定义特征分组
        precision_features = [0, 1, 2, 5]      # 学历、技能匹配、经验、证书 → 精确匹配
        diversity_features = [3, 8, 9, 10, 11]  # 项目、沟通、创新、文化、潜力 → 多样性

        if not hasattr(self, '_weight_adjustments'):
            # 初始化累计调整量（相对于默认权重的偏移）
            self._weight_adjustments = np.zeros(12, dtype=np.float64)

        if adjustment == "increase_diversity":
            # 用户不满意率高 → 推荐结果太同质化 → 增加多样性维度权重
            for i in diversity_features:
                self._weight_adjustments[i] += step
            for i in precision_features:
                self._weight_adjustments[i] -= step * 0.5
        elif adjustment == "increase_precision":
            # 用户满意率高 → 保持并强化当前策略 → 微调精确匹配权重
            for i in precision_features:
                self._weight_adjustments[i] += step * 0.3
        else:
            return

        # 限制累计调整范围 [-0.15, +0.15]，避免权重漂移过远
        self._weight_adjustments = np.clip(self._weight_adjustments, -0.15, 0.15)

        # 如果是加权平均模式，立即生效
        if not self.is_trained:
            logger.info(f"Weight adjustments applied (weighted mode): "
                       f"adjustments={self._weight_adjustments.round(4).tolist()}")

        # 记录调整历史供分析
        if not hasattr(self, '_adjustment_history'):
            self._adjustment_history = []
        self._adjustment_history.append({
            "feedback": feedback,
            "step": step,
            "cumulative_adjustments": self._weight_adjustments.tolist(),
        })
        # 只保留最近 100 次调整记录
        if len(self._adjustment_history) > 100:
            self._adjustment_history = self._adjustment_history[-50:]

    # ══════════════════════════════════════════════════════════════════════════
    # 动态特征评分（LLM 提取的 extra_constraints 自动生成匹配分数）
    # ══════════════════════════════════════════════════════════════════════════

    def compute_dynamic_feature_scores(self, candidate: Dict[str, Any],
                                        extra_constraints: Dict[str, Any]) -> Dict[str, float]:
        """根据 LLM 提取的动态约束，计算候选人在每个动态维度上的匹配分数
        
        参数:
            candidate: 候选人完整数据（含 extra_attributes 字段）
            extra_constraints: LLM 提取的动态约束，格式如：
                {"gpa": {"operator": ">=", "value": 3.0},
                 "target_job": {"operator": "contains", "value": "DevOps"}}
        
        返回:
            动态特征分数字典，如：
                {"gpa_match": 0.85, "target_job_match": 0.70}
            
            每个分数在 [0, 1] 范围内，表示候选人在该维度上满足查询要求的程度。
        """
        if not extra_constraints:
            return {}
        
        # 获取候选人的动态属性
        candidate_attrs = candidate.get("extra_attributes", {})
        if not candidate_attrs:
            # 如果候选人数据中没有 extra_attributes，尝试从数据库加载
            from backend.database.models import hr_db
            candidate_attrs = hr_db.get_extra_attributes(candidate.get("id", 0))
        
        scores = {}
        
        for attr_key, condition in extra_constraints.items():
            if not isinstance(condition, dict):
                continue
            
            operator = condition.get("operator", "==")
            target_value = condition.get("value")
            if target_value is None:
                continue
            
            feature_key = f"{attr_key}_match"
            candidate_value = candidate_attrs.get(attr_key)
            
            if candidate_value is None:
                # 候选人没有该属性 → 给低分但不是0（可能数据缺失）
                scores[feature_key] = 0.15
                continue
            
            # 根据不同操作符计算匹配分数
            scores[feature_key] = self._score_dynamic_attribute(
                candidate_value, operator, target_value, attr_key
            )
        
        return scores

    def _score_dynamic_attribute(self, candidate_value: Any, operator: str,
                                  target_value: Any, attr_key: str) -> float:
        """计算单个动态属性的匹配分数 [0, 1]
        
        评分策略：
        - 完全满足约束 → 高分 (0.8-1.0)
        - 接近满足 → 中等分 (0.4-0.7)
        - 完全不满足 → 低分 (0.1-0.3)
        """
        try:
            if operator == "contains":
                # 字符串包含匹配
                cand_str = str(candidate_value).lower()
                target_str = str(target_value).lower()
                if target_str in cand_str:
                    return 0.95  # 完全包含
                # 部分匹配（如 target="DevOps"，candidate="DevOps工程师"）
                # 计算字符重叠度
                overlap = sum(1 for c in target_str if c in cand_str)
                ratio = overlap / max(len(target_str), 1)
                return max(0.1, min(0.6, ratio * 0.6))
            
            elif operator == "==":
                cand_str = str(candidate_value).strip().lower()
                target_str = str(target_value).strip().lower()
                if cand_str == target_str:
                    return 0.95
                # 部分匹配
                if target_str in cand_str or cand_str in target_str:
                    return 0.7
                return 0.15
            
            elif operator in (">=", "<=", ">", "<"):
                # 数值比较 → 计算满足程度
                cand_num = float(candidate_value)
                target_num = float(target_value)
                
                if operator == ">=":
                    if cand_num >= target_num:
                        # 满足：超出越多分越高（但有上限）
                        excess_ratio = (cand_num - target_num) / max(target_num, 1)
                        return min(1.0, 0.8 + excess_ratio * 0.2)
                    else:
                        # 不满足：差距越大分越低
                        deficit_ratio = (target_num - cand_num) / max(target_num, 1)
                        return max(0.1, 0.7 - deficit_ratio * 0.6)
                
                elif operator == "<=":
                    if cand_num <= target_num:
                        return min(1.0, 0.8 + (target_num - cand_num) / max(target_num, 1) * 0.2)
                    else:
                        deficit_ratio = (cand_num - target_num) / max(target_num, 1)
                        return max(0.1, 0.7 - deficit_ratio * 0.6)
                
                elif operator == ">":
                    if cand_num > target_num:
                        return min(1.0, 0.85 + (cand_num - target_num) / max(target_num, 1) * 0.15)
                    else:
                        return max(0.1, 0.5 - abs(cand_num - target_num) / max(target_num, 1) * 0.4)
                
                elif operator == "<":
                    if cand_num < target_num:
                        return min(1.0, 0.85 + (target_num - cand_num) / max(target_num, 1) * 0.15)
                    else:
                        return max(0.1, 0.5 - abs(cand_num - target_num) / max(target_num, 1) * 0.4)
            
            return 0.5  # 未知操作符给中间值
            
        except (ValueError, TypeError):
            return 0.3  # 类型转换失败给低分


# 全局实例
catboost_matcher = CatBoostMatcher()
