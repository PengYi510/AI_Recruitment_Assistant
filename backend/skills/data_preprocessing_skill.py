"""数据预处理Skill - 清洗、特征工程、向量嵌入"""
import logging
import numpy as np
from typing import Dict, Any, List
from backend.skills.base_skill import BaseSkill
from backend.models.multimodal_fusion import multimodal_fusion
from backend.vector_db.client import vector_db
from backend.database.models import hr_db

logger = logging.getLogger(__name__)


class DataPreprocessingSkill(BaseSkill):
    """数据预处理Skill: 清洗、特征工程、多模态特征提取、向量嵌入"""

    def __init__(self):
        super().__init__(name="data_preprocessing", description="数据清洗和特征工程")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "preprocess_all")
        if action == "build_vectors":
            return await self._build_vector_index()
        return await self._preprocess_candidates()

    async def _preprocess_candidates(self) -> Dict[str, Any]:
        """预处理所有候选人数据"""
        candidates = hr_db.search_candidates(limit=1000)
        processed = 0
        for cand in candidates:
            # 数据清洗
            self._clean_candidate(cand)
            processed += 1
        return {"processed": processed}

    async def _build_vector_index(self) -> Dict[str, Any]:
        """构建向量索引 — 使用真实BGE-M3 embedding批量编码"""
        candidates = hr_db.search_candidates(limit=1000)

        # 收集所有候选人文本
        texts = []
        valid_candidates = []
        for cand in candidates:
            full_cand = hr_db.get_candidate(cand["id"])
            if not full_cand:
                continue
            text = self._candidate_to_text(full_cand)
            texts.append(text)
            valid_candidates.append((cand["id"], full_cand))

        if not texts:
            return {"indexed": 0, "total_vectors": vector_db.get_collection_count()}

        # 批量编码（利用GPU/CPU批处理加速）
        logger.info(f"Building vector index for {len(texts)} candidates...")
        embeddings = multimodal_fusion.extract_text_features_batch(texts)

        # 写入向量数据库
        indexed = 0
        for i, (cand_id, full_cand) in enumerate(valid_candidates):
            embedding = embeddings[i].flatten().tolist()
            metadata = {
                "name": full_cand.get("name", ""),
                "education": full_cand.get("highest_education", ""),
                "work_years": full_cand.get("work_years", 0),
                "skills": ",".join(s["skill_name"] for s in full_cand.get("skills", [])[:5]),
            }
            vector_db.add_candidate(cand_id, embedding, metadata)
            indexed += 1

        logger.info(f"Vector index built: {indexed} candidates indexed")
        return {
            "indexed": indexed,
            "total_vectors": vector_db.get_collection_count(),
            "using_real_embedding": multimodal_fusion.is_using_real_model
        }

    def _clean_candidate(self, cand: Dict[str, Any]):
        """数据清洗"""
        if cand.get("age") and (cand["age"] < 18 or cand["age"] > 65):
            cand["age"] = None
        if cand.get("work_years") and cand["work_years"] < 0:
            cand["work_years"] = 0

    def _candidate_to_text(self, cand: Dict[str, Any]) -> str:
        """将候选人信息转为文本（含完整教育经历路径）"""
        parts = [
            f"姓名:{cand.get('name','')}",
        ]
        # 优先使用多段教育经历
        edu_history = cand.get("education_history", [])
        if edu_history:
            edu_segs = []
            for edu in edu_history:
                seg = f"{edu.get('degree','')}/{edu.get('school','')}/{edu.get('major','')}"
                if edu.get('school_tier'):
                    seg += f"({edu.get('school_tier')})"
                if edu.get('is_fulltime') == 0:
                    seg += "(非全日制)"
                edu_segs.append(seg)
            parts.append(f"教育经历:{'→'.join(edu_segs)}")
        else:
            parts.append(f"学历:{cand.get('highest_education','')}")
        parts.append(f"工作年限:{cand.get('work_years',0)}年")
        parts.append(f"技能:{','.join(s['skill_name'] for s in cand.get('skills', []))}")
        for exp in cand.get("work_experiences", []):
            parts.append(f"经历:{exp.get('company_name','')} {exp.get('position','')}")
        for proj in cand.get("projects", []):
            parts.append(f"项目:{proj.get('project_name','')} {proj.get('technologies','')}")
        return " ".join(parts)
