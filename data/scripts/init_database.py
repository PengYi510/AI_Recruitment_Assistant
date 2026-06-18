"""数据库初始化脚本 - 将合成简历数据导入SQLite和ChromaDB

运行方法: cd hr_agent_mt && python -m data.scripts.init_database

功能:
1. 从 full_resume_dataset_1000.json 读取1000条简历数据
2. 导入到 SQLite 数据库（6张核心表: 候选人、教育经历、技能、获奖证书、工作经历、项目）
3. 生成文本特征向量并存入 ChromaDB 向量库（1主collection + metadata过滤）

注意: 此脚本为幂等操作，重复运行会先清空再重新导入。
"""

import sys
import json
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import SYNTHETIC_DIR
from backend.database.models import hr_db
from backend.vector_db.client import vector_db
from backend.models.multimodal_fusion import multimodal_fusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset() -> list:
    """加载合成数据集"""
    dataset_path = SYNTHETIC_DIR / "full_resume_dataset_1000.json"
    if not dataset_path.exists():
        logger.error(f"数据集文件不存在: {dataset_path}")
        logger.info("请先运行: python -m data.scripts.generate_full_dataset")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"加载数据集: {len(data)} 条简历")
    return data


def clear_databases():
    """清空现有数据"""
    import sqlite3
    from backend.config import DB_PATH

    logger.info("清空 SQLite 数据库...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM projects")
    conn.execute("DELETE FROM work_experiences")
    conn.execute("DELETE FROM awards_certificates")
    conn.execute("DELETE FROM skills")
    conn.execute("DELETE FROM education_history")
    conn.execute("DELETE FROM candidates")
    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('candidates','skills','work_experiences','projects','education_history','awards_certificates')")
    conn.commit()
    conn.close()

    logger.info("清空 ChromaDB 向量库...")
    vector_db.reset()


def build_candidate_text(candidate: dict) -> str:
    """构建候选人的完整结构化简历文本（用于生成向量嵌入）

    将候选人的所有结构化信息拼接为一份完整的简历文本，格式清晰、信息密集，
    确保向量嵌入能充分捕捉教育背景、技术能力、工作经历等语义信息。

    文本格式示例:
        个人简历：候选人0700，男，32岁，后端开发工程师，8年工作经验，坐标成都，在职看机会，期望薪资49000元/月。
        教育经历：2009.09-2013.06，燕山大学，网络工程，本科（普通本科/全日制）；
        2013.09-2016.06，四川大学，人工智能，硕士（985/全日制）。
        技术栈：Git(精通)、RabbitMQ(熟练)、Redis(熟练)、Python(熟练)、Spring(了解)。
        获奖证书：全国大学生数学建模竞赛国家二等奖(国家级)、AWS认证架构师(企业级)。
        工作经历：美团-后端开发工程师(11个月)-优化系统性能；拼多多-UI设计师(40个月)。
        项目经历：分布式网关(Python,RabbitMQ,Redis)-主要开发者；高并发服务(Spring,Redis,RabbitMQ)-负责人。
    """
    sections = []

    # ━━ 个人信息 ━━
    personal_parts = []
    name = candidate.get('name', '')
    if name:
        personal_parts.append(name)
    gender = candidate.get('gender', '')
    if gender:
        personal_parts.append(gender)
    age = candidate.get('age')
    if age:
        personal_parts.append(f"{age}岁")
    position = candidate.get('current_position', '')
    if position:
        personal_parts.append(position)
    work_years = candidate.get('work_years', 0)
    if work_years:
        personal_parts.append(f"{work_years}年工作经验")
    location = candidate.get('location', '')
    if location:
        personal_parts.append(f"坐标{location}")
    job_status = candidate.get('job_status', '')
    if job_status:
        personal_parts.append(job_status)
    salary = candidate.get('expected_salary')
    if salary:
        personal_parts.append(f"期望薪资{salary}元/月")

    sections.append(f"个人简历：{'，'.join(personal_parts)}。")

    # ━━ 教育经历时间线 ━━
    edu_history = candidate.get("education_history", [])
    if edu_history:
        edu_lines = []
        for edu in edu_history:
            start = edu.get('start_date', '').replace('-', '.')
            end = edu.get('end_date', '').replace('-', '.')
            school = edu.get('school', '')
            major = edu.get('major', '')
            degree = edu.get('degree', '')
            tier = edu.get('school_tier', '')
            ft_tag = "全日制" if edu.get("is_fulltime", True) else "非全日制"
            tier_str = f"{tier}/" if tier else ""
            edu_lines.append(f"{start}-{end}，{school}，{major}，{degree}（{tier_str}{ft_tag}）")
        sections.append(f"教育经历：{'；'.join(edu_lines)}。")
    else:
        # 无详细教育经历时用顶层字段
        edu_level = candidate.get('highest_education', candidate.get('education_level', ''))
        school = candidate.get('school', '')
        major = candidate.get('major', '')
        if school:
            sections.append(f"教育经历：{school}，{major}，{edu_level}。")

    # ━━ 技术栈（按熟练度排序）━━
    skills = candidate.get("skills", [])
    if skills:
        proficiency_map = {5: "精通", 4: "熟练", 3: "熟悉", 2: "掌握", 1: "了解"}
        sorted_skills = sorted(skills, key=lambda s: s.get('proficiency', 0), reverse=True)
        skill_strs = [f"{s['skill_name']}({proficiency_map.get(s.get('proficiency', 1), '了解')})"
                      for s in sorted_skills]
        sections.append(f"技术栈：{'、'.join(skill_strs)}。")

    # ━━ 获奖证书 ━━
    awards = candidate.get("awards_certificates", [])
    if awards:
        award_strs = [f"{a.get('name', '')}({a.get('level', '')})" for a in awards]
        sections.append(f"获奖证书：{'、'.join(award_strs)}。")

    # ━━ 工作经历时间线 ━━
    work_exps = candidate.get("work_experiences", [])
    if work_exps:
        exp_lines = []
        for exp in work_exps:
            company = exp.get('company_name', '')
            pos = exp.get('position', '')
            duration = exp.get('duration_months', 0)
            desc = exp.get('description', '')
            duration_str = f"({duration}个月)" if duration else ""
            desc_str = f"-{desc}" if desc else ""
            exp_lines.append(f"{company}-{pos}{duration_str}{desc_str}")
        sections.append(f"工作经历：{'；'.join(exp_lines)}。")

    # ━━ 项目经历 ━━
    projects = candidate.get("projects", [])
    if projects:
        proj_lines = []
        for proj in projects:
            proj_name = proj.get('project_name', '')
            role = proj.get('role', '')
            techs = proj.get("technologies", [])
            tech_str = ",".join(techs) if isinstance(techs, list) else str(techs)
            proj_lines.append(f"{proj_name}({tech_str})-{role}")
        sections.append(f"项目经历：{'；'.join(proj_lines)}。")

    return "\n".join(sections)


def import_candidates(dataset: list):
    """导入候选人数据到 SQLite 和 ChromaDB

    分两阶段：
    1. 逐条插入 SQLite（6张核心表的结构化数据）
    2. 批量生成 BGE-M3 向量嵌入（1024d）并存入 ChromaDB
    """
    total = len(dataset)
    success_count = 0
    error_count = 0

    # 阶段1: 导入 SQLite 结构化数据
    logger.info("阶段1: 导入结构化数据到 SQLite（6张核心表）...")
    candidate_ids = []  # 记录成功插入的candidate_id与原始数据映射

    for i, candidate in enumerate(dataset):
        try:
            # 1. 插入 candidates 主表（人物基础信息）
            candidate_id = hr_db.insert_candidate({
                "name": candidate.get("name", f"候选人{i+1:04d}"),
                "gender": candidate.get("gender"),
                "birth_date": candidate.get("birth_date"),
                "age": candidate.get("age"),
                "phone": candidate.get("phone"),
                "email": candidate.get("email"),
                "address": candidate.get("address"),
                "current_position": candidate.get("current_position"),
                "work_years": candidate.get("work_years", 0),
                "current_salary": candidate.get("current_salary"),
                "expected_salary": candidate.get("expected_salary"),
                "job_status": candidate.get("job_status"),
                "location": candidate.get("location"),
                "highest_education": candidate.get("highest_education",
                                                   candidate.get("education_level")),
                "summary": candidate.get("summary", ""),
            })

            # 2. 插入教育经历表
            for edu in candidate.get("education_history", []):
                hr_db.insert_education_history(candidate_id, {
                    "degree": edu.get("degree"),
                    "school": edu.get("school"),
                    "major": edu.get("major"),
                    "start_date": edu.get("start_date"),
                    "end_date": edu.get("end_date"),
                    "is_fulltime": edu.get("is_fulltime", True),
                    "school_tier": edu.get("school_tier"),
                })

            # 3. 插入技术栈表
            for skill in candidate.get("skills", []):
                hr_db.insert_skill(
                    candidate_id,
                    skill.get("skill_name", ""),
                    skill.get("proficiency", 3),
                    skill.get("category")
                )

            # 4. 插入获奖资格证书表
            for award in candidate.get("awards_certificates", []):
                hr_db.insert_award_certificate(candidate_id, {
                    "type": award.get("type", "award"),
                    "name": award.get("name"),
                    "level": award.get("level"),
                    "date": award.get("date"),
                    "role": award.get("role"),
                    "description": award.get("description"),
                    "image_path": award.get("image_path"),
                })

            # 5. 插入工作经历表
            for exp in candidate.get("work_experiences", []):
                hr_db.insert_work_experience(candidate_id, {
                    "company_name": exp.get("company_name"),
                    "position": exp.get("position"),
                    "location": exp.get("location"),
                    "start_date": exp.get("start_date"),
                    "end_date": exp.get("end_date"),
                    "duration_months": exp.get("duration_months"),
                    "description": exp.get("description"),
                })

            # 6. 插入项目经历表
            for proj in candidate.get("projects", []):
                techs = proj.get("technologies", [])
                hr_db.insert_project(candidate_id, {
                    "project_name": proj.get("project_name"),
                    "role": proj.get("role"),
                    "start_date": proj.get("start_date"),
                    "end_date": proj.get("end_date"),
                    "duration_months": proj.get("duration_months"),
                    "description": proj.get("description"),
                    "technologies": ", ".join(techs) if isinstance(techs, list) else str(techs),
                })

            candidate_ids.append((candidate_id, candidate))
            success_count += 1
            if (i + 1) % 100 == 0:
                logger.info(f"  SQLite进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

        except Exception as e:
            error_count += 1
            logger.error(f"导入候选人 {i+1} 失败: {e}")

    # 阶段2: 批量生成向量嵌入并存入 ChromaDB
    logger.info(f"阶段2: 批量生成向量嵌入 ({len(candidate_ids)} 条候选人)...")
    logger.info(f"  使用真实模型: {multimodal_fusion.is_using_real_model}")

    # 构建所有候选人文本
    texts = [build_candidate_text(cand) for _, cand in candidate_ids]

    # 批量编码（batch_size=32，充分利用批处理加速）
    embeddings = multimodal_fusion.extract_text_features_batch(texts)

    # 存入 ChromaDB（带metadata和document）
    for j, (candidate_id, candidate) in enumerate(candidate_ids):
        embedding = embeddings[j].flatten().tolist()

        # 构建metadata供结构化过滤
        skills_list = [s.get("skill_name", "") for s in candidate.get("skills", [])]
        schools_list = [e.get("school", "") for e in candidate.get("education_history", [])]

        metadata = {
            "name": candidate.get("name", ""),
            "highest_education": candidate.get("highest_education",
                                               candidate.get("education_level", "")),
            "work_years": candidate.get("work_years", 0),
            "current_position": candidate.get("current_position", ""),
            "location": candidate.get("location", ""),
            "skills_text": ",".join(skills_list),
            "school_list": ",".join(schools_list),
        }

        # document存储完整简历文本（可用于BM25回溯）
        document = texts[j]

        vector_db.add_candidate(
            candidate_id=candidate_id,
            embedding=embedding,
            metadata=metadata,
            document=document
        )
        if (j + 1) % 100 == 0:
            logger.info(f"  ChromaDB进度: {j+1}/{len(candidate_ids)}")

    logger.info(f"向量索引构建完成: {len(candidate_ids)} 条")

    return success_count, error_count


def verify_import():
    """验证导入结果"""
    candidate_count = hr_db.get_all_candidates_count()
    vector_count = vector_db.get_collection_count()
    logger.info(f"验证结果:")
    logger.info(f"  SQLite 候选人数: {candidate_count}")
    logger.info(f"  ChromaDB 向量数: {vector_count}")

    # 测试查询
    test_candidates = hr_db.search_candidates(filters={"min_work_years": 5}, limit=5)
    logger.info(f"  测试查询(5年+经验): 找到 {len(test_candidates)} 条结果")
    for c in test_candidates[:3]:
        logger.info(f"    - {c['name']}: {c['highest_education']} {c.get('location', '')}, {c['work_years']}年经验")

    return candidate_count > 0 and vector_count > 0


def main():
    logger.info("=" * 60)
    logger.info("智能招聘匹配系统 - 数据库初始化 (新版6表+向量库)")
    logger.info("=" * 60)

    # 加载数据
    dataset = load_dataset()

    # 清空旧数据
    clear_databases()

    # 导入数据
    logger.info(f"开始导入 {len(dataset)} 条候选人数据...")
    success, errors = import_candidates(dataset)
    logger.info(f"导入完成: 成功 {success} 条, 失败 {errors} 条")

    # 验证
    if verify_import():
        logger.info("数据库初始化成功！系统已就绪。")
    else:
        logger.error("数据库初始化验证失败，请检查错误日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
