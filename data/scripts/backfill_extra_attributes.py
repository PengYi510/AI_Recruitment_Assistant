"""批量回填 extra_attributes 数据

为所有现有候选人生成动态扩展属性（GPA、身高体重、兴趣爱好、目标岗位等），
模拟真实简历中的多样化信息分布。

运行方法: cd hr_agent_mt && python -m data.scripts.backfill_extra_attributes
"""

import sys
import random
import sqlite3
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_extra_attributes(candidate_id: int, age: int, education_level: str, work_years: int) -> dict:
    """为单个候选人生成动态扩展属性。

    使用 candidate_id 作为随机种子，确保幂等（重复运行结果一致）。
    根据候选人的年龄、学历、工作年限调整生成概率。
    """
    random.seed(candidate_id * 7 + 31)  # 确定性随机
    attrs = {}

    is_student = work_years <= 1
    highest_edu = education_level or "本科"

    # GPA（学生/应届更常写，约60%学生写、30%社招写）
    gpa_prob = 0.6 if is_student else 0.3
    if random.random() < gpa_prob:
        gpa_style = random.choice(["4.0", "5.0", "100"])
        if gpa_style == "4.0":
            gpa = round(random.uniform(2.8, 4.0), 2)
            attrs["gpa"] = f"{gpa}/4.0"
        elif gpa_style == "5.0":
            gpa = round(random.uniform(3.5, 5.0), 2)
            attrs["gpa"] = f"{gpa}/5.0"
        else:
            gpa = random.randint(72, 98)
            attrs["gpa"] = f"{gpa}/100"

    # 爱好（约40%的简历会写）
    if random.random() < 0.4:
        hobby_pool = ["篮球", "足球", "游泳", "跑步", "健身", "羽毛球", "乒乓球",
                      "阅读", "摄影", "旅行", "音乐", "吉他", "钢琴", "绘画",
                      "编程", "开源贡献", "写博客", "电竞", "桌游", "烹饪",
                      "登山", "骑行", "瑜伽", "滑雪", "潜水"]
        hobbies = random.sample(hobby_pool, random.randint(2, 5))
        attrs["hobbies"] = ",".join(hobbies)

    # 目标岗位（约50%会写）
    if random.random() < 0.5:
        target_pool = ["后端开发工程师", "Java高级工程师", "算法工程师",
                       "数据开发工程师", "全栈工程师", "架构师",
                       "大数据开发", "AI工程师", "NLP算法工程师",
                       "推荐算法工程师", "搜索工程师", "数据分析师",
                       "前端开发工程师", "移动端开发", "DevOps工程师",
                       "测试开发工程师", "安全工程师", "产品经理"]
        attrs["target_job"] = random.choice(target_pool)

    # 民族（约15%会写）
    if random.random() < 0.15:
        ethnicity_pool = ["汉族", "汉族", "汉族", "汉族", "汉族",
                          "回族", "满族", "壮族", "苗族", "维吾尔族", "藏族", "蒙古族"]
        attrs["ethnicity"] = random.choice(ethnicity_pool)

    # 身高体重（约12%会写）
    if random.random() < 0.12:
        attrs["height_cm"] = random.randint(155, 190)
        attrs["weight_kg"] = random.randint(45, 95)

    # 婚姻状况（年龄大的更可能写）
    if age and age >= 26 and random.random() < 0.15:
        attrs["marital_status"] = random.choices(
            ["未婚", "已婚", "离异"], weights=[0.5, 0.45, 0.05], k=1)[0]

    # 政治面貌（约20%会写）
    if random.random() < 0.20:
        attrs["political_status"] = random.choices(
            ["群众", "共青团员", "中共党员"], weights=[0.3, 0.4, 0.3], k=1)[0]

    # 语言能力（约35%会写）
    if random.random() < 0.35:
        lang_pool = ["英语CET-4", "英语CET-6", "英语CET-6(560+)", "英语雅思7.0",
                     "英语托福100+", "日语N2", "日语N1", "德语B1", "法语A2"]
        langs = random.sample(lang_pool, random.randint(1, 2))
        attrs["languages"] = ",".join(langs)

    # 自我评价（约30%会写）
    if random.random() < 0.30:
        eval_pool = [
            "具有良好的团队协作能力和沟通能力，善于解决复杂技术问题",
            "热爱技术，持续学习，有较强的自驱力和责任心",
            "注重代码质量，熟悉敏捷开发流程，有良好的工程素养",
            "逻辑思维强，善于抽象建模，对分布式系统有深入理解",
            "具备全栈视野，能独立负责从需求分析到上线的完整流程",
            "对AI/大模型技术充满热情，持续关注前沿论文和开源项目",
            "善于数据驱动决策，具备较强的业务理解能力",
            "有丰富的大型项目经验，擅长系统设计和性能优化",
        ]
        attrs["self_evaluation"] = random.choice(eval_pool)

    return attrs


def backfill():
    """批量回填所有候选人的 extra_attributes"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 获取所有候选人基本信息
    candidates = conn.execute(
        "SELECT id, age, education_level, work_years FROM candidates"
    ).fetchall()
    total = len(candidates)
    logger.info(f"共 {total} 条候选人需要处理")

    # 先清空已有的 extra_attributes（幂等操作）
    conn.execute("DELETE FROM candidate_extra_attributes")
    conn.commit()
    logger.info("已清空旧的 extra_attributes 数据")

    # 批量生成并插入
    batch_size = 500
    inserted_count = 0
    candidates_with_attrs = 0

    for i, cand in enumerate(candidates):
        cid = cand["id"]
        age = cand["age"] or 25
        edu = cand["education_level"] or "本科"
        work_years = cand["work_years"] or 0

        attrs = generate_extra_attributes(cid, age, edu, work_years)

        if attrs:
            candidates_with_attrs += 1
            for key, value in attrs.items():
                # 判断类型
                if isinstance(value, bool):
                    attr_type = "bool"
                elif isinstance(value, (int, float)):
                    attr_type = "number"
                elif isinstance(value, list):
                    attr_type = "list"
                    value = str(value)
                else:
                    attr_type = "string"

                conn.execute(
                    "INSERT INTO candidate_extra_attributes (candidate_id, attr_key, attr_value, attr_type) VALUES (?, ?, ?, ?)",
                    (cid, key, str(value), attr_type)
                )
                inserted_count += 1

        if (i + 1) % batch_size == 0:
            conn.commit()
            logger.info(f"  进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    conn.commit()
    conn.close()

    logger.info(f"回填完成:")
    logger.info(f"  总候选人: {total}")
    logger.info(f"  有属性的候选人: {candidates_with_attrs} ({candidates_with_attrs/total*100:.1f}%)")
    logger.info(f"  插入属性条目: {inserted_count}")


def verify():
    """验证回填结果"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total_attrs = conn.execute("SELECT COUNT(*) FROM candidate_extra_attributes").fetchone()[0]
    unique_candidates = conn.execute("SELECT COUNT(DISTINCT candidate_id) FROM candidate_extra_attributes").fetchone()[0]
    total_candidates = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    logger.info(f"验证结果:")
    logger.info(f"  总属性条目: {total_attrs}")
    logger.info(f"  有属性的候选人: {unique_candidates}/{total_candidates}")

    # 按属性类型统计
    attr_stats = conn.execute(
        "SELECT attr_key, COUNT(*) as cnt FROM candidate_extra_attributes GROUP BY attr_key ORDER BY cnt DESC"
    ).fetchall()
    logger.info(f"  各属性分布:")
    for row in attr_stats:
        pct = row["cnt"] / total_candidates * 100
        logger.info(f"    {row['attr_key']}: {row['cnt']} ({pct:.1f}%)")

    # 展示几个样例
    sample_ids = conn.execute("SELECT DISTINCT candidate_id FROM candidate_extra_attributes LIMIT 3").fetchall()
    for sid_row in sample_ids:
        sid = sid_row["candidate_id"]
        attrs = conn.execute(
            "SELECT attr_key, attr_value FROM candidate_extra_attributes WHERE candidate_id = ?", (sid,)
        ).fetchall()
        logger.info(f"  样例 candidate_id={sid}: {dict((r['attr_key'], r['attr_value']) for r in attrs)}")

    conn.close()


def main():
    logger.info("=" * 60)
    logger.info("批量回填 extra_attributes（GPA、身高体重、兴趣爱好等）")
    logger.info("=" * 60)

    backfill()
    verify()

    logger.info("完成！")


if __name__ == "__main__":
    main()
