"""数据迁移脚本 - 修复 school_tier 和 gender 数据

功能:
1. 将 education_history 表中的 "QS Top 50/100/200/500" 格式更新为具体排名 "QS N"
2. 为缺失 gender 的候选人随机回填性别数据

运行方法: cd hr_agent_mt && python -m data.scripts.migrate_school_tier_and_gender
"""

import sys
import json
import random
import sqlite3
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 海外院校排名知识库
OVERSEAS_KB_PATH = Path(__file__).parent.parent / "knowledge" / "overseas_school_rankings.json"


def _build_school_rank_map() -> dict:
    """构建 校名(归一化) -> best_rank 的映射"""
    import re
    if not OVERSEAS_KB_PATH.exists():
        logger.error(f"海外院校知识库不存在: {OVERSEAS_KB_PATH}")
        return {}

    data = json.loads(OVERSEAS_KB_PATH.read_text(encoding="utf-8"))
    rank_map = {}  # 归一化校名 -> best_rank

    def norm(name):
        if not name:
            return ""
        s = str(name).strip()
        s = re.sub(r"[（(\[【].*?[）)\]】]", "", s)
        s = re.sub(r"\s+", "", s)
        return s.lower()

    for uni in data.get("universities", []):
        best_rank = uni.get("best_rank")
        if not isinstance(best_rank, int) or best_rank <= 0 or best_rank > 500:
            continue
        for key in ("name_en", "name_cn", "display"):
            name = uni.get(key)
            n = norm(name)
            if n and n not in rank_map:
                rank_map[n] = best_rank

    logger.info(f"构建校名排名映射: {len(rank_map)} 条")
    return rank_map


def migrate_school_tier():
    """将 education_history 中的 QS Top N 格式更新为 QS {具体排名}"""
    import re

    rank_map = _build_school_rank_map()
    if not rank_map:
        logger.warning("无法构建排名映射，跳过 school_tier 迁移")
        return

    def norm(name):
        if not name:
            return ""
        s = str(name).strip()
        s = re.sub(r"[（(\[【].*?[）)\]】]", "", s)
        s = re.sub(r"\s+", "", s)
        return s.lower()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 查找所有 QS Top N 格式的记录
    rows = conn.execute(
        "SELECT id, school, school_tier FROM education_history WHERE school_tier LIKE 'QS Top%'"
    ).fetchall()

    logger.info(f"找到 {len(rows)} 条需要迁移的 QS Top N 记录")

    updated = 0
    not_found = 0
    for row in rows:
        edu_id = row["id"]
        school = row["school"]
        school_norm = norm(school)

        # 尝试精确匹配
        best_rank = rank_map.get(school_norm)

        # 尝试包含匹配
        if best_rank is None:
            for kb_name, rank in rank_map.items():
                if kb_name and len(kb_name) >= 3 and (kb_name in school_norm or school_norm in kb_name):
                    best_rank = rank
                    break

        if best_rank is not None:
            new_tier = f"QS {best_rank}"
            conn.execute(
                "UPDATE education_history SET school_tier = ? WHERE id = ?",
                (new_tier, edu_id)
            )
            updated += 1
        else:
            not_found += 1
            if not_found <= 10:
                logger.warning(f"  未找到排名: {school} (原tier: {row['school_tier']})")

    conn.commit()
    conn.close()
    logger.info(f"school_tier 迁移完成: 更新 {updated} 条, 未找到排名 {not_found} 条")


def migrate_gender():
    """为缺失 gender 的候选人回填性别"""
    conn = sqlite3.connect(DB_PATH)

    # 查找缺失 gender 的候选人
    rows = conn.execute(
        "SELECT id, name FROM candidates WHERE gender IS NULL OR gender = ''"
    ).fetchall()

    logger.info(f"找到 {len(rows)} 条缺失 gender 的候选人")

    if not rows:
        conn.close()
        return

    # 使用确定性随机（基于候选人ID）回填性别，男女比例约 6:4
    updated = 0
    for row in rows:
        cid = row[0]
        random.seed(cid)  # 确定性随机，重复运行结果一致
        gender = "男" if random.random() < 0.6 else "女"
        conn.execute("UPDATE candidates SET gender = ? WHERE id = ?", (gender, cid))
        updated += 1

    conn.commit()
    conn.close()
    logger.info(f"gender 回填完成: 更新 {updated} 条")


def verify():
    """验证迁移结果"""
    conn = sqlite3.connect(DB_PATH)

    # 验证 school_tier
    old_format = conn.execute(
        "SELECT COUNT(*) FROM education_history WHERE school_tier LIKE 'QS Top%'"
    ).fetchone()[0]
    new_format = conn.execute(
        "SELECT COUNT(*) FROM education_history WHERE school_tier LIKE 'QS %' AND school_tier NOT LIKE 'QS Top%'"
    ).fetchone()[0]
    logger.info(f"验证 school_tier: 旧格式剩余 {old_format} 条, 新格式 {new_format} 条")

    # 显示几个新格式示例
    samples = conn.execute(
        "SELECT school, school_tier FROM education_history WHERE school_tier LIKE 'QS %' AND school_tier NOT LIKE 'QS Top%' LIMIT 5"
    ).fetchall()
    for s in samples:
        logger.info(f"  示例: {s[0]} -> {s[1]}")

    # 验证 gender
    null_gender = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE gender IS NULL OR gender = ''"
    ).fetchone()[0]
    logger.info(f"验证 gender: 仍缺失 {null_gender} 条")

    conn.close()


def main():
    logger.info("=" * 60)
    logger.info("数据迁移: school_tier 格式升级 + gender 回填")
    logger.info("=" * 60)

    migrate_school_tier()
    migrate_gender()
    verify()

    logger.info("迁移完成！")


if __name__ == "__main__":
    main()
