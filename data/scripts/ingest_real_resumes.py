"""真实简历批量入库脚本（完全解耦，不依赖任何生成器）。

把真实简历喂入系统的唯一推荐入口。支持两种来源：

1) 一个目录里的简历文件（.pdf / .docx / .txt / .pptx）：
     python -m data.scripts.ingest_real_resumes --dir path/to/resumes

2) 一个 JSON 文件，元素是 {"resume_text": "..."} 或 {"file_path": "..."}：
     python -m data.scripts.ingest_real_resumes --json path/to/list.json

每条简历都会走完整链路：
    LLM 结构化提取 → 权威院校层级归一化(school_tier.py) → 写 SQLite
    → BGE-M3 向量化 → 写 ChromaDB

特点：
- 不清空已有数据，纯追加（可与生成简历、历史数据共存）。
- 单条失败不影响其余，最后打印汇总。
- 与生成流程零耦合：本脚本完全不 import 任何 resume_generator。
"""

import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.skills.resume_extraction_skill import ResumeExtractionSkill

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 支持的简历文件后缀（与 resume_extraction_skill 解析能力一致）
SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".pptx", ".doc", ".md"}


def _collect_from_dir(dir_path: Path) -> list:
    """从目录收集简历文件，返回 [{"file_path": ...}, ...]。"""
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error(f"目录不存在或不是目录: {dir_path}")
        sys.exit(1)
    items = []
    for p in sorted(dir_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            items.append({"file_path": str(p)})
    logger.info(f"从目录收集到 {len(items)} 份简历文件: {dir_path}")
    return items


def _collect_from_json(json_path: Path) -> list:
    """从 JSON 收集简历来源。元素支持 resume_text 或 file_path，也支持纯字符串。"""
    if not json_path.exists():
        logger.error(f"JSON 文件不存在: {json_path}")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("resumes") or raw.get("data") or []
    items = []
    for el in raw:
        if isinstance(el, str):
            items.append({"resume_text": el})
        elif isinstance(el, dict):
            if el.get("resume_text"):
                items.append({"resume_text": el["resume_text"]})
            elif el.get("file_path"):
                items.append({"file_path": el["file_path"]})
            elif el.get("text"):
                items.append({"resume_text": el["text"]})
    logger.info(f"从 JSON 收集到 {len(items)} 份简历来源: {json_path}")
    return items


async def ingest(items: list) -> dict:
    """逐条入库，返回汇总统计。"""
    skill = ResumeExtractionSkill()
    total = len(items)
    ok, failed = 0, 0
    failed_detail = []
    candidate_ids = []

    for idx, item in enumerate(items, 1):
        label = item.get("file_path") or f"text#{idx}"
        try:
            result = await skill.extract_and_index(
                file_path=item.get("file_path", ""),
                resume_text=item.get("resume_text", ""),
            )
            if result.get("success"):
                cid = result.get("candidate_id")
                candidate_ids.append(cid)
                vec = "✓向量" if result.get("vector_indexed") else "✗向量"
                ok += 1
                logger.info(f"[{idx}/{total}] OK  candidate_id={cid}  {vec}  ({label})")
            else:
                failed += 1
                err = result.get("error", "unknown")
                failed_detail.append({"label": label, "error": err})
                logger.error(f"[{idx}/{total}] FAIL {err}  ({label})")
        except Exception as e:
            failed += 1
            failed_detail.append({"label": label, "error": str(e)})
            logger.error(f"[{idx}/{total}] EXC  {e}  ({label})")

    return {
        "total": total,
        "ok": ok,
        "failed": failed,
        "candidate_ids": candidate_ids,
        "failed_detail": failed_detail,
    }


def main():
    parser = argparse.ArgumentParser(description="真实简历批量入库（解耦，不依赖生成器）")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", type=str, help="简历文件所在目录")
    g.add_argument("--json", type=str, help="简历来源 JSON 文件")
    args = parser.parse_args()

    if args.dir:
        items = _collect_from_dir(Path(args.dir))
    else:
        items = _collect_from_json(Path(args.json))

    if not items:
        logger.warning("没有可入库的简历，退出。")
        return

    summary = asyncio.run(ingest(items))

    logger.info("=" * 50)
    logger.info(f"入库完成: 共 {summary['total']} 份，成功 {summary['ok']}，失败 {summary['failed']}")
    if summary["failed_detail"]:
        logger.info("失败明细：")
        for d in summary["failed_detail"]:
            logger.info(f"  - {d['label']}: {d['error']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
