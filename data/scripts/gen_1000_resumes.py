"""批量生成模拟简历（分批 + 并发 + 断点续跑 + 进度日志）。

用法：
    python -m data.scripts.gen_1000_resumes --total 1000 --batch 50 --concurrency 5

特点：
- 分批调用 ResumeGeneratorSkill.execute，每批内部并发（Semaphore 限流），
  避免一次性 gather 上千个任务把 LLM/DB 打爆。
- 追加模式：id_offset 自动取数据库当前 max(id)，新文件/邮箱编号从 max+1 起，
  与 DB 自增主键对齐，不覆盖已有简历。
- 进度落盘到 data/scripts/.gen_progress.json，中断后重跑会自动从已完成份数继续。
- 单批异常不致命，记录后继续下一批。
"""

import io
import sys
import json
import time
import asyncio
import argparse
import sqlite3
import logging
from pathlib import Path

# 控制台 UTF-8，避免 Windows GBK 乱码
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.skills.resume_generator_skill import ResumeGeneratorSkill  # noqa: E402
from backend.config import DB_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gen_1000")

PROGRESS_FILE = Path(__file__).parent / ".gen_progress.json"
# study_abroad 模式使用独立进度文件，避免与普通批次进度互相覆盖
ABROAD_PROGRESS_FILE = Path(__file__).parent / ".gen_progress_abroad.json"


def _db_count() -> int:
    try:
        c = sqlite3.connect(str(DB_PATH))
        n = c.execute("select count(*) from candidates").fetchone()[0]
        c.close()
        return int(n)
    except Exception as e:
        logger.warning(f"读取 DB 计数失败: {e}")
        return 0


def _db_max_id() -> int:
    try:
        c = sqlite3.connect(str(DB_PATH))
        r = c.execute("select max(id) from candidates").fetchone()[0]
        c.close()
        return int(r or 0)
    except Exception as e:
        logger.warning(f"读取 DB max(id) 失败: {e}")
        return 0


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": 0, "all_candidate_ids": []}


def _save_progress(p: dict) -> None:
    try:
        PROGRESS_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"写进度文件失败: {e}")


async def main(total: int, batch: int, concurrency: int, study_abroad: bool = False) -> None:
    global PROGRESS_FILE
    if study_abroad:
        PROGRESS_FILE = ABROAD_PROGRESS_FILE
    skill = ResumeGeneratorSkill()

    progress = _load_progress()
    done = int(progress.get("done", 0))
    all_ids = list(progress.get("all_candidate_ids", []))

    start_count = _db_count()
    mode_str = "留学专属(study_abroad)" if study_abroad else "常规"
    logger.info(f"开始[{mode_str}]：目标 {total} 份，每批 {batch}，并发 {concurrency}")
    logger.info(f"断点续跑：已完成 {done} 份；当前 DB candidates={start_count}")

    t0 = time.time()
    while done < total:
        this_batch = min(batch, total - done)
        # 文件/邮箱编号偏移：按当前 DB max(id) 对齐，保证唯一不覆盖
        id_offset = _db_max_id()
        logger.info(f"--- 批次：生成 {this_batch} 份（已完成 {done}/{total}，id_offset={id_offset}）---")
        try:
            res = await skill.execute({
                "count": this_batch,
                "use_llm": True,
                "save_to_db": True,
                "save_text": True,
                "concurrency": concurrency,
                "id_offset": id_offset,
                "study_abroad": study_abroad,
            })
            got = res.get("extracted_to_db", 0)
            cids = res.get("candidate_ids", [])
            all_ids.extend(cids)
            done += this_batch
            progress["done"] = done
            progress["all_candidate_ids"] = all_ids
            _save_progress(progress)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            logger.info(
                f"批次完成：入库 {got} 份；累计 {done}/{total}；"
                f"DB candidates={_db_count()}；用时 {elapsed:.0f}s；ETA {eta:.0f}s"
            )
        except Exception as e:
            logger.error(f"批次异常（已完成 {done}），稍后重跑会自动续：{e}", exc_info=True)
            # 不抛出，等待短暂后继续，避免单批失败终止全程
            await asyncio.sleep(3)

    final_count = _db_count()
    logger.info("=" * 60)
    logger.info(f"全部完成：目标 {total} 份")
    logger.info(f"DB candidates：{start_count} -> {final_count}（新增 {final_count - start_count}）")
    logger.info(f"本轮成功入库 candidate_ids 数量：{len(all_ids)}")
    logger.info(f"总用时：{time.time() - t0:.0f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1000, help="总生成份数")
    ap.add_argument("--batch", type=int, default=50, help="每批份数")
    ap.add_argument("--concurrency", type=int, default=5, help="批内并发度")
    ap.add_argument("--reset", action="store_true", help="清空进度文件，从头开始计数")
    ap.add_argument("--study-abroad", action="store_true", help="生成留学专属简历（海外院校用 英文(中文)）")
    args = ap.parse_args()

    _pf = ABROAD_PROGRESS_FILE if args.study_abroad else PROGRESS_FILE
    if args.reset and _pf.exists():
        _pf.unlink()
        logger.info("已重置进度文件")

    asyncio.run(main(args.total, args.batch, args.concurrency, study_abroad=args.study_abroad))
