"""附件测试问题端到端测评脚本。

流程：
1. 解析附件原文 -> 71 条 query（简单20 / 复杂31 / 虚构JD20 / 真实JD30）。
2. 逐条调用 Harness（与 /chat 同一执行内核），拿到自然语言回答 + 结构化候选人列表。
3. 对返回的候选人回查 SQLite，校验关键硬约束（学历层级/全日制/城市/技能/工作年限/学校等），
   判断"候选人是不是真的那么回事"。
4. 输出 JSON 报告 + 控制台摘要。

用法：
    python -m data.scripts.eval_attachment --attach "<附件路径>" --section simple --limit 5
    python -m data.scripts.eval_attachment --attach "<附件路径>"   # 全量
"""
from __future__ import annotations

import io
import re
import sys
import json
import time
import asyncio
import argparse
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import DB_PATH  # noqa: E402
from backend.harness.harness import harness as harness_controller  # noqa: E402
from agents.main_agent import _extract_candidates_list  # noqa: E402
from backend.skills.skill_registry import register_all_skills  # noqa: E402

# 关键：脱离 HTTP 服务直接调用 harness 时，必须先注册所有 skill，
# 否则 GeneratorAgent.get_skill 全部返回 None，结果恒为空。
register_all_skills()

REPORT_PATH = Path(__file__).parent / "eval_report.json"


# ─────────────────────────── 附件解析 ───────────────────────────
def parse_attachment(path: str) -> Dict[str, List[str]]:
    """把附件切分为四类 query。

    结构（依据实际附件）：
    - 行首 "一、简单查询" 起，1..20 单行问题
    - 随后一段 1..31 复杂查询（单行，31=筛选前3个候选人）
    - 随后 虚构JD：以 "数字. 标题" 起的多行块（共20个）
    - 随后 真实大厂JD：以 "数字. 标题" 起的多行块（共30个）
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    simple: List[str] = []
    complex_q: List[str] = []
    # 简单查询：第一段 1..20
    i = 0
    # 找到第一处以 "1. " 或 "1." 开头的单行
    single_re = re.compile(r"^\s*(\d{1,2})\s*[\.、]\s*(.+)$")
    # 收集所有"单行编号问题"段落（简单20在前，复杂31在后）
    single_blocks: List[List[str]] = []
    cur: List[str] = []
    in_jd = False
    for ln in lines:
        s = ln.strip()
        m = single_re.match(s)
        # JD 块的特征：编号标题后面跟"社招/校招/正式岗/部门介绍/岗位职责"等
        if m and not in_jd:
            num = int(m.group(1))
            body = m.group(2).strip()
            if num == 1 and cur:
                single_blocks.append(cur)
                cur = []
            cur.append(body)
        else:
            # 非编号行：若当前块已积累且遇到空白，暂不收尾（复杂查询块是连续的）
            pass
    if cur:
        single_blocks.append(cur)

    # single_blocks 里：第一块=简单20，第二块=复杂31（可能 JD 标题也被误并入，靠数量/内容裁剪）
    if single_blocks:
        simple = single_blocks[0][:20]
    if len(single_blocks) > 1:
        complex_q = single_blocks[1][:31]

    # ── JD 块解析（虚构20 + 真实30）──────────────────────────────
    # JD 区在"复杂查询"之后。用复杂查询最后一条作为锚点，从其后开始扫描 JD，
    # 避免把前面的"简单/复杂查询"单行编号误判为 JD 标题。
    jd_start_line = 0
    if complex_q:
        anchor = complex_q[-1].strip()  # 如 "筛选前3个候选人"
        for li, ln in enumerate(lines):
            if anchor and anchor in ln.strip():
                jd_start_line = li + 1
                break
    jd_text = "\n".join(lines[jd_start_line:]) if jd_start_line else text
    jd_blocks = _parse_jd_blocks(jd_text)
    fake_jd = jd_blocks[:20]
    real_jd = jd_blocks[20:50]

    return {
        "simple": simple,
        "complex": complex_q,
        "fake_jd": fake_jd,
        "real_jd": real_jd,
    }


def _parse_jd_blocks(text: str) -> List[str]:
    """提取所有 JD 文本块。JD 以 '数字. 岗位标题' 开头，块内含部门/城市/职责/要求等多行。"""
    lines = text.split("\n")
    blocks: List[str] = []
    cur: List[str] = []
    title_re = re.compile(r"^\s*\d{1,2}\s*[\.、]\s*\S+")
    query_verbs = ("找", "筛选", "查找", "推荐", "寻找")
    # JD 头部锚：标题后第一个非空行的"招聘性质"标识（区别于岗位职责/要求里的编号行）
    # 实际文本形如：社招-正式岗 / 校招-正式岗 / 正式岗-社招 / 实习-日常实习 / 实习-暑期实习
    head_anchor = ("正式岗", "社招", "校招", "实习", "兼职", "日常实习", "暑期实习")
    # 职位名特征词：真正的 JD 标题正文一定包含其一（"岗位职责/要求"内部编号行是描述句，不含）
    job_words = ("工程师", "专员", "经理", "实习生", "分析师", "设计师", "HRBP",
                 "BP", "顾问", "主管", "总监", "架构师", "研发", "运营", "产品",
                 "助理", "专家", "讲师", "导师")

    def looks_like_jd_start(idx: int) -> bool:
        title = lines[idx].strip()
        if not title_re.match(title):
            return False
        body = title.split(".", 1)[-1].split("、", 1)[-1].strip()
        # 查询句排除：以查询动词开头且不含职位词（避免误杀"推荐算法工程师"这类岗位名）
        if any(body.startswith(v) for v in query_verbs) and not any(w in body for w in job_words):
            return False  # 查询句，非岗位标题
        # 核心判据：JD 标题是"岗位名"——较短、不是长句、且包含职位特征词。
        # 岗位职责/要求/亮点段内部的编号行是描述长句（含逗号/句号、长度大），予以排除。
        if len(body) > 25:
            return False
        if any(p in body for p in ("，", "。", "、", "；")):
            return False
        if not any(w in body for w in job_words):
            return False
        # 双保险：标题后第一个非空行通常是招聘性质标识（实习/社招/校招/正式岗…）
        for k in range(idx + 1, min(idx + 5, len(lines))):
            nxt = lines[k].strip()
            if not nxt:
                continue
            return any(a in nxt for a in head_anchor)
        return False

    i = 0
    n = len(lines)
    while i < n:
        if looks_like_jd_start(i):
            # 收集到下一个 JD 起点或文件尾
            j = i + 1
            block = [lines[i].strip()]
            while j < n and not looks_like_jd_start(j):
                block.append(lines[j])
                j += 1
            blocks.append("\n".join([b for b in block]).strip())
            i = j
        else:
            i += 1
    return blocks


# ─────────────────────────── DB 回查 ───────────────────────────
class DB:
    def __init__(self):
        self.c = sqlite3.connect(str(DB_PATH))
        self.c.row_factory = sqlite3.Row

    def candidate(self, cid: int) -> Optional[Dict[str, Any]]:
        r = self.c.execute("select * from candidates where id=?", (cid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["skills"] = [x["skill_name"] for x in
                       self.c.execute("select skill_name from skills where candidate_id=?", (cid,)).fetchall()]
        d["educations"] = [dict(x) for x in
                           self.c.execute("select degree,school,major,is_fulltime,school_tier,start_date,end_date "
                                          "from education_history where candidate_id=?", (cid,)).fetchall()]
        return d

    def close(self):
        self.c.close()


# ─────────────────────────── 执行单条 query ───────────────────────────
def run_one(query: str, idx_tag: str) -> Dict[str, Any]:
    context = {"session_id": f"eval_{idx_tag}", "emp_id": "eval_bot",
               "memory_context": "", "active_entities": {}}
    loop = asyncio.new_event_loop()
    try:
        hr = loop.run_until_complete(harness_controller.execute(query, context))
    finally:
        loop.close()
    final = hr.get("result", {}) if hr.get("success") else {}
    cands = _extract_candidates_list(final) if final else []
    norm = []
    for c in cands[:10]:
        data = c.get("data", c) if isinstance(c, dict) else {}
        cid = data.get("id") or c.get("candidate_id")
        norm.append({"id": cid, "name": data.get("name", ""),
                     "score": c.get("score", c.get("match_score", 0))})
    return {
        "success": hr.get("success", False),
        "error": hr.get("error", ""),
        "candidate_count": len(cands),
        "top": norm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attach", required=True, help="附件路径")
    ap.add_argument("--section", default="all",
                    choices=["all", "simple", "complex", "fake_jd", "real_jd"])
    ap.add_argument("--limit", type=int, default=0, help="每类只跑前 N 条（0=全部）")
    args = ap.parse_args()

    parsed = parse_attachment(args.attach)
    print("解析结果：", {k: len(v) for k, v in parsed.items()})

    sections = ["simple", "complex", "fake_jd", "real_jd"] if args.section == "all" else [args.section]
    db = DB()
    report: Dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sections": {}}

    for sec in sections:
        queries = parsed[sec]
        if args.limit > 0:
            queries = queries[:args.limit]
        print(f"\n========== 区块 {sec}（{len(queries)} 条）==========")
        sec_results = []
        for qi, q in enumerate(queries):
            tag = f"{sec}_{qi+1}"
            preview = q.replace("\n", " ")[:40]
            t0 = time.time()
            try:
                res = run_one(q, tag)
            except Exception as e:
                res = {"success": False, "error": f"EXC: {e}", "candidate_count": 0, "top": []}
            dt = time.time() - t0
            # 回查 DB：补充 TOP 候选人结构化字段
            for t in res["top"]:
                if t["id"]:
                    info = db.candidate(int(t["id"]))
                    if info:
                        t["db"] = {
                            "education_level": info["education_level"],
                            "school": info["school"],
                            "job_status": info["job_status"],
                            "work_years": info["work_years"],
                            "skills": info["skills"][:15],
                            "educations": info["educations"],
                        }
            print(f"[{tag}] q='{preview}' -> ok={res['success']} "
                  f"命中={res['candidate_count']} top_ids={[t['id'] for t in res['top'][:5]]} "
                  f"({dt:.1f}s)", flush=True)
            sec_results.append({"query": q, "result": res, "latency_s": round(dt, 1)})
            # 增量落盘，避免中途中断丢失已完成结果
            report["sections"][sec] = sec_results
            REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["sections"][sec] = sec_results

    db.close()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {REPORT_PATH}")


if __name__ == "__main__":
    main()
