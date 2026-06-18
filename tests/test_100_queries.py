"""100个测试问题批量测试 - 20个简单 + 80个复杂
评估RAG检索系统的召回率和排序质量
"""
import sys, os, asyncio, json, time, io
from datetime import datetime
# 强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.models import hr_db
from backend.skills.rag_retrieval_skill import RAGRetrievalSkill

# ═══════════════════════════════════════════════════════════════════════════════
# 20 个简单查询 (单一约束)
# ═══════════════════════════════════════════════════════════════════════════════
SIMPLE_QUERIES = [
    # 单学校约束
    {"query": "浙大毕业的", "expect_constraint": {"school": "浙江大学"}},
    {"query": "清华大学的候选人", "expect_constraint": {"school": "清华大学"}},
    {"query": "北大毕业生", "expect_constraint": {"school": "北京大学"}},
    {"query": "成都大学的同学", "expect_constraint": {"school": "成都大学"}},
    {"query": "电子科大的", "expect_constraint": {"school": "电子科技大学"}},
    # 单年限约束
    {"query": "5年以上经验的", "expect_constraint": {"min_work_years": 5}},
    {"query": "至少10年工作经验", "expect_constraint": {"min_work_years": 10}},
    {"query": "3年以上经验", "expect_constraint": {"min_work_years": 3}},
    # 单技能约束
    {"query": "会Java的人", "expect_constraint": {"required_skills": ["Java"]}},
    {"query": "Python开发", "expect_constraint": {"required_skills": ["Python"]}},
    {"query": "会用Docker的", "expect_constraint": {"required_skills": ["Docker"]}},
    {"query": "Redis经验", "expect_constraint": {"required_skills": ["Redis"]}},
    {"query": "Spring框架开发者", "expect_constraint": {"required_skills": ["Spring"]}},
    # 单学历约束
    {"query": "硕士学历的", "expect_constraint": {"highest_education": "硕士"}},
    {"query": "博士候选人", "expect_constraint": {"highest_education": "博士"}},
    # 单岗位/方向约束（软约束，靠BM25匹配）
    {"query": "后端开发", "expect_constraint": {}},
    {"query": "前端工程师", "expect_constraint": {}},
    {"query": "算法工程师", "expect_constraint": {}},
    {"query": "产品经理", "expect_constraint": {}},
    {"query": "数据分析师", "expect_constraint": {}},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 80 个复杂查询 (多约束组合)
# ═══════════════════════════════════════════════════════════════════════════════
COMPLEX_QUERIES = [
    # 学校 + 年限
    {"query": "川大毕业的，3年以上后端开发经验的", "expect_constraint": {"school": "四川大学", "min_work_years": 3}},
    {"query": "北大5年以上工作经验的算法工程师", "expect_constraint": {"school": "北京大学", "min_work_years": 5}},
    {"query": "浙大毕业，至少8年Java经验", "expect_constraint": {"school": "浙江大学", "min_work_years": 8}},
    {"query": "哈工大10年以上经验的高级工程师", "expect_constraint": {"school": "哈尔滨工业大学", "min_work_years": 10}},
    {"query": "电子科大5年以上经验", "expect_constraint": {"school": "电子科技大学", "min_work_years": 5}},
    # 学校 + 学历
    {"query": "清华硕士", "expect_constraint": {"school": "清华大学", "highest_education": "硕士"}},
    {"query": "北大博士", "expect_constraint": {"school": "北京大学", "highest_education": "博士"}},
    {"query": "浙大本科毕业的", "expect_constraint": {"school": "浙江大学", "highest_education": "本科"}},
    {"query": "复旦硕士研究生", "expect_constraint": {"school": "复旦大学", "highest_education": "硕士"}},
    {"query": "武大硕士以上学历", "expect_constraint": {"school": "武汉大学", "highest_education": "硕士"}},
    # 学校 + 技能
    {"query": "浙大会Python的", "expect_constraint": {"school": "浙江大学", "required_skills": ["Python"]}},
    {"query": "北邮Java开发", "expect_constraint": {"school": "北京邮电大学", "required_skills": ["Java"]}},
    {"query": "成电会Docker的工程师", "expect_constraint": {"school": "电子科技大学", "required_skills": ["Docker"]}},
    {"query": "西电Redis经验", "expect_constraint": {"school": "西安电子科技大学", "required_skills": ["Redis"]}},
    {"query": "南大Spring开发", "expect_constraint": {"school": "南京大学", "required_skills": ["Spring"]}},
    # 年限 + 技能
    {"query": "5年以上Java后端", "expect_constraint": {"min_work_years": 5, "required_skills": ["Java"]}},
    {"query": "3年以上Python开发经验", "expect_constraint": {"min_work_years": 3, "required_skills": ["Python"]}},
    {"query": "至少7年Docker和Kubernetes经验", "expect_constraint": {"min_work_years": 7, "required_skills": ["Docker", "Kubernetes"]}},
    {"query": "10年以上分布式系统经验", "expect_constraint": {"min_work_years": 10}},
    {"query": "5年以上Redis MySQL经验", "expect_constraint": {"min_work_years": 5, "required_skills": ["Redis", "MySQL"]}},
    # 年限 + 学历
    {"query": "硕士5年以上经验", "expect_constraint": {"highest_education": "硕士", "min_work_years": 5}},
    {"query": "博士3年以上工作经验", "expect_constraint": {"highest_education": "博士", "min_work_years": 3}},
    {"query": "本科10年以上资深工程师", "expect_constraint": {"highest_education": "本科", "min_work_years": 10}},
    {"query": "硕士至少8年经验的架构师", "expect_constraint": {"highest_education": "硕士", "min_work_years": 8}},
    {"query": "博士5年以上算法经验", "expect_constraint": {"highest_education": "博士", "min_work_years": 5}},
    # 三重约束: 学校 + 年限 + 技能/方向
    {"query": "浙大毕业5年以上Java后端", "expect_constraint": {"school": "浙江大学", "min_work_years": 5, "required_skills": ["Java"]}},
    {"query": "川大3年以上Python算法工程师", "expect_constraint": {"school": "四川大学", "min_work_years": 3, "required_skills": ["Python"]}},
    {"query": "北大至少5年经验的数据分析师", "expect_constraint": {"school": "北京大学", "min_work_years": 5}},
    {"query": "成都大学3年以上前端React开发", "expect_constraint": {"school": "成都大学", "min_work_years": 3, "required_skills": ["React"]}},
    {"query": "杭电5年以上Go语言后端", "expect_constraint": {"school": "杭州电子科技大学", "min_work_years": 5, "required_skills": ["Go"]}},
    # 三重约束: 学校 + 学历 + 年限
    {"query": "清华硕士5年以上经验", "expect_constraint": {"school": "清华大学", "highest_education": "硕士", "min_work_years": 5}},
    {"query": "浙大博士3年以上研究经验", "expect_constraint": {"school": "浙江大学", "highest_education": "博士", "min_work_years": 3}},
    {"query": "北邮硕士至少3年工作经验", "expect_constraint": {"school": "北京邮电大学", "highest_education": "硕士", "min_work_years": 3}},
    {"query": "电子科大硕士5年以上", "expect_constraint": {"school": "电子科技大学", "highest_education": "硕士", "min_work_years": 5}},
    {"query": "西南交大本科7年以上后端", "expect_constraint": {"school": "西南交通大学", "highest_education": "本科", "min_work_years": 7}},
    # 四重约束: 学校 + 学历 + 年限 + 技能
    {"query": "浙大硕士3年以上Java Spring后端", "expect_constraint": {"school": "浙江大学", "highest_education": "硕士", "min_work_years": 3, "required_skills": ["Java", "Spring"]}},
    {"query": "北大博士5年以上PyTorch算法", "expect_constraint": {"school": "北京大学", "highest_education": "博士", "min_work_years": 5, "required_skills": ["PyTorch"]}},
    {"query": "川大硕士5年以上Redis后端开发", "expect_constraint": {"school": "四川大学", "highest_education": "硕士", "min_work_years": 5, "required_skills": ["Redis"]}},
    {"query": "清华本科8年以上分布式系统经验", "expect_constraint": {"school": "清华大学", "highest_education": "本科", "min_work_years": 8}},
    {"query": "复旦硕士3年以上Python数据分析", "expect_constraint": {"school": "复旦大学", "highest_education": "硕士", "min_work_years": 3, "required_skills": ["Python"]}},
    # 公司经验约束 (软约束，BM25匹配)
    {"query": "在美团工作过的后端", "expect_constraint": {}},
    {"query": "百度出来的算法工程师", "expect_constraint": {}},
    {"query": "有阿里工作经验的Java开发", "expect_constraint": {"required_skills": ["Java"]}},
    {"query": "腾讯3年以上经验的前端", "expect_constraint": {"min_work_years": 3}},
    {"query": "字节跳动出来的5年以上后端", "expect_constraint": {"min_work_years": 5}},
    {"query": "华为工作过的硕士", "expect_constraint": {"highest_education": "硕士"}},
    {"query": "小米的产品经理", "expect_constraint": {}},
    {"query": "网易游戏开发", "expect_constraint": {}},
    {"query": "拼多多做过数据的", "expect_constraint": {}},
    {"query": "bilibili的前端开发", "expect_constraint": {}},
    # 自然语言表达变体
    {"query": "帮我找一个后端大牛，最好10年以上经验", "expect_constraint": {"min_work_years": 10}},
    {"query": "我需要一个会Kubernetes的运维工程师", "expect_constraint": {"required_skills": ["Kubernetes"]}},
    {"query": "有没有做过推荐系统的算法同学", "expect_constraint": {}},
    {"query": "想找个全栈工程师，前后端都会的", "expect_constraint": {}},
    {"query": "给我推荐几个机器学习方向的博士", "expect_constraint": {"highest_education": "博士"}},
    {"query": "找个搞NLP的，至少5年经验", "expect_constraint": {"min_work_years": 5}},
    {"query": "有大厂经验的前端leader，8年以上", "expect_constraint": {"min_work_years": 8}},
    {"query": "技术总监级别的人选，15年以上", "expect_constraint": {"min_work_years": 15}},
    {"query": "刚毕业1-2年的应届生", "expect_constraint": {}},
    {"query": "中级Java开发，3到5年经验", "expect_constraint": {"min_work_years": 3, "required_skills": ["Java"]}},
    # 多技能组合
    {"query": "Java+Spring+MySQL+Redis的后端", "expect_constraint": {"required_skills": ["Java", "Spring", "MySQL", "Redis"]}},
    {"query": "Python+TensorFlow+PyTorch的算法", "expect_constraint": {"required_skills": ["Python", "PyTorch"]}},
    {"query": "React+TypeScript+Node.js全栈", "expect_constraint": {"required_skills": ["React", "TypeScript", "Node.js"]}},
    {"query": "Go+Docker+Kubernetes云原生开发", "expect_constraint": {"required_skills": ["Go", "Docker", "Kubernetes"]}},
    {"query": "Python+Spark+Kafka大数据工程师", "expect_constraint": {"required_skills": ["Python", "Kafka"]}},
    # 薪资相关（软约束）
    {"query": "月薪5万以上的高级后端", "expect_constraint": {}},
    {"query": "年薪百万级别的架构师", "expect_constraint": {}},
    {"query": "性价比高的3年Java开发", "expect_constraint": {"min_work_years": 3, "required_skills": ["Java"]}},
    # 综合复杂场景
    {"query": "浙大或者清华毕业的硕士，5年以上后端经验", "expect_constraint": {"highest_education": "硕士", "min_work_years": 5}},
    {"query": "985院校毕业的博士，做过深度学习", "expect_constraint": {"highest_education": "博士"}},
    {"query": "有美团或者阿里经验的后端，至少5年", "expect_constraint": {"min_work_years": 5}},
    {"query": "精通微服务架构的Java高级开发，8年以上", "expect_constraint": {"min_work_years": 8, "required_skills": ["Java"]}},
    {"query": "有大规模分布式系统经验的架构师", "expect_constraint": {}},
    {"query": "做过电商推荐系统的算法工程师", "expect_constraint": {}},
    {"query": "移动端iOS和Android都会的", "expect_constraint": {"required_skills": ["iOS", "Android"]}},
    {"query": "DevOps方向，熟悉CI/CD流水线", "expect_constraint": {}},
    {"query": "测试开发工程师，会自动化测试框架", "expect_constraint": {}},
    {"query": "UI设计师转产品经理的", "expect_constraint": {}},
    {"query": "南大硕士，做过gRPC微服务的后端开发", "expect_constraint": {"school": "南京大学", "highest_education": "硕士", "required_skills": ["gRPC"]}},
    {"query": "浙江工业大学5年以上经验的全栈工程师", "expect_constraint": {"school": "浙江工业大学", "min_work_years": 5}},
]

assert len(SIMPLE_QUERIES) == 20, f"Simple queries count: {len(SIMPLE_QUERIES)}"
assert len(COMPLEX_QUERIES) == 80, f"Complex queries count: {len(COMPLEX_QUERIES)}"


async def run_single_test(skill: RAGRetrievalSkill, query_info: dict, idx: int) -> dict:
    """运行单个测试查询"""
    query = query_info["query"]
    expected = query_info.get("expect_constraint", {})
    
    start_time = time.time()
    try:
        result = await skill.execute({"query": query, "top_k": 20})
        elapsed = time.time() - start_time
    except Exception as e:
        return {
            "idx": idx, "query": query, "status": "ERROR",
            "error": str(e), "elapsed": 0
        }
    
    detected = result.get("constraints_detected", {})
    candidates = result.get("candidates", [])
    constraint_count = result.get("constraint_matched_count", 0)
    
    # 评估约束提取准确性
    constraint_correct = True
    for key, val in expected.items():
        if key not in detected:
            constraint_correct = False
            break
        if key == "school" and val != detected.get(key):
            constraint_correct = False
            break
        if key == "min_work_years" and val != detected.get(key):
            constraint_correct = False
            break
        if key == "highest_education" and val != detected.get(key):
            constraint_correct = False
            break
    
    # 评估结果质量：检查top-10中满足硬约束的比例
    top10 = candidates[:10]
    constraint_match_in_top10 = sum(1 for c in top10 if c.get("constraint_match", False))
    
    return {
        "idx": idx,
        "query": query,
        "status": "OK",
        "elapsed": elapsed,
        "constraint_detected": detected,
        "constraint_expected": expected,
        "constraint_correct": constraint_correct,
        "total_found": len(candidates),
        "constraint_matched_total": constraint_count,
        "constraint_match_in_top10": constraint_match_in_top10,
        "top5_ids": [c["candidate_id"] for c in candidates[:5]],
        "top5_scores": [round(c["score"], 4) for c in candidates[:5]],
    }


async def run_all_tests():
    """批量运行所有100个测试"""
    skill = RAGRetrievalSkill()
    all_queries = SIMPLE_QUERIES + COMPLEX_QUERIES
    
    print(f"{'='*70}")
    print(f"开始批量测试 - 共 {len(all_queries)} 个查询")
    print(f"  简单查询: {len(SIMPLE_QUERIES)} 个")
    print(f"  复杂查询: {len(COMPLEX_QUERIES)} 个")
    print(f"{'='*70}")
    
    results = []
    start_total = time.time()
    
    for i, q in enumerate(all_queries):
        category = "简单" if i < 20 else "复杂"
        print(f"  [{i+1:3d}/100] [{category}] {q['query'][:40]:40s}", end="", flush=True)
        r = await run_single_test(skill, q, i + 1)
        results.append(r)
        
        if r["status"] == "OK":
            mark = "OK" if r["constraint_correct"] else "FAIL"
            print(f" | {mark:4s} | {r['elapsed']:.2f}s | matched={r['constraint_matched_total']:3d} | top10_hit={r['constraint_match_in_top10']}")
        else:
            print(f" | ERROR: {r['error'][:50]}")
    
    total_time = time.time() - start_total
    
    # ═══════════════════════════════════════════════════════════════════════
    # 统计分析
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("测试结果统计")
    print(f"{'='*70}")
    
    ok_results = [r for r in results if r["status"] == "OK"]
    error_results = [r for r in results if r["status"] == "ERROR"]
    
    print(f"\n总体:")
    print(f"  成功: {len(ok_results)}/100")
    print(f"  失败: {len(error_results)}/100")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  平均耗时: {total_time/100:.2f}s/query")
    
    # 约束提取准确率
    has_constraint = [r for r in ok_results if r["constraint_expected"]]
    constraint_correct = [r for r in has_constraint if r["constraint_correct"]]
    print(f"\n约束提取准确率:")
    print(f"  有预期约束的查询: {len(has_constraint)}")
    print(f"  正确提取: {len(constraint_correct)}/{len(has_constraint)} = {len(constraint_correct)/max(len(has_constraint),1)*100:.1f}%")
    
    # 约束匹配覆盖率（有硬约束时，top10里有多少命中了硬约束）
    has_match = [r for r in ok_results if r["constraint_matched_total"] > 0]
    if has_match:
        avg_top10_hit = sum(r["constraint_match_in_top10"] for r in has_match) / len(has_match)
        print(f"\n硬约束召回率 (有SQL预过滤的查询):")
        print(f"  有硬约束匹配的查询: {len(has_match)}")
        print(f"  平均top10命中数: {avg_top10_hit:.1f}")
        print(f"  top10至少有1个命中: {sum(1 for r in has_match if r['constraint_match_in_top10'] > 0)}/{len(has_match)}")
    
    # 简单 vs 复杂
    simple_ok = [r for r in ok_results if r["idx"] <= 20]
    complex_ok = [r for r in ok_results if r["idx"] > 20]
    
    if simple_ok:
        avg_simple = sum(r["elapsed"] for r in simple_ok) / len(simple_ok)
        print(f"\n简单查询 (1-20):")
        print(f"  平均耗时: {avg_simple:.2f}s")
        simple_correct = sum(1 for r in simple_ok if r["constraint_correct"])
        print(f"  约束提取正确: {simple_correct}/{len(simple_ok)}")
    
    if complex_ok:
        avg_complex = sum(r["elapsed"] for r in complex_ok) / len(complex_ok)
        print(f"\n复杂查询 (21-100):")
        print(f"  平均耗时: {avg_complex:.2f}s")
        complex_correct = sum(1 for r in complex_ok if r["constraint_correct"])
        print(f"  约束提取正确: {complex_correct}/{len(complex_ok)}")
    
    # 找出问题查询
    print(f"\n{'='*70}")
    print("约束提取失败的查询:")
    print(f"{'='*70}")
    failed = [r for r in ok_results if not r["constraint_correct"] and r["constraint_expected"]]
    for r in failed[:20]:
        print(f"  [{r['idx']:3d}] {r['query']}")
        print(f"       期望: {r['constraint_expected']}")
        print(f"       实际: {r['constraint_detected']}")
    
    # 保存完整结果到日志
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "logs", f"test_100_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_queries": 100,
            "total_time": total_time,
            "success_count": len(ok_results),
            "error_count": len(error_results),
            "constraint_accuracy": len(constraint_correct) / max(len(has_constraint), 1),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {log_path}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
