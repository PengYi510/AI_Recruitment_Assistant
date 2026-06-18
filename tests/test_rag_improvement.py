"""验证RAG检索改进效果 - 川大查询测试"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.models import hr_db
from backend.skills.rag_retrieval_skill import RAGRetrievalSkill


def check_database():
    """检查数据库中的川大毕业生"""
    print("=" * 60)
    print("数据库中的四川大学毕业生:")
    print("=" * 60)
    with hr_db._get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT eh.candidate_id, eh.school, c.work_years, 
                   c.education_level, c.name
            FROM education_history eh 
            JOIN candidates c ON c.id = eh.candidate_id 
            WHERE eh.school LIKE '%四川大学%'
            ORDER BY c.work_years DESC
        """).fetchall()
        for r in rows:
            print(f"  ID={r[0]:4d} | {r[4]:12s} | {r[1]} | {r[3]} | {r[2]}年经验")
        print(f"\n  共 {len(rows)} 人")
        
        # 3年以上的
        print("\n其中 3年以上经验的:")
        filtered = [r for r in rows if r[2] and r[2] >= 3]
        for r in filtered:
            print(f"  ID={r[0]:4d} | {r[4]:12s} | {r[2]}年 | {r[3]}")
        print(f"  共 {len(filtered)} 人满足年限要求")
    return rows


async def test_rag_retrieval():
    """测试RAG检索 - 川大毕业 + 3年以上后端"""
    skill = RAGRetrievalSkill()
    query = "找个川大毕业的，3年以上后端开发经验的"
    
    print("\n" + "=" * 60)
    print(f"测试查询: {query}")
    print("=" * 60)
    
    result = await skill.execute({"query": query, "top_k": 20})
    
    print(f"\n检测到的硬约束: {result['constraints_detected']}")
    print(f"硬约束匹配数量: {result['constraint_matched_count']}")
    print(f"返回候选人数: {result['total_found']}")
    
    print("\n排名结果 (Top 20):")
    print("-" * 90)
    for i, cand in enumerate(result['candidates'][:20]):
        cid = cand['candidate_id']
        data = cand.get('data', {})
        name = data.get('name', '?')
        years = data.get('work_years', 0)
        # 检查是否川大 - 从 education_history 和主表 school 字段
        edu_list = data.get('education_history', [])
        schools = [e.get('school', '') for e in edu_list]
        main_school = data.get('school', '')
        is_scu = any('四川大学' in s for s in schools) or '四川大学' in (main_school or '')
        marker = " ★川大★" if is_scu else ""
        constraint = " [硬约束匹配]" if cand.get('constraint_match') else ""
        print(f"  #{i+1:2d} | ID={cid:4d} | {name:12s} | {years}年 | score={cand['score']:.4f}{marker}{constraint}")
    
    # 检查川大候选人的排名
    print("\n\n川大候选人在结果中的位置:")
    scu_found = False
    for i, cand in enumerate(result['candidates']):
        cid = cand['candidate_id']
        data = cand.get('data', {})
        edu_list = data.get('education_history', [])
        schools = [e.get('school', '') for e in edu_list]
        main_school = data.get('school', '')
        if any('四川大学' in s for s in schools) or '四川大学' in (main_school or ''):
            scu_found = True
            name = data.get('name', '?')
            years = data.get('work_years', 0)
            print(f"  排名 #{i+1} | ID={cid} | {name} | {years}年 | score={cand['score']:.4f}")
    
    if not scu_found:
        print("  ⚠️ 未在结果中找到川大候选人!")


if __name__ == "__main__":
    check_database()
    asyncio.run(test_rag_retrieval())
