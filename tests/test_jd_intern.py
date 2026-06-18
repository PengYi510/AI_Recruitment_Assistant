"""测试JD长文本（实习生场景）的匹配效果"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import requests
import json

jd_text = (
    "HR数据挖掘岗（实习生） 实习-日常实习 人力资源平台 北京市 "
    "岗位职责 1. 参与HRAI底层数据体系建设，为AI技术在HR领域应用提供高质量数据，"
    "包括大规模数据开发、人才特征处理、训练/评测数据集构建等。 "
    "2. 负责HR相关数据分析及可视化，通过深入的数据分析与洞察，"
    "为公司HR和管理者的人才管理决策提供数据支撑。 "
    "3. 负责HR领域数据挖掘/机器学习算法研发，开发相应的智能化产品，提高人才管理决策效率。 "
    "4. 研究HR领域最前沿的AI大模型技术及产品，完成相关技术的开发并落地应用。 "
    "5. 与HR各个业务方协作，完成HRAI创新产品的探索和AICoding开发。 "
    "岗位基本要求 1. 2026届及以后的本科或研究生，计算机科学、软件工程、信息管理、统计学及相关专业。 "
    "2. 具有强悍的编程能力，熟练掌握Python、Java、C/C++至少一种编程能力，熟悉常用数据结构及算法。 "
    "3. 熟练使用SQL，熟悉Hive、MySQL等至少一种数据库。 "
    "4. 具有数据挖掘、机器学习、文本挖掘、自然语言处理、大模型等相关经验，熟悉Sklearn、PyTorch者优先。 "
    "5. 善于结构化思考，习惯于将复杂的问题结构化，并通过流程化的方式运营和解决。 "
    "6. 具有较强的学习和总结能力，良好的沟通技能、团队合作能力，可以承受一定的工作压力。"
)

print(f"JD长度: {len(jd_text)} 字")
print("=" * 60)

resp = requests.post('http://localhost:8003/chat', json={
    'session_id': 'test_intern_001',
    'message_id': 'msg_001',
    'emp_id': 'test_user',
    'query': jd_text
}, timeout=120)

result = resp.json()
print(f"HTTP Status: {resp.status_code}")
print("=" * 60)
print("Answer:")
print(result.get('answer', '')[:3000])
print("=" * 60)
print("Suggestions:", result.get('suggestions', []))
print("Steps:", json.dumps(result.get('steps', []), ensure_ascii=False, indent=2))
