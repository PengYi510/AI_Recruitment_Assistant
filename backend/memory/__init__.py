"""双层长期记忆系统

Layer 1 - PersistentMemory (显式长期记忆):
    用户明确要求"记住"的规则/偏好，长期不变，仅当用户显式修改时才更新。
    例如："记住，QS前50等于985"、"以后搜索默认只看社招"

Layer 2 - AdaptiveMemory (智能自适应记忆):
    Agent 自动识别的重要信息，带有重要性评分和时间衰减机制。
    定期重新总结、调整分数、增删内容。
    例如：用户最近频繁搜索 Java 岗位、用户偏好看 SHAP 解释
"""

from backend.memory.persistent_memory import PersistentMemoryStore, persistent_memory
from backend.memory.adaptive_memory import AdaptiveMemoryStore, adaptive_memory
from backend.memory.memory_loader import MemoryLoader, memory_loader

__all__ = [
    "PersistentMemoryStore",
    "persistent_memory",
    "AdaptiveMemoryStore",
    "adaptive_memory",
    "MemoryLoader",
    "memory_loader",
]
