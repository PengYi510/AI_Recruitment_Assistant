"""BaseSkill - 所有Skill的基类"""
import logging, time
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BaseSkill(ABC):
    """Skill基类 - 统一接口、输入验证、错误处理、日志记录"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.execution_count = 0
        self.total_time = 0.0

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Any:
        """执行Skill逻辑 - 子类必须实现"""
        pass

    async def run(self, params: Dict[str, Any]) -> Any:
        """带验证和错误处理的执行入口"""
        start = time.time()
        try:
            self._validate_input(params)
            result = await self.execute(params)
            self.execution_count += 1
            elapsed = time.time() - start
            self.total_time += elapsed
            logger.info(f"[Skill:{self.name}] Completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"[Skill:{self.name}] Error: {e}")
            raise

    def _validate_input(self, params: Dict[str, Any]):
        """输入验证 - 子类可覆盖"""
        if not isinstance(params, dict):
            raise ValueError(f"Skill {self.name}: params must be dict")

    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        avg_time = self.total_time / max(self.execution_count, 1)
        return {"name": self.name, "executions": self.execution_count, "avg_time": round(avg_time, 3)}
