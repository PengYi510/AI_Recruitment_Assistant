"""Harness状态管理"""
import time, logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    REWORKING = "reworking"

@dataclass
class SubTask:
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    complexity: float = 0.0
    result: Any = None
    iterations: int = 0
    max_iterations: int = 3
    error_message: str = ""

@dataclass
class TaskState:
    task_id: str
    user_query: str
    complexity_score: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    sub_tasks: List[SubTask] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 3
    plan: Dict[str, Any] = field(default_factory=dict)
    generation_result: Any = None
    evaluation_result: Dict[str, Any] = field(default_factory=dict)
    final_result: Any = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    error_log: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "user_query": self.user_query,
                "complexity_score": self.complexity_score, "status": self.status.value,
                "current_iteration": self.current_iteration, "max_iterations": self.max_iterations}

    def add_history(self, event: str, detail: str = ""):
        self.history.append({"event": event, "detail": detail, "timestamp": time.time()})

    def mark_completed(self):
        self.status = TaskStatus.COMPLETED
        self.end_time = time.time()

    def mark_failed(self, error: str):
        self.status = TaskStatus.FAILED
        self.end_time = time.time()
        self.error_log.append(error)
