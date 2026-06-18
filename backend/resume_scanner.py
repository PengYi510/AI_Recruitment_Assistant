"""简历自动扫描入库后台线程

功能:
- 定时扫描 resume_data/ 文件夹中的新简历文件
- 执行时间: 每天 0:00, 4:00, 8:00, 12:00, 16:00, 20:00
- 支持格式: PDF, DOCX, MD, TXT, PPTX
- 文件命名规范: 用户名年月日_小时_分钟_秒_用户id_xxx_xxxx.pdf/md/docx/pptx/txt
- 按周创建子文件夹: 开始日期——结束日期
- 记录上次入库时间戳，系统离线后恢复时自动补扫

开发者可修改的 LLM 配置在文件顶部。
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

# ══════════════════════════════════════════════════════════════════════════════
# 【开发者可修改区域】LLM API 配置
# 修改以下变量即可切换 LLM 服务（当前使用 LongCat API）
# ══════════════════════════════════════════════════════════════════════════════
from backend.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, PROJECT_ROOT

# 如需覆盖 config.py 中的默认值，取消注释以下行并填入新值:
# LLM_API_KEY = "your-api-key-here"
# LLM_BASE_URL = "https://your-llm-api-endpoint/v1/openai/native"
# LLM_MODEL = "your-model-name"

# ══════════════════════════════════════════════════════════════════════════════
# 扫描配置
# ══════════════════════════════════════════════════════════════════════════════
SCAN_HOURS = [0, 4, 8, 12, 16, 20]  # 每天执行扫描的小时
RESUME_DATA_DIR = PROJECT_ROOT / "resume_data"
SCANNER_STATE_FILE = PROJECT_ROOT / "data" / "scanner_state.json"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".pptx"}

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 周文件夹管理
# ══════════════════════════════════════════════════════════════════════════════

def get_week_folder_name(dt: datetime) -> str:
    """根据日期获取所属周文件夹名称: 开始日期——结束日期
    
    周一为一周的开始，周日为结束。
    """
    weekday = dt.weekday()  # 0=Monday
    week_start = dt - timedelta(days=weekday)
    week_end = week_start + timedelta(days=6)
    return f"{week_start.strftime('%Y%m%d')}——{week_end.strftime('%Y%m%d')}"


def ensure_resume_data_dir() -> Path:
    """确保 resume_data 目录存在，并创建当前周的子文件夹"""
    RESUME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建当前周的子文件夹
    now = datetime.now()
    week_folder = RESUME_DATA_DIR / get_week_folder_name(now)
    week_folder.mkdir(parents=True, exist_ok=True)
    
    return RESUME_DATA_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 扫描状态持久化
# ══════════════════════════════════════════════════════════════════════════════

class ScannerState:
    """管理扫描器状态（上次扫描时间、已处理文件列表）"""
    
    def __init__(self):
        self.state_file = SCANNER_STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """从文件加载状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Scanner state file corrupted, resetting.")
        return {
            "last_scan_time": None,
            "processed_files": [],  # 已成功入库的文件路径列表
            "failed_files": [],     # 入库失败的文件路径列表
            "scan_history": []      # 扫描历史记录
        }
    
    def _save(self):
        """保存状态到文件"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Failed to save scanner state: {e}")
    
    @property
    def last_scan_time(self) -> Optional[str]:
        return self._state.get("last_scan_time")
    
    @last_scan_time.setter
    def last_scan_time(self, value: str):
        self._state["last_scan_time"] = value
        self._save()
    
    @property
    def processed_files(self) -> List[str]:
        return self._state.get("processed_files", [])
    
    @property
    def failed_files(self) -> List[str]:
        return self._state.get("failed_files", [])
    
    def mark_processed(self, file_path: str):
        """标记文件为已处理"""
        if file_path not in self._state["processed_files"]:
            self._state["processed_files"].append(file_path)
        # 从失败列表中移除（如果之前失败过）
        if file_path in self._state["failed_files"]:
            self._state["failed_files"].remove(file_path)
        self._save()
    
    def mark_failed(self, file_path: str):
        """标记文件为处理失败"""
        if file_path not in self._state["failed_files"]:
            self._state["failed_files"].append(file_path)
        self._save()
    
    def is_processed(self, file_path: str) -> bool:
        """检查文件是否已处理"""
        return file_path in self._state["processed_files"]
    
    def add_scan_record(self, record: Dict[str, Any]):
        """添加扫描历史记录"""
        self._state.setdefault("scan_history", [])
        self._state["scan_history"].append(record)
        # 只保留最近100条记录
        if len(self._state["scan_history"]) > 100:
            self._state["scan_history"] = self._state["scan_history"][-100:]
        self._save()
    
    @property
    def scan_history(self) -> List[Dict[str, Any]]:
        return self._state.get("scan_history", [])


# ══════════════════════════════════════════════════════════════════════════════
# 简历扫描与入库核心逻辑
# ══════════════════════════════════════════════════════════════════════════════

def discover_new_resumes(state: ScannerState) -> List[Path]:
    """扫描 resume_data 目录，发现未处理的简历文件"""
    new_files = []
    
    if not RESUME_DATA_DIR.exists():
        return new_files
    
    # 递归扫描所有子文件夹
    for file_path in RESUME_DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # 跳过隐藏文件和临时文件
        if file_path.name.startswith(".") or file_path.name.startswith("~"):
            continue
        # 检查是否已处理
        rel_path = str(file_path.relative_to(PROJECT_ROOT))
        if not state.is_processed(rel_path):
            new_files.append(file_path)
    
    # 按修改时间排序（先处理旧文件）
    new_files.sort(key=lambda f: f.stat().st_mtime)
    return new_files


async def ingest_single_resume(file_path: Path) -> Dict[str, Any]:
    """入库单个简历文件
    
    流程: 文件 → 文本提取 → LLM结构化抽取 → 数据库 + 向量库
    """
    from backend.skills.resume_extraction_skill import ResumeExtractionSkill
    
    skill = ResumeExtractionSkill()
    
    try:
        result = await skill.execute({
            "file_path": str(file_path),
            "save_to_db": True
        })
        return result
    except Exception as e:
        logger.error(f"Failed to ingest resume {file_path}: {e}")
        return {"success": False, "error": str(e)}


async def run_scan_cycle(state: ScannerState) -> Dict[str, Any]:
    """执行一次完整的扫描入库周期"""
    scan_start = datetime.now()
    logger.info(f"[ResumeScanner] 开始扫描周期: {scan_start.isoformat()}")
    
    # 确保目录结构存在
    ensure_resume_data_dir()
    
    # 发现新文件
    new_files = discover_new_resumes(state)
    
    if not new_files:
        logger.info("[ResumeScanner] 没有发现新的简历文件")
        state.last_scan_time = scan_start.isoformat()
        state.add_scan_record({
            "time": scan_start.isoformat(),
            "found": 0,
            "success": 0,
            "failed": 0
        })
        return {"found": 0, "success": 0, "failed": 0}
    
    logger.info(f"[ResumeScanner] 发现 {len(new_files)} 个新简历文件")
    
    success_count = 0
    failed_count = 0
    
    for file_path in new_files:
        rel_path = str(file_path.relative_to(PROJECT_ROOT))
        logger.info(f"[ResumeScanner] 正在处理: {file_path.name}")
        
        try:
            result = await ingest_single_resume(file_path)
            if result.get("success"):
                state.mark_processed(rel_path)
                success_count += 1
                logger.info(f"[ResumeScanner] ✅ 入库成功: {file_path.name} (candidate_id={result.get('candidate_id')})")
            else:
                state.mark_failed(rel_path)
                failed_count += 1
                logger.warning(f"[ResumeScanner] ❌ 入库失败: {file_path.name} - {result.get('error')}")
        except Exception as e:
            state.mark_failed(rel_path)
            failed_count += 1
            logger.error(f"[ResumeScanner] ❌ 处理异常: {file_path.name} - {e}")
    
    # 更新状态
    state.last_scan_time = scan_start.isoformat()
    state.add_scan_record({
        "time": scan_start.isoformat(),
        "found": len(new_files),
        "success": success_count,
        "failed": failed_count
    })
    
    logger.info(f"[ResumeScanner] 扫描完成: 发现={len(new_files)}, 成功={success_count}, 失败={failed_count}")
    return {"found": len(new_files), "success": success_count, "failed": failed_count}


# ══════════════════════════════════════════════════════════════════════════════
# 后台调度线程
# ══════════════════════════════════════════════════════════════════════════════

class ResumeScannerThread(threading.Thread):
    """后台简历扫描线程（daemon 模式）
    
    在指定时间点（0/4/8/12/16/20时）自动执行扫描。
    启动时如果发现上次扫描时间距今超过4小时，立即执行一次补扫。
    """
    
    def __init__(self):
        super().__init__(daemon=True, name="ResumeScannerThread")
        self.state = ScannerState()
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def stop(self):
        """停止扫描线程"""
        self._stop_event.set()
    
    def _should_scan_now(self) -> bool:
        """判断当前是否应该执行扫描"""
        now = datetime.now()
        current_hour = now.hour
        
        # 检查是否在扫描时间点（允许5分钟误差）
        if current_hour in SCAN_HOURS and now.minute < 5:
            return True
        
        return False
    
    def _need_catchup_scan(self) -> bool:
        """判断是否需要补扫（系统离线后恢复）"""
        last_scan = self.state.last_scan_time
        if not last_scan:
            return True  # 从未扫描过
        
        try:
            last_dt = datetime.fromisoformat(last_scan)
            hours_since = (datetime.now() - last_dt).total_seconds() / 3600
            return hours_since >= 4  # 超过4小时未扫描
        except (ValueError, TypeError):
            return True
    
    def _run_async_scan(self):
        """在独立事件循环中运行异步扫描"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_scan_cycle(self.state))
            return result
        except Exception as e:
            logger.error(f"[ResumeScannerThread] 扫描异常: {e}")
            return None
        finally:
            loop.close()
    
    def run(self):
        """线程主循环"""
        logger.info("[ResumeScannerThread] 后台简历扫描线程已启动")
        logger.info(f"[ResumeScannerThread] 扫描目录: {RESUME_DATA_DIR}")
        logger.info(f"[ResumeScannerThread] 扫描时间: 每天 {SCAN_HOURS} 时")
        
        # 确保目录存在
        ensure_resume_data_dir()
        
        # 启动时检查是否需要补扫
        if self._need_catchup_scan():
            logger.info("[ResumeScannerThread] 检测到需要补扫（系统可能离线过），立即执行...")
            self._run_async_scan()
        
        # 主循环：每60秒检查一次是否到了扫描时间
        last_scan_hour = -1
        while not self._stop_event.is_set():
            now = datetime.now()
            
            # 避免同一小时重复扫描
            if self._should_scan_now() and now.hour != last_scan_hour:
                logger.info(f"[ResumeScannerThread] 到达扫描时间 {now.hour}:00，开始扫描...")
                self._run_async_scan()
                last_scan_hour = now.hour
            
            # 每60秒检查一次
            self._stop_event.wait(timeout=60)
        
        logger.info("[ResumeScannerThread] 后台简历扫描线程已停止")


# ══════════════════════════════════════════════════════════════════════════════
# 手动触发接口（供管理面板调用）
# ══════════════════════════════════════════════════════════════════════════════

async def manual_scan() -> Dict[str, Any]:
    """手动触发一次扫描（供管理面板使用）"""
    state = ScannerState()
    return await run_scan_cycle(state)


async def manual_ingest_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """手动入库指定文件列表（供管理面板使用）"""
    state = ScannerState()
    results = []
    
    for fp in file_paths:
        file_path = Path(fp)
        if not file_path.exists():
            results.append({"file": fp, "success": False, "error": "文件不存在"})
            continue
        
        result = await ingest_single_resume(file_path)
        if result.get("success"):
            rel_path = str(file_path.relative_to(PROJECT_ROOT)) if file_path.is_relative_to(PROJECT_ROOT) else fp
            state.mark_processed(rel_path)
        
        results.append({"file": fp, **result})
    
    return results


def get_pending_resumes() -> List[Dict[str, Any]]:
    """获取待入库的简历文件列表"""
    state = ScannerState()
    pending = []
    
    if not RESUME_DATA_DIR.exists():
        return pending
    
    for file_path in RESUME_DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.name.startswith(".") or file_path.name.startswith("~"):
            continue
        
        rel_path = str(file_path.relative_to(PROJECT_ROOT))
        if not state.is_processed(rel_path):
            stat = file_path.stat()
            pending.append({
                "path": str(file_path),
                "rel_path": rel_path,
                "name": file_path.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_path.suffix.lower(),
                "failed": rel_path in state.failed_files
            })
    
    return pending


def get_scanner_status() -> Dict[str, Any]:
    """获取扫描器状态信息"""
    state = ScannerState()
    return {
        "last_scan_time": state.last_scan_time,
        "processed_count": len(state.processed_files),
        "failed_count": len(state.failed_files),
        "pending_count": len(get_pending_resumes()),
        "scan_hours": SCAN_HOURS,
        "resume_data_dir": str(RESUME_DATA_DIR),
        "recent_history": state.scan_history[-10:] if state.scan_history else []
    }


# ══════════════════════════════════════════════════════════════════════════════
# 全局线程实例
# ══════════════════════════════════════════════════════════════════════════════

_scanner_thread: Optional[ResumeScannerThread] = None


def start_scanner_thread():
    """启动后台扫描线程（幂等，重复调用不会创建多个线程）"""
    global _scanner_thread
    if _scanner_thread is not None and _scanner_thread.is_alive():
        logger.info("[ResumeScanner] 扫描线程已在运行，跳过")
        return
    
    _scanner_thread = ResumeScannerThread()
    _scanner_thread.start()
    logger.info("[ResumeScanner] 后台扫描线程已启动")


def stop_scanner_thread():
    """停止后台扫描线程"""
    global _scanner_thread
    if _scanner_thread is not None and _scanner_thread.is_alive():
        _scanner_thread.stop()
        _scanner_thread.join(timeout=5)
        logger.info("[ResumeScanner] 后台扫描线程已停止")
    _scanner_thread = None
