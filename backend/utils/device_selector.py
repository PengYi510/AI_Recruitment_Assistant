"""GPU/CPU 设备自动选择工具模块

优先级策略：
  GPU:0 → GPU:1 → GPU:2 → ... → CPU

选择逻辑：
  1. 检查 torch 是否可用且 CUDA 可用
  2. 遍历所有 GPU 设备（从 GPU:0 开始），选择第一个有足够
     可用显存（默认阈值 512MB）的 GPU
  3. 如果所有 GPU 显存都不足，或没有 GPU，降级到 CPU
  4. 选择结果缓存，后续调用直接返回（除非手动刷新）

用法：
    from backend.utils.device_selector import get_device, get_device_str
    device = get_device()            # -> torch.device 对象
    device_str = get_device_str()    # -> "cuda:0" / "cpu" 字符串
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 缓存
_cached_device = None
_cached_device_str: Optional[str] = None

# 最低可用显存阈值（字节），默认 512 MB
MIN_FREE_MEMORY_BYTES = 512 * 1024 * 1024


def _select_device(min_free: int = MIN_FREE_MEMORY_BYTES) -> str:
    """遍历所有 GPU，返回第一个有足够显存的设备字符串；都不满足则返回 'cpu'。"""
    try:
        import torch
    except ImportError:
        logger.warning("[DeviceSelector] PyTorch 未安装，使用 CPU")
        return "cpu"

    if not torch.cuda.is_available():
        logger.info("[DeviceSelector] CUDA 不可用，使用 CPU")
        return "cpu"

    gpu_count = torch.cuda.device_count()
    logger.info(f"[DeviceSelector] 检测到 {gpu_count} 个 GPU 设备")

    for idx in range(gpu_count):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(idx)
            free_mb = free_mem / (1024 * 1024)
            total_mb = total_mem / (1024 * 1024)
            logger.info(f"[DeviceSelector] GPU:{idx} — "
                        f"空闲 {free_mb:.0f}MB / 总计 {total_mb:.0f}MB")
            if free_mem >= min_free:
                device_str = f"cuda:{idx}"
                logger.info(f"[DeviceSelector] ✓ 选择 {device_str}"
                            f"（空闲 {free_mb:.0f}MB ≥ 阈值 {min_free / 1024 / 1024:.0f}MB）")
                return device_str
            else:
                logger.warning(f"[DeviceSelector] ✗ GPU:{idx} 显存不足"
                               f"（空闲 {free_mb:.0f}MB < 阈值 {min_free / 1024 / 1024:.0f}MB），跳过")
        except Exception as e:
            logger.warning(f"[DeviceSelector] GPU:{idx} 查询显存失败: {e}，跳过")

    logger.info("[DeviceSelector] 所有 GPU 显存不足或不可用，降级到 CPU")
    return "cpu"


def get_device_str(refresh: bool = False) -> str:
    """返回设备字符串（如 'cuda:0' 或 'cpu'），结果会被缓存。

    Args:
        refresh: 为 True 时重新检测设备，忽略缓存
    """
    global _cached_device_str
    if _cached_device_str is not None and not refresh:
        return _cached_device_str
    _cached_device_str = _select_device()
    return _cached_device_str


def get_device(refresh: bool = False):
    """返回 torch.device 对象。

    Args:
        refresh: 为 True 时重新检测设备，忽略缓存

    Returns:
        torch.device 对象（若 torch 未安装则返回字符串 'cpu'）
    """
    global _cached_device
    if _cached_device is not None and not refresh:
        return _cached_device

    device_str = get_device_str(refresh=refresh)
    try:
        import torch
        _cached_device = torch.device(device_str)
    except ImportError:
        _cached_device = device_str  # fallback: 返回纯字符串
    return _cached_device


def reset_cache():
    """清除缓存，下次调用 get_device() 时重新检测。"""
    global _cached_device, _cached_device_str
    _cached_device = None
    _cached_device_str = None
