"""BLIP 视觉编码器 - 真实图像语义特征提取

为多模态分层融合提供真实的视觉特征（替换早期的哈希模拟）。

模型选择与回退策略：
  1. 首选 BLIP-3 (Salesforce/xgen-mm-phi3-mini-instruct-r-v1)；
     但其官方远程代码基于旧版 transformers API，在 transformers>=5.x 上
     AutoModel 工厂无法识别 XGenMMConfig，故无法直接加载。
  2. 回退到真 BLIP 家族视觉编码器 BLIP-base
     (Salesforce/blip-image-captioning-base) 的原生 BlipVisionModel，
     hidden_size=768，正好等于 IMAGE_EMBEDDING_DIM，无需额外投影。
  3. 若运行环境无法加载任何视觉模型（缺依赖/无网络/无权重），
     则降级为确定性哈希向量，并将 USE_REAL_IMAGE 置为 False，
     供上层与实验数据如实标注。

特征带磁盘缓存（按图片绝对路径的 mtime+size 生成 key），
因证书图种类有限，缓存命中率极高，离线实验与在线服务均可复用。
"""
import os
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from backend.config import IMAGE_EMBEDDING_DIM, PROJECT_ROOT

logger = logging.getLogger(__name__)

# HuggingFace 镜像（中国大陆可达）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 模型标识
_BLIP_BASE_ID = "Salesforce/blip-image-captioning-base"

# ModelScope 本地缓存路径（国内下载稳定，仅含 pytorch_model.bin，无 safetensors）
_MODELSCOPE_LOCAL = os.path.join(
    os.path.expanduser("~"), ".cache", "modelscope", "hub", "models",
    "Salesforce", "blip-image-captioning-base",
)


def _resolve_model_source() -> str:
    """优先返回本地 ModelScope 缓存目录（若权重已下载），否则返回 HF repo id。"""
    bin_path = os.path.join(_MODELSCOPE_LOCAL, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return _MODELSCOPE_LOCAL
    return _BLIP_BASE_ID

# 特征磁盘缓存目录
_FEATURE_CACHE_DIR = PROJECT_ROOT / "data" / "image_feature_cache"
_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 运行时状态 ────────────────────────────────────────────────────────────────
_vision_model = None
_processor = None
_torch = None
_LOAD_ATTEMPTED = False
USE_REAL_IMAGE = False          # 是否成功加载真实视觉模型
ACTIVE_IMAGE_MODEL = "hash"     # "blip-base" / "hash"


def _try_load_model() -> bool:
    """延迟加载 BLIP-base 视觉编码器（仅尝试一次）。"""
    global _vision_model, _processor, _torch, _LOAD_ATTEMPTED
    global USE_REAL_IMAGE, ACTIVE_IMAGE_MODEL

    if _LOAD_ATTEMPTED:
        return USE_REAL_IMAGE
    _LOAD_ATTEMPTED = True

    try:
        import torch
        from transformers import BlipForConditionalGeneration, AutoProcessor
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[BLIP] torch/transformers 不可用，图像特征降级为哈希: {e}")
        return False

    src = _resolve_model_source()
    try:
        logger.info(f"[BLIP] 正在加载完整 BLIP 模型并取其视觉编码器: {src}")
        # 加载完整 BLIP（权重键前缀匹配），再取 vision_model 子模块；
        # 直接加载 BlipVisionModel 会因键前缀(vision_model.*)不匹配而全部随机初始化。
        full = BlipForConditionalGeneration.from_pretrained(
            src, dtype=torch.float32, low_cpu_mem_usage=True,
            use_safetensors=False,
        )
        model = full.vision_model
        model.eval()
        proc = AutoProcessor.from_pretrained(src)
        _vision_model = model
        _processor = proc
        _torch = torch
        USE_REAL_IMAGE = True
        ACTIVE_IMAGE_MODEL = "blip-base"
        hidden = getattr(model.config, "hidden_size", "?")
        logger.info(f"[BLIP] 视觉编码器加载成功(预训练权重) hidden_size={hidden} (source={src})")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[BLIP] 视觉编码器加载失败，图像特征降级为哈希: {e}")
        return False


def _hash_feature(image_path: str) -> np.ndarray:
    """确定性哈希向量（降级方案，与历史行为一致）。"""
    np.random.seed(hash(image_path) % 2**31)
    feat = np.random.randn(1, IMAGE_EMBEDDING_DIM).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat


def _cache_key(image_path: str) -> str:
    """基于路径 + mtime + size 的缓存 key。"""
    try:
        st = os.stat(image_path)
        sig = f"{os.path.abspath(image_path)}|{int(st.st_mtime)}|{st.st_size}|blip-base"
    except OSError:
        sig = f"{image_path}|missing|blip-base"
    return hashlib.md5(sig.encode("utf-8")).hexdigest()


def _project_to_dim(vec: np.ndarray) -> np.ndarray:
    """对齐到 IMAGE_EMBEDDING_DIM 并 L2 归一化。"""
    vec = vec.reshape(1, -1).astype(np.float32)
    d = vec.shape[1]
    if d > IMAGE_EMBEDDING_DIM:
        vec = vec[:, :IMAGE_EMBEDDING_DIM]
    elif d < IMAGE_EMBEDDING_DIM:
        vec = np.pad(vec, ((0, 0), (0, IMAGE_EMBEDDING_DIM - d)))
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def encode_image(image_path: str) -> np.ndarray:
    """提取图像语义特征，返回 (1, IMAGE_EMBEDDING_DIM)。

    优先用真实 BLIP 视觉编码器；不可用或图片缺失时回退哈希。
    成功的真实特征会写磁盘缓存。
    """
    if not image_path:
        return _hash_feature("")

    # 图片不存在 -> 回退哈希（按路径确定性，保持实验可复现）
    if not os.path.exists(image_path):
        return _hash_feature(image_path)

    if not _try_load_model():
        return _hash_feature(image_path)

    # 磁盘缓存命中
    ckey = _cache_key(image_path)
    cpath = _FEATURE_CACHE_DIR / f"{ckey}.npy"
    if cpath.exists():
        try:
            return np.load(cpath)
        except Exception:  # noqa: BLE001
            pass

    # 真实推理
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = _processor(images=img, return_tensors="pt")
        with _torch.no_grad():
            out = _vision_model(pixel_values=inputs["pixel_values"])
        pooled = out.pooler_output.detach().cpu().numpy()  # (1, hidden=768)
        feat = _project_to_dim(pooled)
        try:
            np.save(cpath, feat)
        except Exception:  # noqa: BLE001
            pass
        return feat
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[BLIP] 图像编码失败({image_path})，回退哈希: {e}")
        return _hash_feature(image_path)


def get_status() -> dict:
    """返回当前图像编码器状态（供日志/实验标注）。"""
    _try_load_model()
    return {
        "use_real_image": USE_REAL_IMAGE,
        "active_model": ACTIVE_IMAGE_MODEL,
        "model_id": _BLIP_BASE_ID if USE_REAL_IMAGE else None,
        "image_embedding_dim": IMAGE_EMBEDDING_DIM,
    }
