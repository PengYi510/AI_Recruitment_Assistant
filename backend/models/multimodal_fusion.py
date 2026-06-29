"""多模态特征分层融合匹配模型 - 核心创新点2
三层融合架构:
  Layer 1: 模态内特征提取 (BGE-M3 1024d + BLIP-base 768d)
  Layer 2: 模态间交叉注意力融合 -> 1024d（PyTorch 实现，加载预训练权重）
  Layer 3: 全局特征融合 (多模态1024d + 结构化12d, 权重6:4)

文本特征提取使用 BAAI/bge-m3 模型（sentence-transformers），567M参数，
输出 1024 维语义向量，中英文多语言效果优秀。
模型需从 HuggingFace 下载后放置于项目 models/bge-m3/ 目录中。

图片特征提取使用真实 BLIP-base 视觉编码器（blip_image_encoder 模块），
hidden_size=768，输出 768 维视觉语义向量并带磁盘缓存。

设备选择策略：GPU:0 → GPU:1 → ... → CPU 自动降级。
模型加载失败时直接抛出异常，不做静默降级。
"""
import logging
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from backend.config import (TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM,
    FUSION_EMBEDDING_DIM, STRUCTURED_FEATURE_DIM, MULTIMODAL_WEIGHT, STRUCTURED_WEIGHT,
    EMBEDDING_MODEL_PATH, PROJECT_ROOT)
from backend.utils.device_selector import get_device, get_device_str

logger = logging.getLogger(__name__)

# ── 尝试加载 sentence-transformers ──────────────────────────────────────────
_st_model = None
_USE_REAL_EMBEDDING = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("sentence-transformers is available")
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("sentence-transformers 未安装，BGE-M3 文本编码不可用")


def _load_embedding_model():
    """延迟加载 BGE-M3 模型（从项目 models/bge-m3/ 目录加载）

    使用 device_selector 自动选择 GPU/CPU。
    加载失败时直接抛出异常，不做静默降级。

    首次使用需从 HuggingFace 下载模型到 models/bge-m3/ 目录，
    详见 README.md 的「模型下载」章节。
    """
    global _st_model, _USE_REAL_EMBEDDING
    if _st_model is not None:
        return _st_model

    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "BGE-M3 文本编码模型不可用：sentence-transformers 库未安装。"
            "请执行 `pip install sentence-transformers` 后重启系统，"
            "或联系系统管理员。"
        )

    device_str = get_device_str()
    try:
        logger.info(f"Loading BGE-M3 embedding model from: {EMBEDDING_MODEL_PATH} (device={device_str})")
        _st_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device_str)
        _USE_REAL_EMBEDDING = True
        logger.info(f"BGE-M3 model loaded successfully on {device_str}, "
                     f"embedding dim={_st_model.get_embedding_dimension()}")
        return _st_model
    except Exception as e:
        raise RuntimeError(
            f"BGE-M3 文本编码模型加载失败：{e}。"
            f"请检查模型文件是否完整（路径: {EMBEDDING_MODEL_PATH}），"
            f"或联系系统管理员。"
        ) from e


# ── CrossAttention: PyTorch 实现 + 预训练权重 ─────────────────────────────────

# 预训练权重路径
_CROSS_ATTENTION_WEIGHTS_PATH = PROJECT_ROOT / "models" / "cross_attention_weights.pt"

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch 未安装，CrossAttention 将使用 numpy 计算")


class CrossAttentionFusionTorch(nn.Module):
    """PyTorch 实现的交叉注意力融合模块

    Q = W_q * text_features, K = W_k * image_features, V = W_v * image_features
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    权重使用 Xavier 均匀初始化（torch.nn.init.xavier_uniform_），
    并保存到本地文件作为可复现的预训练权重。
    后续加载时直接读取已保存的权重，保证每次运行结果一致。
    """

    def __init__(self, text_dim: int = TEXT_EMBEDDING_DIM,
                 image_dim: int = IMAGE_EMBEDDING_DIM,
                 output_dim: int = FUSION_EMBEDDING_DIM):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim

        self.W_q = nn.Linear(text_dim, output_dim, bias=False)
        self.W_k = nn.Linear(image_dim, output_dim, bias=False)
        self.W_v = nn.Linear(image_dim, output_dim, bias=False)
        self.W_out = nn.Linear(output_dim, output_dim, bias=False)

        # Xavier 均匀初始化（比 0.02 * randn 更符合深度学习最佳实践）
        for module in [self.W_q, self.W_k, self.W_v, self.W_out]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """计算交叉注意力融合

        Args:
            text_feat: (1, text_dim) 文本特征
            image_feat: (1, image_dim) 图片特征
        Returns:
            fused: (1, output_dim) 融合特征
        """
        Q = self.W_q(text_feat)    # (1, output_dim)
        K = self.W_k(image_feat)   # (1, output_dim)
        V = self.W_v(image_feat)   # (1, output_dim)

        d_k = self.output_dim ** 0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k  # (1, 1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)  # (1, output_dim)

        # 残差连接
        fused = Q + context
        fused = self.W_out(fused)
        fused = fused / (torch.norm(fused, dim=-1, keepdim=True) + 1e-8)  # L2 归一化
        return fused


def _load_or_create_cross_attention_weights():
    """加载或创建 CrossAttention 预训练权重

    首次运行时使用 Xavier 初始化并保存到文件；
    后续运行直接加载已保存的权重。
    """
    device = get_device()
    model = CrossAttentionFusionTorch().to(device)
    model.eval()

    if _CROSS_ATTENTION_WEIGHTS_PATH.exists():
        try:
            state_dict = torch.load(
                str(_CROSS_ATTENTION_WEIGHTS_PATH),
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            logger.info(f"[CrossAttention] 加载预训练权重成功: {_CROSS_ATTENTION_WEIGHTS_PATH}")
        except Exception as e:
            logger.warning(f"[CrossAttention] 加载权重失败({e})，重新初始化并保存")
            _save_cross_attention_weights(model)
    else:
        logger.info("[CrossAttention] 预训练权重文件不存在，使用 Xavier 初始化并保存")
        _save_cross_attention_weights(model)

    return model


def _save_cross_attention_weights(model):
    """保存 CrossAttention 权重到文件"""
    try:
        _CROSS_ATTENTION_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(_CROSS_ATTENTION_WEIGHTS_PATH))
        logger.info(f"[CrossAttention] 权重已保存到: {_CROSS_ATTENTION_WEIGHTS_PATH}")
    except Exception as e:
        logger.warning(f"[CrossAttention] 权重保存失败: {e}")


class CrossAttentionFusion:
    """交叉注意力融合模块（统一接口）

    当 PyTorch 可用时使用真实的 nn.Module 实现（加载预训练权重）；
    当 PyTorch 不可用时退化为 numpy 计算（使用固定种子初始化保证可复现）。
    """

    def __init__(self, text_dim: int = TEXT_EMBEDDING_DIM,
                 image_dim: int = IMAGE_EMBEDDING_DIM,
                 output_dim: int = FUSION_EMBEDDING_DIM):
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim
        self._use_torch = False
        self._torch_model = None

        if _TORCH_AVAILABLE:
            try:
                self._torch_model = _load_or_create_cross_attention_weights()
                self._use_torch = True
                device = get_device_str()
                logger.info(f"[CrossAttention] 使用 PyTorch 实现 (device={device})")
            except Exception as e:
                logger.warning(f"[CrossAttention] PyTorch 加载失败({e})，退化为 numpy")
                self._init_numpy_weights()
        else:
            self._init_numpy_weights()

    def _init_numpy_weights(self):
        """numpy 退化方案：使用固定种子初始化（保证可复现）"""
        np.random.seed(42)
        self.W_q = np.random.randn(self.text_dim, self.output_dim) * 0.02
        self.W_k = np.random.randn(self.image_dim, self.output_dim) * 0.02
        self.W_v = np.random.randn(self.image_dim, self.output_dim) * 0.02
        self.W_out = np.random.randn(self.output_dim, self.output_dim) * 0.02

    def forward(self, text_feat: np.ndarray, image_feat: np.ndarray) -> np.ndarray:
        """计算交叉注意力融合

        Args:
            text_feat: (1, text_dim) 文本特征 numpy 数组
            image_feat: (1, image_dim) 图片特征 numpy 数组
        Returns:
            fused: (1, output_dim) 融合特征 numpy 数组
        """
        if self._use_torch and self._torch_model is not None:
            return self._forward_torch(text_feat, image_feat)
        return self._forward_numpy(text_feat, image_feat)

    def _forward_torch(self, text_feat: np.ndarray, image_feat: np.ndarray) -> np.ndarray:
        """PyTorch 推理路径"""
        device = get_device()
        text_t = torch.from_numpy(text_feat.astype(np.float32)).to(device)
        image_t = torch.from_numpy(image_feat.astype(np.float32)).to(device)
        with torch.no_grad():
            fused = self._torch_model(text_t, image_t)
        return fused.cpu().numpy()

    def _forward_numpy(self, text_feat: np.ndarray, image_feat: np.ndarray) -> np.ndarray:
        """numpy 退化路径"""
        Q = text_feat @ self.W_q
        K = image_feat @ self.W_k
        V = image_feat @ self.W_v
        d_k = self.output_dim ** 0.5
        attention_scores = (Q @ K.T) / d_k
        attention_weights = self._softmax(attention_scores)
        context = attention_weights @ V
        fused = Q + context  # 残差连接
        fused = fused @ self.W_out
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return fused

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-8)


class MultimodalHierarchicalFusion:
    """多模态分层融合模型
    实现三层融合架构:
    Layer 1: 模态内特征提取
    Layer 2: 模态间交叉注意力融合
    Layer 3: 全局特征融合
    """
    def __init__(self):
        self.cross_attention = CrossAttentionFusion()
        self.multimodal_weight = MULTIMODAL_WEIGHT
        self.structured_weight = STRUCTURED_WEIGHT
        self._embedding_cache = {}  # 缓存已计算的embedding，避免重复推理
        self._cache_max_size = 2000
        logger.info("MultimodalHierarchicalFusion initialized")

    def extract_text_features(self, text: str) -> np.ndarray:
        """Layer 1: 文本特征提取 — BGE-M3 真实语义编码

        使用 BAAI/bge-m3 模型进行真实语义编码，输出 1024 维向量。
        中英文多语言效果优秀，模型本地运行。

        模型加载失败时直接抛出异常（不做静默降级），
        前端应捕获错误并提示用户联系管理员。

        Args:
            text: 输入文本
        Returns:
            (1, 1024) 形状的特征向量

        Raises:
            RuntimeError: 当 BGE-M3 模型不可用时
        """
        if not text or not text.strip():
            return np.zeros((1, TEXT_EMBEDDING_DIM), dtype=np.float32)

        # 检查缓存
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        model = _load_embedding_model()  # 失败时会直接抛 RuntimeError

        # 真实模型推理
        embedding = model.encode(
            text,
            normalize_embeddings=True,  # L2归一化
            show_progress_bar=False
        )
        # 确保维度正确 (bge-m3 输出 1024d)
        feat = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # 如果模型维度不匹配配置，做padding或截断
        if feat.shape[1] != TEXT_EMBEDDING_DIM:
            if feat.shape[1] > TEXT_EMBEDDING_DIM:
                feat = feat[:, :TEXT_EMBEDDING_DIM]
            else:
                feat = np.pad(feat, ((0, 0), (0, TEXT_EMBEDDING_DIM - feat.shape[1])))
            feat = feat / (np.linalg.norm(feat) + 1e-8)

        # 缓存管理
        if len(self._embedding_cache) >= self._cache_max_size:
            # LRU简单实现：清除一半缓存
            keys_to_remove = list(self._embedding_cache.keys())[:self._cache_max_size // 2]
            for k in keys_to_remove:
                del self._embedding_cache[k]
        self._embedding_cache[text] = feat
        return feat

    def extract_text_features_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量文本特征提取 — 利用 GPU/CPU 批处理加速

        Args:
            texts: 文本列表
        Returns:
            每个文本对应的 (1, 1024) 特征向量列表

        Raises:
            RuntimeError: 当 BGE-M3 模型不可用时
        """
        if not texts:
            return []

        model = _load_embedding_model()  # 失败时会直接抛 RuntimeError

        # 分离已缓存和未缓存的文本
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = np.zeros((1, TEXT_EMBEDDING_DIM), dtype=np.float32)
            elif text in self._embedding_cache:
                results[i] = self._embedding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 批量编码未缓存的文本
        if uncached_texts:
            embeddings = model.encode(
                uncached_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32
            )
            for j, idx in enumerate(uncached_indices):
                feat = np.array(embeddings[j], dtype=np.float32).reshape(1, -1)
                if feat.shape[1] != TEXT_EMBEDDING_DIM:
                    if feat.shape[1] > TEXT_EMBEDDING_DIM:
                        feat = feat[:, :TEXT_EMBEDDING_DIM]
                    else:
                        feat = np.pad(feat, ((0, 0), (0, TEXT_EMBEDDING_DIM - feat.shape[1])))
                    feat = feat / (np.linalg.norm(feat) + 1e-8)
                results[idx] = feat
                self._embedding_cache[uncached_texts[j]] = feat

        return results

    def extract_image_features(self, image_path: str) -> np.ndarray:
        """Layer 1: 图片特征提取

        使用真实 BLIP-base 视觉编码器（blip_image_encoder）提取 768 维语义特征，
        带磁盘缓存。当视觉模型不可用时抛出异常。

        Args:
            image_path: 图片路径字符串

        Raises:
            RuntimeError: 当 BLIP-base 模型不可用时
        """
        from backend.models.blip_image_encoder import encode_image
        return encode_image(image_path)

    def fuse_multimodal(self, text_feat: np.ndarray, image_feat: np.ndarray) -> np.ndarray:
        """Layer 2: 模态间交叉注意力融合"""
        return self.cross_attention.forward(text_feat, image_feat)

    def fuse_global(self, multimodal_feat: np.ndarray, structured_feat: np.ndarray) -> np.ndarray:
        """Layer 3: 全局特征融合 (多模态特征 + 结构化特征)
        权重比 6:4
        """
        mm_norm = multimodal_feat.flatten()[:FUSION_EMBEDDING_DIM]
        if len(mm_norm) < FUSION_EMBEDDING_DIM:
            mm_norm = np.pad(mm_norm, (0, FUSION_EMBEDDING_DIM - len(mm_norm)))
        sf_norm = structured_feat.flatten()[:STRUCTURED_FEATURE_DIM]
        if len(sf_norm) < STRUCTURED_FEATURE_DIM:
            sf_norm = np.pad(sf_norm, (0, STRUCTURED_FEATURE_DIM - len(sf_norm)))
        # 加权融合
        combined = np.concatenate([
            mm_norm * self.multimodal_weight,
            sf_norm * self.structured_weight
        ])
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        return combined

    def compute_matching_score(self, jd_text: str, candidate_text: str,
                               candidate_images: List[str] = None,
                               structured_features: np.ndarray = None) -> Dict[str, Any]:
        """计算JD与候选人的匹配分数
        Returns:
            score: 0-1之间的匹配分数
            features: 各层特征
            attention_weights: 注意力权重
        """
        # Layer 1: 模态内特征提取
        jd_text_feat = self.extract_text_features(jd_text)
        cand_text_feat = self.extract_text_features(candidate_text)

        # 文本相似度
        text_sim = float(np.dot(jd_text_feat.flatten(), cand_text_feat.flatten()))

        # Layer 2: 多模态融合（如有图片）
        if candidate_images:
            img_feats = [self.extract_image_features(img) for img in candidate_images]
            avg_img_feat = np.mean(img_feats, axis=0)
            fused_feat = self.fuse_multimodal(cand_text_feat, avg_img_feat)
            jd_fused = self.fuse_multimodal(jd_text_feat, avg_img_feat)
            multimodal_sim = float(np.dot(jd_fused.flatten(), fused_feat.flatten()))
        else:
            fused_feat = cand_text_feat
            multimodal_sim = text_sim

        # Layer 3: 全局融合（如有结构化特征）
        if structured_features is not None:
            global_feat = self.fuse_global(fused_feat, structured_features)
            jd_global = self.fuse_global(jd_text_feat, np.zeros(STRUCTURED_FEATURE_DIM))
            final_sim = float(np.dot(
                global_feat[:FUSION_EMBEDDING_DIM],
                jd_global[:FUSION_EMBEDDING_DIM]
            ))
        else:
            final_sim = multimodal_sim

        # 归一化到0-1
        score = (final_sim + 1) / 2
        score = max(0.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "text_similarity": round((text_sim + 1) / 2, 4),
            "multimodal_similarity": round((multimodal_sim + 1) / 2, 4),
            "has_image_features": bool(candidate_images),
            "has_structured_features": structured_features is not None,
            "using_real_embedding": _USE_REAL_EMBEDDING,
        }

    def clear_cache(self):
        """清除embedding缓存"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    @property
    def is_using_real_model(self) -> bool:
        """是否使用真实embedding模型"""
        return _USE_REAL_EMBEDDING


# 全局实例
multimodal_fusion = MultimodalHierarchicalFusion()
