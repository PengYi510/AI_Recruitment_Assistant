"""多模态特征分层融合匹配模型 - 核心创新点2
三层融合架构:
  Layer 1: 模态内特征提取 (BGE-M3 1024d + BLIP-3 768d)
  Layer 2: 模态间交叉注意力融合 -> 1024d
  Layer 3: 全局特征融合 (多模态1024d + 结构化12d, 权重6:4)

文本特征提取使用 BAAI/bge-m3 模型（sentence-transformers），567M参数，
输出 1024 维语义向量，中英文多语言效果优秀。
模型需从 HuggingFace 下载后放置于项目 models/bge-m3/ 目录中。

图片特征提取使用真实 BLIP 视觉编码器（blip_image_encoder 模块）：
首选 BLIP-3，因其远程代码与 transformers>=5.x 不兼容而回退到 BLIP-base
（BlipVisionModel, hidden_size=768），输出 768 维视觉语义向量并带磁盘缓存；
若运行环境无法加载视觉模型则降级为确定性哈希向量。
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from backend.config import (TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM,
    FUSION_EMBEDDING_DIM, STRUCTURED_FEATURE_DIM, MULTIMODAL_WEIGHT, STRUCTURED_WEIGHT,
    EMBEDDING_MODEL_PATH)

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
    logger.warning("sentence-transformers not installed, using hash-based fallback for embeddings")


def _load_embedding_model():
    """延迟加载 BGE-M3 模型（从项目 models/bge-m3/ 目录加载）
    
    首次使用需从 HuggingFace 下载模型到 models/bge-m3/ 目录，
    详见 README.md 的「模型下载」章节。
    """
    global _st_model, _USE_REAL_EMBEDDING
    if _st_model is not None:
        return _st_model
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        logger.info(f"Loading BGE-M3 embedding model from: {EMBEDDING_MODEL_PATH}")
        _st_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        _USE_REAL_EMBEDDING = True
        logger.info(f"BGE-M3 model loaded successfully, embedding dim={_st_model.get_embedding_dimension()}")
        return _st_model
    except Exception as e:
        logger.error(f"Failed to load BGE-M3 model from {EMBEDDING_MODEL_PATH}: {e}, falling back to hash-based embeddings")
        _USE_REAL_EMBEDDING = False
        return None


class CrossAttentionFusion:
    """交叉注意力融合模块
    计算文本特征和图片特征之间的交叉注意力权重
    Q = W_q * text_features, K = W_k * image_features, V = W_v * image_features
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self, text_dim: int = TEXT_EMBEDDING_DIM, image_dim: int = IMAGE_EMBEDDING_DIM,
                 output_dim: int = FUSION_EMBEDDING_DIM):
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim
        np.random.seed(42)
        self.W_q = np.random.randn(text_dim, output_dim) * 0.02
        self.W_k = np.random.randn(image_dim, output_dim) * 0.02
        self.W_v = np.random.randn(image_dim, output_dim) * 0.02
        self.W_out = np.random.randn(output_dim, output_dim) * 0.02

    def forward(self, text_feat: np.ndarray, image_feat: np.ndarray) -> np.ndarray:
        """计算交叉注意力融合
        Args:
            text_feat: (1, text_dim) 文本特征
            image_feat: (1, image_dim) 图片特征
        Returns:
            fused: (1, output_dim) 融合特征
        """
        Q = text_feat @ self.W_q  # (1, output_dim)
        K = image_feat @ self.W_k  # (1, output_dim)
        V = image_feat @ self.W_v  # (1, output_dim)
        d_k = self.output_dim ** 0.5
        attention_scores = (Q @ K.T) / d_k  # (1, 1)
        attention_weights = self._softmax(attention_scores)
        context = attention_weights @ V  # (1, output_dim)
        fused = text_feat @ self.W_q + context  # 残差连接
        fused = fused @ self.W_out
        fused = fused / (np.linalg.norm(fused) + 1e-8)  # L2归一化
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
        
        如果模型不可用，降级为 hash-based 确定性向量（仅用于架构验证）。
        
        Args:
            text: 输入文本
        Returns:
            (1, 1024) 形状的特征向量
        """
        if not text or not text.strip():
            return np.zeros((1, TEXT_EMBEDDING_DIM), dtype=np.float32)

        # 检查缓存
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        model = _load_embedding_model()
        if model is not None:
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
        else:
            # 降级方案：hash-based 确定性向量
            np.random.seed(hash(text) % 2**31)
            feat = np.random.randn(1, TEXT_EMBEDDING_DIM).astype(np.float32)
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
        """
        if not texts:
            return []

        model = _load_embedding_model()
        if model is not None:
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
        else:
            # 降级：逐个计算
            return [self.extract_text_features(t) for t in texts]

    def extract_image_features(self, image_path: str) -> np.ndarray:
        """Layer 1: 图片特征提取

        使用真实 BLIP 视觉编码器（blip_image_encoder）提取 768 维语义特征，
        带磁盘缓存。当视觉模型不可用或图片缺失时，自动回退为确定性哈希向量
        （行为与历史一致，保证实验可复现）。

        Args:
            image_path: 图片路径字符串
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
