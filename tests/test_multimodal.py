"""测试多模态融合"""
import pytest
import numpy as np


class TestMultimodalFusion:
    def test_extract_text_features(self):
        from backend.models.multimodal_fusion import multimodal_fusion
        features = multimodal_fusion.extract_text_features("Python高级工程师")
        assert features.shape == (1, 1024)
        assert not np.all(features == 0)

    def test_extract_image_features(self):
        from backend.models.multimodal_fusion import multimodal_fusion
        features = multimodal_fusion.extract_image_features("nonexist.png")
        assert features.shape == (1, 768)

    def test_cross_attention_fusion(self):
        from backend.models.multimodal_fusion import CrossAttentionFusion
        fusion = CrossAttentionFusion(text_dim=1024, image_dim=768, output_dim=1024)
        text_feat = np.random.rand(1, 1024).astype(np.float32)
        image_feat = np.random.rand(1, 768).astype(np.float32)
        result = fusion.forward(text_feat, image_feat)
        assert result.shape == (1, 1024)

    def test_compute_matching_score(self):
        from backend.models.multimodal_fusion import multimodal_fusion
        structured = np.random.rand(12)
        result = multimodal_fusion.compute_matching_score(
            jd_text="Python工程师",
            candidate_text="5年Python开发经验",
            structured_features=structured
        )
        assert "score" in result
        assert 0 <= result["score"] <= 1

    def test_matching_with_images(self):
        from backend.models.multimodal_fusion import multimodal_fusion
        structured = np.random.rand(12)
        result = multimodal_fusion.compute_matching_score(
            jd_text="AI工程师",
            candidate_text="深度学习专家",
            candidate_images=["img1.png", "img2.png"],
            structured_features=structured
        )
        assert "score" in result
        assert "multimodal_similarity" in result
        assert result["has_image_features"] is True

    def test_fuse_multimodal(self):
        from backend.models.multimodal_fusion import multimodal_fusion
        text_feat = np.random.rand(1, 1024).astype(np.float32)
        image_feat = np.random.rand(1, 768).astype(np.float32)
        fused = multimodal_fusion.fuse_multimodal(text_feat, image_feat)
        assert fused.shape == (1, 1024)

    def test_fuse_global(self):
        from backend.models.multimodal_fusion import multimodal_fusion
        mm_feat = np.random.rand(1, 1024).astype(np.float32)
        struct_feat = np.random.rand(12).astype(np.float32)
        global_feat = multimodal_fusion.fuse_global(mm_feat, struct_feat)
        assert global_feat.shape[0] == 1024 + 12
