"""ChromaDB向量数据库客户端 - 1主collection + 丰富metadata过滤方案"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from backend.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed, using in-memory fallback")


class VectorDBClient:
    """ChromaDB向量数据库客户端 - 1个主collection + metadata结构化过滤

    设计思路:
    - 1个主collection: candidates_collection，存储每个候选人的完整简历文本向量
    - metadata中存储结构化字段（用于where条件过滤）:
        - name, highest_education, work_years, current_position, location
        - skills_text (技能列表拼接), school_list (所有院校拼接)
    - document中存储完整简历文本（可用于全文检索回溯）
    """

    def __init__(self):
        self._collection = None
        self._fallback_store = []  # 降级内存存储
        if CHROMA_AVAILABLE:
            try:
                persist_dir = str(CHROMA_PERSIST_DIR)
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=persist_dir)
                self._collection = self._client.get_or_create_collection(
                    name=CHROMA_COLLECTION,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"ChromaDB initialized: {persist_dir}")
            except Exception as e:
                logger.error(f"ChromaDB init failed: {e}, falling back to memory")
                self._collection = None
        else:
            logger.warning("Using fallback in-memory vector store")

    def add_candidate(self, candidate_id: int, embedding: List[float],
                      metadata: Dict[str, Any] = None, document: str = None):
        """添加候选人向量

        Args:
            candidate_id: 候选人SQL表ID
            embedding: BGE-M3生成的1024维向量
            metadata: 结构化字段（用于where过滤）
            document: 完整简历文本（用于BM25回溯）
        """
        doc_id = f"candidate_{candidate_id}"

        # 清理metadata: ChromaDB要求metadata值为str/int/float/bool
        clean_meta = {}
        if metadata:
            for k, v in metadata.items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)

        if self._collection:
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[clean_meta],
                documents=[document or f"candidate {candidate_id}"]
            )
        else:
            self._fallback_store.append({
                "id": doc_id, "candidate_id": candidate_id,
                "embedding": embedding, "metadata": clean_meta,
                "document": document or ""
            })

    def search_similar(self, query_embedding: List[float], top_k: int = 20,
                       where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """向量相似度搜索（支持metadata条件过滤）

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            where: ChromaDB where条件过滤，例如:
                {"highest_education": "硕士"}
                {"work_years": {"$gte": 3}}
                {"$and": [{"work_years": {"$gte": 3}}, {"location": "北京"}]}
        """
        if self._collection:
            try:
                query_params = {
                    "query_embeddings": [query_embedding],
                    "n_results": min(top_k, self.get_collection_count()),
                    "include": ["metadatas", "distances", "documents"]
                }
                if where:
                    query_params["where"] = where

                results = self._collection.query(**query_params)
                output = []
                for i, doc_id in enumerate(results["ids"][0]):
                    cid = int(doc_id.replace("candidate_", ""))
                    output.append({
                        "candidate_id": cid,
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "document": results["documents"][0][i] if results.get("documents") else ""
                    })
                return output
            except Exception as e:
                logger.error(f"ChromaDB search failed: {e}")
                return []
        else:
            # 降级: 内存暴力搜索
            import numpy as np
            if not self._fallback_store:
                return []
            query = np.array(query_embedding)
            scored = []
            for item in self._fallback_store:
                # 简单where过滤
                if where and not self._match_where(item["metadata"], where):
                    continue
                emb = np.array(item["embedding"])
                cos_sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8)
                scored.append((item["candidate_id"], 1 - cos_sim, item["metadata"], item.get("document", "")))
            scored.sort(key=lambda x: x[1])
            return [{"candidate_id": s[0], "distance": float(s[1]), "metadata": s[2], "document": s[3]}
                    for s in scored[:top_k]]

    def _match_where(self, metadata: Dict, where: Dict) -> bool:
        """简易where条件匹配（fallback用）"""
        for key, value in where.items():
            if key.startswith("$"):
                continue  # 跳过$and/$or等复杂条件
            meta_val = metadata.get(key)
            if isinstance(value, dict):
                # 范围条件
                if "$gte" in value and (meta_val is None or meta_val < value["$gte"]):
                    return False
                if "$lte" in value and (meta_val is None or meta_val > value["$lte"]):
                    return False
            else:
                if meta_val != value:
                    return False
        return True

    def get_collection_count(self) -> int:
        if self._collection:
            return self._collection.count()
        return len(self._fallback_store)

    def delete_candidate(self, candidate_id: int):
        doc_id = f"candidate_{candidate_id}"
        if self._collection:
            self._collection.delete(ids=[doc_id])
        else:
            self._fallback_store = [s for s in self._fallback_store if s["id"] != doc_id]

    def reset(self):
        """重置向量数据库"""
        if self._collection:
            self._client.delete_collection(CHROMA_COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self._fallback_store = []


# 全局实例
vector_db = VectorDBClient()
