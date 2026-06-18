# API Documentation - API接口文档

## 基础信息

| 服务 | 基础URL | 说明 |
|------|---------|------|
| 后端 API | `http://localhost:8003` | FastAPI + Uvicorn，对话处理、Harness调度 |
| 前端 API | `http://localhost:9033` | Flask，用户界面、SSE流式推送、会话管理 |

- 数据格式: JSON
- 认证方式: 无需登录，通过 session_id 区分会话

## 后端接口（端口 8003）

### 1. POST /chat - 统一问答接口

**功能**: 支持多轮对话的统一入口，Harness驱动智能匹配

**请求体**:
```json
{
  "session_id": "string (必填) - 会话ID，同一会话保持一致",
  "message_id": "string (必填) - 消息ID",
  "emp_id": "string (必填) - 员工工号",
  "query": "string (必填) - 自然语言查询"
}
```

**响应**:
```json
{
  "answer": "string - AI回复内容",
  "candidates": [...],
  "session_id": "string"
}
```

### 2. GET /health - 健康检查

**响应**: `{"status": "ok"}`

### 3. DELETE /session/{session_id} - 清除会话

**响应**: `{"status": "cleared"}`

---

## 前端接口（端口 9033）

### 4. POST /api/match - 智能匹配

**功能**: Harness驱动的智能人岗匹配

**请求体**:
```json
{
  "query": "string (必填) - 自然语言招聘需求描述",
  "context": {
    "filters": {},
    "top_k": 10,
    "include_explanation": true
  }
}
```

**响应**:
```json
{
  "matched_candidates": [
    {
      "candidate_id": 1,
      "candidate": {
        "name": "张三",
        "education_level": "硕士",
        "school": "清华大学",
        "work_years": 5,
        "skills": [{"skill_name": "Python", "proficiency": 5}]
      },
      "match_score": 0.9234,
      "fusion_score": 0.8856,
      "catboost_score": 0.9012,
      "text_similarity": 0.8734,
      "multimodal_similarity": 0.7521,
      "structured_features": [0.8, 0.9, ...]
    }
  ],
  "total_evaluated": 150,
  "top_k": 10,
  "method": "multimodal_hierarchical_fusion",
  "history_id": 42,
  "latency_ms": 1234.56
}
```

### 2. POST /api/generate - 生成合成数据

**功能**: 生成多模态合成简历数据集

**请求体**:
```json
{
  "count": 100
}
```

**响应**:
```json
{
  "generated": 100,
  "total_in_db": 500
}
```

### 3. POST /api/preprocess - 数据预处理

**功能**: 数据清洗并构建向量索引

**请求体**: 无需请求体

**响应**:
```json
{
  "preprocess": {"processed": 500},
  "vector_index": {"indexed": 500, "total_vectors": 500}
}
```

### 4. POST /api/explain - SHAP可解释性

**功能**: 获取候选人匹配的可解释性分析

**请求体**:
```json
{
  "candidate_id": 1,
  "features": [0.8, 0.9, ...],
  "match_score": 0.92,
  "level": "all"
}
```

`level`可选值: `all`, `global`, `individual`, `interaction`, `nlp`

**响应**:
```json
{
  "global_explanation": {
    "feature_importance": {"skill_match": 0.18, ...}
  },
  "individual_explanation": {
    "shap_values": {"skill_match": 0.072, ...},
    "base_value": 0.5,
    "prediction": 0.82
  },
  "interaction_explanation": {
    "top_interactions": [
      {"feature_1": "skill_match", "feature_2": "experience_match", "interaction_strength": 0.034}
    ]
  },
  "nlp_explanation": {
    "explanation": "该候选人综合匹配度为92%...",
    "top_features": [["skill_match", 0.144], ...],
    "detailed": false
  }
}
```

### 5. POST /api/feedback - 提交反馈

**功能**: 提交匹配结果反馈，触发模型权重调整

**请求体**:
```json
{
  "history_id": 42,
  "feedback": 1
}
```

`feedback`: 1=满意, 0=不满意

**响应**:
```json
{
  "status": "feedback_recorded",
  "history_id": 42
}
```

### 6. GET /api/stats - 系统统计

**功能**: 获取系统运行统计信息

**响应**:
```json
{
  "database": {
    "total_queries": 1234,
    "positive_feedback": 1050,
    "satisfaction_rate": 0.8511,
    "avg_latency_ms": 892.34
  },
  "candidates_count": 5000
}
```

### 7. GET /api/shap/explain/{candidate_id} - SHAP四层解释

**功能**: 获取候选人的 SHAP 可解释性分析数据（全局重要性 + 特征交互 + 自然语言解释）

**路径参数**: `candidate_id` (int) - 候选人ID

**响应**:
```json
{
  "global_explanation": {
    "feature_importance": {
      "skill_match": 0.18,
      "education_match": 0.15,
      "experience_years": 0.12,
      "...": "..."
    }
  },
  "interaction_explanation": {
    "top_interactions": [
      {
        "feature_1": "skill_match",
        "feature_2": "experience_match",
        "interaction_strength": 0.034
      }
    ]
  },
  "nlp_explanation": {
    "explanation": "该候选人综合匹配度为70%...",
    "top_features": [["skill_match", 0.144], ...],
    "detailed": false
  }
}
```

### 8. POST /api/sessions - 创建新会话

**响应**:
```json
{
  "session_id": "uuid-string",
  "created_at": "2026-06-15T10:00:00"
}
```

### 9. GET /api/sessions - 获取会话列表

**响应**:
```json
{
  "sessions": [
    {"session_id": "...", "title": "...", "created_at": "...", "updated_at": "..."}
  ]
}
```

### 10. DELETE /api/sessions/{session_id} - 删除会话

**响应**: `{"status": "deleted"}`

### 11. POST /api/chat - 发送消息（SSE流式返回）

**功能**: 前端统一对话入口，通过 SSE (Server-Sent Events) 流式返回结果

**请求体**:
```json
{
  "session_id": "string",
  "message": "string - 用户输入"
}
```

**响应**: SSE 流，每个 event 的 data 为 JSON 字符串

### 12. GET /api/sessions/{session_id}/history - 获取会话历史

**响应**:
```json
{
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ]
}
```

### 13. GET /api/health - 前端健康检查

**响应**: `{"status": "ok"}`

## 错误处理

所有接口在出错时返回:
```json
{
  "error": "错误描述信息"
}
```

HTTP状态码:
- 200: 成功
- 400: 请求参数错误
- 500: 服务器内部错误

## 数据模型

### Candidate (候选人)
| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | 主键 |
| name | string | 姓名 |
| gender | string | 性别 |
| age | int | 年龄 |
| education_level | string | 学历 |
| school | string | 学校 |
| major | string | 专业 |
| work_years | int | 工作年限 |
| skills | array | 技能列表 |
| work_experiences | array | 工作经历 |
| projects | array | 项目经历 |
| multimodal | array | 多模态数据 |

### MatchResult (匹配结果)
| 字段 | 类型 | 说明 |
|------|------|------|
| candidate_id | int | 候选人ID |
| match_score | float | 综合匹配分 |
| fusion_score | float | 多模态融合分 |
| catboost_score | float | 结构化匹配分 |
| text_similarity | float | 文本相似度 |
| structured_features | array | 12维结构化特征 |
