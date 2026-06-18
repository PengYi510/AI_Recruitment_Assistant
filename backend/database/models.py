"""SQLite数据库模型 - 6张核心表设计（人物基础信息、教育经历、技术栈、获奖资格证书、工作经历、项目经历）"""
import sqlite3
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from contextlib import contextmanager
from backend.config import DB_PATH

logger = logging.getLogger(__name__)


class HRDatabase:
    """HR招聘智能匹配系统数据库 - 多表归一化设计"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表结构 - 6张核心业务表 + 辅助表"""
        with self._get_conn() as conn:
            conn.executescript("""
                -- ═══════════════════════════════════════════════════════════════
                -- 表1: 人物基础信息表 (candidates)
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    gender TEXT,
                    birth_date TEXT,
                    age INTEGER,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    current_position TEXT,
                    work_years INTEGER DEFAULT 0,
                    current_salary REAL,
                    expected_salary REAL,
                    job_status TEXT,
                    location TEXT,
                    current_city TEXT,
                    hometown TEXT,
                    highest_education TEXT,
                    summary TEXT,
                    resume_raw_text TEXT,
                    vector_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 表2: 教育经历表 (education_history)
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS education_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    degree TEXT NOT NULL,
                    school TEXT NOT NULL,
                    major TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    is_fulltime INTEGER DEFAULT 1,
                    school_tier TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 表3: 技术栈表 (skills)
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    skill_name TEXT NOT NULL,
                    proficiency INTEGER DEFAULT 3,
                    category TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 表4: 获奖资格证书表 (awards_certificates)
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS awards_certificates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    level TEXT,
                    date TEXT,
                    role TEXT,
                    description TEXT,
                    image_path TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 表5: 历史工作经历表 (work_experiences)
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS work_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    company_name TEXT,
                    position TEXT,
                    location TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    duration_months INTEGER,
                    description TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 表6: 历史项目经历表 (projects)
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    project_name TEXT,
                    role TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    duration_months INTEGER,
                    description TEXT,
                    technologies TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 表7: 论文发表 (publications)
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS publications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    venue TEXT,
                    venue_type TEXT,
                    venue_rank TEXT,
                    sci_zone TEXT,
                    year INTEGER,
                    authors TEXT,
                    author_position TEXT,
                    doi TEXT,
                    abstract TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 表8: 国际会议经历 (conferences)
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS conferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    conference_name TEXT NOT NULL,
                    conference_rank TEXT,
                    year INTEGER,
                    location TEXT,
                    role TEXT,
                    paper_title TEXT,
                    description TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 表9: 动态扩展属性表 (candidate_extra_attributes)
                -- 用于存储 GPA、爱好、目标岗位、民族、身高、体重等不固定字段
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS candidate_extra_attributes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    attr_key TEXT NOT NULL,
                    attr_value TEXT,
                    attr_type TEXT DEFAULT 'str',
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 辅助表: 匹配历史记录
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS matching_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    candidate_ids TEXT,
                    scores TEXT,
                    feedback INTEGER,
                    success INTEGER DEFAULT 0,
                    latency_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 辅助表: Harness任务
                -- ═══════════════════════════════════════════════════════════════
                CREATE TABLE IF NOT EXISTS harness_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE NOT NULL,
                    query TEXT,
                    status TEXT DEFAULT 'pending',
                    complexity REAL DEFAULT 0.5,
                    iterations INTEGER DEFAULT 0,
                    max_iterations INTEGER DEFAULT 3,
                    result_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- ═══════════════════════════════════════════════════════════════
                -- 索引
                -- ═══════════════════════════════════════════════════════════════
                CREATE INDEX IF NOT EXISTS idx_education_history_candidate ON education_history(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_skills_candidate ON skills(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_awards_certs_candidate ON awards_certificates(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_work_exp_candidate ON work_experiences(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_projects_candidate ON projects(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_publications_candidate ON publications(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_publications_venue_rank ON publications(venue_rank);
                CREATE INDEX IF NOT EXISTS idx_conferences_candidate ON conferences(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_conferences_rank ON conferences(conference_rank);
                CREATE INDEX IF NOT EXISTS idx_history_created ON matching_history(created_at);
                CREATE INDEX IF NOT EXISTS idx_extra_attrs_candidate ON candidate_extra_attributes(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_extra_attrs_key ON candidate_extra_attributes(attr_key);
            """)
            # ── 增量迁移：为已有的 candidates 表添加 resume_raw_text 列 ──
            existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(candidates)")}
            if "resume_raw_text" not in existing_cols:
                conn.execute("ALTER TABLE candidates ADD COLUMN resume_raw_text TEXT")

    # ═══════════════════════════════════════════════════════════════════════════
    # 插入方法
    # ═══════════════════════════════════════════════════════════════════════════

    def insert_candidate(self, data: Dict[str, Any]) -> int:
        """插入候选人基础信息（自适应实际表结构，兼容新旧 schema）"""
        # 新旧字段名映射：新 schema 字段 -> 旧 schema 字段
        # 旧库列名: education_level/school/major/graduation_year
        # 新库列名: highest_education/...（无 school/major，挪到 education_history 表）
        alias = {
            "highest_education": "education_level",
            "education_level": "highest_education",
        }
        with self._get_conn() as conn:
            # 读取实际存在的列
            existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(candidates)")}

            cols, vals = [], []
            for key, value in data.items():
                target = None
                if key in existing_cols:
                    target = key
                elif key in alias and alias[key] in existing_cols:
                    target = alias[key]
                if target and target not in cols:
                    cols.append(target)
                    vals.append(value)

            if not cols:
                raise ValueError("insert_candidate: 没有可写入的列")

            placeholders = ", ".join("?" for _ in cols)
            sql = f"INSERT INTO candidates ({', '.join(cols)}) VALUES ({placeholders})"
            cursor = conn.execute(sql, tuple(vals))
            return cursor.lastrowid

    def insert_education_history(self, candidate_id: int, data: Dict[str, Any]) -> int:
        """插入一段教育经历"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO education_history (candidate_id, degree, school, major,
                   start_date, end_date, is_fulltime, school_tier) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (candidate_id, data.get("degree"), data.get("school"), data.get("major"),
                 data.get("start_date"), data.get("end_date"),
                 1 if data.get("is_fulltime", True) else 0,
                 data.get("school_tier")))
            return cursor.lastrowid

    def _insert_adaptive(self, conn, table: str, data: Dict[str, Any]) -> int:
        """按目标表实际存在的列写入，自动忽略表中不存在的字段（兼容 schema 漂移）"""
        existing_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        cols, vals = [], []
        for key, value in data.items():
            if key in existing_cols and key != "id":
                cols.append(key)
                vals.append(value)
        if not cols:
            raise ValueError(f"_insert_adaptive: 表 {table} 没有可写入的列")
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        return conn.execute(sql, tuple(vals)).lastrowid

    def insert_skill(self, candidate_id: int, skill_name: str, proficiency: int = 3,
                     category: str = None) -> int:
        """插入技能（自适应表结构，category 列不存在时自动忽略）"""
        with self._get_conn() as conn:
            return self._insert_adaptive(conn, "skills", {
                "candidate_id": candidate_id, "skill_name": skill_name,
                "proficiency": proficiency, "category": category,
            })

    def insert_award_certificate(self, candidate_id: int, data: Dict[str, Any]) -> int:
        """插入获奖/资格证书"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO awards_certificates (candidate_id, type, name, level,
                   date, role, description, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (candidate_id, data.get("type", "award"), data.get("name"),
                 data.get("level"), data.get("date"), data.get("role"),
                 data.get("description"), data.get("image_path")))
            return cursor.lastrowid

    def insert_work_experience(self, candidate_id: int, data: Dict[str, Any]) -> int:
        """插入工作经历（自适应表结构，缺失列自动忽略）"""
        with self._get_conn() as conn:
            return self._insert_adaptive(conn, "work_experiences", {
                "candidate_id": candidate_id,
                "company_name": data.get("company_name"), "position": data.get("position"),
                "location": data.get("location"), "start_date": data.get("start_date"),
                "end_date": data.get("end_date"), "duration_months": data.get("duration_months"),
                "description": data.get("description"),
            })

    def insert_project(self, candidate_id: int, data: Dict[str, Any]) -> int:
        """插入项目经历（自适应表结构，缺失列自动忽略）"""
        with self._get_conn() as conn:
            return self._insert_adaptive(conn, "projects", {
                "candidate_id": candidate_id,
                "project_name": data.get("project_name"), "role": data.get("role"),
                "start_date": data.get("start_date"), "end_date": data.get("end_date"),
                "duration_months": data.get("duration_months"),
                "description": data.get("description"), "technologies": data.get("technologies"),
            })

    def insert_publication(self, candidate_id: int, data: Dict[str, Any]) -> int:
        """插入论文发表记录"""
        with self._get_conn() as conn:
            return self._insert_adaptive(conn, "publications", {
                "candidate_id": candidate_id,
                "title": data.get("title"),
                "venue": data.get("venue"),
                "venue_type": data.get("venue_type"),
                "venue_rank": data.get("venue_rank"),
                "sci_zone": data.get("sci_zone"),
                "year": data.get("year"),
                "authors": data.get("authors"),
                "author_position": data.get("author_position"),
                "doi": data.get("doi"),
                "abstract": data.get("abstract"),
            })

    def insert_conference(self, candidate_id: int, data: Dict[str, Any]) -> int:
        """插入国际会议经历"""
        with self._get_conn() as conn:
            return self._insert_adaptive(conn, "conferences", {
                "candidate_id": candidate_id,
                "conference_name": data.get("conference_name"),
                "conference_rank": data.get("conference_rank"),
                "year": data.get("year"),
                "location": data.get("location"),
                "role": data.get("role"),
                "paper_title": data.get("paper_title"),
                "description": data.get("description"),
            })

    def insert_extra_attribute(self, candidate_id: int, attr_key: str,
                               attr_value: str, attr_type: str = "str") -> int:
        """插入一条动态扩展属性（GPA、爱好、目标岗位、民族、身高、体重等）"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO candidate_extra_attributes
                   (candidate_id, attr_key, attr_value, attr_type) VALUES (?, ?, ?, ?)""",
                (candidate_id, attr_key, attr_value, attr_type))
            return cursor.lastrowid

    def insert_extra_attributes_batch(self, candidate_id: int,
                                      attrs: Dict[str, Any]) -> None:
        """批量插入动态扩展属性

        attrs: {"gpa": "3.8", "hobbies": "篮球,游泳", "target_job": "后端开发", ...}
        """
        if not attrs:
            return
        with self._get_conn() as conn:
            for key, value in attrs.items():
                if value is None:
                    continue
                # 推断类型
                if isinstance(value, (int, float)):
                    attr_type = "number"
                elif isinstance(value, bool):
                    attr_type = "bool"
                elif isinstance(value, list):
                    attr_type = "list"
                    value = json.dumps(value, ensure_ascii=False)
                else:
                    attr_type = "str"
                conn.execute(
                    """INSERT INTO candidate_extra_attributes
                       (candidate_id, attr_key, attr_value, attr_type) VALUES (?, ?, ?, ?)""",
                    (candidate_id, key, str(value), attr_type))

    def get_extra_attributes(self, candidate_id: int) -> Dict[str, Any]:
        """获取候选人的所有动态扩展属性，返回 dict"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT attr_key, attr_value, attr_type FROM candidate_extra_attributes WHERE candidate_id = ?",
                (candidate_id,)).fetchall()
            result = {}
            for r in rows:
                key, val, atype = r["attr_key"], r["attr_value"], r["attr_type"]
                if atype == "number":
                    try:
                        val = float(val) if "." in val else int(val)
                    except (ValueError, TypeError):
                        pass
                elif atype == "bool":
                    val = val.lower() in ("true", "1", "yes")
                elif atype == "list":
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
                result[key] = val
            return result

    def update_resume_raw_text(self, candidate_id: int, raw_text: str) -> None:
        """更新候选人的简历原始文本"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE candidates SET resume_raw_text = ? WHERE id = ?",
                (raw_text, candidate_id))

    # ═══════════════════════════════════════════════════════════════════════════
    # 查询方法
    # ═══════════════════════════════════════════════════════════════════════════

    def get_candidate(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        """获取完整候选人信息(含教育经历、技能、获奖证书、工作经历、项目)"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
            if not row:
                return None
            cand = dict(row)

            # 兼容新旧表结构：确保 highest_education 字段始终有值
            if not cand.get("highest_education") and cand.get("education_level"):
                cand["highest_education"] = cand["education_level"]
            elif not cand.get("education_level") and cand.get("highest_education"):
                cand["education_level"] = cand["highest_education"]

            cand["education_history"] = [dict(r) for r in
                conn.execute("SELECT * FROM education_history WHERE candidate_id = ? ORDER BY start_date",
                            (candidate_id,)).fetchall()]
            cand["skills"] = [dict(r) for r in
                conn.execute("SELECT * FROM skills WHERE candidate_id = ?", (candidate_id,)).fetchall()]
            cand["awards_certificates"] = [dict(r) for r in
                conn.execute("SELECT * FROM awards_certificates WHERE candidate_id = ?", (candidate_id,)).fetchall()]
            cand["work_experiences"] = [dict(r) for r in
                conn.execute("SELECT * FROM work_experiences WHERE candidate_id = ?", (candidate_id,)).fetchall()]
            cand["projects"] = [dict(r) for r in
                conn.execute("SELECT * FROM projects WHERE candidate_id = ?", (candidate_id,)).fetchall()]
            cand["publications"] = [dict(r) for r in
                conn.execute("SELECT * FROM publications WHERE candidate_id = ? ORDER BY year DESC",
                            (candidate_id,)).fetchall()]
            cand["conferences"] = [dict(r) for r in
                conn.execute("SELECT * FROM conferences WHERE candidate_id = ? ORDER BY year DESC",
                            (candidate_id,)).fetchall()]
            # 动态扩展属性
            cand["extra_attributes"] = self.get_extra_attributes(candidate_id)
            return cand

    def get_candidate_by_name_number(self, name_number: int) -> Optional[Dict[str, Any]]:
        """按名字中的数字查找候选人（如 name_number=2186 → 查找 name 包含 '2186' 的候选人）

        前端从聊天消息中提取"候选人_2186"中的数字，但这个数字不一定等于数据库 ID。
        此方法通过名字匹配找到正确的候选人。
        """
        with self._get_conn() as conn:
            # 精确匹配：候选人_XXXX 格式
            patterns = [
                f"候选人_{name_number}",
                f"候选人_{name_number:04d}",  # 带前导零
            ]
            for pattern in patterns:
                row = conn.execute("SELECT id FROM candidates WHERE name = ?", (pattern,)).fetchone()
                if row:
                    return self.get_candidate(row[0])

            # 模糊匹配：名字中包含该数字
            row = conn.execute(
                "SELECT id FROM candidates WHERE name LIKE ?",
                (f"%{name_number}%",)
            ).fetchone()
            if row:
                return self.get_candidate(row[0])

        return None

    def search_candidates(self, filters: Dict[str, Any] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """搜索候选人"""
        query = "SELECT * FROM candidates"
        params = []
        conditions = []
        if filters:
            if "highest_education" in filters:
                conditions.append("highest_education = ?")
                params.append(filters["highest_education"])
            if "education_level" in filters:
                conditions.append("highest_education = ?")
                params.append(filters["education_level"])
            if "min_work_years" in filters:
                conditions.append("work_years >= ?")
                params.append(filters["min_work_years"])
            if "school" in filters:
                # 需要联合education_history表查询
                conditions.append("""id IN (
                    SELECT candidate_id FROM education_history WHERE school LIKE ?
                )""")
                params.append(f"%{filters['school']}%")
            if "location" in filters:
                conditions.append("location LIKE ?")
                params.append(f"%{filters['location']}%")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" LIMIT {limit}"
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_all_candidates_count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    # ═══════════════════════════════════════════════════════════════════════════
    # 匹配历史与反馈
    # ═══════════════════════════════════════════════════════════════════════════

    def insert_matching_history(self, query: str, candidate_ids: List[int],
                                scores: List[float], latency_ms: float) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO matching_history (query, candidate_ids, scores, latency_ms)
                   VALUES (?, ?, ?, ?)""",
                (query, json.dumps(candidate_ids), json.dumps(scores), latency_ms))
            return cursor.lastrowid

    def update_feedback(self, history_id: int, feedback: int):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE matching_history SET feedback = ?, success = ? WHERE id = ?",
                (feedback, 1 if feedback >= 1 else 0, history_id))

    def get_recent_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM matching_history WHERE feedback IS NOT NULL ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_performance_stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM matching_history").fetchone()[0]
            positive = conn.execute(
                "SELECT COUNT(*) FROM matching_history WHERE success = 1").fetchone()[0]
            avg_latency = conn.execute(
                "SELECT AVG(latency_ms) FROM matching_history WHERE latency_ms > 0").fetchone()[0]
            return {
                "total_queries": total,
                "positive_feedback": positive,
                "satisfaction_rate": round(positive / total, 4) if total > 0 else 0,
                "avg_latency_ms": round(avg_latency or 0, 2)
            }

    def record_feedback_standalone(self, session_id: str, message_id: str,
                                    feedback: int, query: str = "") -> Optional[int]:
        """独立记录用户反馈（来自前端点赞/点踩按钮）。

        与 update_feedback 不同，此方法直接插入一条新记录（不依赖已有 matching_history 行）。
        用于前端反馈接口 /api/feedback 的持久化入口。
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO matching_history (query, candidate_ids, scores, feedback, success, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (query or f"[feedback] session={session_id} msg={message_id}",
                 json.dumps([]), json.dumps([]),
                 feedback, 1 if feedback >= 1 else 0, 0))
            return cursor.lastrowid

    def save_system_feedback(self, task_type: str, complexity_score: float,
                             iterations: int, success: bool, response_time: float):
        """保存系统反馈到harness_tasks表"""
        import uuid
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO harness_tasks (task_id, query, status, complexity, iterations, max_iterations)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4())[:8], task_type,
                 "completed" if success else "failed",
                 complexity_score, iterations, 3))


# 全局实例
hr_db = HRDatabase()
