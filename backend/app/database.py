from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from app.schemas import AgentRunResponse


class SQLiteRunStore:
    """SQLite storage engine for persisting agent execution runs, planning traces, tool logs, and reports."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return closing(conn)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL,
                    self_corrections_json TEXT NOT NULL,
                    evaluation_json TEXT NOT NULL,
                    structured_report_json TEXT NOT NULL,
                    markdown_report TEXT NOT NULL,
                    vector_chunks_indexed INTEGER DEFAULT 0,
                    total_duration_ms REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_run(self, run: AgentRunResponse) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_runs (
                    run_id, goal, model_used, status, plan_json,
                    tool_calls_json, self_corrections_json, evaluation_json,
                    structured_report_json, markdown_report,
                    vector_chunks_indexed, total_duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.goal,
                    run.model_used,
                    run.status,
                    run.plan.model_dump_json(),
                    json.dumps([tc.model_dump() for tc in run.tool_calls]),
                    json.dumps([sc.model_dump() for sc in run.self_corrections]),
                    run.evaluation.model_dump_json(),
                    run.structured_report.model_dump_json(),
                    run.markdown_report,
                    run.vector_chunks_indexed,
                    run.total_duration_ms,
                    run.created_at,
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[AgentRunResponse]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_response(row)

    def list_runs(self, limit: int = 25) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT run_id, goal, model_used, status,
                       self_corrections_json, tool_calls_json,
                       vector_chunks_indexed, total_duration_ms, created_at
                FROM agent_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            summaries = []
            for r in rows:
                corrections = json.loads(r["self_corrections_json"])
                tool_calls = json.loads(r["tool_calls_json"])
                summaries.append(
                    {
                        "run_id": r["run_id"],
                        "goal": r["goal"],
                        "model_used": r["model_used"],
                        "status": r["status"],
                        "tool_calls_count": len(tool_calls),
                        "self_corrections_count": len(corrections),
                        "vector_chunks_indexed": r["vector_chunks_indexed"],
                        "total_duration_ms": r["total_duration_ms"],
                        "created_at": r["created_at"],
                    }
                )
            return summaries

    def _row_to_response(self, row: sqlite3.Row) -> AgentRunResponse:
        return AgentRunResponse(
            run_id=row["run_id"],
            goal=row["goal"],
            model_used=row["model_used"],
            status=row["status"],
            plan=json.loads(row["plan_json"]),
            tool_calls=json.loads(row["tool_calls_json"]),
            self_corrections=json.loads(row["self_corrections_json"]),
            evaluation=json.loads(row["evaluation_json"]),
            structured_report=json.loads(row["structured_report_json"]),
            markdown_report=row["markdown_report"],
            vector_chunks_indexed=row["vector_chunks_indexed"],
            total_duration_ms=row["total_duration_ms"],
            created_at=row["created_at"],
        )
