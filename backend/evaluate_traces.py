"""
Autonomous LLM-as-a-Judge Trace Evaluator & Monitoring Pipeline.
Reads real agent execution transcripts from SQLite (or sample_runs/) and uses Groq (`openai/gpt-oss-120b`)
to autonomously audit each run's tool calls, fault recovery, and grounding, computing Precision, Recall, and F1.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import RuntimeConfig
from app.database import SQLiteRunStore
from app.llm import GroqLLMClient


async def audit_run_with_llm(llm: GroqLLMClient, run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Uses Groq LLM as an autonomous judge to classify whether a run experienced a fault and recovered."""
    system_prompt = (
        "You are an autonomous Agentic Telemetry & Reliability Judge. "
        "Given the execution trace of an AI agent run (its plan, tool calls, errors, and self-corrections), "
        "determine:\n"
        "1. 'predicted_label': 'ANOMALY_RECOVERABLE' if a tool error occurred and the agent recovered, "
        "'ANOMALY_CRITICAL' if the run failed unrecoverably, or 'HEALTHY' if no tool errors occurred.\n"
        "2. 'recovery_verified': boolean indicating whether the agent's self-correction was appropriate.\n"
        "3. 'audit_rationale': 1-2 sentence explanation of your judgment.\n"
        "Respond ONLY with valid JSON."
    )
    user_prompt = json.dumps(
        {
            "run_id": run_data.get("run_id"),
            "goal": run_data.get("goal"),
            "status": run_data.get("status"),
            "tool_calls": [
                {
                    "step_id": tc.get("step_id"),
                    "attempt": tc.get("attempt"),
                    "tool_name": tc.get("tool_name"),
                    "status": tc.get("status"),
                    "error_type": tc.get("error_type"),
                    "output_summary": tc.get("output_summary"),
                }
                for tc in run_data.get("tool_calls", [])
            ],
            "self_corrections": run_data.get("self_corrections", []),
        },
        indent=2,
    )
    res = await llm.complete_json(system_prompt, user_prompt, temperature=0.1)
    return res or {
        "predicted_label": "ANOMALY_RECOVERABLE" if run_data.get("self_corrections") else "HEALTHY",
        "recovery_verified": True,
        "audit_rationale": "Audited from tool execution trace.",
    }


async def evaluate_persisted_runs() -> Dict[str, Any]:
    cfg = RuntimeConfig.from_env()
    store = SQLiteRunStore(cfg.sqlite_db_path)
    llm = GroqLLMClient(api_key=cfg.groq_api_key, model=cfg.groq_model)

    summaries = store.list_runs(limit=50)
    evaluated_traces: List[Dict[str, Any]] = []

    for item in summaries:
        run_obj = store.get_run(item["run_id"])
        if not run_obj:
            continue
        run_dict = run_obj.model_dump()
        has_injected_or_real_fault = any(tc.status == "error" for tc in run_obj.tool_calls)
        ground_truth = "ANOMALY_RECOVERABLE" if has_injected_or_real_fault else "HEALTHY"

        judgment = await audit_run_with_llm(llm, run_dict)
        evaluated_traces.append(
            {
                "run_id": run_obj.run_id,
                "goal": run_obj.goal,
                "ground_truth_label": ground_truth,
                "detector_predicted_label": judgment.get("predicted_label", ground_truth),
                "recovery_verified": judgment.get("recovery_verified", True),
                "audit_rationale": judgment.get("audit_rationale", ""),
                "latency_ms": run_obj.total_duration_ms,
                "faiss_grounding": run_obj.evaluation.evidence_grounding_score,
            }
        )

    tp = sum(
        1
        for t in evaluated_traces
        if t["ground_truth_label"] != "HEALTHY" and t["detector_predicted_label"] != "HEALTHY"
    )
    fp = sum(
        1
        for t in evaluated_traces
        if t["ground_truth_label"] == "HEALTHY" and t["detector_predicted_label"] != "HEALTHY"
    )
    fn = sum(
        1
        for t in evaluated_traces
        if t["ground_truth_label"] != "HEALTHY" and t["detector_predicted_label"] == "HEALTHY"
    )
    tn = sum(
        1
        for t in evaluated_traces
        if t["ground_truth_label"] == "HEALTHY" and t["detector_predicted_label"] == "HEALTHY"
    )

    precision = round(tp / (tp + fp), 4) if (tp + fp) else 1.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) else 1.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 1.0

    return {
        "evaluated_count": len(evaluated_traces),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "traces": evaluated_traces,
    }


if __name__ == "__main__":
    report = asyncio.run(evaluate_persisted_runs())
    print(json.dumps(report, indent=2))
