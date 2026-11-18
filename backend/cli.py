import argparse
import asyncio
import json
from pathlib import Path

from app.agent.orchestrator import AutonomousAgentOrchestrator
from app.config import RuntimeConfig
from app.database import SQLiteRunStore
from app.schemas import AgentRunRequest, FaultInjectionConfig


async def run_cli():
    parser = argparse.ArgumentParser(
        description="Autonomous Agentic AI System CLI Runner"
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="High-level natural language goal for the agent",
    )
    parser.add_argument(
        "--inject-fault",
        action="store_true",
        help="Deliberately inject a tool failure to demonstrate self-correction",
    )
    parser.add_argument(
        "--fault-tool",
        type=str,
        default="serper_web_search",
        choices=["serper_web_search", "tavily_web_crawler", "faiss_rag_memory", "calculator_evaluator"],
        help="Target tool for fault injection",
    )
    parser.add_argument(
        "--fault-mode",
        type=str,
        default="http_503_rate_limit",
        choices=["timeout", "http_503_rate_limit", "malformed_payload", "empty_results"],
        help="Type of fault to inject",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional path prefix to export .json and .md run transcripts",
    )
    args = parser.parse_args()

    cfg = RuntimeConfig.from_env()
    store = SQLiteRunStore(cfg.sqlite_db_path)
    orchestrator = AutonomousAgentOrchestrator(cfg)

    req = AgentRunRequest(
        goal=args.goal,
        fault_injection=FaultInjectionConfig(
            enabled=args.inject_fault,
            target_tool=args.fault_tool,
            failure_mode=args.fault_mode,
            fail_count=1,
        ),
    )

    print(f"\n[AGENT START] Goal: {args.goal}")
    print(f"[MODEL] {cfg.groq_model} | Fault Injection: {args.inject_fault} ({args.fault_tool} -> {args.fault_mode})")

    async def print_event(ev: dict):
        etype = ev.get("event")
        if etype == "plan_created":
            plan = ev["plan"]
            print(f"\n=== VISIBLE PLANNING TRACE ({plan['domain_classification']}) ===")
            print(f"Reasoning: {plan['reasoning_trace']}")
            for s in plan["steps"]:
                print(f"  Step {s['step_id']}: [{s['tool_name']}] {s['title']} -> query='{s['tool_query']}'")
        elif etype == "tool_call_logged":
            tc = ev["tool_call"]
            print(f"  -> Tool Call (Step {tc['step_id']}, Attempt {tc['attempt']}): {tc['tool_name']} [{tc['status'].upper()}] ({tc['latency_ms']}ms)")
            print(f"     Summary: {tc['output_summary']}")
        elif etype == "self_correction_triggered":
            sc = ev["self_correction"]
            print(f"  [SELF-CORRECTION] Step {sc['step_id']} failed on {sc['failed_tool']}!")
            print(f"     Diagnosis: {sc['agent_diagnosis']}")
            print(f"     Recovery: Strategy={sc['recovery_strategy']} -> Rerouting to '{sc['recovery_tool']}' with query='{sc['recovery_action']}'")

    result = await orchestrator.execute_goal(req, on_event=print_event)
    store.save_run(result)

    print(f"\n=== RUN COMPLETED ({result.run_id}) in {result.total_duration_ms}ms ===")
    print(f"FAISS Chunks Indexed: {result.vector_chunks_indexed} | Self-Corrections: {len(result.self_corrections)}")

    if args.output_prefix:
        out_path = Path(args.output_prefix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_file = out_path.with_suffix(".json")
        md_file = out_path.with_suffix(".md")
        json_file.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
        md_file.write_text(result.markdown_report, encoding="utf-8")
        print(f"Saved transcripts to:\n  - {json_file}\n  - {md_file}")


if __name__ == "__main__":
    asyncio.run(run_cli())
