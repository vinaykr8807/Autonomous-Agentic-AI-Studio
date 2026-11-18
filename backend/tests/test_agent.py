import asyncio
import tempfile
from pathlib import Path
import pytest

from app.agent.orchestrator import AutonomousAgentOrchestrator
from app.config import RuntimeConfig
from app.database import SQLiteRunStore
from app.schemas import AgentRunRequest, FaultInjectionConfig
from app.tools import CalculatorEvaluatorTool, FAISSRagTool, FaultInjector, ToolExecutionError
from app.vector_store import FAISSVectorMemory


def test_calculator_tool_expressions():
    async def _run():
        calc = CalculatorEvaluatorTool()
        res1 = await calc.execute("round((3 * 180.5) + 250, 2)")
        assert res1["result"] == 791.5

        res2 = await calc.execute("cagr(100, 200, 3)")
        assert 25.9 < res2["result"] < 26.1

        res3 = await calc.execute("mean([120, 150, 180])")
        assert res3["result"] == 150

    asyncio.run(_run())


def test_faiss_vector_memory_indexing_and_search():
    async def _run():
        memory = FAISSVectorMemory(model_name="all-MiniLM-L6-v2")
        added = memory.add_documents(
            [
                {
                    "title": "Groq LPU Inference Architecture",
                    "url": "https://groq.com/lpu-architecture",
                    "content": "Groq Language Processing Units (LPUs) deliver deterministic low-latency token generation using SRAM on-chip memory.",
                    "source_tool": "tavily_web_crawler",
                },
                {
                    "title": "Tokyo Ramen & Robotics Guide",
                    "url": "https://example.org/tokyo-guide",
                    "content": "Visit Akihabara and Odaiba for robotics exhibitions and Shinjuku for Michelin-guide ramen shops under $15.",
                    "source_tool": "serper_web_search",
                },
            ]
        )
        assert added == 2
        assert memory.total_chunks == 2

        rag_tool = FAISSRagTool(memory)
        query_out = await rag_tool.execute("deterministic SRAM inference latency LPU", top_k=2)
        assert len(query_out["top_matches"]) == 2
        assert "Groq" in query_out["top_matches"][0]["title"]
        assert query_out["top_matches"][0]["similarity_score"] > query_out["top_matches"][1]["similarity_score"]

    asyncio.run(_run())


def test_fault_injector_triggers_exact_count():
    injector = FaultInjector(
        FaultInjectionConfig(
            enabled=True,
            target_tool="serper_web_search",
            failure_mode="http_503_rate_limit",
            fail_count=1,
        )
    )
    with pytest.raises(ToolExecutionError) as exc_info:
        injector.check_and_inject("serper_web_search")
    assert exc_info.value.is_injected is True
    assert exc_info.value.error_type == "HTTP503ServiceUnavailable"

    # Second call passes without raising since fail_count=1
    injector.check_and_inject("serper_web_search")


def test_end_to_end_orchestrator_with_fault_recovery_and_sqlite():
    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_runs.db")
            cfg = RuntimeConfig.from_env(overrides={"sqlite_db_path": db_path})
            orchestrator = AutonomousAgentOrchestrator(cfg)
            request = AgentRunRequest(
                goal="Given the company Groq, produce a competitive-landscape brief using public web data.",
                fault_injection=FaultInjectionConfig(
                    enabled=True,
                    target_tool="serper_web_search",
                    failure_mode="http_503_rate_limit",
                    fail_count=1,
                ),
            )

            result = await orchestrator.execute_goal(request)
            assert result.status == "completed"
            assert len(result.plan.steps) >= 3
            assert len(result.self_corrections) >= 1
            assert result.self_corrections[0].failed_tool == "serper_web_search"
            assert result.self_corrections[0].recovery_tool == "tavily_web_crawler"
            assert result.self_corrections[0].resolved is True
            assert result.vector_chunks_indexed > 0
            assert "Executive Summary" in result.markdown_report

            store = SQLiteRunStore(db_path)
            store.save_run(result)
            loaded = store.get_run(result.run_id)
            assert loaded is not None
            assert loaded.run_id == result.run_id
            assert len(loaded.self_corrections) == len(result.self_corrections)

    asyncio.run(_run())
