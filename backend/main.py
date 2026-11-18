import asyncio
import json
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import AutonomousAgentOrchestrator
from app.config import RuntimeConfig
from app.database import SQLiteRunStore
from app.schemas import AgentRunRequest, AgentRunResponse

app = FastAPI(
    title="Autonomous Agentic AI System API",
    description=(
        "FastAPI backend for the Agentic AI Engineer Intern Assignment. "
        "Powered by Groq (openai/gpt-oss-120b), Serper Web Search, Tavily Web Crawler, "
        "SentenceTransformers + FAISS Vector RAG, AST Calculator, and SQLite."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

default_config = RuntimeConfig.from_env()
db_store = SQLiteRunStore(default_config.sqlite_db_path)

PRESET_GOALS = [
    {
        "id": "ai_developments",
        "title": "Appendix A.1 — Weekly Tech & AI Developments",
        "goal": "Research and summarize the top 3 developments in Agentic AI and open-weight reasoning models from the last week, including performance benchmarks.",
        "recommended_fault": {
            "enabled": True,
            "target_tool": "serper_web_search",
            "failure_mode": "timeout",
            "fail_count": 1,
        },
    },
    {
        "id": "competitive_brief",
        "title": "Appendix A.4 — Competitive Landscape Brief",
        "goal": "Given the company Groq, produce a competitive-landscape brief comparing its LPU inference architecture against NVIDIA and Cerebras using public web data.",
        "recommended_fault": {
            "enabled": True,
            "target_tool": "serper_web_search",
            "failure_mode": "http_503_rate_limit",
            "fail_count": 1,
        },
    },
    {
        "id": "tokyo_itinerary",
        "title": "Appendix A.3 — 3-Day City Itinerary & Budget",
        "goal": "Plan a 3-day itinerary for Tokyo respecting a $1500 total budget and interests in robotics, historic temples, and ramen.",
        "recommended_fault": {
            "enabled": True,
            "target_tool": "tavily_web_crawler",
            "failure_mode": "malformed_payload",
            "fail_count": 1,
        },
    },
    {
        "id": "github_code_quality",
        "title": "Appendix A.2 — Repository Architecture & Quality Review",
        "goal": "Analyze key architectural patterns, common code quality bottlenecks, and remediation best practices for production FastAPI + FAISS RAG repositories.",
        "recommended_fault": {
            "enabled": False,
            "target_tool": "serper_web_search",
            "failure_mode": "timeout",
            "fail_count": 1,
        },
    },
]


@app.get("/api/health")
async def health_check() -> Dict:
    cfg = RuntimeConfig.from_env()
    return {
        "status": "ok",
        "llm_provider": "Groq API",
        "llm_model": cfg.groq_model,
        "embedding_model": cfg.embedding_model,
        "vector_store": "FAISS (IndexFlatIP)",
        "storage": "SQLite",
        "configured_keys": {
            "groq_api_key": bool(cfg.groq_api_key and cfg.groq_api_key != "your_groq_api_key_here"),
            "serper_api_key": bool(cfg.serper_api_key and cfg.serper_api_key != "your_serper_api_key_here"),
            "tavily_api_key": bool(cfg.tavily_api_key and cfg.tavily_api_key != "your_tavily_api_key_here"),
        },
    }


@app.get("/api/presets")
async def get_presets() -> List[Dict]:
    return PRESET_GOALS


@app.post("/api/runs", response_model=AgentRunResponse)
async def create_agent_run(request: AgentRunRequest) -> AgentRunResponse:
    cfg = RuntimeConfig.from_env(overrides=request.api_keys)
    orchestrator = AutonomousAgentOrchestrator(cfg)
    result = await orchestrator.execute_goal(request)
    db_store.save_run(result)
    return result


@app.post("/api/runs/stream")
async def stream_agent_run(request: AgentRunRequest):
    """Streams live planning, tool calls, self-corrections, and final report via Server-Sent Events (SSE)."""
    cfg = RuntimeConfig.from_env(overrides=request.api_keys)
    orchestrator = AutonomousAgentOrchestrator(cfg)
    queue: asyncio.Queue = asyncio.Queue()

    async def on_event(event_data: Dict):
        await queue.put(event_data)

    async def runner():
        try:
            result = await orchestrator.execute_goal(request, on_event=on_event)
            db_store.save_run(result)
            await queue.put({"event": "final_result", "result": result.model_dump()})
        except Exception as exc:
            await queue.put({"event": "fatal_error", "message": str(exc)})
        finally:
            await queue.put(None)

    asyncio.create_task(runner())

    async def event_generator():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/runs")
async def list_saved_runs(limit: int = 25) -> List[Dict]:
    return db_store.list_runs(limit=limit)


@app.get("/api/runs/{run_id}", response_model=AgentRunResponse)
async def get_saved_run(run_id: str) -> AgentRunResponse:
    run = db_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in SQLite database.")
    return run
