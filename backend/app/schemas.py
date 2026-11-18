from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class FaultInjectionConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether deliberate failure injection is active")
    target_tool: Literal[
        "serper_web_search",
        "tavily_web_crawler",
        "faiss_rag_memory",
        "calculator_evaluator",
        "any",
    ] = Field(default="serper_web_search", description="Which tool should fail initially")
    failure_mode: Literal[
        "timeout",
        "http_503_rate_limit",
        "malformed_payload",
        "empty_results",
    ] = Field(default="timeout", description="Type of failure to simulate on first invocation")
    fail_count: int = Field(default=1, ge=1, le=3, description="Number of times to fail before allowing recovery")


class PlanStep(BaseModel):
    step_id: int
    title: str
    description: str
    tool_name: Literal[
        "serper_web_search",
        "tavily_web_crawler",
        "faiss_rag_memory",
        "calculator_evaluator",
    ]
    tool_query: str
    expected_outcome: str
    status: Literal["pending", "running", "completed", "corrected", "failed"] = "pending"


class ExecutionPlan(BaseModel):
    goal: str
    domain_classification: str
    reasoning_trace: str
    assumptions: List[str] = Field(default_factory=list)
    steps: List[PlanStep]


class ToolCallRecord(BaseModel):
    call_id: str
    step_id: int
    attempt: int
    tool_name: str
    tool_input: Dict[str, Any]
    status: Literal["success", "error", "fallback_success"]
    is_injected_fault: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    output_summary: str = ""
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: str


class SelfCorrectionRecord(BaseModel):
    correction_id: str
    step_id: int
    failed_tool: str
    failure_reason: str
    agent_diagnosis: str
    recovery_strategy: Literal[
        "retry_with_modified_params",
        "cross_tool_fallback",
        "query_reformulation",
        "graceful_degradation",
    ]
    recovery_action: str
    recovery_tool: str
    resolved: bool
    timestamp: str


class SelfEvaluationMetrics(BaseModel):
    goal_coverage_score: float = Field(ge=0.0, le=1.0)
    evidence_grounding_score: float = Field(ge=0.0, le=1.0)
    plan_adherence_ratio: float = Field(ge=0.0, le=1.0)
    resilience_verified: bool
    critique_notes: str


class StructuredReportSummary(BaseModel):
    executive_summary: str
    key_findings: List[Dict[str, str]]
    quantitative_insights: List[str] = Field(default_factory=list)
    what_the_agent_did_and_why: List[str]
    sources_cited: List[Dict[str, str]]
    limitations_and_assumptions: List[str]


class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=5, description="High-level natural language goal for the agent")
    fault_injection: FaultInjectionConfig = Field(default_factory=FaultInjectionConfig)
    api_keys: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional runtime overrides for groq_api_key, serper_api_key, tavily_api_key",
    )


class AgentRunResponse(BaseModel):
    run_id: str
    goal: str
    model_used: str
    status: Literal["completed", "partial", "failed"]
    plan: ExecutionPlan
    tool_calls: List[ToolCallRecord]
    self_corrections: List[SelfCorrectionRecord]
    evaluation: SelfEvaluationMetrics
    structured_report: StructuredReportSummary
    markdown_report: str
    vector_chunks_indexed: int
    total_duration_ms: float
    created_at: str
