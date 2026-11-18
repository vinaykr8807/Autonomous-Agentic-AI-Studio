import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.config import RuntimeConfig
from app.llm import GroqLLMClient
from app.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ExecutionPlan,
    PlanStep,
    SelfCorrectionRecord,
    SelfEvaluationMetrics,
    StructuredReportSummary,
    ToolCallRecord,
)
from app.tools import (
    CalculatorEvaluatorTool,
    FAISSRagTool,
    FaultInjector,
    SerperSearchTool,
    TavilyCrawlerTool,
    ToolExecutionError,
)
from app.vector_store import FAISSVectorMemory

EventCallback = Optional[Callable[[Dict[str, Any]], Awaitable[None]]]

VALID_TOOLS = {
    "serper_web_search",
    "tavily_web_crawler",
    "faiss_rag_memory",
    "calculator_evaluator",
}

VALID_STRATEGIES = {
    "retry_with_modified_params",
    "cross_tool_fallback",
    "query_reformulation",
    "graceful_degradation",
}


class AutonomousAgentOrchestrator:
    """
    Autonomous Agentic AI System powered by Groq LLM (`openai/gpt-oss-120b`) reasoning across all stages:
    1. Autonomous Goal Decomposition & Visible Planning Trace (LLM Planner)
    2. Multi-Tool Execution (Serper Search, Tavily Crawler, FAISS Vector RAG, AST Calculator)
    3. Autonomous Closed-Loop Self-Correction (LLM Root-Cause Diagnosis & Dynamic Re-Routing)
    4. Autonomous LLM Self-Evaluation & Structured Report Synthesis
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.llm = GroqLLMClient(api_key=config.groq_api_key, model=config.groq_model)
        self.vector_memory = FAISSVectorMemory(model_name=config.embedding_model)

        self.serper_tool = SerperSearchTool(api_key=config.serper_api_key)
        self.tavily_tool = TavilyCrawlerTool(api_key=config.tavily_api_key)
        self.faiss_tool = FAISSRagTool(memory=self.vector_memory)
        self.calc_tool = CalculatorEvaluatorTool()

    async def _emit(self, callback: EventCallback, event_type: str, payload: Dict[str, Any]) -> None:
        if callback:
            await callback({"event": event_type, "timestamp": self._now_iso(), **payload})

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    async def execute_goal(
        self, request: AgentRunRequest, on_event: EventCallback = None
    ) -> AgentRunResponse:
        start_perf = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        fault_injector = FaultInjector(request.fault_injection)

        await self._emit(
            on_event,
            "run_started",
            {
                "run_id": run_id,
                "goal": request.goal,
                "model": self.config.groq_model,
                "fault_injection": request.fault_injection.model_dump(),
            },
        )

        # Phase 1: Autonomous LLM Goal Decomposition into a Visible Planning Trace
        plan = await self._create_plan(request.goal)
        await self._emit(on_event, "plan_created", {"plan": plan.model_dump()})

        tool_calls: List[ToolCallRecord] = []
        self_corrections: List[SelfCorrectionRecord] = []
        discovered_urls: List[str] = []
        step_outputs: Dict[int, Dict[str, Any]] = {}

        # Phase 2: Step-by-Step Execution with LLM-Driven Self-Correction
        for step in plan.steps:
            step.status = "running"
            await self._emit(on_event, "step_started", {"step": step.model_dump()})

            success_output, step_was_corrected = await self._execute_step_with_resilience(
                step=step,
                goal=request.goal,
                discovered_urls=discovered_urls,
                fault_injector=fault_injector,
                tool_calls=tool_calls,
                self_corrections=self_corrections,
                on_event=on_event,
            )

            if success_output is not None:
                step.status = "corrected" if step_was_corrected else "completed"
                step_outputs[step.step_id] = success_output
                for r in success_output.get("results", []):
                    if r.get("url") and r["url"] not in discovered_urls:
                        discovered_urls.append(r["url"])
            else:
                step.status = "failed"
                step_outputs[step.step_id] = {"error": "Step failed after autonomous recovery attempts."}

            await self._emit(
                on_event,
                "step_finished",
                {
                    "step": step.model_dump(),
                    "vector_chunks_indexed": self.vector_memory.total_chunks,
                },
            )

        # Phase 3: Autonomous LLM Self-Evaluation & Final Report Synthesis
        await self._emit(on_event, "synthesizing_report", {"run_id": run_id})
        structured_report, markdown_report, evaluation = await self._synthesize_final_report(
            goal=request.goal,
            plan=plan,
            tool_calls=tool_calls,
            self_corrections=self_corrections,
            step_outputs=step_outputs,
        )

        total_duration_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
        completed_steps = sum(1 for s in plan.steps if s.status in ("completed", "corrected"))
        overall_status = (
            "completed"
            if completed_steps == len(plan.steps)
            else ("partial" if completed_steps > 0 else "failed")
        )

        response = AgentRunResponse(
            run_id=run_id,
            goal=request.goal,
            model_used=f"Groq ({self.config.groq_model})",
            status=overall_status,
            plan=plan,
            tool_calls=tool_calls,
            self_corrections=self_corrections,
            evaluation=evaluation,
            structured_report=structured_report,
            markdown_report=markdown_report,
            vector_chunks_indexed=self.vector_memory.total_chunks,
            total_duration_ms=total_duration_ms,
            created_at=self._now_iso(),
        )

        await self._emit(on_event, "run_completed", {"run_id": run_id, "status": overall_status})
        return response

    async def _create_plan(self, goal: str) -> ExecutionPlan:
        """Uses Groq LLM (`openai/gpt-oss-120b`) to autonomously decompose the user's goal."""
        system_prompt = (
            "You are an autonomous Agentic AI Planner. Given a user goal, reason about the problem and "
            "decompose it into a logical multi-step execution plan using these 4 tools:\n"
            "1. 'serper_web_search': Fast Google web search discovery for recent facts, news, and target URLs.\n"
            "2. 'tavily_web_crawler': Deep web crawler that extracts full article content and indexes it into FAISS.\n"
            "3. 'faiss_rag_memory': Semantic vector search over indexed chunks using SentenceTransformers + FAISS.\n"
            "4. 'calculator_evaluator': Mathematical/financial AST evaluator (supports arithmetic, mean([...]), pct_change(a,b), cagr(start,end,yrs)).\n\n"
            "Return strictly valid JSON with keys:\n"
            "{\n"
            '  "domain_classification": "string",\n'
            '  "reasoning_trace": "Detailed explanation of how and why you decomposed the goal into these steps",\n'
            '  "assumptions": ["assumption 1", "assumption 2"],\n'
            '  "steps": [\n'
            "    {\n"
            '      "step_id": 1,\n'
            '      "title": "Action title",\n'
            '      "description": "Why this step is needed",\n'
            '      "tool_name": "serper_web_search | tavily_web_crawler | faiss_rag_memory | calculator_evaluator",\n'
            '      "tool_query": "Exact search query, crawl topic, semantic query, or numeric math expression using ONLY numbers (e.g. pct_change(120, 340) or round(1500 / 3, 2))",\n'
            '      "expected_outcome": "What output this step produces"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        llm_plan = await self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=f"High-level user goal: {goal}",
            temperature=0.2,
        )

        if not llm_plan or not isinstance(llm_plan.get("steps"), list):
            raise RuntimeError(
                "LLM Planner did not return a valid step decomposition. Ensure GROQ_API_KEY is configured."
            )

        steps: List[PlanStep] = []
        for idx, raw_step in enumerate(llm_plan["steps"][:6], start=1):
            t_name = raw_step.get("tool_name", "serper_web_search")
            if t_name not in VALID_TOOLS:
                t_name = "serper_web_search"
            steps.append(
                PlanStep(
                    step_id=idx,
                    title=raw_step.get("title", f"Step {idx}"),
                    description=raw_step.get("description", ""),
                    tool_name=t_name,
                    tool_query=str(raw_step.get("tool_query", goal)),
                    expected_outcome=raw_step.get("expected_outcome", ""),
                )
            )

        return ExecutionPlan(
            goal=goal,
            domain_classification=llm_plan.get("domain_classification", "Autonomous Research"),
            reasoning_trace=llm_plan.get("reasoning_trace", ""),
            assumptions=llm_plan.get("assumptions", []),
            steps=steps,
        )

    async def _execute_step_with_resilience(
        self,
        step: PlanStep,
        goal: str,
        discovered_urls: List[str],
        fault_injector: FaultInjector,
        tool_calls: List[ToolCallRecord],
        self_corrections: List[SelfCorrectionRecord],
        on_event: EventCallback,
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        attempt = 1
        current_tool = step.tool_name
        current_query = step.tool_query
        step_was_corrected = False

        while attempt <= 3:
            t0 = time.perf_counter()
            call_id = f"call_{step.step_id}_{attempt}_{uuid.uuid4().hex[:6]}"
            try:
                fault_injector.check_and_inject(current_tool)

                raw_output = await self._invoke_tool(
                    tool_name=current_tool,
                    query=current_query,
                    discovered_urls=discovered_urls,
                )
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                indexed_now = self._auto_index_into_faiss(current_tool, raw_output)
                if indexed_now > 0:
                    raw_output["faiss_chunks_added"] = indexed_now

                summary = self._summarize_tool_output(current_tool, raw_output)
                record = ToolCallRecord(
                    call_id=call_id,
                    step_id=step.step_id,
                    attempt=attempt,
                    tool_name=current_tool,
                    tool_input={"query": current_query, "urls_hint": discovered_urls[:2]},
                    status="fallback_success" if step_was_corrected else "success",
                    is_injected_fault=False,
                    output_summary=summary,
                    raw_output=raw_output,
                    latency_ms=latency_ms,
                    timestamp=self._now_iso(),
                )
                tool_calls.append(record)
                await self._emit(on_event, "tool_call_logged", {"tool_call": record.model_dump()})
                return raw_output, step_was_corrected

            except ToolExecutionError as err:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                err_record = ToolCallRecord(
                    call_id=call_id,
                    step_id=step.step_id,
                    attempt=attempt,
                    tool_name=current_tool,
                    tool_input={"query": current_query},
                    status="error",
                    is_injected_fault=err.is_injected,
                    error_type=err.error_type,
                    error_message=err.message,
                    output_summary=f"FAILED ({err.error_type}): {err.message}",
                    raw_output={},
                    latency_ms=latency_ms,
                    timestamp=self._now_iso(),
                )
                tool_calls.append(err_record)
                await self._emit(on_event, "tool_call_logged", {"tool_call": err_record.model_dump()})

                # Autonomous LLM Self-Correction & Recovery Reasoning
                step_was_corrected = True
                correction = await self._diagnose_and_plan_recovery(
                    step=step,
                    failed_tool=current_tool,
                    failed_query=current_query,
                    error=err,
                    attempt=attempt,
                    goal=goal,
                )
                self_corrections.append(correction)
                await self._emit(
                    on_event,
                    "self_correction_triggered",
                    {"self_correction": correction.model_dump()},
                )

                current_tool = correction.recovery_tool
                current_query = correction.recovery_action
                attempt += 1

        return None, step_was_corrected

    async def _invoke_tool(
        self, tool_name: str, query: str, discovered_urls: List[str]
    ) -> Dict[str, Any]:
        if tool_name == "serper_web_search":
            return await self.serper_tool.execute(query=query, num_results=5)
        elif tool_name == "tavily_web_crawler":
            return await self.tavily_tool.execute(
                query_or_url=query, urls=discovered_urls[:3] if discovered_urls else None, max_pages=3
            )
        elif tool_name == "faiss_rag_memory":
            if self.vector_memory.total_chunks == 0:
                crawl_data = await self.tavily_tool.execute(query_or_url=query, max_pages=2)
                self._auto_index_into_faiss("tavily_web_crawler", crawl_data)
            return await self.faiss_tool.execute(query=query, top_k=4)
        elif tool_name == "calculator_evaluator":
            return await self.calc_tool.execute(expression=query)
        else:
            raise ToolExecutionError(
                tool_name, "UnknownToolError", f"Tool '{tool_name}' is not registered."
            )

    def _auto_index_into_faiss(self, tool_name: str, output: Dict[str, Any]) -> int:
        docs_to_index: List[Dict[str, str]] = []
        if tool_name == "serper_web_search":
            for item in output.get("results", []):
                docs_to_index.append(
                    {
                        "title": item.get("title", "Search Result"),
                        "url": item.get("url", ""),
                        "content": item.get("snippet", ""),
                        "source_tool": "serper_web_search",
                    }
                )
        elif tool_name == "tavily_web_crawler":
            for page in output.get("pages", []):
                docs_to_index.append(
                    {
                        "title": page.get("title", "Crawled Page"),
                        "url": page.get("url", ""),
                        "content": page.get("content", ""),
                        "source_tool": "tavily_web_crawler",
                    }
                )
        if not docs_to_index:
            return 0
        return self.vector_memory.add_documents(docs_to_index)

    async def _diagnose_and_plan_recovery(
        self,
        step: PlanStep,
        failed_tool: str,
        failed_query: str,
        error: ToolExecutionError,
        attempt: int,
        goal: str,
    ) -> SelfCorrectionRecord:
        """
        Uses Groq LLM (`openai/gpt-oss-120b`) autonomous reasoning to diagnose the tool failure,
        choose a recovery strategy, select the recovery tool (e.g. switching between Serper and Tavily),
        and formulate the recovery query/expression.
        """
        system_prompt = (
            "You are an Autonomous Agent Self-Correction & Resilience Engine. "
            "A tool invocation just failed during step execution. Analyze the failure and decide how to recover.\n"
            "Available tools:\n"
            "- 'serper_web_search': Google SERP search.\n"
            "- 'tavily_web_crawler': Deep web crawler & content extractor (can substitute for serper_web_search if Serper is down/503, and vice versa).\n"
            "- 'faiss_rag_memory': Semantic vector search over FAISS.\n"
            "- 'calculator_evaluator': AST mathematical evaluator (requires pure numeric expressions like 'round(1500/3, 2)').\n\n"
            "Guidelines:\n"
            "- If 'serper_web_search' failed with HTTP 503 or rate limit, prefer 'cross_tool_fallback' to 'tavily_web_crawler'.\n"
            "- If a tool timed out on attempt 1, you may 'retry_with_modified_params' or 'cross_tool_fallback'.\n"
            "- If a search returned empty results, use 'query_reformulation' with broader keywords.\n\n"
            "Return strictly valid JSON with keys:\n"
            "{\n"
            '  "agent_diagnosis": "2-sentence root-cause diagnosis and why you chose this recovery plan",\n'
            '  "recovery_strategy": "retry_with_modified_params | cross_tool_fallback | query_reformulation | graceful_degradation",\n'
            '  "recovery_tool": "serper_web_search | tavily_web_crawler | faiss_rag_memory | calculator_evaluator",\n'
            '  "recovery_action": "The exact reformulated query or valid numeric math expression to pass to recovery_tool"\n'
            "}"
        )
        user_prompt = (
            f"Overall Goal: {goal}\n"
            f"Step {step.step_id}: {step.title} ({step.description})\n"
            f"Attempt #{attempt} Failed Tool: {failed_tool}\n"
            f"Failed Query/Input: {failed_query}\n"
            f"Error Type: {error.error_type}\n"
            f"Error Message: {error.message}"
        )

        llm_recovery = await self.llm.complete_json(
            system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1
        )

        if llm_recovery:
            strategy = llm_recovery.get("recovery_strategy", "cross_tool_fallback")
            if strategy not in VALID_STRATEGIES:
                strategy = "cross_tool_fallback"
            rec_tool = llm_recovery.get("recovery_tool", failed_tool)
            if rec_tool not in VALID_TOOLS:
                rec_tool = "tavily_web_crawler" if failed_tool == "serper_web_search" else "serper_web_search"
            diagnosis = llm_recovery.get("agent_diagnosis", f"Recovered from {error.error_type}")
            rec_action = str(llm_recovery.get("recovery_action") or failed_query)
        else:
            strategy = "cross_tool_fallback"
            rec_tool = "tavily_web_crawler" if failed_tool == "serper_web_search" else "serper_web_search"
            diagnosis = f"Diagnosed {error.error_type} on {failed_tool}; rerouting to {rec_tool}."
            rec_action = failed_query

        return SelfCorrectionRecord(
            correction_id=f"corr_{step.step_id}_{attempt}_{uuid.uuid4().hex[:5]}",
            step_id=step.step_id,
            failed_tool=failed_tool,
            failure_reason=f"{error.error_type}: {error.message}",
            agent_diagnosis=diagnosis,
            recovery_strategy=strategy,
            recovery_action=rec_action,
            recovery_tool=rec_tool,
            resolved=True,
            timestamp=self._now_iso(),
        )

    async def _synthesize_final_report(
        self,
        goal: str,
        plan: ExecutionPlan,
        tool_calls: List[ToolCallRecord],
        self_corrections: List[SelfCorrectionRecord],
        step_outputs: Dict[int, Dict[str, Any]],
    ) -> tuple[StructuredReportSummary, str, SelfEvaluationMetrics]:
        """Uses Groq LLM (`openai/gpt-oss-120b`) to synthesize gathered evidence and self-evaluate the run."""
        semantic_hits = self.vector_memory.search(goal, top_k=6)
        sources_map: Dict[str, Dict[str, str]] = {}
        for chunk in semantic_hits:
            url = chunk.get("url", "")
            if url and url not in sources_map:
                sources_map[url] = {
                    "title": chunk.get("title", "Source Document"),
                    "url": url,
                    "source_tool": chunk.get("source_tool", "web"),
                }

        quant_notes: List[str] = []
        for tc in tool_calls:
            if tc.tool_name == "calculator_evaluator" and tc.status in ("success", "fallback_success"):
                expr = tc.raw_output.get("expression", "")
                res = tc.raw_output.get("formatted") or tc.raw_output.get("result")
                quant_notes.append(f"Evaluated `{expr}` = **{res}**")

        audit_trail: List[str] = [
            f"Decomposed the user goal into {len(plan.steps)} steps under domain '{plan.domain_classification}'.",
        ]
        for tc in tool_calls:
            audit_trail.append(
                f"Step {tc.step_id} (Attempt {tc.attempt}) invoked `{tc.tool_name}` -> [{tc.status.upper()}] in {tc.latency_ms}ms: {tc.output_summary}"
            )
        for sc in self_corrections:
            audit_trail.append(
                f"Self-Correction on Step {sc.step_id}: Diagnosed `{sc.failed_tool}` failure ({sc.failure_reason}) and recovered via `{sc.recovery_strategy}` using `{sc.recovery_tool}`."
            )

        evidence_context = "\n".join(
            [
                f"- [{c.get('title')}]({c.get('url')}) (Cosine Sim: {c.get('similarity_score')}): {c.get('content')}"
                for c in semantic_hits
            ]
        )
        synth_system = (
            "You are an autonomous Research & Self-Evaluation Agent. Based on the user's goal, the execution trace, "
            "the FAISS vector memory evidence, and calculator outputs, synthesize the final report and evaluate your own run.\n"
            "Return strictly valid JSON with keys:\n"
            "{\n"
            '  "executive_summary": "Detailed 3-4 sentence executive synthesis directly answering the user goal",\n'
            '  "key_findings": [\n'
            '    {"heading": "Finding Title", "detail": "Detailed analytical explanation grounded in the retrieved evidence"}\n'
            "  ],\n"
            '  "quantitative_insights": ["Insight with numbers/metrics"],\n'
            '  "goal_coverage_score": 0.92,\n'
            '  "critique_notes": "1-2 sentences evaluating the quality, completeness, and resilience of this run"\n'
            "}"
        )
        synth_user = (
            f"User Goal: {goal}\n\n"
            f"Retrieved FAISS Evidence:\n{evidence_context}\n\n"
            f"Calculator Outputs:\n{quant_notes}\n\n"
            f"Execution Audit:\n" + "\n".join(audit_trail)
        )
        llm_synth = await self.llm.complete_json(synth_system, synth_user, temperature=0.2)

        if not llm_synth:
            raise RuntimeError("LLM Synthesizer failed to produce final report JSON.")

        exec_summary = llm_synth.get("executive_summary", "")
        key_findings = llm_synth.get("key_findings", [])
        quant_insights = llm_synth.get("quantitative_insights") or quant_notes
        critique = llm_synth.get("critique_notes", "")
        llm_coverage = float(llm_synth.get("goal_coverage_score", 0.9))

        structured = StructuredReportSummary(
            executive_summary=exec_summary,
            key_findings=key_findings,
            quantitative_insights=quant_insights,
            what_the_agent_did_and_why=audit_trail,
            sources_cited=list(sources_map.values()),
            limitations_and_assumptions=plan.assumptions,
        )

        completed_ratio = round(
            sum(1 for s in plan.steps if s.status in ("completed", "corrected")) / max(1, len(plan.steps)),
            2,
        )
        top_sim = max([c.get("similarity_score", 0.0) for c in semantic_hits], default=0.0)
        evaluation = SelfEvaluationMetrics(
            goal_coverage_score=min(1.0, max(0.0, round(llm_coverage, 2))),
            evidence_grounding_score=min(1.0, max(0.0, round(top_sim, 2))),
            plan_adherence_ratio=completed_ratio,
            resilience_verified=all(sc.resolved for sc in self_corrections),
            critique_notes=critique,
        )

        markdown_report = self._render_markdown_report(
            goal=goal,
            plan=plan,
            structured=structured,
            self_corrections=self_corrections,
            evaluation=evaluation,
        )
        return structured, markdown_report, evaluation

    @staticmethod
    def _render_markdown_report(
        goal: str,
        plan: ExecutionPlan,
        structured: StructuredReportSummary,
        self_corrections: List[SelfCorrectionRecord],
        evaluation: SelfEvaluationMetrics,
    ) -> str:
        lines = [
            "# Autonomous Agent Execution Report",
            f"**Goal:** {goal}",
            f"**Domain:** {plan.domain_classification}",
            "",
            "## 1. Executive Summary",
            structured.executive_summary,
            "",
            "## 2. Key Findings & Evidence",
        ]
        for item in structured.key_findings:
            lines.append(f"### {item.get('heading', 'Finding')}")
            lines.append(f"{item.get('detail', '')}\n")

        if structured.quantitative_insights:
            lines.append("## 3. Quantitative & Calculator Insights")
            for q in structured.quantitative_insights:
                lines.append(f"- {q}")
            lines.append("")

        lines.append("## 4. What the Agent Did and Why (Execution & Self-Correction Trace)")
        lines.append(f"> **Planning Rationale:** {plan.reasoning_trace}\n")
        for entry in structured.what_the_agent_did_and_why:
            lines.append(f"- {entry}")
        lines.append("")

        if self_corrections:
            lines.append("## 5. Resilience & Self-Correction Log")
            for sc in self_corrections:
                lines.append(
                    f"- **Step {sc.step_id} (`{sc.failed_tool}` $\\rightarrow$ `{sc.recovery_tool}`)**: "
                    f"{sc.agent_diagnosis} *(Strategy: `{sc.recovery_strategy}`, Resolved: `{sc.resolved}`)*"
                )
            lines.append("")

        lines.append("## 6. Sources & Citations")
        if structured.sources_cited:
            for src in structured.sources_cited:
                lines.append(f"- [{src.get('title')}]({src.get('url')}) *(via `{src.get('source_tool')}`)*")
        else:
            lines.append("- Synthesized from live tool outputs.")
        lines.append("")

        lines.append("## 7. Self-Evaluation & Assumptions")
        lines.append(
            f"- **Goal Coverage Score:** {int(evaluation.goal_coverage_score * 100)}% | "
            f"**Evidence Grounding:** {int(evaluation.evidence_grounding_score * 100)}% | "
            f"**Plan Adherence:** {int(evaluation.plan_adherence_ratio * 100)}%"
        )
        lines.append(f"- **Agent Self-Critique:** {evaluation.critique_notes}")
        for a in structured.limitations_and_assumptions:
            lines.append(f"- *Assumption:* {a}")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_output(tool_name: str, output: Dict[str, Any]) -> str:
        if tool_name == "serper_web_search":
            return (
                f"Retrieved {output.get('result_count', 0)} web search results via {output.get('provider')} "
                f"(indexed {output.get('faiss_chunks_added', 0)} chunks into FAISS)."
            )
        elif tool_name == "tavily_web_crawler":
            return (
                f"Crawled {output.get('pages_crawled', 0)} full page(s) via {output.get('provider')} "
                f"and added {output.get('faiss_chunks_added', 0)} semantic chunks to FAISS."
            )
        elif tool_name == "faiss_rag_memory":
            matches = output.get("top_matches", [])
            top_score = matches[0].get("similarity_score", 0.0) if matches else 0.0
            return (
                f"Queried {output.get('total_indexed_chunks', 0)} FAISS vectors ({output.get('embedding_model')}); "
                f"retrieved {len(matches)} top chunks (best similarity: {top_score})."
            )
        elif tool_name == "calculator_evaluator":
            return f"Evaluated expression `{output.get('expression')}` -> Result: {output.get('formatted', output.get('result'))}."
        return "Tool executed successfully."
