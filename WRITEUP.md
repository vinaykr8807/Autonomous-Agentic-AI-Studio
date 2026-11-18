# Engineering Write-Up: Autonomous Agentic AI System
**Deliverable #4 — Design Decisions, Limitations, and Future Roadmap (1-Page Summary)**

---

## 1. System Architecture & Key Design Decisions

### A. Explicit Two-Phase Planning vs. Pure Unbounded ReAct
Rather than letting the LLM call tools opportunistically without a blueprint, our orchestrator (`AutonomousAgentOrchestrator`) separates **Phase 1: Goal Decomposition (`_create_plan`)** from **Phase 2: Resilient Step Execution (`_execute_step_with_resilience`)**.
- **Why:** Using **Groq (`openai/gpt-oss-120b`)**, the agent first emits an explicit, human-inspectable `ExecutionPlan` containing its domain classification, intermediate reasoning trace, explicit assumptions, and a sequenced tool DAG. This satisfies the rubric's requirement for a *visible planning trace before acting* and prevents infinite tool loops.

### B. Complementary Multi-Tool Ecosystem (4 Distinct Tools)
To operate on live inference-time data without any pre-curated dataset, we designed four orthogonal tools with strict input/output validation:
1. **`serper_web_search` (Serper.dev Google Search API):** Optimized for low-latency broad discovery—fetching recent news, SERP snippets, knowledge-graph attributes, and candidate URLs.
2. **`tavily_web_crawler` (Tavily Extract & Deep Search API):** Optimized for deep content extraction—crawling target URLs discovered by Serper to pull clean, noise-free article bodies.
3. **`faiss_rag_memory` (`sentence-transformers/all-MiniLM-L6-v2` + `FAISS IndexFlatIP`):** Every snippet and crawled page is automatically chunked (700 chars, 100-char overlap), SHA-256 deduplicated, embedded into 384-dimensional normalized vectors, and indexed into FAISS. Downstream steps query FAISS via cosine similarity so the final synthesis is grounded in high-relevance evidence chunks rather than overflowing the prompt window.
4. **`calculator_evaluator` (Safe AST Mathematical Evaluator):** Parses arithmetic, statistical (`mean`, `median`), and financial (`cagr`, `pct_change`) expressions via Python's `ast` module without `eval()`, guaranteeing deterministic numerical accuracy.

### C. Closed-Loop Self-Correction & Fault Injection Engine
Real-world tools fail due to rate limits, timeouts, malformed schemas, or overly narrow queries. We built a first-class **`FaultInjector`** (toggleable via UI, API, and CLI) alongside a diagnostic recovery engine (`_diagnose_and_plan_recovery`):
- **Cross-Tool Provider Failover:** When `serper_web_search` encounters an `HTTP503ServiceUnavailable` or `MalformedPayloadError`, the agent diagnoses the upstream outage and dynamically reroutes the step to `tavily_web_crawler` (and vice versa).
- **Backoff Retry & Query Reformulation:** Transient `TimeoutError` faults trigger a parameter-cleaned retry, while `EmptyResultSetError` faults strip stopwords/punctuation to broaden keyword recall.
- **Full Auditability in SQLite:** Every failed attempt, fault flag, diagnosis, and recovery action is persisted in SQLite (`agent_runs.db`) and streamed via **Server-Sent Events (SSE)** to the React dashboard.

---

## 2. Current Limitations

1. **Sequential Step Execution:** While steps share state (e.g., URLs discovered in Step 1 feed Step 2's crawler, and both feed Step 3's FAISS index), independent sub-queries within a single step are currently executed sequentially rather than fanned out concurrently.
2. **Ephemeral Per-Run FAISS Index:** Each run initializes a dedicated `FAISSVectorMemory` instance so evidence from unrelated goals does not pollute cosine retrieval, whereas cross-session long-term episodic memory is stored at the transcript level in SQLite.
3. **Anti-Bot / Paywall Constraints:** Certain publisher domains block automated crawlers; while Tavily and our fallback parser mitigate this across multiple URLs, heavily paywalled financial data still relies on secondary summary coverage.

---

## 3. What We Would Do Differently With More Time

1. **Dynamic DAG Re-Planning & Parallel Fan-Out:** Upgrade the linear step list into a topological DAG executor (`asyncio.gather`) where the agent can spawn parallel `tavily_web_crawler` workers across 5–10 domains simultaneously and dynamically insert new steps mid-run if the self-evaluator detects low coverage (`goal_coverage_score < 0.80`).
2. **Hybrid BM25 + FAISS Reciprocal Rank Fusion (RRF):** Combine dense `all-MiniLM-L6-v2` embeddings in FAISS with sparse lexical BM25 scoring + a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) to further improve precision on exact entity names, version numbers, and ticker symbols.
3. **Sandboxed Code Execution Container:** Extend `calculator_evaluator` into an isolated WebAssembly/Docker Python sandbox capable of generating charts and running pandas/scipy data pipelines on live API tables.
