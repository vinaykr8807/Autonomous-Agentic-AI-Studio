import React, { useState, useEffect } from 'react'

const API_BASE = 'http://127.0.0.1:8000'

const TOOL_BADGES = {
  serper_web_search: { label: 'Serper Search API', color: '#38bdf8' },
  tavily_web_crawler: { label: 'Tavily Web Crawler', color: '#a78bfa' },
  faiss_rag_memory: { label: 'FAISS + SentenceTransformers', color: '#34d399' },
  calculator_evaluator: { label: 'AST Calculator', color: '#fbbf24' },
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [presets, setPresets] = useState([])
  const [history, setHistory] = useState([])

  const [goal, setGoal] = useState(
    'Given the company Groq, produce a competitive-landscape brief comparing its LPU inference architecture against NVIDIA and Cerebras using public web data.'
  )
  const [faultEnabled, setFaultEnabled] = useState(true)
  const [faultTool, setFaultTool] = useState('serper_web_search')
  const [faultMode, setFaultMode] = useState('http_503_rate_limit')

  const [showKeyModal, setShowKeyModal] = useState(false)
  const [apiKeys, setApiKeys] = useState({
    groq_api_key: '',
    serper_api_key: '',
    tavily_api_key: '',
  })

  const [isRunning, setIsRunning] = useState(false)
  const [livePlan, setLivePlan] = useState(null)
  const [liveToolCalls, setLiveToolCalls] = useState([])
  const [liveCorrections, setLiveCorrections] = useState([])
  const [liveChunksCount, setLiveChunksCount] = useState(0)
  const [activeRunResult, setActiveRunResult] = useState(null)
  const [activeTab, setActiveTab] = useState('trace') // 'trace' | 'report' | 'json'
  const [errorBanner, setErrorBanner] = useState('')

  useEffect(() => {
    fetchInitialData()
  }, [])

  async function fetchInitialData() {
    try {
      const [hRes, pRes, rRes] = await Promise.all([
        fetch(`${API_BASE}/api/health`),
        fetch(`${API_BASE}/api/presets`),
        fetch(`${API_BASE}/api/runs`),
      ])
      if (hRes.ok) setHealth(await hRes.json())
      if (pRes.ok) setPresets(await pRes.json())
      if (rRes.ok) {
        const runs = await rRes.json()
        setHistory(runs)
        if (runs.length > 0 && !activeRunResult) {
          loadSavedRun(runs[0].run_id)
        }
      }
    } catch (err) {
      console.warn('Backend not reachable yet:', err)
    }
  }

  async function loadSavedRun(runId) {
    try {
      const res = await fetch(`${API_BASE}/api/runs/${runId}`)
      if (res.ok) {
        const data = await res.json()
        setActiveRunResult(data)
        setLivePlan(data.plan)
        setLiveToolCalls(data.tool_calls || [])
        setLiveCorrections(data.self_corrections || [])
        setLiveChunksCount(data.vector_chunks_indexed || 0)
      }
    } catch (err) {
      console.error('Error loading run:', err)
    }
  }

  function applyPreset(preset) {
    setGoal(preset.goal)
    if (preset.recommended_fault) {
      setFaultEnabled(preset.recommended_fault.enabled)
      setFaultTool(preset.recommended_fault.target_tool)
      setFaultMode(preset.recommended_fault.failure_mode)
    }
  }

  async function handleRunAgent(e) {
    e.preventDefault()
    if (!goal.trim() || isRunning) return

    setIsRunning(true)
    setErrorBanner('')
    setLivePlan(null)
    setLiveToolCalls([])
    setLiveCorrections([])
    setLiveChunksCount(0)
    setActiveRunResult(null)
    setActiveTab('trace')

    const payload = {
      goal,
      fault_injection: {
        enabled: faultEnabled,
        target_tool: faultTool,
        failure_mode: faultMode,
        fail_count: 1,
      },
      api_keys: Object.fromEntries(
        Object.entries(apiKeys).filter(([_, v]) => v && v.trim() !== '')
      ),
    }

    try {
      const response = await fetch(`${API_BASE}/api/runs/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(`Backend HTTP ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const chunk of parts) {
          const line = chunk.trim()
          if (!line.startsWith('data:')) continue
          const eventData = JSON.parse(line.replace(/^data:\s*/, ''))

          if (eventData.event === 'plan_created') {
            setLivePlan(eventData.plan)
          } else if (eventData.event === 'step_started') {
            setLivePlan((prev) => {
              if (!prev) return prev
              return {
                ...prev,
                steps: prev.steps.map((s) =>
                  s.step_id === eventData.step.step_id ? eventData.step : s
                ),
              }
            })
          } else if (eventData.event === 'tool_call_logged') {
            setLiveToolCalls((prev) => [...prev, eventData.tool_call])
          } else if (eventData.event === 'self_correction_triggered') {
            setLiveCorrections((prev) => [...prev, eventData.self_correction])
          } else if (eventData.event === 'step_finished') {
            setLiveChunksCount(eventData.vector_chunks_indexed || 0)
            setLivePlan((prev) => {
              if (!prev) return prev
              return {
                ...prev,
                steps: prev.steps.map((s) =>
                  s.step_id === eventData.step.step_id ? eventData.step : s
                ),
              }
            })
          } else if (eventData.event === 'final_result') {
            setActiveRunResult(eventData.result)
            fetchInitialData()
          } else if (eventData.event === 'fatal_error') {
            setErrorBanner(eventData.message || 'Execution failed')
          }
        }
      }
    } catch (err) {
      setErrorBanner(`Connection error: ${err.message}. Ensure FastAPI server is running on port 8000.`)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="app-shell">
      {/* Top Header */}
      <header className="top-nav">
        <div className="brand">
          <div className="brand-orb" />
          <div>
            <h1>Autonomous Agentic AI Studio</h1>
            <p className="subtitle">
              Groq (<code>openai/gpt-oss-120b</code>) &bull; Serper Search &bull; Tavily Crawler &bull; SentenceTransformers + FAISS &bull; SQLite
            </p>
          </div>
        </div>
        <div className="nav-actions">
          <div className="status-pills">
            <span className="pill pill-green">
              LLM: {health?.llm_model || 'openai/gpt-oss-120b'}
            </span>
            <span className="pill pill-blue">
              Embeddings: {health?.embedding_model || 'all-MiniLM-L6-v2'}
            </span>
            <span className="pill pill-purple">
              FAISS Chunks: {liveChunksCount}
            </span>
          </div>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => setShowKeyModal(!showKeyModal)}
          >
            {showKeyModal ? 'Close API Keys' : 'Configure API Keys'}
          </button>
        </div>
      </header>

      {/* Optional Runtime API Key Drawer */}
      {showKeyModal && (
        <section className="api-key-drawer">
          <div className="drawer-header">
            <h3>Runtime API Keys Override (Optional — can also be set in <code>backend/.env</code>)</h3>
            <span>If left blank, backend uses environment variables or live public search/crawler fallbacks.</span>
          </div>
          <div className="key-grid">
            <label>
              <span>GROQ_API_KEY (for <code>openai/gpt-oss-120b</code>)</span>
              <input
                type="password"
                placeholder="gsk_..."
                value={apiKeys.groq_api_key}
                onChange={(e) => setApiKeys({ ...apiKeys, groq_api_key: e.target.value })}
              />
            </label>
            <label>
              <span>SERPER_API_KEY (Serper.dev Google Search)</span>
              <input
                type="password"
                placeholder="Serper API key..."
                value={apiKeys.serper_api_key}
                onChange={(e) => setApiKeys({ ...apiKeys, serper_api_key: e.target.value })}
              />
            </label>
            <label>
              <span>TAVILY_API_KEY (Tavily Web Crawler)</span>
              <input
                type="password"
                placeholder="tvly-..."
                value={apiKeys.tavily_api_key}
                onChange={(e) => setApiKeys({ ...apiKeys, tavily_api_key: e.target.value })}
              />
            </label>
          </div>
        </section>
      )}

      <main className="workspace-grid">
        {/* Left Sidebar: Goal Input, Presets, Fault Injection & SQLite History */}
        <aside className="left-panel">
          <form onSubmit={handleRunAgent} className="control-card">
            <div className="section-label">1. High-Level User Goal</div>
            <textarea
              rows={4}
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="Enter any complex multi-step research, competitive intelligence, or planning goal..."
              className="goal-textarea"
            />

            <div className="preset-chips">
              {presets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className="preset-chip"
                  onClick={() => applyPreset(p)}
                >
                  {p.title}
                </button>
              ))}
            </div>

            <div className="fault-box">
              <div className="fault-header">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={faultEnabled}
                    onChange={(e) => setFaultEnabled(e.target.checked)}
                  />
                  <span>Deliberate Fault Injection (Req #4 Self-Correction)</span>
                </label>
              </div>
              {faultEnabled && (
                <div className="fault-controls">
                  <div>
                    <label>Target Tool</label>
                    <select value={faultTool} onChange={(e) => setFaultTool(e.target.value)}>
                      <option value="serper_web_search">serper_web_search</option>
                      <option value="tavily_web_crawler">tavily_web_crawler</option>
                      <option value="calculator_evaluator">calculator_evaluator</option>
                    </select>
                  </div>
                  <div>
                    <label>Simulated Failure Mode</label>
                    <select value={faultMode} onChange={(e) => setFaultMode(e.target.value)}>
                      <option value="http_503_rate_limit">HTTP 503 Rate Limit / Unavailable (Triggers Cross-Tool Fallback)</option>
                      <option value="timeout">Upstream Read Timeout 10s (Triggers Backoff Retry)</option>
                      <option value="malformed_payload">Malformed JSON Schema (Triggers Fallback Provider)</option>
                      <option value="empty_results">0 Matching Records (Triggers Query Reformulation)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            <button type="submit" className="btn-primary" disabled={isRunning}>
              {isRunning ? 'Autonomous Agent Executing Plan...' : 'Launch Autonomous Agent Run'}
            </button>
          </form>

          {/* SQLite Run History */}
          <div className="control-card history-card">
            <div className="section-label">SQLite Persisted Run History ({history.length})</div>
            <div className="history-list">
              {history.length === 0 && (
                <div className="empty-hint">No saved runs yet. Execute a goal above!</div>
              )}
              {history.map((item) => (
                <div
                  key={item.run_id}
                  className={`history-item ${activeRunResult?.run_id === item.run_id ? 'active' : ''}`}
                  onClick={() => loadSavedRun(item.run_id)}
                >
                  <div className="history-title">{item.goal}</div>
                  <div className="history-meta">
                    <span>{item.run_id}</span>
                    <span>{item.tool_calls_count} tools</span>
                    {item.self_corrections_count > 0 && (
                      <span className="badge-recovered">
                        {item.self_corrections_count} self-corrected
                      </span>
                    )}
                    <span>{Math.round(item.total_duration_ms)}ms</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Right Panel: Live Planning Trace, Tool Calls, Self-Correction, and Structured Report */}
        <section className="right-panel">
          {errorBanner && <div className="error-banner">{errorBanner}</div>}

          <div className="tabs-bar">
            <button
              className={`tab-btn ${activeTab === 'trace' ? 'active' : ''}`}
              onClick={() => setActiveTab('trace')}
            >
              1. Visible Planning Trace &amp; Tool Orchestration
            </button>
            <button
              className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`}
              onClick={() => setActiveTab('report')}
            >
              2. Final Structured Report &amp; Self-Evaluation
            </button>
            <button
              className={`tab-btn ${activeTab === 'json' ? 'active' : ''}`}
              onClick={() => setActiveTab('json')}
            >
              3. Full JSON Transcript (Deliverable #3)
            </button>
          </div>

          {activeTab === 'trace' && (
            <div className="trace-container">
              {/* Planning Trace Card */}
              {livePlan ? (
                <div className="panel-card">
                  <div className="card-header-row">
                    <div>
                      <span className="eyebrow">PHASE 1: AUTONOMOUS DECOMPOSITION</span>
                      <h2>Visible Planning Trace ({livePlan.domain_classification})</h2>
                    </div>
                    <span className="step-count-badge">{livePlan.steps?.length || 0} Planned Steps</span>
                  </div>
                  <p className="reasoning-box">
                    <strong>Planner Reasoning:</strong> {livePlan.reasoning_trace}
                  </p>

                  <div className="steps-timeline">
                    {livePlan.steps?.map((step) => {
                      const badge = TOOL_BADGES[step.tool_name] || {
                        label: step.tool_name,
                        color: '#94a3b8',
                      }
                      return (
                        <div key={step.step_id} className={`step-card status-${step.status}`}>
                          <div className="step-top">
                            <span className="step-num">Step {step.step_id}</span>
                            <span className="step-title">{step.title}</span>
                            <span
                              className="tool-tag"
                              style={{ borderColor: badge.color, color: badge.color }}
                            >
                              {badge.label}
                            </span>
                            <span className={`status-tag status-${step.status}`}>
                              {step.status.toUpperCase()}
                            </span>
                          </div>
                          <p className="step-desc">{step.description}</p>
                          <div className="step-query">
                            <code>Query / Expression: {step.tool_query}</code>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              ) : (
                <div className="panel-card empty-state">
                  <h3>Ready to Decompose Goal</h3>
                  <p>
                    Click <strong>Launch Autonomous Agent Run</strong> to generate a visible multi-step
                    planning trace, invoke Serper, Tavily, FAISS Vector RAG, and Calculator tools,
                    and observe live fault recovery.
                  </p>
                </div>
              )}

              {/* Self-Correction Events Card */}
              {liveCorrections.length > 0 && (
                <div className="panel-card correction-panel">
                  <span className="eyebrow eyebrow-amber">
                    PHASE 2B: AUTONOMOUS SELF-CORRECTION &amp; RECOVERY
                  </span>
                  <h2>Detected Tool Failures &amp; Recovery Decisions ({liveCorrections.length})</h2>
                  {liveCorrections.map((sc) => (
                    <div key={sc.correction_id} className="correction-item">
                      <div className="correction-header">
                        <span className="corr-badge">Step {sc.step_id} Self-Corrected</span>
                        <span>
                          Failed Tool: <code>{sc.failed_tool}</code> &rarr; Recovery Tool:{' '}
                          <code>{sc.recovery_tool}</code>
                        </span>
                        <span className="strategy-pill">Strategy: {sc.recovery_strategy}</span>
                      </div>
                      <p className="corr-reason">
                        <strong>Trigger Error:</strong> {sc.failure_reason}
                      </p>
                      <p className="corr-diag">
                        <strong>Agent Diagnosis:</strong> {sc.agent_diagnosis}
                      </p>
                      <div className="corr-action">
                        <strong>Recovery Action Executed:</strong> <code>{sc.recovery_action}</code>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Step-by-Step Tool Execution Logs */}
              {liveToolCalls.length > 0 && (
                <div className="panel-card">
                  <span className="eyebrow">PHASE 2A: LIVE TOOL INVOCATIONS</span>
                  <h2>Tool Call Execution Log ({liveToolCalls.length} Invocations)</h2>
                  <div className="tool-logs-list">
                    {liveToolCalls.map((tc) => (
                      <div key={tc.call_id} className={`tool-log-card status-${tc.status}`}>
                        <div className="tool-log-header">
                          <strong>
                            Step {tc.step_id} (Attempt #{tc.attempt}) — <code>{tc.tool_name}</code>
                          </strong>
                          <div className="tool-log-badges">
                            {tc.is_injected_fault && (
                              <span className="fault-injected-pill">INJECTED FAULT</span>
                            )}
                            <span className={`status-tag status-${tc.status}`}>
                              {tc.status.toUpperCase()}
                            </span>
                            <span className="latency-pill">{tc.latency_ms} ms</span>
                          </div>
                        </div>
                        <div className="tool-log-summary">{tc.output_summary}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'report' && (
            <div className="report-container">
              {activeRunResult ? (
                <>
                  {/* Self-Evaluation Scorecard */}
                  <div className="panel-card eval-scorecard">
                    <span className="eyebrow">AGENT SELF-EVALUATION METRICS</span>
                    <div className="metrics-row">
                      <div className="metric-box">
                        <span className="metric-val">
                          {Math.round(activeRunResult.evaluation.goal_coverage_score * 100)}%
                        </span>
                        <span className="metric-label">Goal Coverage</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-val">
                          {Math.round(activeRunResult.evaluation.evidence_grounding_score * 100)}%
                        </span>
                        <span className="metric-label">FAISS Cosine Grounding</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-val">
                          {Math.round(activeRunResult.evaluation.plan_adherence_ratio * 100)}%
                        </span>
                        <span className="metric-label">Plan Completion</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-val">{activeRunResult.vector_chunks_indexed}</span>
                        <span className="metric-label">FAISS Chunks Embedded</span>
                      </div>
                    </div>
                    <p className="critique-text">
                      <strong>Self-Critique:</strong> {activeRunResult.evaluation.critique_notes}
                    </p>
                  </div>

                  {/* Structured Final Report */}
                  <div className="panel-card structured-report">
                    <span className="eyebrow">FINAL STRUCTURED REPORT</span>
                    <h2>Executive Summary</h2>
                    <p className="exec-summary-p">
                      {activeRunResult.structured_report.executive_summary}
                    </p>

                    <h3>Key Findings (Grounded in FAISS Vector Memory)</h3>
                    <div className="findings-grid">
                      {activeRunResult.structured_report.key_findings.map((f, i) => (
                        <div key={i} className="finding-card">
                          <h4>{f.heading}</h4>
                          <p>{f.detail}</p>
                        </div>
                      ))}
                    </div>

                    {activeRunResult.structured_report.quantitative_insights?.length > 0 && (
                      <>
                        <h3>Quantitative &amp; Calculator Verification</h3>
                        <ul className="report-list">
                          {activeRunResult.structured_report.quantitative_insights.map((q, i) => (
                            <li key={i}>{q}</li>
                          ))}
                        </ul>
                      </>
                    )}

                    <h3>What the Agent Did and Why (Decision Audit)</h3>
                    <ul className="report-list">
                      {activeRunResult.structured_report.what_the_agent_did_and_why.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>

                    <h3>Sources &amp; Live Citations</h3>
                    <ul className="report-list">
                      {activeRunResult.structured_report.sources_cited.map((s, i) => (
                        <li key={i}>
                          <a href={s.url} target="_blank" rel="noreferrer">
                            {s.title}
                          </a>{' '}
                          <span className="source-tool-chip">via {s.source_tool}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </>
              ) : (
                <div className="panel-card empty-state">
                  <h3>No Final Report Yet</h3>
                  <p>Run a goal or select a completed run from the SQLite history sidebar.</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'json' && (
            <div className="panel-card">
              <span className="eyebrow">COMPLETE SERIALIZED TRANSCRIPT</span>
              <h2>Raw JSON Run Artifact</h2>
              <pre className="json-viewer">
                {activeRunResult
                  ? JSON.stringify(activeRunResult, null, 2)
                  : '// Run the agent to inspect full JSON transcript'}
              </pre>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
