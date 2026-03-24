import { startTransition, useDeferredValue, useEffect, useState } from "react"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts"
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation
} from "d3-force"

const API_BASE = import.meta.env.VITE_DASHBOARD_API_BASE ?? "http://localhost:8001"

const TAB_DEFS = [
  { id: "overview", label: "Overview" },
  { id: "live", label: "Live Monitor" },
  { id: "analytics", label: "Fraud Analytics" },
  { id: "performance", label: "Performance" }
]

const RISK_COLORS = {
  LOW: "#1F7A8C",
  MEDIUM: "#F59E0B",
  HIGH: "#C2410C"
}

const EMPTY_DASHBOARD = {
  metadata: { source: "loading", generated_at: null, warning: null },
  overview: { kpis: [], timeline: [], risk_distribution: [], top_flagged_users: [] },
  live: { updated_at: null, transactions: [] },
  analytics: {
    alerts_by_merchant: [],
    amount_vs_risk: [],
    anomaly_rules: [],
    network: { nodes: [], links: [] }
  },
  performance: {
    cards: [],
    confusion_matrix: [],
    rule_weights: [],
    benchmark_models: [],
    pnl: { saved_inr: 0, lost_inr: 0, net_inr: 0, reported_monthly_loss_inr: 0 }
  }
}

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    throw new Error(`Dashboard request failed: ${response.status}`)
  }
  return response.json()
}

function App() {
  const [activeTab, setActiveTab] = useState("overview")
  const [dashboard, setDashboard] = useState(EMPTY_DASHBOARD)
  const [status, setStatus] = useState("Connecting to dashboard backend...")
  const [error, setError] = useState(null)
  const deferredTransactions = useDeferredValue(dashboard.live.transactions)

  useEffect(() => {
    let mounted = true

    async function loadSnapshot() {
      try {
        const snapshot = await fetchJson("/dashboard/snapshot")
        if (!mounted) {
          return
        }
        startTransition(() => {
          setDashboard(snapshot)
          setError(null)
          setStatus(
            snapshot.metadata.warning
              ? `Showing fallback dashboard data. ${snapshot.metadata.warning}`
              : `Connected to ${snapshot.metadata.source}`
          )
        })
      } catch (loadError) {
        if (mounted) {
          setError(loadError.message)
          setStatus("Dashboard backend unavailable")
        }
      }
    }

    async function refreshLive() {
      try {
        const live = await fetchJson("/dashboard/live")
        if (!mounted) {
          return
        }
        startTransition(() => {
          setDashboard((current) => ({ ...current, live }))
        })
      } catch {
        if (mounted) {
          setStatus("Live feed paused until the backend responds again")
        }
      }
    }

    loadSnapshot()
    const snapshotInterval = window.setInterval(loadSnapshot, 15000)
    const liveInterval = window.setInterval(refreshLive, 2000)

    return () => {
      mounted = false
      window.clearInterval(snapshotInterval)
      window.clearInterval(liveInterval)
    }
  }, [])

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Sentinel-UPI Analytics Deck</p>
          <h1>Fraud pressure, model behavior, and live payment risk in one surface.</h1>
        </div>
        <div className="hero-meta">
          <div className="hero-stat">
            <span>Data Source</span>
            <strong>{dashboard.metadata.source}</strong>
          </div>
          <div className="hero-stat">
            <span>Last Refresh</span>
            <strong>{formatTimestamp(dashboard.live.updated_at || dashboard.metadata.generated_at)}</strong>
          </div>
        </div>
      </header>

      <section className="status-bar">
        <div>
          <span className="status-dot" />
          {status}
        </div>
        {error ? <div className="error-pill">{error}</div> : null}
      </section>

      <nav className="tabs">
        {TAB_DEFS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={tab.id === activeTab ? "tab active" : "tab"}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="tab-content">
        {activeTab === "overview" ? <OverviewTab data={dashboard.overview} /> : null}
        {activeTab === "live" ? <LiveMonitorTab transactions={deferredTransactions} /> : null}
        {activeTab === "analytics" ? <AnalyticsTab data={dashboard.analytics} /> : null}
        {activeTab === "performance" ? <PerformanceTab data={dashboard.performance} /> : null}
      </main>
    </div>
  )
}

function OverviewTab({ data }) {
  return (
    <section className="dashboard-grid">
      <div className="card span-12 kpi-grid">
        {data.kpis.map((kpi) => (
          <article key={kpi.label} className="kpi-card">
            <span>{kpi.label}</span>
            <strong>{formatMetric(kpi)}</strong>
          </article>
        ))}
      </div>

      <div className="card span-8">
        <CardHeader title="Safe vs Flagged in Last 30 Minutes" subtitle="Area profile of the hot path" />
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={data.timeline}>
              <defs>
                <linearGradient id="safeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#1F7A8C" stopOpacity={0.65} />
                  <stop offset="95%" stopColor="#1F7A8C" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="flaggedGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#C2410C" stopOpacity={0.65} />
                  <stop offset="95%" stopColor="#C2410C" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
              <XAxis dataKey="minute" stroke="#9FB3C8" tickLine={false} axisLine={false} minTickGap={18} />
              <YAxis stroke="#9FB3C8" tickLine={false} axisLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="safe" stackId="1" stroke="#1F7A8C" fill="url(#safeGradient)" />
              <Area type="monotone" dataKey="flagged" stackId="1" stroke="#C2410C" fill="url(#flaggedGradient)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card span-4">
        <CardHeader title="Risk Distribution" subtitle="Current scoring mix" />
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={data.risk_distribution} dataKey="value" nameKey="name" innerRadius={58} outerRadius={96} paddingAngle={4}>
                {data.risk_distribution.map((entry) => (
                  <Cell key={entry.name} fill={RISK_COLORS[entry.name] || "#9CA3AF"} />
                ))}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
          <div className="legend">
            {data.risk_distribution.map((entry) => (
              <div key={entry.name} className="legend-row">
                <span className="legend-swatch" style={{ background: RISK_COLORS[entry.name] }} />
                <span>{entry.name}</span>
                <strong>{entry.value}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card span-12">
        <CardHeader title="Top Flagged Users" subtitle="Most frequently blocked senders" />
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.top_flagged_users}>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
              <XAxis dataKey="upi_id" stroke="#9FB3C8" tickLine={false} axisLine={false} interval={0} angle={-18} textAnchor="end" height={70} />
              <YAxis stroke="#9FB3C8" tickLine={false} axisLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" fill="#F97316" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  )
}

function LiveMonitorTab({ transactions }) {
  return (
    <section className="dashboard-grid">
      <div className="card span-12">
        <CardHeader title="Live Transaction Feed" subtitle="Auto-refresh every 2 seconds" />
        <div className="table-wrap">
          <table className="live-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>TXN ID</th>
                <th>User</th>
                <th>Amount</th>
                <th>Merchant</th>
                <th>Risk Score</th>
                <th>Flags</th>
                <th>Decision</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((transaction) => (
                <tr key={transaction.txn_id} className={`row-${transaction.risk_level.toLowerCase()}`}>
                  <td>{transaction.time}</td>
                  <td>{transaction.txn_id}</td>
                  <td>{transaction.user}</td>
                  <td>{formatCurrency(transaction.amount)}</td>
                  <td>
                    <div>{transaction.merchant}</div>
                    <small>{humanizeText(transaction.merchant_type)}</small>
                  </td>
                  <td>
                    <div className="risk-bar-row">
                      <div className="risk-bar">
                        <div
                          className={`risk-fill ${transaction.risk_level.toLowerCase()}`}
                          style={{ width: `${transaction.risk_score * 100}%` }}
                        />
                      </div>
                      <strong>{Math.round(transaction.risk_score * 100)}%</strong>
                    </div>
                  </td>
                  <td>
                    <div className="flag-list">
                      {transaction.flags.map((flag) => (
                        <span key={`${transaction.txn_id}-${flag}`} className="flag-pill">
                          {humanizeText(flag)}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <span className={`decision-pill ${transaction.decision.toLowerCase()}`}>
                      {transaction.decision}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}

function AnalyticsTab({ data }) {
  return (
    <section className="dashboard-grid">
      <div className="card span-4">
        <CardHeader title="Alerts by Merchant Category" subtitle="Blocked volume by merchant type" />
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.alerts_by_merchant}>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
              <XAxis dataKey="merchant_type" stroke="#9FB3C8" tickLine={false} axisLine={false} tickFormatter={humanizeText} interval={0} angle={-14} textAnchor="end" height={70} />
              <YAxis stroke="#9FB3C8" tickLine={false} axisLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" fill="#0F766E" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card span-4">
        <CardHeader title="Amount vs Risk Score" subtitle="Actual fraud marked in orange" />
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" />
              <XAxis dataKey="amount" name="Amount" stroke="#9FB3C8" tickLine={false} axisLine={false} />
              <YAxis dataKey="risk_score" name="Risk" stroke="#9FB3C8" tickLine={false} axisLine={false} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "4 4" }} />
              <Scatter data={data.amount_vs_risk.filter((item) => item.label === "Fraud")} fill="#F97316" />
              <Scatter data={data.amount_vs_risk.filter((item) => item.label === "Legitimate")} fill="#1F7A8C" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card span-4">
        <CardHeader title="Anomaly Rule Pressure" subtitle="Most active trigger families" />
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.anomaly_rules} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" horizontal={false} />
              <XAxis type="number" stroke="#9FB3C8" tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="rule" stroke="#9FB3C8" tickLine={false} axisLine={false} width={120} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" fill="#FACC15" radius={[0, 10, 10, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card span-12">
        <CardHeader title="Flagged Money-Mule Topology" subtitle="D3 force graph of high-risk links" />
        <NetworkGraph graph={data.network} />
      </div>
    </section>
  )
}

function PerformanceTab({ data }) {
  const benchmarkModels = data.benchmark_models ?? []

  return (
    <section className="dashboard-grid">
      <div className="card span-12 kpi-grid">
        {data.cards.map((card) => (
          <article key={card.label} className="kpi-card">
            <span>{card.label}</span>
            <strong>{Number(card.value).toFixed(4)}</strong>
          </article>
        ))}
      </div>

      <div className="card span-4">
        <CardHeader title="Confusion Matrix" subtitle="Held-out test split" />
        <div className="confusion-grid">
          {data.confusion_matrix.map((cell) => (
            <div key={cell.label} className={`confusion-cell ${cell.label.toLowerCase()}`}>
              <span>{cell.label}</span>
              <strong>{cell.value}</strong>
            </div>
          ))}
        </div>
      </div>

      <div className="card span-4">
        <CardHeader title="Detection Rule Weights" subtitle="Relative contribution in flagged flow" />
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.rule_weights} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" horizontal={false} />
              <XAxis type="number" stroke="#9FB3C8" tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="rule" stroke="#9FB3C8" tickLine={false} axisLine={false} width={120} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="weight" fill="#FB7185" radius={[0, 10, 10, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card span-4 pnl-card">
        <CardHeader title="Cost-Sensitive P and L" subtitle="Monthly impact estimate" />
        <div className="pnl-metrics">
          <div>
            <span>Estimated Saved</span>
            <strong>{formatCurrency(data.pnl.saved_inr)}</strong>
          </div>
          <div>
            <span>Estimated Lost</span>
            <strong>{formatCurrency(data.pnl.lost_inr)}</strong>
          </div>
          <div>
            <span>Net Position</span>
            <strong>{formatCurrency(data.pnl.net_inr)}</strong>
          </div>
          <div>
            <span>Reported Monthly Loss</span>
            <strong>{formatCurrency(data.pnl.reported_monthly_loss_inr)}</strong>
          </div>
        </div>
      </div>

      <div className="card span-12">
        <CardHeader title="Benchmark Comparison" subtitle="All five models on the same held-out split" />
        <div className="table-wrap">
          <table className="live-table benchmark-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>AUC-ROC</th>
                <th>Latency (ms)</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkModels.length ? (
                benchmarkModels.map((model) => (
                  <tr
                    key={model.name}
                    className={model.name === "Sentinel-UPI" ? "benchmark-row active" : "benchmark-row"}
                  >
                    <td className="benchmark-model">
                      <strong>{model.name}</strong>
                    </td>
                    <td>
                      <span className="benchmark-chip">{model.type}</span>
                    </td>
                    <td>{formatBenchmarkMetric(model.precision)}</td>
                    <td>{formatBenchmarkMetric(model.recall)}</td>
                    <td>{formatBenchmarkMetric(model.f1_score)}</td>
                    <td>{formatBenchmarkMetric(model.auc_roc)}</td>
                    <td>{formatBenchmarkMetric(model.latency_ms)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="7" className="benchmark-empty">
                    Benchmark results are not available yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}

function NetworkGraph({ graph }) {
  const [layout, setLayout] = useState({ nodes: [], links: [] })

  useEffect(() => {
    if (!graph.nodes.length) {
      setLayout({ nodes: [], links: [] })
      return
    }

    const nodes = graph.nodes.map((node) => ({ ...node }))
    const links = graph.links.map((link) => ({ ...link }))
    const simulation = forceSimulation(nodes)
      .force("link", forceLink(links).id((node) => node.id).distance(90))
      .force("charge", forceManyBody().strength(-180))
      .force("center", forceCenter(360, 170))
      .force("collision", forceCollide(28))

    for (let tick = 0; tick < 80; tick += 1) {
      simulation.tick()
    }
    simulation.stop()
    setLayout({ nodes, links })
  }, [graph])

  return (
    <div className="network-panel">
      <svg viewBox="0 0 720 340" className="network-svg" role="img" aria-label="Fraud network">
        {layout.links.map((link, index) => (
          <line
            key={`${link.source.id ?? link.source}-${link.target.id ?? link.target}-${index}`}
            x1={link.source.x}
            y1={link.source.y}
            x2={link.target.x}
            y2={link.target.y}
            stroke={link.decision === "BLOCK" ? "#F97316" : "#1F7A8C"}
            strokeOpacity="0.55"
            strokeWidth={Math.max(1.5, link.risk_score * 4)}
          />
        ))}
        {layout.nodes.map((node) => (
          <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
            <circle
              r={node.type === "merchant" ? 16 : 12}
              fill={RISK_COLORS[node.risk_level] || "#E2E8F0"}
              stroke="#0F172A"
              strokeWidth="2"
            />
            <text x="20" y="4" className="network-label">
              {node.id}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}

function CardHeader({ title, subtitle }) {
  return (
    <div className="card-header">
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  )
}

function formatMetric(metric) {
  if (metric.prefix) {
    return `${metric.prefix}${Number(metric.value).toLocaleString()}`
  }
  if (metric.suffix) {
    return `${metric.value}${metric.suffix}`
  }
  return Number(metric.value).toLocaleString()
}

function formatTimestamp(value) {
  if (!value) {
    return "Waiting for data"
  }
  return new Date(value).toLocaleTimeString()
}

function formatCurrency(value) {
  return `INR ${Number(value).toLocaleString()}`
}

function formatBenchmarkMetric(value) {
  return Number(value ?? 0).toFixed(4)
}

function humanizeText(value) {
  return value.replaceAll("_", " ")
}

const tooltipStyle = {
  background: "rgba(10, 15, 28, 0.92)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "16px",
  color: "#F8FAFC"
}

export default App
