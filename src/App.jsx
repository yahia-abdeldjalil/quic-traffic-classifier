// quic-frontend/src/App.jsx
import { useMemo, useState } from 'react'
import {
  Activity,
  ArrowUpDown,
  Clock3,
  Database,
  Network,
  Radio,
  ShieldCheck,
  Sparkles,
  Dice1,
  Dices
} from 'lucide-react'

const FLOW_END_REASONS = [
  { value: 0, label: 'Idle Timeout' },
  { value: 1, label: 'Active Timeout' },
  { value: 2, label: 'Other' },
]

const INITIAL_FORM = {
  avg_ipt: 120,
  std_ipt: 40,

  avg_size: 512,
  std_size: 180,

  ul_dl_ratio_direction: 0.55,
  ul_dl_ratio_bytes: 0.65,
  ul_dl_ratio_packets: 0.58,

  ppi_len: 30,
  ppi_roundtrips: 4,
  ppi_duration: 3200,

  flow_end_reason: 0,
}

function MetricCard({ icon: Icon, title, value, subtitle }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
      <div className="mb-3 flex items-center gap-3">
        <div className="rounded-xl bg-cyan-500/15 p-2">
          <Icon className="h-5 w-5 text-cyan-400" />
        </div>

        <h3 className="text-sm font-medium text-zinc-300">
          {title}
        </h3>
      </div>

      <div className="text-2xl font-bold text-white">
        {value}
      </div>

      <p className="mt-1 text-xs text-zinc-500">
        {subtitle}
      </p>
    </div>
  )
}

function SliderField({
  label,
  min,
  max,
  step,
  value,
  onChange,
  suffix = '',
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-zinc-200">
          {label}
        </label>

        <span className="rounded-lg bg-zinc-800 px-3 py-1 text-xs text-cyan-300">
          {value}
          {suffix}
        </span>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-zinc-700 accent-cyan-400"
      />
    </div>
  )
}

function App() {
  const [form, setForm] = useState(INITIAL_FORM)
  const [strategy, setStrategy] = useState('avg')
  const [loading, setLoading] = useState(false)

  const [result, setResult] = useState(null)

  const [error, setError] = useState(null)

  const backendUrl = useMemo(
    () => 'http://localhost:8000/predict',
    []
  )
  const [pcapFile, setPcapFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  async function handlePcapUpload(file) {
    setPcapFile(file)
  }
  async function handlePcapSubmit() {
    if (!pcapFile) return

    setUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", pcapFile)
      formData.append("strategy", strategy)
      console.log("Uploading PCAP file to backend for prediction...");
      for (let pair of formData.entries()) {
        console.log(pair[0], pair[1])
      }
      const response = await fetch("http://127.0.0.1:8000/pcap", {
        method: "POST",
        body: formData,
      })
      .then(r=>r.text())
      .then(console.log)
      .catch(console.error)
      console.log("STATUS:", response.status)
      console.log("OK:", response.ok)
      const data = await response.json()
      console.log("Received response from backend:", data)
      setResult(data)

    } catch (err) {
      setError(err.message || "PCAP prediction failed")
    } finally {
      setUploading(false)
    }
  }
  function updateField(key, value) {
    setForm((prev) => ({
      ...prev,
      [key]: value,
    }))
  }

  async function handlePredict(e) {
    e.preventDefault()

    setLoading(true)
    setError(null)

    try {
      console.log('Sending payload to backend:', form)
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features: form,
          strategy: strategy,
        }),
      })

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`)
      }

      const data = await response.json()
      console.log('Received response from backend:', data)
      setResult(data)
    } catch (err) {
      console.error(err)

      setError(
        err.message || 'Prediction request failed'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-[#050816] text-white">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(6,182,212,0.15),transparent_40%)]" />

      <div className="relative mx-auto max-w-7xl px-6 py-10">
        {/* HERO */}
        <section className="mb-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-300">
            <Sparkles className="h-4 w-4" />
            QUIC Traffic Intelligence Platform
          </div>

          <div className="mt-6 grid gap-10 lg:grid-cols-2">
            <div>
              <h1 className="max-w-3xl text-5xl font-black leading-tight tracking-tight">
                Encrypted QUIC Application Classification
              </h1>

              <p className="mt-6 max-w-2xl text-lg leading-8 text-zinc-400">
                Real-time inference using a weighted ensemble
                of machine learning classifiers trained on
                CESNET_QUIC22 network flow statistics and
                packet histograms.
              </p>

              <div className="mt-8 flex flex-wrap gap-4">
                <div className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3">
                  <div className="text-xs uppercase tracking-widest text-zinc-500">
                    Models
                  </div>

                  <div className="mt-1 text-xl font-bold">
                    Ensemble ML
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3">
                  <div className="text-xs uppercase tracking-widest text-zinc-500">
                    Dataset
                  </div>

                  <div className="mt-1 text-xl font-bold">
                    CESNET_QUIC22
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3">
                  <div className="text-xs uppercase tracking-widest text-zinc-500">
                    Features
                  </div>

                  <div className="mt-1 text-xl font-bold">
                    133
                  </div>
                </div>
              </div>
            </div>

            {/* RESULT PANEL */}
            <div className="rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl">
              <div className="mb-8 flex items-center gap-3">
                <div className="rounded-2xl bg-cyan-500/15 p-3">
                  <ShieldCheck className="h-7 w-7 text-cyan-400" />
                </div>

                <div>
                  <h2 className="text-2xl font-bold">
                    Prediction Output
                  </h2>

                  <p className="text-zinc-400">
                    Ensemble inference response
                  </p>
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Ensemble Strategy
                </label>

                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full rounded-xl border border-zinc-700 bg-zinc-900 p-3"
                >
                  <option value="avg">
                    Hybrid (GA + PSO)
                  </option>

                  <option value="ga">
                    Genetic Algorithm
                  </option>

                  <option value="pso">
                    Particle Swarm Optimization
                  </option>
                </select>
              </div>
              {/* {!result && (
                <div className="rounded-2xl border border-dashed border-zinc-700 p-10 text-center">
                  <Network className="mx-auto mb-4 h-12 w-12 text-zinc-600" />

                  <p className="text-zinc-400">
                    Submit flow statistics to classify
                    encrypted QUIC traffic.
                  </p>
                </div>
              )} */}
              {!result && (
                <div className="rounded-2xl border border-dashed border-zinc-700 p-10 text-center space-y-6">
                  <Network className="mx-auto h-12 w-12 text-zinc-600" />

                  <p className="text-zinc-400">
                    Upload a QUIC PCAP file for classification
                  </p>

                  {/* FILE INPUT */}
                  <input
                    type="file"
                    accept=".pcap,.pcapng"
                    onChange={(e) => {
                      setPcapFile(e.target.files[0])
                    }}
                    className="block w-full text-sm text-zinc-400 file:mr-4 file:rounded-xl file:border-0 file:bg-cyan-500 file:px-4 file:py-2 file:text-black hover:file:bg-cyan-400"
                  />

                  {/* SELECTED FILE */}
                  {pcapFile && (
                    <p className="text-sm text-cyan-300">
                      Selected: {pcapFile.name}
                    </p>
                  )}

                  {/* UPLOAD BUTTON */}
                  <button
                    onClick={handlePcapSubmit}
                    disabled={!pcapFile || uploading}
                    className="inline-flex items-center gap-2 rounded-2xl bg-cyan-500 px-6 py-3 font-semibold text-black transition hover:bg-cyan-400 disabled:opacity-50"
                  >
                    {uploading ? "Analyzing PCAP..." : "Run PCAP Inference"}
                  </button>
                </div>
              )}
              {uploading && (
                <div className="mt-4 text-cyan-300 text-sm">
                  Extracting QUIC features and running ensemble...
                </div>
              )}
              {/* Prediction Result */}
              {result && (
                <div className="mt-8 grid gap-6">
                  <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-6">
                    <p className="text-sm text-zinc-400 mb-2">
                      Predicted Application
                    </p>

                    <h2 className="text-3xl font-bold text-white">
                      {result.prediction.app_name}
                    </h2>

                    <div className="mt-4 flex flex-wrap gap-4 text-sm">
                      <div className="rounded-xl bg-zinc-800 px-4 py-2">
                        <span className="text-zinc-400">App ID:</span>{" "}
                        <span className="text-white font-semibold">
                          {result.prediction.app_id}
                        </span>
                      </div>

                      <div className="rounded-xl bg-zinc-800 px-4 py-2">
                        <span className="text-zinc-400">Confidence:</span>{" "}
                        <span className="text-emerald-400 font-semibold">
                          {(result.prediction.confidence * 100).toFixed(2)}%
                        </span>
                      </div>

                      <div className="rounded-xl bg-zinc-800 px-4 py-2">
                        <span className="text-zinc-400">Latency:</span>{" "}
                        <span className="text-cyan-400 font-semibold">
                          {result.latency_ms.toFixed(2)} ms
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Top 5 Predictions */}
                  <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">
                      Top 5 Predictions
                    </h3>

                    <div className="space-y-3">
                      {result.top5_predictions.map((item, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between rounded-xl bg-zinc-800/70 px-4 py-3"
                        >
                          <div>
                            <p className="font-medium text-white">
                              {item.app_name}
                            </p>

                            <p className="text-xs text-zinc-400">
                              App ID: {item.app_id}
                            </p>
                          </div>

                          <div className="text-emerald-400 font-semibold">
                            {(item.probability * 100).toFixed(2)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-6 rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
                  {error}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* MAIN GRID */}
        <section className="grid gap-8 lg:grid-cols-[1.5fr_0.8fr]">
          {/* FORM */}
          <form
            onSubmit={handlePredict}
            className="rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl"
          >
            <div className="mb-8 flex items-center gap-3">
              <div className="rounded-2xl bg-cyan-500/15 p-3">
                <Radio className="h-7 w-7 text-cyan-400" />
              </div>

              <div>
                <h2 className="text-2xl font-bold">
                  Flow Feature Controls
                </h2>

                <p className="text-zinc-400">
                  Generate synthetic QUIC flow statistics
                </p>
              </div>
            </div>

            <div className="grid gap-8 md:grid-cols-2">
              <div className="space-y-7">
                <SliderField
                  label="Average IPT"
                  min={1}
                  max={2000}
                  step={1}
                  value={form.avg_ipt}
                  suffix=" ms"
                  onChange={(e) =>
                    updateField(
                      'avg_ipt',
                      Number(e.target.value)
                    )
                  }
                />

                <SliderField
                  label="IPT Standard Deviation"
                  min={1}
                  max={1000}
                  step={1}
                  value={form.std_ipt}
                  suffix=" ms"
                  onChange={(e) =>
                    updateField(
                      'std_ipt',
                      Number(e.target.value)
                    )
                  }
                />

                <SliderField
                  label="Average Packet Size"
                  min={32}
                  max={1500}
                  step={1}
                  value={form.avg_size}
                  suffix=" B"
                  onChange={(e) =>
                    updateField(
                      'avg_size',
                      Number(e.target.value)
                    )
                  }
                />

                <SliderField
                  label="Packet Size Standard Deviation"
                  min={1}
                  max={800}
                  step={1}
                  value={form.std_size}
                  suffix=" B"
                  onChange={(e) =>
                    updateField(
                      'std_size',
                      Number(e.target.value)
                    )
                  }
                />
              </div>

              <div className="space-y-7">
                <SliderField
                  label="UL/DL Direction Ratio"
                  min={0}
                  max={1}
                  step={0.01}
                  value={form.ul_dl_ratio_direction}
                  onChange={(e) =>
                    updateField(
                      'ul_dl_ratio_direction',
                      Number(e.target.value)
                    )
                  }
                />

                <SliderField
                  label="UL/DL Byte Ratio"
                  min={0}
                  max={1}
                  step={0.01}
                  value={form.ul_dl_ratio_bytes}
                  onChange={(e) =>
                    updateField(
                      'ul_dl_ratio_bytes',
                      Number(e.target.value)
                    )
                  }
                />

                <SliderField
                  label="UL/DL Packet Ratio"
                  min={0}
                  max={1}
                  step={0.01}
                  value={form.ul_dl_ratio_packets}
                  onChange={(e) =>
                    updateField(
                      'ul_dl_ratio_packets',
                      Number(e.target.value)
                    )
                  }
                />

                <SliderField
                  label="PPI Duration"
                  min={100}
                  max={20000}
                  step={50}
                  value={form.ppi_duration}
                  suffix=" ms"
                  onChange={(e) =>
                    updateField(
                      'ppi_duration',
                      Number(e.target.value)
                    )
                  }
                />
              </div>
            </div>

            <div className="mt-10 grid gap-8 md:grid-cols-3">
              <div className="space-y-3">
                <label className="text-sm font-medium text-zinc-300">
                  PPI Length
                </label>

                <input
                  type="number"
                  min={2}
                  max={30}
                  value={form.ppi_len}
                  onChange={(e) =>
                    updateField(
                      'ppi_len',
                      Number(e.target.value)
                    )
                  }
                  className="w-full rounded-2xl border border-zinc-700 bg-zinc-900/70 px-4 py-3 outline-none transition focus:border-cyan-400"
                />
              </div>

              <div className="space-y-3">
                <label className="text-sm font-medium text-zinc-300">
                  PPI Roundtrips
                </label>

                <input
                  type="number"
                  min={0}
                  max={30}
                  value={form.ppi_roundtrips}
                  onChange={(e) =>
                    updateField(
                      'ppi_roundtrips',
                      Number(e.target.value)
                    )
                  }
                  className="w-full rounded-2xl border border-zinc-700 bg-zinc-900/70 px-4 py-3 outline-none transition focus:border-cyan-400"
                />
              </div>

              <div className="space-y-3">
                <label className="text-sm font-medium text-zinc-300">
                  Flow End Reason
                </label>

                <select
                  value={form.flow_end_reason}
                  onChange={(e) =>
                    updateField(
                      'flow_end_reason',
                      Number(e.target.value)
                    )
                  }
                  className="w-full rounded-2xl border border-zinc-700 bg-zinc-900/70 px-4 py-3 outline-none transition focus:border-cyan-400"
                >
                  {FLOW_END_REASONS.map((item) => (
                    <option
                      key={item.value}
                      value={item.value}
                    >
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="mt-10 flex flex-wrap items-center gap-4">
              <button
                type="submit"
                disabled={loading}
                className="hover:cursor-pointer inline-flex items-center gap-3 rounded-2xl bg-cyan-500 px-8 py-4 font-semibold text-black transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Sparkles className="h-5 w-5" />

                {loading
                  ? 'Running Inference...'
                  : 'Predict Application'}
              </button>

              <button
                type="button"
                onClick={() => {
                  setForm(INITIAL_FORM)
                  setResult(null)
                  setError(null)
                }}
                className="hover:cursor-pointer inline-flex items-center gap-3 rounded-2xl border border-zinc-700 bg-zinc-900/70 px-8 py-4 font-semibold text-zinc-200 transition hover:border-zinc-500"
              >
                <ArrowUpDown className="h-5 w-5" />
                Reset
              </button>
              <button
                type="button"
                onClick={() => {
                  setForm({
                    avg_ipt: Math.floor(Math.random() * 2000) + 1,
                    std_ipt: Math.floor(Math.random() * 1000) + 1,
                    avg_size: Math.floor(Math.random() * (1500 - 32 + 1)) + 32,
                    std_size: Math.floor(Math.random() * 800) + 1,
                    ul_dl_ratio_direction: parseFloat((Math.random()).toFixed(2)),
                    ul_dl_ratio_bytes: parseFloat((Math.random()).toFixed(2)),
                    ul_dl_ratio_packets: parseFloat((Math.random()).toFixed(2)),
                    ppi_duration: Math.floor(Math.random() * (20000 - 100 + 1)) + 100,
                    ppi_len: Math.floor(Math.random() * (30 - 2 + 1)) + 2,
                    ppi_roundtrips: Math.floor(Math.random() * 31),
                    // flow_end_reason must be a random choice of 0 or 1 or 2
                    flow_end_reason: Math.floor(Math.random() * FLOW_END_REASONS.length),
                  })
                  setResult(null)
                  setError(null)
                }}
                className="hover:cursor-pointer inline-flex items-center gap-3 rounded-2xl border border-zinc-700 bg-zinc-900/70 px-8 py-4 font-semibold text-zinc-200 transition hover:border-zinc-500"
              >
                {/* <ArrowUpDown className="h-5 w-5" /> */}
                {/* suggest dice flower icon */}
                <Dices className="h-5 w-5" />
                Randomize
              </button>
            </div>
          </form>

          {/* SIDE PANEL */}
          <aside className="space-y-6">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
              <h3 className="mb-4 text-xl font-bold">
                Backend Payload
              </h3>

              <pre className="overflow-auto rounded-2xl bg-black/40 p-4 text-xs text-cyan-300">
                {JSON.stringify(
                  {
                    features: form,
                    strategy: strategy,
                  },
                  null,
                  2
                )}
              </pre>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
              <h3 className="mb-4 text-xl font-bold">
                Model Pipeline
              </h3>

              <ul className="space-y-4 text-sm text-zinc-300">
                <li>
                  • Synthetic feature generation
                </li>

                <li>
                  • StandardScaler normalization
                </li>

                <li>
                  • Multi-model probability inference
                </li>

                <li>
                  • Weighted soft-voting ensemble
                </li>

                <li>
                  • Human-readable app classification
                </li>
              </ul>
            </div>
          </aside>
        </section>
      </div>
    </main>
  )
}

export default App
