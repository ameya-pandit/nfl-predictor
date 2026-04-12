import { useState, useEffect } from "react";

const API_BASE = "https://nfl-predictor-api-production.up.railway.app";

const WEEKS = Array.from({ length: 19 }, (_, i) => i + 4);

const WEEK_LABEL = (w) => {
  if (w === 19) return "WC";
  if (w === 20) return "DIV";
  if (w === 21) return "CONF";
  if (w === 22) return "SB";
  return `W${w}`;
};

const WEEK_FULL = (w) => {
  if (w === 19) return "Wild Card";
  if (w === 20) return "Divisional Round";
  if (w === 21) return "Conference Championships";
  if (w === 22) return "Super Bowl";
  return `WEEK ${w}`;
};

const TEAM_COLORS = {
  ARI: "#97233F", ATL: "#A71930", BAL: "#241773", BUF: "#00338D",
  CAR: "#0085CA", CHI: "#0B162A", CIN: "#FB4F14", CLE: "#311D00",
  DAL: "#003594", DEN: "#FB4F14", DET: "#0076B6", GB: "#203731",
  HOU: "#03202F", IND: "#002C5F", JAX: "#006778", KC: "#E31837",
  LA: "#003594", LAC: "#0080C6", LV: "#000000", MIA: "#008E97",
  MIN: "#4F2683", NE: "#002244", NO: "#D3BC8D", NYG: "#0B2265",
  NYJ: "#125740", PHI: "#004C54", PIT: "#FFB612", SEA: "#002244",
  SF: "#AA0000", TB: "#D50A0A", TEN: "#4B92DB", WAS: "#5A1414",
};

const CONFIDENCE_LABELS = {
  high_confidence: { label: "High", color: "#22c55e" },
  moderate: { label: "Moderate", color: "#f59e0b" },
  toss_up: { label: "Toss-up", color: "#6b7280" },
};

function TeamBlock({ team, isWinner }) {
  const color = TEAM_COLORS[team] || "#333";
  return (
    <div className="flex flex-col items-center gap-2 flex-1">
      <div
        className="w-16 h-16 rounded-full flex items-center justify-center text-white font-black text-lg tracking-widest border-2"
        style={{
          background: color,
          borderColor: isWinner ? "#f0c040" : "transparent",
          boxShadow: isWinner ? `0 0 18px ${color}99` : "none",
        }}
      >
        {team}
      </div>
      {isWinner && (
        <span className="text-xs font-bold tracking-widest uppercase text-yellow-400">
          Predicted
        </span>
      )}
    </div>
  );
}

function GameCard({ game }) {
  const conf = CONFIDENCE_LABELS[game.confidence_tier] || CONFIDENCE_LABELS.toss_up;
  const homeWins = game.predicted_winner === game.home_team;
  const absMargin = Math.abs(game.predicted_margin).toFixed(1);
  const isCompleted = game.actual_margin !== null && game.actual_margin !== undefined;
  const correct = game.correct_pick === 1;
  const winnerColor = TEAM_COLORS[game.predicted_winner] || "#333";

  const shadeOpacity = 0.45;

  // Boost dark colors by blending with white
  const r = parseInt(winnerColor.slice(1,3), 16);
  const g = parseInt(winnerColor.slice(3,5), 16);
  const b = parseInt(winnerColor.slice(5,7), 16);
  const brightness = (r * 299 + g * 587 + b * 114) / 1000;
  const boostedColor = brightness < 100
  ? `rgb(${Math.min(Math.round(r*2.5),255)}, ${Math.min(Math.round(g*2.5),255)}, ${Math.min(Math.round(b*2.5),255)})`
  : winnerColor;

  return (
    <div
      className="rounded-2xl flex flex-col gap-4 border transition-all duration-200 hover:scale-[1.01] overflow-hidden relative"
      style={{
        borderColor: isCompleted
          ? correct ? "rgba(34,197,94,0.8)" : "rgba(239,68,68,0.8)"
          : "rgba(255,255,255,0.08)",
        borderWidth: isCompleted ? "3px" : "1px",
      }}
    >
      {/* Winner shade background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: homeWins
            ? `linear-gradient(to right, ${boostedColor} 0%, transparent 50%)`
            : `linear-gradient(to left, ${boostedColor} 0%, transparent 50%)`,
          opacity: shadeOpacity,
        }}
      />

      {/* Teams row */}
      <div className="flex items-center justify-between gap-4 p-5 relative">
        <TeamBlock team={game.home_team} isWinner={homeWins} />
        <div className="flex flex-col items-center gap-1">
          <span className="text-xs text-gray-500 uppercase tracking-widest font-semibold">vs</span>
          <span className="text-2xl font-black text-white tabular-nums">{absMargin}</span>
          <span className="text-xs text-gray-500">pts</span>
        </div>
        <TeamBlock team={game.away_team} isWinner={!homeWins} />
      </div>

      {/* Bottom bar */}
      <div className="flex items-center justify-between px-5 pb-5 border-t border-white/5 pt-3 relative">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ background: conf.color }} />
          <span className="text-xs font-semibold tracking-wide" style={{ color: conf.color }}>
            {conf.label} Confidence
          </span>
        </div>
        {isCompleted && (
          <span
            className="text-xs font-bold px-2 py-0.5 rounded-full"
            style={{
              background: correct ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)",
              color: correct ? "#22c55e" : "#ef4444",
            }}
          >
            {correct ? "✓ Correct" : "✗ Wrong"} · {Math.abs(game.actual_margin).toFixed(0)} pts
          </span>
        )}
      </div>
    </div>
  );
}
function PerformanceBar({ label, wins, losses, accuracy }) {
  const pct = Math.round((accuracy || 0) * 100);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between items-baseline">
        <span className="text-xs uppercase tracking-widest text-gray-400 font-semibold">{label}</span>
        <span className="text-2xl font-black text-white">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: pct >= 60 ? "#22c55e" : pct >= 50 ? "#f59e0b" : "#ef4444",
          }}
        />
      </div>
      <span className="text-xs text-gray-500">{wins}W – {losses}L</span>
    </div>
  );
}

export default function App() {
  const [week, setWeek] = useState(4);
  const [predictions, setPredictions] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [availableWeeks, setAvailableWeeks] = useState(new Set());
  const [seasonRecord, setSeasonRecord] = useState(null);

  // Discover which weeks have data on mount
  useEffect(() => {
    const found = new Set();
    Promise.all(
      WEEKS.map((w) =>
        fetch(`${API_BASE}/predictions?week=${w}`)
          .then((r) => r.json())
          .then((data) => { if (Array.isArray(data) && data.length > 0) found.add(w); })
          .catch(() => {})
      )
    ).then(() => {
      setAvailableWeeks(new Set(found));
      const first = WEEKS.find((w) => found.has(w));
      if (first) setWeek(first);
    });
  }, []);

  // Fetch season-to-date record from the last available week
  useEffect(() => {
    if (availableWeeks.size === 0) return;
    const lastWeek = Math.max(...availableWeeks);
    fetch(`${API_BASE}/performance?week=${lastWeek}`)
      .then((r) => r.json())
      .then((data) => { if (data && !data.detail) setSeasonRecord(data); })
      .catch(() => {});
  }, [availableWeeks]);

  useEffect(() => {
    if (!availableWeeks.has(week)) return;
    setLoading(true);
    setPredictions([]);
    setPerformance(null);

    fetch(`${API_BASE}/predictions?week=${week}`)
      .then((r) => r.json())
      .then((data) => setPredictions(Array.isArray(data) ? data : []))
      .catch(() => setPredictions([]));

    fetch(`${API_BASE}/performance?week=${week}`)
      .then((r) => r.json())
      .then((data) => { if (data && !data.detail) setPerformance(data); })
      .catch(() => setPerformance(null))
      .finally(() => setLoading(false));
  }, [week, availableWeeks]);

  const hasResults = predictions.some(
    (g) => g.actual_margin !== null && g.actual_margin !== undefined
  );

  return (
    <div className="min-h-screen text-white" style={{ background: "#0a0a0f", fontFamily: "'DM Mono', monospace" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;900&display=swap" rel="stylesheet" />

      {/* Header */}
      <header className="border-b border-white/5 px-6 py-5">
        <div className="max-w-5xl mx-auto flex items-baseline justify-between">
          <h1 style={{ fontSize: "2.2rem", letterSpacing: "0.08em", color: "#f0c040", fontWeight: 900 }}>
            MODEL THE NFL
          </h1>
          {seasonRecord && (
            <div className="flex items-baseline gap-2">
              <span className="text-xs text-gray-500 uppercase tracking-widest">2025 Season</span>
              <span className="text-sm font-bold text-white tabular-nums">
                {seasonRecord.cumulative_wins}W – {seasonRecord.cumulative_losses}L
              </span>
              <span
                className="text-sm font-black tabular-nums"
                style={{ color: seasonRecord.cumulative_accuracy >= 0.6 ? "#22c55e" : "#f59e0b" }}
              >
                ({Math.round(seasonRecord.cumulative_accuracy * 100)}%)
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Week tabs */}
      <div className="border-b border-white/5 sticky top-0 z-10" style={{ background: "#0a0a0f" }}>
        <div className="max-w-5xl mx-auto px-6 overflow-x-auto">
          <div className="flex gap-1 py-2" style={{ minWidth: "max-content" }}>
            {WEEKS.map((w) => {
              const available = availableWeeks.has(w);
              const active = week === w;
              return (
                <button
                  key={w}
                  onClick={() => available && setWeek(w)}
                  disabled={!available}
                  className="px-3 py-1.5 rounded-lg text-xs font-bold tracking-widest uppercase transition-all duration-150"
                  style={{
                    background: active ? "#f0c040" : "transparent",
                    color: active ? "#0a0a0f" : available ? "#9ca3af" : "#2d2d2d",
                    border: active ? "none" : "1px solid rgba(255,255,255,0.06)",
                    cursor: available ? "pointer" : "not-allowed",
                    opacity: available ? 1 : 0.4,
                  }}
                >
                  {WEEK_LABEL(w)}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <main className="max-w-5xl mx-auto px-6 py-8 flex flex-col gap-8">
        {/* Week title */}
        <div className="flex items-baseline gap-3">
          <h2 style={{ fontSize: "1.6rem", letterSpacing: "0.06em", color: "#fff", fontWeight: 900 }}>
            {WEEK_FULL(week)}
          </h2>
          <span className="text-xs text-gray-500">{predictions.length} matchups</span>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <div className="w-4 h-4 border border-gray-600 border-t-yellow-400 rounded-full animate-spin" />
            Loading predictions...
          </div>
        )}

        {/* Performance section */}
        {hasResults && performance && (
          <div
            className="rounded-2xl p-6 border border-white/8 flex flex-col gap-6"
            style={{ background: "rgba(255,255,255,0.02)" }}
          >
            <h3 style={{ fontSize: "1.2rem", letterSpacing: "0.06em", color: "#f0c040", fontWeight: 900 }}>
              PERFORMANCE
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <PerformanceBar
                label={WEEK_FULL(week)}
                wins={performance.weekly_wins}
                losses={performance.weekly_losses}
                accuracy={performance.weekly_accuracy}
              />
              <PerformanceBar
                label="Season to date"
                wins={performance.cumulative_wins}
                losses={performance.cumulative_losses}
                accuracy={performance.cumulative_accuracy}
              />
            </div>
          </div>
        )}

        {/* Games grid */}
        {!loading && predictions.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {predictions.map((game, i) => (
              <GameCard key={i} game={game} />
            ))}
          </div>
        )}

        {!loading && predictions.length === 0 && availableWeeks.has(week) && (
          <div className="text-gray-600 text-sm">No predictions available for this week.</div>
        )}
      </main>
    </div>
  );
}