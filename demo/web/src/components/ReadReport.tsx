"use client";

import type { Fit, Read, Stat } from "@/lib/types";

const COLORS: Record<string, string> = {
  "Loose-passive": "#f59e0b",
  "Passive limper": "#84cc16",
  "Weak-tight": "#38bdf8",
  TAG: "#a78bfa",
  Nit: "#94a3b8",
  "Short-stacker": "#f472b6",
};
const FALLBACK = ["#fb7185", "#2dd4bf", "#facc15", "#60a5fa", "#c084fc", "#4ade80"];

export function styleColor(style: string, i = 0): string {
  return COLORS[style] ?? FALLBACK[i % FALLBACK.length];
}

const pct = (p: number) => `${Math.round(p * 100)}%`;

/** PokerBot's read of one player: the verdict, every style's probability, how the read moved
 * hand by hand, and the player's stats against each style's typical numbers. */
export default function ReadReport({
  read,
  about,
  stats,
  trajectory,
  name,
}: {
  read: Read;
  about?: string;
  stats?: Stat[] | null;
  trajectory?: { hand: number; probs: Record<string, number> }[] | null;
  name?: string;
}) {
  const styles = Object.keys(read.probs);
  return (
    <div className="space-y-6">
      <div>
        <div className="text-xs font-medium tracking-wider text-white/50 uppercase">
          PokerBot reads {name ?? "you"} as
        </div>
        <div className="mt-1 flex flex-wrap items-baseline gap-3">
          <span className="text-3xl font-bold" style={{ color: styleColor(read.style) }}>
            {read.style}
          </span>
          <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-sm text-white/80">
            {pct(read.confidence)} sure · {read.hands} hands
          </span>
        </div>
        {about && <p className="mt-2 text-sm text-white/70">{about}</p>}
        {read.fit && <FitNote fit={read.fit} style={read.style} name={name} />}
        {read.hands < 50 && (
          <p className="mt-2 text-xs text-amber-200/80">
            Early read: it usually firms up after about 50 hands.
          </p>
        )}
      </div>

      <section>
        <h4 className="mb-2 text-sm font-semibold text-white/80">How likely each style is</h4>
        <div className="space-y-1.5">
          {styles.map((s, i) => (
            <div key={s} className="flex items-center gap-3 text-sm">
              <span className="w-32 shrink-0 truncate text-white/75">{s}</span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full transition-[width] duration-500"
                  style={{ width: pct(read.probs[s]), background: styleColor(s, i) }}
                />
              </div>
              <span className="w-10 text-right text-white/60 tabular-nums">{pct(read.probs[s])}</span>
            </div>
          ))}
        </div>
      </section>

      {trajectory && trajectory.length > 1 && <Trajectory points={trajectory} styles={styles} />}
      {stats && stats.length > 0 && <Stats stats={stats} top={read.style} styles={styles} />}
    </div>
  );
}

const FIT_COLORS: Record<Fit["level"], string> = {
  well: "bg-emerald-400/15 text-emerald-200",
  loose: "bg-amber-400/15 text-amber-200",
  none: "bg-rose-400/15 text-rose-200",
};

function joinWords(parts: string[]): string {
  return parts.length < 2 ? (parts[0] ?? "") : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

/** How typical the player is of the style they were read as: the classifier always names the
 * nearest style, even for someone unlike every style it knows. */
function FitNote({ fit, style, name }: { fit: Fit; style: string; name?: string }) {
  const who = name ?? "you";
  const ways = joinWords(fit.extremes.map((e) => (name ? e.they : e.you)));
  let text: string;
  if (fit.level === "well") {
    text = `${name ? `${name}'s` : "Your"} numbers are typical of a real ${style}.`;
  } else if (!ways) {
    text = `Closest style: ${style}, but ${who} ${name ? "is" : "are"} an unusual one.`;
  } else if (fit.level === "loose") {
    text = `Closest style: ${style}, but ${who} ${ways} than most players of this type.`;
  } else {
    text = `Closest style: ${style}, but ${who} ${ways} than any real player of this type.`;
  }
  return (
    <div className="mt-3 space-y-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${FIT_COLORS[fit.level]}`}
          title={`Distance ${fit.distance.toFixed(1)} over ${fit.stats} stats; ${(fit.p * 100).toPrecision(2)}% of real players of this style sit at least this far out`}
        >
          {fit.label}
        </span>
        {fit.alias && (
          <span className="rounded-full bg-rose-500/20 px-2.5 py-0.5 text-xs font-medium text-rose-100">
            Demo label: {fit.alias}
          </span>
        )}
      </div>
      <p className="text-sm text-white/70">{text}</p>
      {fit.alias && (
        <p className="text-xs text-white/50">
          Plays more hands and raises more before the flop than every style PokerBot learned from real players.
        </p>
      )}
    </div>
  );
}

function Trajectory({ points, styles }: { points: { hand: number; probs: Record<string, number> }[]; styles: string[] }) {
  const W = 520;
  const H = 150;
  const pad = { l: 30, r: 8, t: 8, b: 20 };
  const maxHand = points[points.length - 1].hand;
  const x = (h: number) => pad.l + ((h - 1) / Math.max(1, maxHand - 1)) * (W - pad.l - pad.r);
  const y = (p: number) => pad.t + (1 - p) * (H - pad.t - pad.b);
  return (
    <section>
      <h4 className="mb-2 text-sm font-semibold text-white/80">How the read changed, hand by hand</h4>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Read over time">
        {[0, 0.5, 1].map((g) => (
          <g key={g}>
            <line x1={pad.l} x2={W - pad.r} y1={y(g)} y2={y(g)} stroke="rgba(255,255,255,0.08)" />
            <text x={pad.l - 6} y={y(g) + 3} textAnchor="end" fontSize="9" fill="rgba(255,255,255,0.45)">
              {pct(g)}
            </text>
          </g>
        ))}
        <text x={W - pad.r} y={H - 4} textAnchor="end" fontSize="9" fill="rgba(255,255,255,0.45)">
          hand {maxHand}
        </text>
        <text x={pad.l} y={H - 4} fontSize="9" fill="rgba(255,255,255,0.45)">
          hand 1
        </text>
        {styles.map((s, i) => (
          <polyline
            key={s}
            fill="none"
            stroke={styleColor(s, i)}
            strokeWidth={i === 0 ? 2.4 : 1.3}
            strokeOpacity={i === 0 ? 1 : 0.6}
            points={points.map((pt) => `${x(pt.hand)},${y(pt.probs[s] ?? 0)}`).join(" ")}
          />
        ))}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-white/60">
        {styles.map((s, i) => (
          <span key={s} className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: styleColor(s, i) }} />
            {s}
          </span>
        ))}
      </div>
    </section>
  );
}

function Stats({ stats, top, styles }: { stats: Stat[]; top: string; styles: string[] }) {
  return (
    <section>
      <h4 className="mb-1 text-sm font-semibold text-white/80">Your numbers vs each style</h4>
      <p className="mb-3 text-xs text-white/50">
        Each dot is a style&apos;s typical rate; the white marker is you. The read weighs all of these at once.
      </p>
      <div className="space-y-3">
        {stats.map((st) => {
          const values = Object.values(st.styles);
          const hi = Math.max(0.5, ...values, st.rate ?? 0) * 1.1;
          const pos = (v: number) => `${Math.min(100, (v / hi) * 100)}%`;
          return (
            <div key={st.key} className="text-sm">
              <div className="flex items-baseline justify-between">
                <span className="text-white/75">{st.label}</span>
                <span className="text-xs text-white/55 tabular-nums">
                  {st.rate === null ? "not seen yet" : `${pct(st.rate)} (${st.x}/${st.n})`}
                  <span className="text-white/35"> · {top} {pct(st.styles[top] ?? 0)}</span>
                </span>
              </div>
              <div className="relative mt-1.5 h-3 rounded-full bg-white/5">
                {styles.map((s, i) =>
                  st.styles[s] === undefined ? null : (
                    <span
                      key={s}
                      title={`${s}: ${pct(st.styles[s])}`}
                      className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-1 ring-black/40"
                      style={{ left: pos(st.styles[s]), background: styleColor(s, i) }}
                    />
                  ),
                )}
                {st.rate !== null && (
                  <span
                    title={`You: ${pct(st.rate)}`}
                    className="absolute top-1/2 h-4 w-1 -translate-x-1/2 -translate-y-1/2 rounded-sm bg-white shadow"
                    style={{ left: pos(st.rate) }}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
