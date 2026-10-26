"use client";

import { useState } from "react";
import { fmtBlinds, fmtNum } from "@/lib/api";
import type { SeatKind, Setup } from "@/lib/types";

// Full-ring styles in the server's order (profiles/bots-v3.json "fr"); a bot's style is an index.
export const STYLES = ["Loose-passive", "Passive limper", "Weak-tight", "TAG", "Nit", "Short-stacker"];

type Preset = "mixed" | "pokerbot" | "bots" | "custom";
type SeatChoice = { kind: SeatKind; style: number | null };

const PRESETS: { id: Preset; label: string; hint: string }[] = [
  { id: "mixed", label: "Mixed", hint: "PokerBots and calibrated bots" },
  { id: "pokerbot", label: "All PokerBot", hint: "every opponent is the engine" },
  { id: "bots", label: "All calibrated bots", hint: "bots modeled on real players" },
  { id: "custom", label: "Custom", hint: "pick every seat" },
];

const OPPONENT_KINDS: { kind: SeatKind; label: string }[] = [
  { kind: "pokerbot", label: "PokerBot (engine)" },
  { kind: "bot", label: "Calibrated bot" },
  { kind: "tag", label: "TAG baseline" },
  { kind: "human", label: "Open seat (person)" },
];

/** Seats 2..players for a preset; seat 1 is always you. */
function presetSeats(preset: Preset, players: number, custom: SeatChoice[]): SeatChoice[] {
  const styles = [0, 4, 1, 2, 3, 5];
  return Array.from({ length: players - 1 }, (_, i) => {
    if (preset === "custom") return custom[i] ?? { kind: "pokerbot", style: null };
    if (preset === "pokerbot") return { kind: "pokerbot", style: null };
    if (preset === "bots") return { kind: "bot", style: styles[i % styles.length] };
    // mixed: PokerBots in seats 3 and 6 (when they exist), calibrated bots elsewhere
    const seat = i + 2;
    if (seat === 3 || (seat === 6 && players >= 6)) return { kind: "pokerbot", style: null };
    return { kind: "bot", style: styles[i % styles.length] };
  });
}

export interface SetupState {
  players: number;
  preset: Preset;
  custom: SeatChoice[];
  sb: number;
  bb: number;
  buyin: number;
}

export const DEFAULT_SETUP: SetupState = {
  players: 8,
  preset: "mixed",
  custom: presetSeats("mixed", 8, []),
  sb: 1,
  bb: 2,
  buyin: 200,
};

export function toSetup(s: SetupState): Setup {
  const others = presetSeats(s.preset, s.players, s.custom);
  return {
    players: s.players,
    seats: [{ kind: "human", style: null }, ...others],
    sb: s.sb,
    bb: s.bb,
    buyin: s.buyin,
  };
}

export function setupProblem(s: SetupState): string | null {
  if (!(s.sb > 0 && s.bb > 0)) return "Blinds must be above zero.";
  if (s.sb > s.bb) return "The small blind can't be bigger than the big blind.";
  const depth = s.buyin / s.bb;
  if (!(depth >= 10 && depth <= 1000)) return "The buy-in must be 10 to 1,000 big blinds.";
  return null;
}

export function describeSetup(s: SetupState): string {
  const others = presetSeats(s.preset, s.players, s.custom);
  const count = (k: SeatKind) => others.filter((x) => x.kind === k).length;
  const parts = [
    count("pokerbot") && `${count("pokerbot")} PokerBot${count("pokerbot") > 1 ? "s" : ""}`,
    count("bot") && `${count("bot")} bot${count("bot") > 1 ? "s" : ""}`,
    count("tag") && `${count("tag")} TAG`,
    count("human") && `${count("human")} open seat${count("human") > 1 ? "s" : ""}`,
  ].filter(Boolean);
  return `${s.players} players: you + ${parts.join(" + ")} · blinds ${fmtBlinds(s)} · buy-in ${fmtNum(s.buyin)}`;
}

export function useSetupState(): [SetupState, (s: SetupState) => void] {
  const [s, setS] = useState<SetupState>(() => {
    try {
      const raw = localStorage.getItem("pokerbot-setup");
      return raw ? { ...DEFAULT_SETUP, ...JSON.parse(raw) } : DEFAULT_SETUP;
    } catch {
      return DEFAULT_SETUP;
    }
  });
  const save = (x: SetupState) => {
    setS(x);
    try {
      localStorage.setItem("pokerbot-setup", JSON.stringify(x));
    } catch {}
  };
  return [s, save];
}

export default function TableSetup({ value, onChange }: { value: SetupState; onChange: (s: SetupState) => void }) {
  const s = value;
  const set = (patch: Partial<SetupState>) => onChange({ ...s, ...patch });
  const seats = presetSeats(s.preset, s.players, s.custom);
  const problem = setupProblem(s);
  const setSeat = (i: number, choice: SeatChoice) => {
    const custom = presetSeats(s.preset, 8, s.custom).map((x, j) => (j === i ? choice : x));
    set({ preset: "custom", custom });
  };
  return (
    <div className="space-y-5">
      <Field label="Players at the table">
        <div className="flex flex-wrap gap-1 rounded-lg bg-white/5 p-1">
          {[2, 3, 4, 5, 6, 7, 8].map((n) => (
            <button key={n} onClick={() => set({ players: n })}
              className={`h-8 w-9 rounded-md text-sm font-semibold ${s.players === n ? "bg-white/90 text-black" : "text-white/70 hover:bg-white/10"}`}>
              {n}
            </button>
          ))}
        </div>
        {s.players === 2 && <p className="mt-1.5 text-xs text-teal-200/80">Heads-up: PokerBot plays the evolved exploiter from the research results.</p>}
      </Field>

      <Field label="Opponents">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {PRESETS.map((p) => (
            <button key={p.id} onClick={() => set({ preset: p.id, custom: p.id === "custom" ? presetSeats(s.preset, 8, s.custom) : s.custom })}
              className={`rounded-lg border px-3 py-2 text-left text-sm ${s.preset === p.id ? "border-teal-400 bg-teal-500/15" : "border-white/10 bg-white/5 hover:bg-white/10"}`}>
              <div className="font-semibold">{p.label}</div>
              <div className="text-xs text-white/50">{p.hint}</div>
            </button>
          ))}
        </div>
        <div className="mt-3 grid gap-1.5">
          <SeatRow n={1}><span className="text-white/80">You</span></SeatRow>
          {seats.map((seat, i) => (
            <SeatRow key={i} n={i + 2}>
              <select className="rounded-md bg-black/40 px-2 py-1 text-sm" value={seat.kind}
                onChange={(e) => setSeat(i, { kind: e.target.value as SeatKind, style: e.target.value === "bot" ? null : null })}>
                {OPPONENT_KINDS.map((k) => (
                  <option key={k.kind} value={k.kind}>{k.label}</option>
                ))}
              </select>
              {seat.kind === "bot" && (
                <select className="rounded-md bg-black/40 px-2 py-1 text-sm" value={seat.style ?? ""}
                  onChange={(e) => setSeat(i, { kind: "bot", style: e.target.value === "" ? null : Number(e.target.value) })}>
                  <option value="">Random style</option>
                  {STYLES.map((st, j) => (
                    <option key={st} value={j}>{st}</option>
                  ))}
                </select>
              )}
            </SeatRow>
          ))}
        </div>
      </Field>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Blinds (small / big)">
          <div className="flex items-center gap-2">
            <NumberInput value={s.sb} onChange={(v) => set({ sb: v })} />
            <span className="text-white/40">/</span>
            <NumberInput value={s.bb} onChange={(v) => set({ bb: v })} />
          </div>
        </Field>
        <Field label="Buy-in (starting stack)">
          <div className="flex items-center gap-2">
            <NumberInput value={s.buyin} onChange={(v) => set({ buyin: v })} />
            <span className="text-sm whitespace-nowrap text-white/50">= {s.bb > 0 ? fmtNum(s.buyin / s.bb) : "?"} big blinds</span>
          </div>
        </Field>
      </div>
      <p className="text-xs text-white/45">
        Calibrated bots keep their own buy-in from real data (a short-stacker buys in short), up to yours. Busted players
        rebuy automatically.
      </p>
      {problem && <p className="text-sm text-red-300">{problem}</p>}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-xs font-medium tracking-wider text-white/50 uppercase">{label}</div>
      {children}
    </div>
  );
}

function SeatRow({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg bg-white/5 px-3 py-1.5 text-sm">
      <span className="w-14 text-white/45">Seat {n}</span>
      {children}
    </div>
  );
}

function NumberInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <input type="number" min={0} step="any" value={Number.isFinite(value) ? value : ""}
      onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
      className="w-24 rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-sm outline-none focus:border-teal-400" />
  );
}
