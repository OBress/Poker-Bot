"use client";

import { useEffect, useState } from "react";
import { fetchGame, fetchHistory, fmtBB, fmtDate } from "@/lib/api";
import type { GameRecord, HistorySummary } from "@/lib/types";
import { HistoryRow } from "./Home";
import ReadReport from "./ReadReport";
import { Logo, Pill } from "./ui";

const KIND: Record<string, string> = { human: "Person", pokerbot: "PokerBot", bot: "Calibrated bot", tag: "TAG baseline" };

export default function History({ onBack }: { onBack: () => void }) {
  const [games, setGames] = useState<HistorySummary[] | null>(null);
  const [open, setOpen] = useState<GameRecord | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetchHistory().then(setGames).catch(() => setFailed(true));
  }, []);

  return (
    <div className="min-h-screen bg-[#120d18] px-4 py-6 text-white">
      <div className="mx-auto max-w-4xl">
        <div className="flex items-center gap-4">
          <button className="btn btn-ghost" onClick={open ? () => setOpen(null) : onBack}>
            ← {open ? "All games" : "Home"}
          </button>
          <Logo small />
          <span className="text-white/50">{open ? "Game details" : "Past games"}</span>
        </div>

        {open ? (
          <GameDetail g={open} />
        ) : failed ? (
          <div className="card mt-6 text-sm text-red-200">Couldn&apos;t load your games from the game server.</div>
        ) : games === null ? (
          <div className="mt-6 text-sm text-white/50">Loading…</div>
        ) : games.length === 0 ? (
          <div className="card mt-6 text-sm text-white/55">No games yet. Every game you play is saved here.</div>
        ) : (
          <div className="mt-6 grid gap-2">
            {games.map((g) => (
              <HistoryRow key={g.id} g={g} onClick={() => fetchGame(g.id).then(setOpen).catch(() => setFailed(true))} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function GameDetail({ g }: { g: GameRecord }) {
  const humans = g.players.filter((p) => p.kind === "human" && p.read);
  const [showLog, setShowLog] = useState(false);
  return (
    <div className="mt-6 space-y-6">
      <div className="flex flex-wrap items-center gap-2 text-sm text-white/65">
        <Pill tone={g.mode === "read" ? "teal" : "neutral"}>{g.mode === "read" ? `Read me · ${g.target} hands` : "Cash game"}</Pill>
        <span>{fmtDate(g.started)}</span>
        <span>· {g.hands} hands</span>
        {g.mode === "read" && !g.finished && <span className="text-amber-200">· ended early</span>}
      </div>

      {humans.map((p) => (
        <div key={p.seat} className="card">
          <ReadReport
            name={p.name}
            read={p.read!}
            about={g.styleInfo[p.read!.style]}
            stats={p.stats}
            trajectory={p.trajectory}
          />
        </div>
      ))}

      <div className="card overflow-x-auto p-0">
        <table className="w-full text-sm">
          <thead className="text-left text-xs tracking-wider text-white/45 uppercase">
            <tr>
              <th className="px-4 py-3">Player</th>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3">Actually</th>
              <th className="px-4 py-3">PokerBot&apos;s read</th>
              <th className="px-4 py-3 text-right">Result</th>
            </tr>
          </thead>
          <tbody>
            {[...g.players].sort((a, b) => b.net - a.net).map((p) => {
              const right = p.trueStyle && p.read && p.trueStyle === p.read.style;
              return (
                <tr key={p.seat} className="border-t border-white/5">
                  <td className="px-4 py-2.5 font-medium">{p.name}</td>
                  <td className="px-4 py-2.5 text-white/60">{KIND[p.kind] ?? p.kind}</td>
                  <td className="px-4 py-2.5 text-white/60">{p.trueStyle ?? "—"}</td>
                  <td className="px-4 py-2.5">
                    {p.read ? (
                      <>
                        {p.read.style} <span className="text-white/45">{Math.round(p.read.confidence * 100)}%</span>
                        {p.trueStyle && <span className={right ? "text-emerald-300" : "text-red-300"}>{right ? " ✓" : " ✗"}</span>}
                      </>
                    ) : (
                      <span className="text-white/35">—</span>
                    )}
                  </td>
                  <td className={`px-4 py-2.5 text-right tabular-nums ${p.net >= 0 ? "text-emerald-300" : "text-red-300"}`}>
                    {fmtBB(p.net, g.bigBlind, true)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div>
        <button className="btn btn-ghost" onClick={() => setShowLog((x) => !x)}>
          {showLog ? "Hide" : "Show"} hand log ({g.log.length} lines)
        </button>
        {showLog && (
          <pre className="card mt-3 max-h-96 overflow-y-auto text-xs leading-relaxed whitespace-pre-wrap text-white/70">
            {g.log.join("\n")}
          </pre>
        )}
      </div>
    </div>
  );
}
