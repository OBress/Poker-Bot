"use client";

import { useEffect, useState } from "react";
import { fetchHistory, fmtAmt, fmtBlinds, fmtDate } from "@/lib/api";
import type { ClientMessage, HistorySummary, ServerView } from "@/lib/types";
import TableSetup, { describeSetup, setupProblem, toSetup, useSetupState } from "./TableSetup";
import { Logo, Pill } from "./ui";

type Send = (m: ClientMessage) => void;

export function useName(): [string, (n: string) => void] {
  const [name, setName] = useState(() => {
    try {
      return localStorage.getItem("pokerbot-name") ?? "";
    } catch {
      return "";
    }
  });
  const save = (n: string) => {
    setName(n);
    try {
      localStorage.setItem("pokerbot-name", n);
    } catch {}
  };
  return [name, save];
}

export default function Home({ view, send, name, setName, onPlay, onHistory }: {
  view: ServerView; send: Send; name: string; setName: (n: string) => void;
  onPlay: () => void; onHistory: () => void;
}) {
  const [hands, setHands] = useState(50);
  const [setup, setSetup] = useSetupState();
  const [customizing, setCustomizing] = useState(false);
  const [games, setGames] = useState<HistorySummary[] | null>(null);
  const game = view.game;
  const player = name.trim() || "Player";

  useEffect(() => {
    fetchHistory().then(setGames).catch(() => setGames([]));
  }, [game?.id, game?.finished, game?.hands]);

  const problem = setupProblem(setup);
  const start = (mode: "cash" | "read") => {
    if (problem) {
      setCustomizing(true);
      return;
    }
    send({ type: "newGame", mode, hands: mode === "read" ? hands : undefined, name: player, setup: toSetup(setup) });
    onPlay();
  };
  const resume = (id: string) => {
    send({ type: "resume", id, name: player });
    onPlay();
  };
  const me = view.game && view.you !== null ? view.seats[view.you] : null;
  const unfinished = (games ?? []).filter((g) => g.resumable && g.id !== game?.id).slice(0, 4);
  const recent = (games ?? []).slice(0, 4);

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,#1f3b2e_0%,#120d18_55%)] px-4 py-10 text-white sm:py-14">
      <div className="mx-auto max-w-4xl">
        <header className="text-center">
          <Logo />
          <p className="mx-auto mt-3 max-w-xl text-white/65">
            Play no-limit hold&apos;em against the poker bot from this research project, and find out what it thinks
            of your game.
          </p>
        </header>

        <div className="mx-auto mt-8 max-w-sm">
          <label className="mb-1.5 block text-xs font-medium tracking-wider text-white/50 uppercase">Your name</label>
          <input
            className="w-full rounded-xl border border-white/15 bg-white/5 px-4 py-2.5 text-center text-lg outline-none focus:border-teal-400"
            placeholder="Player"
            value={name}
            maxLength={16}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        {view.game && (
          <div className="card mt-8 flex flex-wrap items-center gap-4 border-teal-500/30 bg-teal-500/[0.06]">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="font-semibold">{view.game.finished ? "Last game" : "Game in progress"}</span>
                <Pill tone="teal">{view.game.mode === "read" ? `Read me · ${view.game.target} hands` : "Cash game"}</Pill>
              </div>
              <div className="mt-1 text-sm text-white/60">
                {view.game.hands} hand{view.game.hands === 1 ? "" : "s"} played
                {view.game.target && !view.game.finished ? ` of ${view.game.target}` : ""} · blinds {fmtBlinds(view.money)}
                {me && <> · you {fmtAmt(me.net, view.money, true)}</>}
              </div>
            </div>
            <button className="btn btn-primary px-5" onClick={onPlay}>
              {view.game.finished ? "See results" : "Continue"}
            </button>
            {!view.game.finished && (
              <button className="btn btn-ghost" onClick={() => send({ type: "endGame" })} title="Close it; you can resume it later">
                Save &amp; close
              </button>
            )}
          </div>
        )}

        {unfinished.length > 0 && (
          <section className="mt-6">
            <h3 className="mb-2 text-sm font-semibold text-white/80">Unfinished games</h3>
            <div className="grid gap-2">
              {unfinished.map((g) => (
                <HistoryRow key={g.id} g={g} onClick={onHistory} onResume={() => resume(g.id)} />
              ))}
            </div>
          </section>
        )}

        <section className="mt-8">
          <div className="card">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">Table</h2>
                <p className="text-sm text-white/60">{describeSetup(setup)}</p>
              </div>
              <button className="btn btn-ghost" onClick={() => setCustomizing((x) => !x)}>
                {customizing ? "Done" : "Customize table"}
              </button>
            </div>
            {customizing && (
              <div className="mt-5 border-t border-white/10 pt-5">
                <TableSetup value={setup} onChange={setSetup} />
              </div>
            )}
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div className="card flex flex-col">
              <div className="text-2xl">🃏</div>
              <h2 className="mt-2 text-xl font-semibold">Cash game</h2>
              <p className="mt-1 flex-1 text-sm text-white/65">
                Play as long as you like. Change who sits where between hands; friends on your network can join.
              </p>
              <button className="btn btn-primary mt-4" disabled={!!problem} onClick={() => start("cash")}>
                Start cash game
              </button>
            </div>
            <div className="card flex flex-col">
              <div className="text-2xl">🔍</div>
              <h2 className="mt-2 text-xl font-semibold">Read me</h2>
              <p className="mt-1 flex-1 text-sm text-white/65">
                Play a set number of hands. At the end PokerBot tells you what kind of player it thinks you are, and why.
              </p>
              <div className="mt-4 flex items-center gap-2">
                <div className="flex rounded-lg bg-white/10 p-1">
                  {[50, 100].map((h) => (
                    <button key={h} onClick={() => setHands(h)}
                      className={`rounded-md px-3 py-1 text-sm font-semibold ${hands === h ? "bg-white/90 text-black" : "text-white/70"}`}>
                      {h} hands
                    </button>
                  ))}
                </div>
                <button className="btn btn-primary flex-1" disabled={!!problem} onClick={() => start("read")}>
                  Start
                </button>
              </div>
            </div>
          </div>
          {view.game && !view.game.finished && (
            <p className="mt-3 text-center text-xs text-white/45">
              Starting a new game saves and closes the current one; you can resume it later.
            </p>
          )}
        </section>

        <section className="mt-10">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="font-semibold text-white/85">Recent games</h3>
            <button className="text-sm text-teal-300 hover:underline" onClick={onHistory}>
              All past games →
            </button>
          </div>
          {games === null ? (
            <div className="text-sm text-white/45">Loading…</div>
          ) : recent.length === 0 ? (
            <div className="card text-sm text-white/55">No games yet. Your games are saved here automatically.</div>
          ) : (
            <div className="grid gap-2">
              {recent.map((g) => (
                <HistoryRow key={g.id} g={g} onClick={onHistory}
                  onResume={g.resumable && g.id !== game?.id ? () => resume(g.id) : undefined} />
              ))}
            </div>
          )}
        </section>

        <footer className="mt-12 text-center text-xs leading-relaxed text-white/40">
          PokerBot heads-up uses the evolved exploiter from the project&apos;s results; at a bigger table it uses a demo
          engine built on the same style classifier. Reads use only what any player at the table could see.
        </footer>
      </div>
    </div>
  );
}

export function HistoryRow({ g, onClick, onResume }: { g: HistorySummary; onClick: () => void; onResume?: () => void }) {
  const h = g.humans[0];
  return (
    <div className="card flex w-full flex-wrap items-center gap-x-4 gap-y-1 p-3.5 hover:bg-white/[0.07]">
      <button className="flex flex-1 flex-wrap items-center gap-x-4 gap-y-1 text-left" onClick={onClick}>
        <span className="w-32 text-sm text-white/55">{fmtDate(g.started)}</span>
        <Pill tone={g.mode === "read" ? "teal" : "neutral"}>{g.mode === "read" ? `Read me · ${g.target}` : "Cash game"}</Pill>
        <span className="text-sm text-white/65">
          {g.hands} hands{g.mode === "read" && !g.finished ? " (unfinished)" : ""}
        </span>
        {h && (
          <span className="ml-auto text-sm">
            {h.name}: <b className={h.net >= 0 ? "text-emerald-300" : "text-red-300"}>{fmtAmt(h.net, g.money, true)}</b>
            {h.read && (
              <span className="text-white/55">
                {" "}
                · read as <b className="text-white/85">{h.read}</b>
              </span>
            )}
          </span>
        )}
      </button>
      {onResume && (
        <button className="btn btn-primary py-1.5" onClick={onResume}>
          Resume
        </button>
      )}
    </div>
  );
}
