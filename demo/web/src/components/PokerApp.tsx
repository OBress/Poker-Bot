"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import type { ClientMessage, Legal, SeatKind, TableView } from "@/lib/types";
import { useTable } from "@/lib/useTable";

const Scene = dynamic(() => import("./scene/Scene"), { ssr: false });

const bbFmt = (chips: number, bb: number) => {
  const v = chips / bb;
  return `${Number.isInteger(v) ? v : v.toFixed(1)} bb`;
};

export default function PokerApp() {
  const { view, status, error, send } = useTable();
  const [reveal, setReveal] = useState(false);
  const [setup, setSetup] = useState(false);
  // the hand log starts open only on wide screens, where it clears the action bar
  const [showLog, setShowLog] = useState(() => typeof window !== "undefined" && window.innerWidth >= 1500);
  if (!view) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#120d18] text-white/80">
        <div className="text-center">
          <div className="text-2xl font-semibold">PokerBot table</div>
          <div className="mt-2 text-sm text-white/60">
            {status === "closed" ? "Can't reach the game server on port 8000. Is it running? Retrying…" : "Connecting…"}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[#120d18] text-white">
      <Scene view={view} reveal={reveal} />

      <TopBar view={view} status={status} send={send} reveal={reveal} setReveal={setReveal} openSetup={() => setSetup(true)} toggleLog={() => setShowLog((x) => !x)} />

      {view.you === null && <JoinPanel view={view} send={send} />}
      {view.legal && <ActionBar legal={view.legal} bb={view.bigBlind} send={send} key={`${view.handNo}-${view.street}-${view.legal.toCall}-${view.legal.minRaiseTo}`} />}
      {view.you !== null && !view.legal && view.running && <Waiting view={view} />}
      {showLog && <Log lines={view.log} />}
      {setup && <SetupPanel view={view} send={send} close={() => setSetup(false)} />}
      {error && (
        <div className="absolute z-[60] top-16 left-1/2 -translate-x-1/2 rounded-lg bg-red-600/90 px-4 py-2 text-sm shadow-lg">{error}</div>
      )}
    </div>
  );
}

type Send = (m: ClientMessage) => void;

function TopBar({ view, status, send, reveal, setReveal, openSetup, toggleLog }: {
  view: TableView; status: string; send: Send; reveal: boolean; setReveal: (b: boolean) => void;
  openSetup: () => void; toggleLog: () => void;
}) {
  const me = view.you !== null ? view.seats[view.you] : null;
  return (
    <div className="absolute z-30 inset-x-0 top-0 flex flex-wrap items-center gap-2 bg-gradient-to-b from-black/70 to-transparent px-4 py-3">
      <div className="mr-2">
        <div className="text-lg leading-tight font-bold">
          Poker<span className="text-teal-300">Bot</span> table
        </div>
        <div className="text-xs text-white/60">
          {view.handNo > 0 ? `Hand #${view.handNo}` : "No hands yet"} · blinds 0.5/1 bb · {view.connected} connected
          {status !== "open" && <span className="text-red-300"> · reconnecting</span>}
        </div>
      </div>
      {view.running ? (
        <button className="btn bg-white/10" onClick={() => send({ type: "pause" })}>Pause after hand</button>
      ) : (
        <button className="btn bg-emerald-600 hover:bg-emerald-500" onClick={() => send({ type: "start" })}>Deal</button>
      )}
      <button className="btn bg-white/10" onClick={openSetup}>Seats</button>
      <button className="btn bg-white/10" onClick={toggleLog}>Log</button>
      <label className="flex cursor-pointer items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-sm">
        <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} />
        Reveal bot styles
      </label>
      <div className="ml-auto flex items-center gap-3 text-sm">
        {me && (
          <>
            <span className="text-white/70">
              You: <b className="text-amber-200">{bbFmt(me.stack, view.bigBlind)}</b>{" "}
              <span className={me.net >= 0 ? "text-emerald-300" : "text-red-300"}>
                ({me.net >= 0 ? "+" : ""}{bbFmt(me.net, view.bigBlind)})
              </span>
            </span>
            <button className="btn bg-white/10" onClick={() => send({ type: "leave" })}>Stand up</button>
          </>
        )}
      </div>
    </div>
  );
}

function JoinPanel({ view, send }: { view: TableView; send: Send }) {
  const [name, setName] = useState(() => {
    try {
      return localStorage.getItem("pokerbot-name") ?? "";
    } catch {
      return "";
    }
  });
  const open = view.seats.filter((s) => s.kind === "empty" || (s.kind === "human" && !s.occupied));
  const sit = (seat: number) => {
    try {
      localStorage.setItem("pokerbot-name", name);
    } catch {}
    send({ type: "sit", seat, name: name.trim() || "Player" });
  };
  return (
    <div className="absolute z-30 bottom-6 left-1/2 w-[min(560px,92vw)] -translate-x-1/2 rounded-2xl border border-white/10 bg-black/75 p-4 shadow-2xl backdrop-blur">
      <div className="text-base font-semibold">Take a seat</div>
      <p className="mt-1 text-xs text-white/60">
        You&apos;re watching. Pick an open seat to play; friends on your network can join from their own browser at this
        page&apos;s address.
      </p>
      <input className="mt-3 w-full rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-sm outline-none focus:border-teal-400" placeholder="Your name" value={name} maxLength={16} onChange={(e) => setName(e.target.value)} />
      <div className="mt-3 flex flex-wrap gap-2">
        {open.length === 0 && <span className="text-sm text-white/60">No open seats: change one under Seats.</span>}
        {open.map((s) => (
          <button key={s.index} className="btn bg-teal-600 hover:bg-teal-500" onClick={() => sit(s.index)}>
            Seat {s.index + 1}
          </button>
        ))}
      </div>
    </div>
  );
}

function Waiting({ view }: { view: TableView }) {
  const actor = view.actor !== null ? view.seats[view.actor] : null;
  if (!actor) return null;
  return (
    <div className="pointer-events-none absolute z-30 bottom-6 left-1/2 -translate-x-1/2 rounded-full bg-black/60 px-4 py-2 text-sm text-white/70">
      {actor.name} is thinking…
    </div>
  );
}

function ActionBar({ legal, bb, send }: { legal: Legal; bb: number; send: Send }) {
  const { toCall, canRaise, minRaiseTo, maxRaiseTo, bet, pot } = legal;
  const [amount, setAmount] = useState(minRaiseTo);
  const presets = useMemo(() => {
    // "raise to" sizes: call first, then add a fraction of the pot after the call
    const potAfter = pot + toCall;
    const to = (f: number) => Math.round(Math.min(maxRaiseTo, Math.max(minRaiseTo, bet + toCall + f * potAfter)));
    return [
      { label: "½ pot", v: to(0.5) },
      { label: "¾ pot", v: to(0.75) },
      { label: "Pot", v: to(1) },
      { label: "All-in", v: maxRaiseTo },
    ];
  }, [pot, toCall, bet, minRaiseTo, maxRaiseTo]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement && e.target.type !== "range") return;
      if (e.key === "f") send({ type: "action", action: "fold" });
      if (e.key === "c") send({ type: "action", action: toCall ? "call" : "check" });
      if (e.key === "r" && canRaise) send({ type: "action", action: "raise", amount });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [send, toCall, canRaise, amount]);

  const isBet = toCall === 0;
  return (
    <div className="absolute z-30 bottom-5 left-1/2 w-[min(760px,96vw)] -translate-x-1/2 rounded-2xl border border-sky-400/40 bg-black/80 p-3 shadow-2xl backdrop-blur">
      <div className="mb-2 flex items-center justify-between text-xs text-white/60">
        <span>Your turn · pot {bbFmt(pot, bb)}{toCall > 0 && <> · {bbFmt(toCall, bb)} to call</>}</span>
        <span>keys: F fold · C {isBet ? "check" : "call"} · R {isBet ? "bet" : "raise"}</span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {toCall > 0 && (
          <button className="btn w-28 bg-red-700 hover:bg-red-600" onClick={() => send({ type: "action", action: "fold" })}>Fold</button>
        )}
        <button className="btn w-36 bg-sky-700 hover:bg-sky-600" onClick={() => send({ type: "action", action: toCall ? "call" : "check" })}>
          {toCall ? `Call ${bbFmt(toCall, bb)}` : "Check"}
        </button>
        {canRaise && (
          <>
            <button className="btn w-44 bg-amber-600 hover:bg-amber-500" onClick={() => send({ type: "action", action: "raise", amount })}>
              {amount >= maxRaiseTo ? "All-in" : isBet ? "Bet" : "Raise to"} {bbFmt(amount, bb)}
            </button>
            <input type="range" className="min-w-32 flex-1 accent-amber-400" min={minRaiseTo} max={maxRaiseTo} step={Math.max(1, Math.round(bb / 10))} value={amount} onChange={(e) => setAmount(Number(e.target.value))} />
            <div className="flex gap-1">
              {presets.map((p) => (
                <button key={p.label} className="rounded-md bg-white/10 px-2 py-1 text-xs hover:bg-white/20" onClick={() => setAmount(p.v)}>
                  {p.label}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Log({ lines }: { lines: string[] }) {
  return (
    <div className="absolute z-30 right-3 bottom-4 flex max-h-[32vh] w-72 flex-col-reverse overflow-y-auto rounded-xl border border-white/10 bg-black/55 p-2 text-xs leading-relaxed text-white/75 backdrop-blur md:flex">
      <div>
        {lines.map((l, i) => (
          <div key={i} className={l.startsWith("Hand #") ? "mt-1.5 font-semibold text-white" : l.includes(" wins ") ? "text-yellow-200" : ""}>
            {l}
          </div>
        ))}
      </div>
    </div>
  );
}

const KIND_LABEL: Record<SeatKind, string> = {
  empty: "Empty",
  human: "Open seat (person)",
  pokerbot: "PokerBot (engine)",
  bot: "Calibrated bot",
  tag: "TAG baseline",
};

function SetupPanel({ view, send, close }: { view: TableView; send: Send; close: () => void }) {
  const handLive = view.street !== null;
  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60" onClick={close}>
      <div className="w-[min(640px,94vw)] rounded-2xl border border-white/10 bg-[#1b1424] p-5 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <div className="text-lg font-semibold">Seats</div>
          <button className="btn bg-white/10" onClick={close}>Done</button>
        </div>
        <p className="mt-1 text-xs text-white/60">
          PokerBot plays the evolved heads-up exploiter when only two players are dealt in, and a multiway engine
          (live style classifier + exploit tilts) otherwise. Calibrated bots are drawn from the full-ring styles fitted to
          real players. {handLive && <b className="text-amber-200">Seat changes apply between hands: pause first.</b>}
        </p>
        <div className="mt-3 grid gap-2">
          {view.seats.map((s) => (
            <div key={s.index} className="flex flex-wrap items-center gap-2 rounded-lg bg-white/5 px-3 py-2 text-sm">
              <span className="w-14 text-white/50">Seat {s.index + 1}</span>
              <select className="rounded-md bg-black/40 px-2 py-1" value={s.kind} disabled={handLive} onChange={(e) => send({ type: "configure", seat: s.index, kind: e.target.value as SeatKind })}>
                {(Object.keys(KIND_LABEL) as SeatKind[]).map((k) => (
                  <option key={k} value={k}>{KIND_LABEL[k]}</option>
                ))}
              </select>
              {s.kind === "bot" && (
                <select className="rounded-md bg-black/40 px-2 py-1" disabled={handLive} value={view.styles.indexOf(s.style ?? "")} onChange={(e) => send({ type: "configure", seat: s.index, kind: "bot", style: Number(e.target.value) })}>
                  {view.styles.map((st, i) => (
                    <option key={st} value={i}>{st}</option>
                  ))}
                </select>
              )}
              <span className="ml-auto truncate text-white/70">{s.occupied ? s.name : ""}</span>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-3 text-sm">
          <span className="text-white/60">Bot speed</span>
          <input type="range" min={0} max={2.5} step={0.1} defaultValue={view.botDelay} onChange={(e) => send({ type: "speed", botDelay: Number(e.target.value) })} />
          <span className="text-white/50">{view.botDelay.toFixed(1)} s per action</span>
        </div>
      </div>
    </div>
  );
}
