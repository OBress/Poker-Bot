"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { fmtAmt, fmtBlinds } from "@/lib/api";
import type { ClientMessage, Legal, Money, ReadDetail, SeatKind, TableView } from "@/lib/types";
import ReadReport from "./ReadReport";
import { Logo, Modal, Pill } from "./ui";

const Scene = dynamic(() => import("./scene/Scene"), { ssr: false });

type Send = (m: ClientMessage) => void;

/** A setting kept in this browser (localStorage), e.g. whether reads show on the table. */
function useSetting(key: string, initial: boolean): [boolean, (v: boolean) => void] {
  const [v, setV] = useState(() => {
    try {
      const s = localStorage.getItem(`pokerbot-${key}`);
      return s === null ? initial : s === "1";
    } catch {
      return initial;
    }
  });
  const set = (x: boolean) => {
    setV(x);
    try {
      localStorage.setItem(`pokerbot-${key}`, x ? "1" : "0");
    } catch {}
  };
  return [v, set];
}

export default function TableScreen({ view, send, name, transport, status, error, active, onHome }: {
  view: TableView; send: Send; name: string; transport: string; status: string; error: string | null;
  active: boolean; // false while another screen covers the table (keys and dialogs stay off)
  onHome: () => void;
}) {
  const [showReads, setShowReads] = useSetting("show-reads", false);
  const [reveal, setReveal] = useSetting("reveal-styles", false);
  const [showLog, setShowLog] = useSetting("show-log", false);
  const [panel, setPanel] = useState<null | "read" | "seats" | "menu">(null);
  const [resultsClosed, setResultsClosed] = useState<string | null>(null);
  const game = view.game;
  const me = view.you !== null ? view.seats[view.you] : null;
  const showResults = active && game.finished && game.mode === "read" && resultsClosed !== game.id;

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[#120d18] text-white">
      <Scene view={view} reveal={reveal} showReads={showReads} />

      {/* top bar */}
      <div className="absolute inset-x-0 top-0 z-30 flex flex-wrap items-center gap-2 bg-gradient-to-b from-black/80 via-black/50 to-transparent px-3 pt-3 pb-6 sm:px-4">
        <button className="btn btn-ghost" onClick={onHome} title="Home: new game, past games">
          ← Menu
        </button>
        <Logo small />
        <Pill tone={game.mode === "read" ? "teal" : "neutral"}>{game.mode === "read" ? "Read me" : "Cash game"}</Pill>
        <HandCounter view={view} />
        <span className="text-sm text-white/45">blinds {fmtBlinds(view.money)}</span>
        {transport === "http" && <span className="text-xs text-white/40" title="WebSocket blocked in this browser; using HTTP">via HTTP</span>}
        {status !== "open" && <span className="text-xs text-red-300">reconnecting…</span>}

        <div className="ml-auto flex flex-wrap items-center gap-2">
          {me && (
            <button className="btn bg-teal-600/90 hover:bg-teal-500" onClick={() => setPanel("read")} disabled={!view.myRead}
              title={view.myRead ? "See how PokerBot classifies you" : "Available after your first hand"}>
              🔍 What does PokerBot think of me?
            </button>
          )}
          <Toggle on={showReads} set={setShowReads} label="Show reads" />
          {!game.finished &&
            (view.running ? (
              <button className="btn btn-ghost" onClick={() => send({ type: "pause" })}>Pause</button>
            ) : (
              <button className="btn btn-primary" onClick={() => send({ type: "start" })}>Deal</button>
            ))}
          <button className="btn btn-ghost" onClick={() => setPanel(panel === "menu" ? null : "menu")} aria-label="More">
            ⋯
          </button>
        </div>
        {panel === "menu" && (
          <div className="absolute top-14 right-3 w-64 rounded-xl border border-white/10 bg-[#1b1424] p-2 shadow-2xl">
            <MenuItem onClick={() => setShowLog(!showLog)}>{showLog ? "Hide" : "Show"} hand log</MenuItem>
            <MenuItem onClick={() => setReveal(!reveal)}>{reveal ? "Hide" : "Reveal"} bots&apos; true styles</MenuItem>
            {game.mode === "cash" && <MenuItem onClick={() => setPanel("seats")}>Seats and bot speed…</MenuItem>}
            {me ? (
              <MenuItem onClick={() => { send({ type: "leave" }); setPanel(null); }}>Stand up (watch)</MenuItem>
            ) : null}
            <MenuItem onClick={() => { send({ type: "endGame" }); onHome(); }}>
              Save &amp; close (resume later)
            </MenuItem>
          </div>
        )}
      </div>

      {/* your status, bottom left */}
      {me && (
        <div className="absolute bottom-4 left-4 z-30 rounded-xl border border-white/10 bg-black/60 px-3.5 py-2 text-sm backdrop-blur">
          <div className="font-semibold">{me.name}</div>
          <div className="text-amber-200">{fmtAmt(me.stack, view.money)}</div>
          <div className={`text-xs ${me.net >= 0 ? "text-emerald-300" : "text-red-300"}`}>{fmtAmt(me.net, view.money, true)} this game</div>
        </div>
      )}

      {view.you === null && !game.finished && <JoinPanel view={view} send={send} name={name} />}
      {active && view.legal && (
        <ActionBar legal={view.legal} money={view.money} send={send}
          key={`${view.handNo}-${view.street}-${view.legal.toCall}-${view.legal.minRaiseTo}`} />
      )}
      {me && !view.legal && <StatusLine view={view} send={send} />}
      {showLog && <Log lines={view.log} />}

      {active && panel === "read" && view.myRead && (
        <Modal title="What PokerBot thinks of you" onClose={() => setPanel(null)} wide>
          <Report detail={view.myRead} />
        </Modal>
      )}
      {active && panel === "seats" && <SetupPanel view={view} send={send} close={() => setPanel(null)} />}
      {showResults && (
        <Results view={view} send={send} name={name} onHome={onHome} close={() => setResultsClosed(game.id)} />
      )}
      {error && (
        <div className="absolute top-20 left-1/2 z-[60] -translate-x-1/2 rounded-lg bg-red-600/90 px-4 py-2 text-sm shadow-lg">{error}</div>
      )}
    </div>
  );
}

function Report({ detail, name }: { detail: ReadDetail; name?: string }) {
  return <ReadReport read={detail} about={detail.about} stats={detail.stats} trajectory={detail.trajectory} name={name} />;
}

function HandCounter({ view }: { view: TableView }) {
  const g = view.game;
  if (g.mode !== "read" || !g.target) {
    return <span className="text-sm text-white/55">Hand {Math.max(1, view.handNo)}</span>;
  }
  const done = Math.min(g.hands, g.target);
  return (
    <div className="flex items-center gap-2 text-sm text-white/70">
      <span className="tabular-nums">
        {done} / {g.target} hands
      </span>
      <div className="h-1.5 w-28 overflow-hidden rounded-full bg-white/15">
        <div className="h-full rounded-full bg-teal-400 transition-[width]" style={{ width: `${(100 * done) / g.target}%` }} />
      </div>
    </div>
  );
}

function Toggle({ on, set, label }: { on: boolean; set: (v: boolean) => void; label: string }) {
  return (
    <button className="btn btn-ghost" onClick={() => set(!on)} aria-pressed={on}>
      <span className={`relative inline-block h-4 w-7 rounded-full transition-colors ${on ? "bg-teal-500" : "bg-white/25"}`}>
        <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${on ? "left-3.5" : "left-0.5"}`} />
      </span>
      {label}
    </button>
  );
}

function MenuItem({ children, onClick, danger = false }: { children: React.ReactNode; onClick: () => void; danger?: boolean }) {
  return (
    <button className={`block w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-white/10 ${danger ? "text-red-300" : "text-white/85"}`} onClick={onClick}>
      {children}
    </button>
  );
}

function StatusLine({ view, send }: { view: TableView; send: Send }) {
  const actor = view.actor !== null ? view.seats[view.actor] : null;
  let text: string | null = null;
  if (view.game.finished) text = "Game over";
  else if (!view.running && !view.street) text = null;
  else if (actor) text = `${actor.name} is thinking…`;
  else if (view.results) text = "Next hand coming up…";
  if (!view.running && !view.street && !view.game.finished) {
    return (
      <div className="absolute bottom-8 left-1/2 z-30 -translate-x-1/2">
        <button className="btn btn-primary px-8 py-3 text-base shadow-xl" onClick={() => send({ type: "start" })}>
          Deal {view.handNo ? "next hand" : "the first hand"}
        </button>
      </div>
    );
  }
  if (!text) return null;
  return (
    <div className="pointer-events-none absolute bottom-8 left-1/2 z-30 -translate-x-1/2 rounded-full bg-black/60 px-4 py-2 text-sm text-white/70">
      {text}
    </div>
  );
}

function JoinPanel({ view, send, name }: { view: TableView; send: Send; name: string }) {
  const open = view.seats.filter((s) => s.kind === "empty" || (s.kind === "human" && !s.occupied));
  return (
    <div className="absolute bottom-6 left-1/2 z-30 w-[min(520px,92vw)] -translate-x-1/2 rounded-2xl border border-white/10 bg-black/80 p-4 shadow-2xl backdrop-blur">
      <div className="font-semibold">You&apos;re watching</div>
      <p className="mt-1 text-sm text-white/60">
        Take an open seat to play as <b className="text-white/85">{name.trim() || "Player"}</b>.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        {open.length === 0 && (
          <span className="text-sm text-white/55">
            No open seats{view.game.mode === "cash" ? ": free one under ⋯ → Seats." : "."}
          </span>
        )}
        {open.map((s) => (
          <button key={s.index} className="btn btn-primary" onClick={() => send({ type: "sit", seat: s.index, name: name.trim() || "Player" })}>
            Sit in seat {s.index + 1}
          </button>
        ))}
      </div>
    </div>
  );
}

function ActionBar({ legal, money, send }: { legal: Legal; money: Money; send: Send }) {
  const bb = 100; // engine chips per big blind
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
      if (e.key === "f" && toCall > 0) send({ type: "action", action: "fold" });
      if (e.key === "c") send({ type: "action", action: toCall ? "call" : "check" });
      if (e.key === "r" && canRaise) send({ type: "action", action: "raise", amount });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [send, toCall, canRaise, amount]);

  const isBet = toCall === 0;
  return (
    <div className="absolute bottom-4 left-1/2 z-30 w-[min(720px,94vw)] -translate-x-1/2 rounded-2xl border border-sky-400/40 bg-black/85 p-3 shadow-2xl backdrop-blur">
      <div className="mb-2.5 flex items-center justify-between text-xs text-white/60">
        <span className="font-semibold text-sky-200">Your turn</span>
        <span>
          pot {fmtAmt(pot, money)}
          {toCall > 0 && <> · {fmtAmt(toCall, money)} to call</>}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <button className="btn btn-danger py-3 text-base" disabled={toCall === 0} onClick={() => send({ type: "action", action: "fold" })}>
          Fold <kbd className="text-xs opacity-60">F</kbd>
        </button>
        <button className="btn bg-sky-700 py-3 text-base hover:bg-sky-600" onClick={() => send({ type: "action", action: toCall ? "call" : "check" })}>
          {toCall ? `Call ${fmtAmt(toCall, money)}` : "Check"} <kbd className="text-xs opacity-60">C</kbd>
        </button>
        <button className="btn bg-amber-600 py-3 text-base hover:bg-amber-500" disabled={!canRaise} onClick={() => send({ type: "action", action: "raise", amount })}>
          {amount >= maxRaiseTo ? "All-in" : isBet ? "Bet" : "Raise to"} {fmtAmt(amount, money)} <kbd className="text-xs opacity-60">R</kbd>
        </button>
      </div>
      {canRaise && (
        <div className="mt-2.5 flex items-center gap-3">
          <input type="range" className="flex-1 accent-amber-400" min={minRaiseTo} max={maxRaiseTo}
            step={Math.max(1, Math.round(bb / 10))} value={amount} onChange={(e) => setAmount(Number(e.target.value))} />
          <div className="flex gap-1">
            {presets.map((p) => (
              <button key={p.label} className="rounded-md bg-white/10 px-2.5 py-1 text-xs hover:bg-white/20" onClick={() => setAmount(p.v)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Log({ lines }: { lines: string[] }) {
  return (
    <div className="absolute top-20 right-3 z-20 flex max-h-[40vh] w-72 flex-col-reverse overflow-y-auto rounded-xl border border-white/10 bg-black/65 p-2.5 text-xs leading-relaxed text-white/75 backdrop-blur">
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
    <Modal title="Seats" onClose={close} footer={<button className="btn btn-primary" onClick={close}>Done</button>}>
      <p className="text-sm text-white/60">
        PokerBot plays the evolved heads-up exploiter when only two players are dealt in, and a multiway engine otherwise.
        Calibrated bots imitate styles fitted to real players.
      </p>
      {handLive && <p className="mt-2 text-sm text-amber-200">Seat changes apply between hands: pause first.</p>}
      <div className="mt-4 grid gap-2">
        {view.seats.map((s) => (
          <div key={s.index} className="flex flex-wrap items-center gap-2 rounded-lg bg-white/5 px-3 py-2 text-sm">
            <span className="w-14 text-white/50">Seat {s.index + 1}</span>
            <select className="rounded-md bg-black/40 px-2 py-1" value={s.kind} disabled={handLive}
              onChange={(e) => send({ type: "configure", seat: s.index, kind: e.target.value as SeatKind })}>
              {(Object.keys(KIND_LABEL) as SeatKind[]).map((k) => (
                <option key={k} value={k}>{KIND_LABEL[k]}</option>
              ))}
            </select>
            {s.kind === "bot" && (
              <select className="rounded-md bg-black/40 px-2 py-1" disabled={handLive} value={view.styles.indexOf(s.style ?? "")}
                onChange={(e) => send({ type: "configure", seat: s.index, kind: "bot", style: Number(e.target.value) })}>
                {view.styles.map((st, i) => (
                  <option key={st} value={i}>{st}</option>
                ))}
              </select>
            )}
            <span className="ml-auto truncate text-white/70">{s.occupied ? s.name : ""}</span>
          </div>
        ))}
      </div>
      <div className="mt-5 flex items-center gap-3 text-sm">
        <span className="text-white/60">Bot speed</span>
        <input type="range" min={0} max={2.5} step={0.1} defaultValue={view.botDelay}
          onChange={(e) => send({ type: "speed", botDelay: Number(e.target.value) })} />
        <span className="text-white/50">{view.botDelay.toFixed(1)} s per action</span>
      </div>
    </Modal>
  );
}

function Results({ view, send, name, onHome, close }: {
  view: TableView; send: Send; name: string; onHome: () => void; close: () => void;
}) {
  const reads = Object.entries(view.finalReads ?? {}).filter(([, d]) => d) as [string, ReadDetail][];
  const mine = view.you !== null ? String(view.you) : null;
  const [tab, setTab] = useState(mine && reads.some(([s]) => s === mine) ? mine : reads[0]?.[0] ?? null);
  const current = reads.find(([s]) => s === tab);
  return (
    <Modal
      wide
      title={`Read challenge complete: ${view.game.target} hands`}
      onClose={close}
      footer={
        <>
          <button className="btn btn-ghost" onClick={onHome}>Home</button>
          <button className="btn btn-ghost" onClick={close}>Look at the table</button>
          <button className="btn btn-primary" onClick={() => send({ type: "newGame", mode: "read", hands: view.game.target ?? 50, name: name.trim() || "Player" })}>
            Play again
          </button>
        </>
      }
    >
      {reads.length > 1 && (
        <div className="mb-5 flex flex-wrap gap-2">
          {reads.map(([s]) => (
            <button key={s} className={`btn ${tab === s ? "btn-primary" : "btn-ghost"}`} onClick={() => setTab(s)}>
              {view.seats[Number(s)].name}
              {s === mine ? " (you)" : ""}
            </button>
          ))}
        </div>
      )}
      {current ? (
        <Report detail={current[1]} name={current[0] === mine ? undefined : view.seats[Number(current[0])].name} />
      ) : (
        <p className="text-sm text-white/60">No player was read in this game.</p>
      )}
    </Modal>
  );
}
