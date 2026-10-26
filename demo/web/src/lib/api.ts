import type { GameRecord, HistorySummary, Money } from "./types";
import { serverBase } from "./useTable";

/** Past games saved by the server (demo/server/history.py). */
export async function fetchHistory(): Promise<HistorySummary[]> {
  const r = await fetch(`${serverBase()}/api/history`, { cache: "no-store" });
  if (!r.ok) throw new Error(`history: ${r.status}`);
  return r.json();
}

export async function fetchGame(id: string): Promise<GameRecord> {
  const r = await fetch(`${serverBase()}/api/history/${encodeURIComponent(id)}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`game ${id}: ${r.status}`);
  return r.json();
}

/** Engine chips (the engine's big blind is always 100) in the game's money units. */
export const fmtAmt = (chips: number, money: Money, sign = false) => {
  const v = (chips / ENGINE_BB) * money.bb;
  const s = v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return `${sign && v > 0 ? "+" : ""}${s}`;
};

export const ENGINE_BB = 100;
export const fmtBlinds = (money: Money) => `${fmtNum(money.sb)}/${fmtNum(money.bb)}`;
export const fmtNum = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 2 });

export const fmtDate = (t: number) =>
  new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
