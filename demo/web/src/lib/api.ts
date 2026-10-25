import type { GameRecord, HistorySummary } from "./types";
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

export const fmtBB = (chips: number, bb: number, sign = false) => {
  const v = chips / bb;
  const s = Number.isInteger(v) ? String(v) : v.toFixed(1);
  return `${sign && v > 0 ? "+" : ""}${s} bb`;
};

export const fmtDate = (t: number) =>
  new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
