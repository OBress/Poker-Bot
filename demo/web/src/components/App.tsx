"use client";

import { useState } from "react";
import { useTable } from "@/lib/useTable";
import History from "./History";
import Home, { useName } from "./Home";
import TableScreen from "./TableScreen";
import { Logo } from "./ui";

type Screen = "home" | "table" | "history";

/** Home (new game, continue, recent games) → the table → past games.
 *
 * While a game exists the table stays mounted underneath the other screens, so going back to
 * the menu and continuing is instant and the 3D scene isn't torn down and rebuilt. */
export default function App() {
  const { view, status, transport, error, send } = useTable();
  const [screen, setScreen] = useState<Screen>("home");
  const [name, setName] = useName();

  if (!view) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#120d18] text-white/80">
        <div className="text-center">
          <Logo />
          <div className="mt-3 text-sm text-white/60">
            {status === "closed" ? "Can't reach the game server. Is it running? Retrying…" : "Connecting…"}
          </div>
        </div>
      </div>
    );
  }
  const atTable = screen === "table" && view.game !== null;
  return (
    <>
      {view.game && (
        <div className={atTable ? "" : "invisible"} aria-hidden={!atTable}>
          <TableScreen view={view} send={send} name={name} transport={transport} status={status} error={error}
            active={atTable} onHome={() => setScreen("home")} />
        </div>
      )}
      {!atTable && (
        <div className="fixed inset-0 z-40 overflow-y-auto bg-[#120d18]">
          {screen === "history" ? (
            <History onBack={() => setScreen("home")} currentId={view.game?.id ?? null}
              onResume={(id) => { send({ type: "resume", id, name: name.trim() || "Player" }); setScreen("table"); }} />
          ) : (
            <Home view={view} send={send} name={name} setName={setName} onPlay={() => setScreen("table")}
              onHistory={() => setScreen("history")} />
          )}
        </div>
      )}
    </>
  );
}
