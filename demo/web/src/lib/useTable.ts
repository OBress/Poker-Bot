"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientMessage, TableView } from "./types";

const PORT = process.env.NEXT_PUBLIC_SERVER_PORT ?? "8000";
const DEV_PORT = "3000";
const WS_GRACE_MS = 2500; // a WebSocket that hasn't opened by then is assumed blocked

/** The game server's base URL. Normally it serves this page too, so it's the page's own
 * origin; under `next dev` on port 3000 it's the game server's port on the same host. */
function serverBase(): string {
  const { protocol, hostname, host, port } = window.location;
  return port === DEV_PORT ? `${protocol}//${hostname}:${PORT}` : `${protocol}//${host}`;
}

function clientId(): string {
  // One player per browser tab: sessionStorage survives reloading the tab (so you keep your
  // seat) but every tab gets its own id, so two tabs never fight over one seat.
  try {
    let id = sessionStorage.getItem("pokerbot-client-id");
    if (!id) {
      id = crypto.randomUUID();
      sessionStorage.setItem("pokerbot-client-id", id);
    }
    return id;
  } catch {
    return crypto.randomUUID();
  }
}

export type Status = "connecting" | "open" | "closed";
export type Transport = "websocket" | "http";

/** Live connection to the demo server: the latest view of the table and a send function.
 *
 * It uses a WebSocket. If that hasn't opened within ``WS_GRACE_MS`` (antivirus web shields,
 * proxies or extensions sometimes stall WebSocket handshakes on localhost), it switches to
 * HTTP long polling for the rest of the page's life. */
export function useTable() {
  const [view, setView] = useState<TableView | null>(null);
  const [status, setStatus] = useState<Status>("connecting");
  const [transport, setTransport] = useState<Transport>("websocket");
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const http = useRef(false);
  const id = useRef("");

  useEffect(() => {
    let stopped = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let everOpened = false;
    id.current = clientId();
    const base = serverBase();

    const show = (v: TableView) => {
      if (v.error) setError(v.error);
      setView(v);
    };

    const poll = async () => {
      http.current = true;
      setTransport("http");
      let since = -1;
      while (!stopped) {
        try {
          const r = await fetch(`${base}/api/poll?cid=${encodeURIComponent(id.current)}&since=${since}`, { cache: "no-store" });
          if (!r.ok) throw new Error(String(r.status));
          const v = (await r.json()) as TableView & { version: number };
          since = v.version;
          setStatus("open");
          show(v);
        } catch {
          setStatus("closed");
          await new Promise((res) => setTimeout(res, 1500));
        }
      }
    };

    const connect = () => {
      setStatus("connecting");
      const socket = new WebSocket(`${base.replace(/^http/, "ws")}/ws`);
      ws.current = socket;
      const grace = setTimeout(() => {
        if (!everOpened && socket.readyState !== WebSocket.OPEN && !stopped) {
          socket.onclose = null;
          socket.close();
          void poll();
        }
      }, WS_GRACE_MS);
      socket.onopen = () => {
        clearTimeout(grace);
        everOpened = true;
        setStatus("open");
        socket.send(JSON.stringify({ clientId: id.current }));
      };
      socket.onmessage = (e) => show(JSON.parse(e.data) as TableView);
      socket.onclose = () => {
        clearTimeout(grace);
        if (stopped || http.current) return;
        setStatus("closed");
        if (!everOpened) void poll(); // refused outright: go straight to the fallback
        else retry = setTimeout(connect, 1500); // the server restarted: reconnect
      };
    };
    // `?http` in the address forces the fallback (for testing, or a browser known to block it)
    if (new URLSearchParams(window.location.search).has("http")) void poll();
    else connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 3500);
    return () => clearTimeout(t);
  }, [error]);

  const send = useCallback((msg: ClientMessage) => {
    if (http.current) {
      void fetch(`${serverBase()}/api/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clientId: id.current, msg }),
      });
    } else if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(msg));
    }
  }, []);

  return { view, status, transport, error, send };
}
