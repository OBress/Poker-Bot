"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientMessage, TableView } from "./types";

const PORT = process.env.NEXT_PUBLIC_SERVER_PORT ?? "8000";
const DEV_PORT = "3000";

/** The game server's WebSocket. Normally it serves this page too, so it's the page's own
 * origin (no cross-site request for browser privacy features to block); under `next dev` on
 * port 3000 it's the game server's port on the same host. */
function serverUrl(): string {
  const { protocol, hostname, host, port } = window.location;
  const scheme = protocol === "https:" ? "wss" : "ws";
  return port === DEV_PORT ? `${scheme}://${hostname}:${PORT}/ws` : `${scheme}://${host}/ws`;
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

/** Live connection to the demo server: the latest view of the table and a send function. */
export function useTable() {
  const [view, setView] = useState<TableView | null>(null);
  const [status, setStatus] = useState<Status>("connecting");
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    let stopped = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const id = clientId();

    const connect = () => {
      setStatus("connecting");
      const socket = new WebSocket(serverUrl());
      ws.current = socket;
      socket.onopen = () => {
        setStatus("open");
        socket.send(JSON.stringify({ clientId: id }));
      };
      socket.onmessage = (e) => {
        const v = JSON.parse(e.data) as TableView;
        if (v.error) setError(v.error);
        setView(v);
      };
      socket.onclose = () => {
        setStatus("closed");
        if (!stopped) retry = setTimeout(connect, 1500);
      };
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      ws.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 3500);
    return () => clearTimeout(t);
  }, [error]);

  const send = useCallback((msg: ClientMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify(msg));
  }, []);

  return { view, status, error, send };
}
