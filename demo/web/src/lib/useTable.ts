"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientMessage, TableView } from "./types";

const PORT = process.env.NEXT_PUBLIC_SERVER_PORT ?? "8000";

function clientId(): string {
  // A stable id per browser, so a reload keeps your seat. `?player=2` gives a tab its own
  // identity, for playing several people from one machine.
  const slot = new URLSearchParams(window.location.search).get("player") ?? "";
  const key = `pokerbot-client-id${slot}`;
  try {
    let id = localStorage.getItem(key);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(key, id);
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
      const socket = new WebSocket(`ws://${window.location.hostname}:${PORT}/ws`);
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
