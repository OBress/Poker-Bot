"""Demo game server: one 8-seat table over WebSockets.

    uv run --extra demo python demo/server/main.py [--port 8000] [--seed 0]

Clients (the Next.js app in ``demo/web``) connect to ``/ws`` and say hello with a stable client
id, so a reload keeps the seat. Every change is pushed to every client as its own view. Bots
think in a worker thread with a short delay so the table feels human; humans get
``TURN_SECONDS`` to act, after which they check or fold.

Listens on all interfaces so other people on your network can join from their browser.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the engine's profiles and programs are addressed from the repo root

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402

from demo.server.table import KINDS, SEATS, Table, parse_action  # noqa: E402
from pokerbot.engine.actions import Action  # noqa: E402

TURN_SECONDS = 45
WEB_PORT = int(os.environ.get("POKERBOT_WEB_PORT", "3000"))  # where the Next.js app runs
BETWEEN_HANDS = 4.0


class Game:
    def __init__(self, seed: int) -> None:
        self.table = Table(seed)
        self.clients: dict[str, WebSocket] = {}
        self.wake = asyncio.Event()
        self.pending: asyncio.Future | None = None
        self.bot_delay = 0.9  # seconds a bot "thinks"
        self.task: asyncio.Task | None = None
        self.error: dict[str, str] = {}
        self.table_deadline: float | None = None  # when the human to act runs out of time

    # ------------------------------------------------------------- clients

    async def broadcast(self) -> None:
        dead = []
        for cid, ws in list(self.clients.items()):
            v = self.table.view(cid)
            v["deadline"] = self.table_deadline
            v["botDelay"] = self.bot_delay
            v["connected"] = len(self.clients)
            if cid in self.error:
                v["error"] = self.error.pop(cid)
            try:
                await ws.send_json(v)
            except Exception:  # noqa: BLE001 - a closed socket is cleaned up below
                dead.append(cid)
        for cid in dead:
            self.clients.pop(cid, None)

    async def handle(self, cid: str, msg: dict) -> None:
        t = self.table
        kind = msg.get("type")
        try:
            if kind == "sit":
                t.sit(cid, str(msg.get("name", "")), int(msg["seat"]))
            elif kind == "leave":
                t.leave(cid)
            elif kind == "configure":
                if self.in_hand():
                    raise ValueError("change seats between hands (pause first)")
                seat, what = int(msg["seat"]), str(msg["kind"])
                if what not in KINDS or not 0 <= seat < SEATS:
                    raise ValueError("bad seat or kind")
                style = msg.get("style")
                t.configure(seat, what, None if style in (None, "") else int(style))
            elif kind == "start":
                t.running = True
                self.wake.set()
            elif kind == "pause":
                t.running = False
            elif kind == "speed":
                self.bot_delay = min(3.0, max(0.0, float(msg["botDelay"])))
            elif kind == "action":
                if t.actor_seat() is None or t.actor_seat() != t.seat_of(cid):
                    raise ValueError("not your turn")
                if self.pending and not self.pending.done():
                    self.pending.set_result(parse_action(msg))
        except (ValueError, KeyError, TypeError) as e:
            self.error[cid] = str(e)
        await self.broadcast()

    def in_hand(self) -> bool:
        h = self.table.hand
        return h is not None and not h.state.finished

    # ------------------------------------------------------------- the game loop

    async def run(self) -> None:
        t = self.table
        while True:
            if not t.running:
                self.wake.clear()
                await self.wake.wait()
                continue
            if not t.start_hand():
                t.running = False
                t.log.append("Need at least two players to deal")
                await self.broadcast()
                continue
            await self.broadcast()
            while self.in_hand():
                seat = t.actor_seat()
                s = t.seats[seat]
                if s.kind == "human":
                    action = await self.wait_human(seat)
                else:
                    await asyncio.sleep(self.bot_delay * (0.6 + 0.8 * t.rng.random()))
                    action = await asyncio.to_thread(t.bot_action)
                t.apply(action)
                await self.broadcast()
            await asyncio.sleep(BETWEEN_HANDS)

    async def wait_human(self, seat: int) -> Action:
        s = self.table.seats[seat]
        if s.client is None or s.client not in self.clients:
            await asyncio.sleep(0.4)
            return Action.fold()  # an empty or disconnected seat checks or folds
        self.pending = asyncio.get_running_loop().create_future()
        self.table_deadline = time.time() + TURN_SECONDS
        await self.broadcast()
        try:
            return await asyncio.wait_for(self.pending, TURN_SECONDS)
        except TimeoutError:
            return Action.fold()  # the engine turns a free fold into a check
        finally:
            self.pending = None
            self.table_deadline = None


def create_app(seed: int = 0) -> FastAPI:
    game: Game | None = None

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal game
        game = Game(seed)
        game.task = asyncio.create_task(game.run())
        yield
        game.task.cancel()

    app = FastAPI(title="Poker-Bot demo", lifespan=lifespan)

    @app.get("/")
    async def root(request: Request) -> RedirectResponse:
        """This server only speaks WebSocket; send browsers to the table on the web port."""
        return RedirectResponse(f"http://{request.url.hostname}:{WEB_PORT}/")

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "clients": len(game.clients) if game else 0}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            hello = await socket.receive_json()
        except WebSocketDisconnect:
            return  # closed before saying hello (e.g. a dev-mode double mount)
        cid = str(hello.get("clientId") or id(socket))[:64]
        old = game.clients.get(cid)
        if old is not None:
            with contextlib.suppress(Exception):
                await old.close()
        game.clients[cid] = socket
        await game.broadcast()
        try:
            while True:
                await game.handle(cid, await socket.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            if game.clients.get(cid) is socket:
                del game.clients[cid]
                if game.pending and game.table.actor_seat() == game.table.seat_of(cid):
                    game.pending.set_result(Action.fold())
            await game.broadcast()

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    uvicorn.run(create_app(args.seed), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
