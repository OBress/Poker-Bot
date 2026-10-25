"""Demo game server: one 8-seat table over WebSockets, plus the built web app.

    uv run --extra demo python demo/server/main.py [--port 8000] [--seed 0]

With the web app built (``npm run build --prefix demo/web``, or ``npm run demo`` which also
starts this server) the table is at http://localhost:8000, page and WebSocket on one origin.

There is at most one game at a time. With none, clients see the home screen; ``newGame``
starts one (a cash game, or a read challenge of 50/100 hands) and replaces any game in
progress. Every game is saved to ``demo/data/games/`` after each hand (``history.py``) and
listed at ``/api/history``.

Clients connect to ``/ws`` (or, when something blocks WebSockets, long-poll ``/api/poll`` and
post to ``/api/send``) with a per-tab client id, so a reload keeps the seat. Every change is
pushed to every client as its own view. Bots think in a worker thread with a short delay so
the table feels human; humans get ``TURN_SECONDS`` to act, after which they check or fold, and
a player who stands up while it's their turn folds at once.

Listens on all interfaces, IPv6 and IPv4, so other people on your network can join from their
browser.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the engine's profiles and programs are addressed from the repo root

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from demo.server import history  # noqa: E402
from demo.server.table import KINDS, SEATS, Table, parse_action  # noqa: E402
from pokerbot.engine.actions import Action  # noqa: E402

TURN_SECONDS = 45
WEB_PORT = int(os.environ.get("POKERBOT_WEB_PORT", "3000"))  # where `next dev` runs
WEB_BUILD = ROOT / "demo/web/out"  # the static site from `next build` (npm run demo)
BETWEEN_HANDS = 3.5
POLL_WAIT = 20.0  # a long poll returns after this long without changes
POLL_TIMEOUT = 45.0  # a polling client that hasn't polled for this long has left
READ_HANDS = (50, 100)  # read-challenge lengths
BOT_DELAY = {"cash": 0.9, "read": 0.5}  # seconds a bot "thinks", by mode


class Game:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.games = 0
        self.table: Table | None = None  # None: everyone is on the home screen
        # every open socket per client id; one person may have several tabs or a stale socket
        self.clients: dict[str, set[WebSocket]] = {}
        self.wake = asyncio.Event()
        self.pending: asyncio.Future | None = None
        self.bot_delay = BOT_DELAY["cash"]
        self.task: asyncio.Task | None = None
        self.error: dict[str, str] = {}
        self.table_deadline: float | None = None  # when the human to act runs out of time
        # HTTP long-polling fallback, for browsers where something blocks the WebSocket
        self.version = 0  # bumped on every change; pollers wait for a newer one
        self.changed = asyncio.Event()
        self.polling: dict[str, float] = {}  # client id -> when it last polled

    # ------------------------------------------------------------- clients

    def online(self) -> set[str]:
        """Client ids with an open WebSocket or a recent poll."""
        now = time.time()
        recent = {cid for cid, t in self.polling.items() if now - t < POLL_TIMEOUT}
        return set(self.clients) | recent

    def view_for(self, cid: str) -> dict:
        v = self.table.view(cid) if self.table else {"type": "state", "game": None, "you": None}
        v["deadline"] = self.table_deadline
        v["botDelay"] = self.bot_delay
        v["connected"] = len(self.online())
        v["version"] = self.version
        if cid in self.error:
            v["error"] = self.error.pop(cid)
        return v

    async def broadcast(self) -> None:
        self.version += 1
        changed, self.changed = self.changed, asyncio.Event()
        changed.set()  # wake every waiting long poll
        for cid, sockets in list(self.clients.items()):
            v = self.view_for(cid)
            for ws in list(sockets):
                try:
                    await ws.send_json(v)
                except Exception:  # noqa: BLE001 - a closed socket just stops receiving
                    sockets.discard(ws)

    def fold_if_waiting(self, seat: int | None) -> None:
        """The player to act left: fold for them now instead of waiting out their clock."""
        p = self.pending
        if seat is not None and p and not p.done() and self.table.actor_seat() == seat:
            p.set_result(Action.fold())

    # ------------------------------------------------------------- games

    def save(self) -> None:
        if self.table and self.table.hand_no:
            history.save(self.table.record())

    def stop_loop(self) -> None:
        if self.task:
            self.task.cancel()
            self.task = None
        if self.pending and not self.pending.done():
            self.pending.cancel()
        self.pending, self.table_deadline = None, None

    def new_game(self, cid: str, name: str, mode: str, hands: int | None) -> None:
        if mode not in ("cash", "read"):
            raise ValueError("unknown mode")
        if mode == "read" and hands not in READ_HANDS:
            raise ValueError(f"a read challenge is {' or '.join(map(str, READ_HANDS))} hands")
        self.stop_loop()
        if self.table:
            self.table.ended = self.table.ended or time.time()
            self.save()
        self.games += 1
        self.table = Table(self.seed * 1000 + self.games + int(time.time()), mode, hands)
        self.table.sit(cid, name, 0)
        self.bot_delay = BOT_DELAY[mode]
        self.table.running = mode == "read"  # a challenge deals at once
        self.task = asyncio.create_task(self.run(self.table))

    def end_game(self) -> None:
        self.stop_loop()
        if self.table:
            self.table.running = False
            self.table.ended = self.table.ended or time.time()
            self.save()
        self.table = None

    async def handle(self, cid: str, msg: dict) -> None:
        kind = msg.get("type")
        try:
            if kind == "newGame":
                hands = msg.get("hands")
                self.new_game(
                    cid,
                    str(msg.get("name", "")),
                    str(msg.get("mode", "cash")),
                    int(hands) if hands else None,
                )
            elif kind == "endGame":
                self.end_game()
            elif self.table is None:
                raise ValueError("no game in progress")
            else:
                self.handle_table(cid, kind, msg)
        except (ValueError, KeyError, TypeError) as e:
            self.error[cid] = str(e)
        await self.broadcast()

    def handle_table(self, cid: str, kind: str | None, msg: dict) -> None:
        t = self.table
        if kind == "sit":
            t.sit(cid, str(msg.get("name", "")), int(msg["seat"]))
        elif kind == "leave":
            self.fold_if_waiting(t.leave(cid))
        elif kind == "configure":
            if t.mode != "cash":
                raise ValueError("seats are fixed in a read challenge")
            if self.in_hand():
                raise ValueError("change seats between hands (pause first)")
            seat, what = int(msg["seat"]), str(msg["kind"])
            if what not in KINDS or not 0 <= seat < SEATS:
                raise ValueError("bad seat or kind")
            style = msg.get("style")
            t.configure(seat, what, None if style in (None, "") else int(style))
        elif kind == "start":
            if t.finished:
                raise ValueError("this game is over: start a new one")
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
        else:
            raise ValueError(f"unknown message {kind!r}")

    def in_hand(self) -> bool:
        h = self.table.hand if self.table else None
        return h is not None and not h.state.finished

    # ------------------------------------------------------------- the game loop

    async def run(self, t: Table) -> None:
        """Deal hands at ``t`` while it's running; one loop per game, cancelled on a new one."""
        while self.table is t:
            if not t.running:
                self.wake.clear()
                await self.wake.wait()
                continue
            if not t.start_hand():
                t.running = False
                if not t.finished:
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
                if self.in_hand() and t.actor_seat() == seat:
                    t.apply(action)
                await self.broadcast()
            await asyncio.to_thread(self.save)
            await self.broadcast()
            if not t.finished:
                await asyncio.sleep(BETWEEN_HANDS)

    async def wait_human(self, seat: int) -> Action:
        s = self.table.seats[seat]
        if s.client is None or s.client not in self.online():
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
        yield
        game.stop_loop()
        game.save()

    app = FastAPI(title="Poker-Bot demo", lifespan=lifespan)
    # `next dev` on port 3000 polls this server cross-origin; it's a local demo, so allow all
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "clients": len(game.online()) if game else 0}

    @app.get("/api/history")
    async def games() -> list[dict]:
        return history.summaries()

    @app.get("/api/history/{game_id}")
    async def game_record(game_id: str) -> dict:
        try:
            return history.load(game_id)
        except (OSError, ValueError) as e:
            raise HTTPException(404, "no such game") from e

    @app.get("/api/poll")
    async def poll(cid: str, since: int = -1) -> dict:
        """Long poll: the client's view as soon as it's newer than ``since`` (or after
        ``POLL_WAIT`` seconds). The fallback when a WebSocket can't be opened."""
        cid = cid[:64]
        new = cid not in game.online()
        game.polling[cid] = time.time()
        if new:
            await game.broadcast()  # everyone sees the new connection count
        elif since >= game.version:
            changed = game.changed
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(changed.wait(), POLL_WAIT)
        game.polling[cid] = time.time()
        return game.view_for(cid)

    @app.post("/api/send")
    async def send(body: dict) -> dict:  # a JSON object: {clientId, msg}
        cid = str(body.get("clientId", ""))[:64]
        game.polling[cid] = time.time()
        await game.handle(cid, dict(body.get("msg") or {}))
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            hello = await socket.receive_json()
        except WebSocketDisconnect:
            return  # closed before saying hello (e.g. a dev-mode double mount)
        cid = str(hello.get("clientId") or id(socket))[:64]
        game.clients.setdefault(cid, set()).add(socket)
        await game.broadcast()
        try:
            while True:
                await game.handle(cid, await socket.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            sockets = game.clients.get(cid, set())
            sockets.discard(socket)
            if not sockets:  # this person's last connection is gone
                game.clients.pop(cid, None)
                if game.table and cid not in game.online():
                    game.fold_if_waiting(game.table.seat_of(cid))
            await game.broadcast()

    if WEB_BUILD.is_dir():
        # the table itself, from this same origin, so the page's WebSocket is same-site
        app.mount("/", StaticFiles(directory=WEB_BUILD, html=True), name="web")
    else:

        @app.get("/")
        async def root(request: Request) -> RedirectResponse:
            """Not built yet: send browsers to `next dev`."""
            return RedirectResponse(f"http://{request.url.hostname}:{WEB_PORT}/")

    return app


def listen_socket(port: int) -> socket.socket:
    """One socket for IPv6 and IPv4 on all interfaces. Browsers may resolve ``localhost`` to
    ``::1`` (as ``next dev`` listens there too); an IPv4-only server is then unreachable."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.bind(("::", port))
    return sock


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default="warning")
    args = ap.parse_args()
    server = uvicorn.Server(uvicorn.Config(create_app(args.seed), log_level=args.log_level))
    server.run(sockets=[listen_socket(args.port)])


if __name__ == "__main__":
    main()
