"""Play our heads-up players against Slumbot over its public API (https://slumbot.com/#/api).

Slumbot (2018 ACPC winner) deals its own cards, so there is no duplicate pairing here: results are
plain winnings per hand, with all-in runouts replaced by their expectation whenever the opponent's
cards are shown (MATH.md §7.2). Rules: blinds 50/100, 200 bb stacks, reset after every hand,
positions alternate.

Our engine does the bookkeeping. Each time it is our turn, the action string Slumbot sends
("b200c/kk/kb400") is replayed into a fresh ``HandState`` with our hole cards, the board seen so
far and placeholder cards for everything still hidden, which yields the same ``DecisionState`` a
table would give the policy. When the hand ends, the full replay (with Slumbot's cards if it
showed them) goes to the policy as an event stream, and its chip result is checked against the
``winnings`` Slumbot reports. The hero always sits in seat 0.

Only the stdlib is used for HTTP. One connection, one request at a time.
"""

from __future__ import annotations

import http.client
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import cards
from pokerbot.engine.events import HoleCards
from pokerbot.engine.state import DecisionState, HandState
from pokerbot.engine.table import Observer, Policy
from pokerbot.eval.allin import allin_adjusted

HOST = "slumbot.com"
STACK = 20_000  # chips, 200 bb
SMALL_BLIND, BIG_BLIND = 50, 100
HERO = 0
_TOKEN = re.compile(r"[kcf]|b\d+")


class SlumbotError(RuntimeError):
    """The server refused a request (``error_msg``) or the transport kept failing."""


class ProtocolError(RuntimeError):
    """The server's state disagrees with our replay of it."""


# --------------------------------------------------------------------------- transport


class SlumbotClient:
    def __init__(
        self,
        host: str = HOST,
        post: Callable[[str, dict], dict] | None = None,
        retries: int = 5,
    ) -> None:
        """``post(path, body) -> response dict`` replaces the HTTPS transport (tests)."""
        self.host = host
        self.token: str | None = None
        self.retries = retries
        self._post = post or self._https_post
        self._conn: http.client.HTTPSConnection | None = None
        self.requests = 0

    def new_hand(self) -> dict:
        return self._call("/slumbot/api/new_hand", {})

    def act(self, incr: str) -> dict:
        return self._call("/slumbot/api/act", {"incr": incr})

    def _call(self, path: str, body: dict) -> dict:
        if self.token:
            body = {"token": self.token, **body}
        r = self._post(path, body)
        self.requests += 1
        if "error_msg" in r:
            raise SlumbotError(r["error_msg"])
        self.token = r.get("token") or self.token
        return r

    def _https_post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPSConnection(self.host, timeout=30)
                self._conn.request("POST", path, data, {"Content-Type": "application/json"})
                resp = self._conn.getresponse()
                raw = resp.read()
                if resp.status == 200:
                    return json.loads(raw)
                last = SlumbotError(f"HTTP {resp.status}: {raw[:200]!r}")
                try:  # the server's own message is final, not transient
                    return {"error_msg": json.loads(raw)["error_msg"]}
                except (ValueError, KeyError, TypeError):
                    pass
            except (OSError, http.client.HTTPException, ValueError) as e:
                last = e
            self._conn = None
            time.sleep(min(2.0**attempt, 30.0))
        raise SlumbotError(f"{path} failed after {self.retries + 1} tries: {last}")


# --------------------------------------------------------------------------- replay


def tokens(action: str) -> list[str]:
    """``b200c/kk/kb400`` -> ``[b200, c, k, k, k, b400]`` (street separators carry no action)."""
    flat = action.replace("/", "")
    toks = _TOKEN.findall(flat)
    if "".join(toks) != flat:
        raise ProtocolError(f"unparseable action string {action!r}")
    return toks


def _deck(hero: Sequence[int], board: Sequence[int], bot: Sequence[int] | None) -> list[int]:
    """Seat 0 = hero, seat 1 = Slumbot, then the five board cards; unknown cards are placeholders
    that can't clash with the known ones."""
    used = {*hero, *board, *(bot or ())}
    spare = iter(c for c in range(52) if c not in used)
    villain = list(bot) if bot else [next(spare), next(spare)]
    runout = [*board, *(next(spare) for _ in range(5 - len(board)))]
    return [*hero, *villain, *runout]


def replay(
    action: str,
    hero: Sequence[int],
    board: Sequence[int],
    client_pos: int,
    bot: Sequence[int] | None = None,
    hand_id: int = 0,
) -> HandState:
    """The hand so far as our engine sees it. ``client_pos`` 0 = we are the big blind."""
    button = 1 if client_pos == 0 else 0  # the small blind is the button, heads-up
    state = HandState(
        [STACK, STACK], button, _deck(hero, board, bot), SMALL_BLIND, BIG_BLIND, hand_id
    )
    for t in tokens(action):
        if state.finished:
            raise ProtocolError(f"action {t!r} after the hand ended in {action!r}")
        if t == "f":
            state.apply(Action.fold())
        elif t in ("c", "k"):
            state.apply(Action.check_call())
        else:
            state.apply(Action.raise_to(int(t[1:])))
    return state


def to_incr(action: Action, state: DecisionState) -> str:
    """Our action as Slumbot's incremental action."""
    legal = state.legal
    if action.type == ActionType.FOLD and legal.to_call > 0:
        return "f"
    if action.type == ActionType.BET_RAISE and legal.can_raise:
        return f"b{min(max(action.amount, legal.min_raise_to), legal.max_raise_to)}"
    return "c" if legal.to_call > 0 else "k"


# --------------------------------------------------------------------------- hands


@dataclass(frozen=True)
class HandRecord:
    index: int
    client_pos: int
    action: str
    won: int  # chips, as reported by Slumbot
    adjusted: float  # chips, all-in runouts as expectation when the cards are known
    showdown: bool
    checked: bool  # our replay's chips equal Slumbot's winnings
    rejected: int = 0  # actions Slumbot refused and we replaced by check/call


def play_hand(client: SlumbotClient, hero: Policy, index: int) -> HandRecord:
    r = client.new_hand()
    rejected = 0
    while r.get("winnings") is None:
        hole = cards("".join(r["hole_cards"]))
        board = cards("".join(r["board"]))
        state = replay(r["action"], hole, board, r["client_pos"], hand_id=index)
        if state.finished or state.actor != HERO:
            raise ProtocolError(f"not our turn after {r['action']!r} (actor {state.actor})")
        decision = state.decision_state()
        incr = to_incr(hero.act(decision), decision)
        try:
            r = client.act(incr)
        except SlumbotError:
            if incr[0] != "b":
                raise
            rejected += 1  # e.g. a raise below Slumbot's minimum
            r = client.act("c" if decision.legal.to_call > 0 else "k")
    hole = cards("".join(r["hole_cards"]))
    board = cards("".join(r["board"]))
    shown = r.get("bot_hole_cards")
    bot = cards("".join(shown)) if shown else None
    final = replay(r["action"], hole, board, r["client_pos"], bot=bot, hand_id=index)
    if not final.finished:
        raise ProtocolError(f"hand over per server but not in our replay: {r['action']!r}")
    won = int(r["winnings"])
    mine = final.payoffs[HERO]
    adjusted = float(won)
    if final.showdown and bot is not None:
        adj = allin_adjusted(final.events, seed=index)
        if adj is not None:
            adjusted = float(adj[HERO])
    if isinstance(hero, Observer):
        for e in final.events:
            if not (isinstance(e, HoleCards) and e.seat != HERO):
                hero.observe(e)
    return HandRecord(
        index, r["client_pos"], r["action"], won, adjusted, final.showdown, mine == won, rejected
    )


# --------------------------------------------------------------------------- results


def summarize(records: Sequence[HandRecord], block: int = 100, seed: int = 0) -> dict:
    """bb/100 (1 chip per hand = 1 bb/100 here, as 1 bb = 100 chips) with a 95% CI from a
    bootstrap over blocks of ``block`` consecutive hands (a learning hero isn't independent
    from one hand to the next)."""
    rng = np.random.default_rng(seed)

    def estimate(values: np.ndarray) -> dict:
        n = len(values) // block
        blocks = values[: n * block].reshape(n, block).mean(axis=1)
        idx = rng.integers(0, n, size=(4000, n))
        se = float(blocks[idx].mean(axis=1).std(ddof=1)) if n > 1 else float("nan")
        return {"bb100": float(values.mean()), "ci": 1.96 * se, "se": se}

    won = np.array([r.won for r in records], dtype=float)
    adj = np.array([r.adjusted for r in records], dtype=float)
    out = {
        "hands": len(records),
        "raw": estimate(won),
        "adjusted": estimate(adj),
        "by_position": {},
        "mismatched": sum(not r.checked for r in records),
        "rejected": sum(r.rejected for r in records),
        "showdowns": sum(r.showdown for r in records),
    }
    for pos, name in ((0, "big_blind"), (1, "small_blind")):
        sel = np.array([r.client_pos == pos for r in records])
        if sel.sum() > 1:
            out["by_position"][name] = {"hands": int(sel.sum()), "bb100": float(adj[sel].mean())}
    return out


def fallback_rate(hero: object) -> tuple[int, int] | None:
    """(decisions the blueprint answered with its heuristic, decisions) for a blueprint hero;
    None for players that don't count them."""
    misses, lookups = getattr(hero, "misses", None), getattr(hero, "lookups", None)
    return (misses, lookups) if lookups else None


def play_session(
    hero: Policy,
    hands: int,
    client: SlumbotClient | None = None,
    log: Callable[[int, list[HandRecord]], None] | None = None,
    sink: Path | None = None,
    every: int = 500,
) -> list[HandRecord]:
    client = client or SlumbotClient()
    records: list[HandRecord] = []
    out = sink.open("a", encoding="utf-8") if sink else None
    try:
        for i in range(hands):
            rec = play_hand(client, hero, i)
            records.append(rec)
            if out:
                out.write(json.dumps(asdict(rec)) + "\n")
                out.flush()
            if log and (i + 1) % every == 0:
                log(i + 1, records)
    finally:
        if out:
            out.close()
    return records


def run_slumbot(hero_spec: str, hands: int, seed: int = 0, every: int = 500) -> dict:
    """A tracked session of ``hands`` hands against Slumbot (kind ``eval``)."""
    from pokerbot.eval.players import make_policy
    from pokerbot.track import Run

    config = dict(
        hero=hero_spec, hands=hands, seed=seed, host=HOST, stack_bb=STACK // BIG_BLIND,
        blinds=[SMALL_BLIND, BIG_BLIND], adjusted="all-in runouts as expectation (MATH.md 7.2)",
    )  # fmt: skip
    name = f"slumbot-{hero_spec.split(':')[0]}"
    with Run.start(name, kind="eval", config=config, seeds={"hero": seed}) as run:
        hero = make_policy(hero_spec, seed)
        sink = run.dir / "artifacts" / "hands.jsonl"

        def log(n: int, records: list[HandRecord]) -> None:
            s = summarize(records)
            run.log_metrics(
                {
                    "slumbot/adjusted_bb100": s["adjusted"]["bb100"],
                    "slumbot/adjusted_ci": s["adjusted"]["ci"],
                    "slumbot/raw_bb100": s["raw"]["bb100"],
                },
                step=n,
            )
            print(
                f"{n:6d} hands  {s['adjusted']['bb100']:+8.1f} +/- {s['adjusted']['ci']:.1f} "
                f"bb/100 (raw {s['raw']['bb100']:+.1f})", flush=True,
            )  # fmt: skip

        records = play_session(hero, hands, SlumbotClient(), log, sink, every)
        s = summarize(records, seed=seed)
        s["fallback"] = fallback_rate(hero)
        s["hero"] = hero_spec
        run.log_metrics(
            {
                "slumbot/hands": s["hands"],
                "slumbot/adjusted_bb100": s["adjusted"]["bb100"],
                "slumbot/adjusted_ci": s["adjusted"]["ci"],
                "slumbot/raw_bb100": s["raw"]["bb100"],
                "slumbot/raw_ci": s["raw"]["ci"],
                "slumbot/mismatched": s["mismatched"],
                "slumbot/rejected": s["rejected"],
            }
        )
        run.finish(**s)
        s["run_id"] = run.dir.name
    return s
