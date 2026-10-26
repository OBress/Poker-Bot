"""Past demo games under ``demo/data/games/`` (git-ignored).

Each game has a JSON record (``<id>.json``: settings, players, reads, log) for the history
screens, and, while it is unfinished, a pickled ``Table`` (``<id>.table.pkl``) to resume it.
Both are written after every hand, so a crash loses at most one hand. ``summaries`` lists
games newest first with what the home screen shows. The pickles are this server's own files,
written by ``save_table`` and only read back from this folder by game id.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GAMES = Path(os.environ.get("POKERBOT_DEMO_GAMES", ROOT / "demo/data/games"))


def save(record: dict, root: Path | None = None) -> Path:
    root = root or GAMES
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(path)  # atomic: a half-written file never replaces a good one
    return path


def _table_path(game_id: str, root: Path) -> Path:
    return root / f"{Path(game_id).name}.table.pkl"  # no path tricks from the client


def save_table(table, root: Path | None = None) -> None:
    """The live game, to resume later; a finished game's state is removed instead."""
    root = root or GAMES
    path = _table_path(table.id, root)
    if table.finished:
        path.unlink(missing_ok=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(table, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)


def load_table(game_id: str, root: Path | None = None):
    with _table_path(game_id, root or GAMES).open("rb") as f:
        return pickle.load(f)  # noqa: S301 - our own saved game, from our own folder


def resumable(game_id: str, root: Path | None = None) -> bool:
    return _table_path(game_id, root or GAMES).exists()


def load(game_id: str, root: Path | None = None) -> dict:
    root = root or GAMES
    path = root / f"{Path(game_id).name}.json"  # no path tricks from the client
    return json.loads(path.read_text(encoding="utf-8"))


def summaries(root: Path | None = None, limit: int = 50) -> list[dict]:
    root = root or GAMES
    out = []
    for path in sorted(root.glob("*.json"), reverse=True)[:limit]:
        resume = resumable(path.stem, root)
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        humans = [p for p in r["players"] if p["kind"] == "human"]
        out.append(
            {
                "id": r["id"],
                "mode": r["mode"],
                "target": r["target"],
                "hands": r["hands"],
                "finished": r["finished"],
                "started": r["started"],
                "ended": r["ended"],
                "bigBlind": r["bigBlind"],
                # money units (older records: amounts in big blinds)
                "money": r.get("money", {"sb": 0.5, "bb": 1.0}),
                "resumable": resume,
                "humans": [
                    {
                        "name": p["name"],
                        "net": p["net"],
                        "read": (p["read"] or {}).get("style"),
                        "confidence": (p["read"] or {}).get("confidence"),
                    }
                    for p in humans
                ],
                "pokerbots": [
                    {"name": p["name"], "net": p["net"]}
                    for p in r["players"]
                    if p["kind"] == "pokerbot"
                ],
            }
        )
    return out
