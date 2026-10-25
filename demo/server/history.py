"""Past demo games, one JSON file each under ``demo/data/games/`` (git-ignored).

A game is saved after every hand (so a crash loses at most one hand) and once more when it
ends. ``summaries`` lists them newest first with what the home screen shows.
"""

from __future__ import annotations

import json
import os
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


def load(game_id: str, root: Path | None = None) -> dict:
    root = root or GAMES
    path = root / f"{Path(game_id).name}.json"  # no path tricks from the client
    return json.loads(path.read_text(encoding="utf-8"))


def summaries(root: Path | None = None, limit: int = 50) -> list[dict]:
    root = root or GAMES
    out = []
    for path in sorted(root.glob("*.json"), reverse=True)[:limit]:
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
