# PokerBot table: 3D demo

Play no-limit hold'em in your browser at an 8-seat 3D table against the project's engine and
calibrated bots. Other people on your network can join too.

- `server/`: a Python game server (FastAPI + WebSockets) that runs the project's own engine
  (`pokerbot.engine.HandState`), bots and classifiers. All game logic and every decision happen
  here.
- `web/`: a Next.js + three.js (react-three-fiber) client. It draws the table and sends your
  actions.

## Run it

From the repository root, install once:

```bash
uv sync --extra demo
```

```bash
npm install --prefix demo/web
```

Then start both servers:

```bash
npm run demo --prefix demo/web
```

Open http://localhost:3000, type a name, take a seat and press **Deal**. If you prefer two
terminals, start the game server and the web app separately:

```bash
uv run --extra demo python demo/server/main.py
```

```bash
npm run dev --prefix demo/web
```

## Who's at the table

Use **Seats** to set each of the 8 seats. Seat changes apply between hands, so pause first.

| Seat | What it is |
|---|---|
| PokerBot (engine) | The project's engine; see below. Its live read of every player shows over their head. |
| Calibrated bot | A rule-based bot drawn from the full-ring styles fitted to real players (`profiles/bots-v3.json`): Loose-passive, Passive limper, Weak-tight, TAG, Nit, Short-stacker. |
| TAG baseline | The fixed tight-aggressive reference bot (`abc_tag`). |
| Open seat (person) | A seat for a human. |
| Empty | Nobody. |

By default you're in seat 1, against two PokerBots and five calibrated bots. **Reveal bot
styles** shows each bot's true style next to PokerBot's read, so you can watch the classifier
work it out.

**How PokerBot plays:**

- **Heads-up (only two players dealt in):** the evolved heads-up exploiter, as evaluated in
  [docs/RESULTS.md](../docs/RESULTS.md). It's the Phase 6 champion over the MCCFR blueprint,
  with the frozen heads-up classifier as its belief source.
- **Multiway:** a demo-only engine, *not* part of the evaluated results. The evolved 6-max
  classifier reads every opponent live, using the full-ring style profiles. A tight-aggressive
  base strategy is then tilted toward the live opponents' weaknesses: more bluffs and steals
  against folders, thinner value bets and fewer bluffs against callers.

PokerBot's memory resets whenever the set of players changes, because both classifiers track
players by seat.

## Playing with other people

The game server listens on all interfaces, IPv4 and IPv6 (port 8000), and `next dev` on port
3000. Friends on your network open `http://<your-computer's-IP>:3000`, pick an open seat and play.
Every browser tab is its own person, and reloading a tab keeps its seat. To play several people
from one machine, just open more tabs.

- **Your turn:** you have 45 seconds; after that you check if you can, otherwise fold. A
  disconnected player checks or folds.
- **Keys:** F fold, C check/call, R bet/raise to the slider amount.
- **Chips:** blinds are 0.5/1 bb, everyone starts with 100 bb (bots with their calibrated
  buy-in), and a busted seat rebuys to 100 bb.

## Notes

- This is a local demo: no accounts, no persistence, one table per server.
- The web app is pinned to Next.js 16.4.0. Its advisories (GHSA-2464-fhr4-w9xw,
  GHSA-p3m5-59mp-jv5r, GHSA-673q-mjm6-r5xr) cover image optimization of SVGs, Server Actions
  and `use cache`, none of which the demo uses. Upgrade to 16.4.3 when your npm
  `min-release-age` allows it (from 2026-11-06):
  `npm i --prefix demo/web next@16.4.3 eslint-config-next@16.4.3`.
- Tests for the server's table logic: `uv run pytest tests/test_demo_table.py`.
