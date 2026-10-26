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

Then build the web app and start the game server, which serves it:

```bash
npm run demo --prefix demo/web
```

Open http://localhost:8000, type a name, take a seat and press **Deal**. The page and its live
connection come from the same address, so browser privacy features and extensions have nothing
cross-site to block. Once built, `uv run --extra demo python demo/server/main.py` alone starts it
again.

If something on your machine stalls WebSocket connections (antivirus web shields, proxies and
some extensions do), the page switches to HTTP polling after 2.5 s and shows "via HTTP" in the
top bar. Add `?http` to the address to force it.

**Developing the client:** `npm run dev:all --prefix demo/web` runs the game server and `next dev`
(hot reload) together; open http://localhost:3000. There the page talks to the game server on
port 8000, a cross-site connection that some browser extensions block. Use a clean browser
profile if it can't connect.

## Modes

The home screen starts a new game, continues or resumes one, or opens your past games.
Starting a new game saves and closes the current one.

**Customize table** (home screen) sets up the next game, in either mode:

- **Players:** 2–8, with you in seat 1. At 2, PokerBot plays the evolved heads-up exploiter.
- **Opponents:** Mixed (PokerBots and calibrated bots), All PokerBot (every opponent is the
  engine), All calibrated bots, or Custom (pick each seat: PokerBot, calibrated bot with a
  chosen or random style, TAG baseline, or an open seat for a friend).
- **Blinds and buy-in:** any amounts, e.g. 5/10 with a 1,000 buy-in (10–1,000 big blinds).
  Everything is shown in those units. The engine itself always plays a 100-chip big blind
  (the heads-up exploiter assumes it), so only the ratios matter to the bots: stack depth in
  big blinds and the small blind's share of the big.

**Resuming:** an unfinished game (a cash game, or a read challenge before its last hand) is
saved after every hand, together with PokerBot's memory. **Save & close** (home screen or ⋯)
closes it, and **Resume** (home screen or past games) reopens it: you take your seat back, and
any other people's seats open up for them to rejoin.

- **Cash game:** play as long as you like at the 8-seat table: you in seat 1, two PokerBots
  and five calibrated bots. Change seats between hands under **⋯ → Seats**.
- **Read me:** play 50 or 100 hands at a fixed table. At the end, PokerBot shows what kind of
  player it thinks you are: each style's probability, how the read moved hand by hand, and your
  stats (VPIP, preflop raises, 3-bets, folds to flop bets and so on) next to each style's
  typical numbers.

In either mode, **What does PokerBot think of me?** in the top bar shows the read on you so far.
**Show reads** puts PokerBot's live read over every player's head; it's off by default.
**⋯ → Reveal bots' true styles** shows what each bot really is, to check the reads against.

Every game is saved to `demo/data/games/` after each hand (git-ignored): a JSON record, and
for an unfinished game a `.table.pkl` with its full state to resume it. **All past games**
lists them; each one shows the reads, every player's result with the read marked right or wrong
for the bots, and the full hand log.

## Who's at the table

| Seat | What it is |
|---|---|
| PokerBot (engine) | The project's engine; see below. |
| Calibrated bot | A rule-based bot drawn from the full-ring styles fitted to real players (`profiles/bots-v3.json`): Loose-passive, Passive limper, Weak-tight, TAG, Nit, Short-stacker. |
| TAG baseline | The fixed tight-aggressive reference bot (`abc_tag`). |
| Open seat (person) | A seat for a human. |
| Empty | Nobody. |

**How PokerBot plays:**

- **Heads-up (only two players dealt in):** the evolved heads-up exploiter, as evaluated in
  [docs/RESULTS.md](../docs/RESULTS.md). It's the Phase 6 champion over the MCCFR blueprint,
  with the frozen heads-up classifier as its belief source.
- **Multiway:** a demo-only engine, *not* part of the evaluated results. The evolved 6-max
  classifier reads every opponent live, using the full-ring style profiles. A tight-aggressive
  base strategy is then tilted toward the live opponents' weaknesses: more bluffs and steals
  against folders, thinner value bets and fewer bluffs against callers.

The reads you see come from the same classifier run over every hand by the table itself, so
they exist even with no PokerBot seated. Reads, and PokerBot's memory, reset whenever the set of
players changes, because the classifiers track players by seat.

## Playing with other people

The game server listens on all interfaces, IPv4 and IPv6, on port 8000. Friends on your network
open `http://<your-computer's-IP>:8000`, pick an open seat and play.
Every browser tab is its own person, and reloading a tab keeps its seat. To play several people
from one machine, just open more tabs.

- **Your turn:** you have 45 seconds; after that you check if you can, otherwise fold. A
  disconnected player checks or folds, and **⋯ → Stand up** folds your hand at once, so the
  game never waits on someone who left.
- **Keys:** F fold, C check/call, R bet/raise to the slider amount.
- **Chips:** blinds are 0.5/1 bb, everyone starts with 100 bb (bots with their calibrated
  buy-in), and a busted seat rebuys to 100 bb.

## Notes

- This is a local demo: no accounts, one game at a time per server. If the server stops, the
  game in progress can be resumed from its last finished hand.
- The web app is pinned to Next.js 16.4.0. Its advisories (GHSA-2464-fhr4-w9xw,
  GHSA-p3m5-59mp-jv5r, GHSA-673q-mjm6-r5xr) cover image optimization of SVGs, Server Actions
  and `use cache`, none of which the demo uses. Upgrade to 16.4.3 when your npm
  `min-release-age` allows it (from 2026-11-06):
  `npm i --prefix demo/web next@16.4.3 eslint-config-next@16.4.3`.
- Tests for the server's table logic: `uv run pytest tests/test_demo_table.py`.
