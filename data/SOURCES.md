# Data sources

Raw and processed data are git-ignored. This file records where everything came from so it can be
re-downloaded exactly.

## HandHQ (via the PHH dataset) — in use

| | |
|---|---|
| What | 21,606,087 real-money online no-limit hold'em hands, July 1–23 2009, 2–9 handed cash games (all downloaded and converted, 0 unreadable files) |
| Sites | Absolute (ABS), Full Tilt (FTP), iPoker (IPN), Ongame (ONG), PokerStars (PS), PartyPoker (PTY) |
| Stakes | 25NL to 1000NL |
| Player names | Obfuscated, but consistent across hands within a site (verified: one player appears in 271 of 1,000 hands in a single file) |
| Hole cards | Hidden (`????`) unless shown at showdown (~46% of showdowns in a sample file reveal cards) |
| Repository | https://github.com/uoftcprg/phh-dataset, folder `data/handhq` |
| Full archive | Zenodo, https://doi.org/10.5281/zenodo.10796885 (record 17136841, 20.3 GB zip including ACPC bot hands we don't need) |
| License | Repository: MIT. Zenodo record: CC BY 4.0. Cite: Kim, J. et al., *A Dataset of Poker Hand Histories*, University of Toronto Computer Poker Student Research Group |
| Format | PHH (`.phhs` = many TOML hand records per file); spec at https://phh.readthedocs.io |

Download (sparse clone of only the HandHQ folder, ~16 GB on disk, 21,782 files):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/uoftcprg/phh-dataset.git data/raw/phh-dataset
```

```bash
git -C data/raw/phh-dataset sparse-checkout set data/handhq
```

Convert to Parquet (`data/processed/handhq/<SITE>-<STAKE>NL.parquet`, one row per hand, 769 MB):

```bash
uv run pokerbot data convert
```

Per-player stat counts (replays every hand in pokerkit, counts stats per MATH.md §2; ~15 min on
28 workers). Writes `data/processed/player_stats.parquet` with columns `pid, stat, n, x`:

```bash
uv run pokerbot data stats
```

Hands are skipped (not guessed) when the blinds don't identify positions: seat 1 must post the
small blind and seat 2 a larger big blind. That excludes 65,295 hands (0.30%: missing small blind or
reversed posts). Extra dead-blind posts and antes are fine; pokerkit handles them.

iPoker (all 6.0M IPN hands, 28%) has no recorded stack sizes (stored as infinity). Those hands are
replayed with a placeholder stack of 10,000 bb (`UNKNOWN_STACK_BB`): actions, pots and payoffs stay
exact, but stack-depth stat splits skip them and all-in flags are missing. iPoker files also show
one player's hole cards on every hand (the recording account); we ignore dealt hole cards.
iPoker also records no winnings and usually no revealed cards at showdown, and all-in hands have
no showdown records at all. So for any hand contested to the end, the winner is only treated as
known if some cards were revealed (`HandEnd.payoffs_known`); W$SD skips unknown outcomes. WTSD
is computed from who never folded, so it doesn't depend on showdown records. A few PartyPoker
high-stakes hands (up to ~5% of contested hands at 600NL) are affected the same way.

Stats are keyed by (site, player, table format): heads-up (2 dealt), short-handed (3–6), full
ring (7–10). Units with ≥ 500 hands: 20,274 short-handed, 11,557 full ring, 2,447 heads-up.

Coverage report (players with enough hands, as days of data accumulate):

```bash
uv run pokerbot data coverage
```

### How much is enough

A player is identified by (site, obfuscated name). Totals: 280,389 players; 107,166 with ≥ 100 hands;
**34,301 with ≥ 500**; 19,359 with ≥ 1,000; 4,371 with ≥ 5,000.

| Data through | Hands | Players ≥ 500 hands | Gain per extra day |
|---|---|---|---|
| Day 1 | 1.3M | 3,106 | — |
| Day 3 | 4.1M | 9,035 | ~3,000 |
| Day 7 | 9.6M | 18,761 | ~1,900 |
| Day 14 | 15.2M | 26,494 | ~1,000 |
| Day 23 (all) | 21.6M | 34,301 | ~400–1,000 |

The clustering model (MATH.md §3) with ~25 features and K ≈ 8 full-covariance components has
~2,800 parameters; ~10 players per parameter is ~28k players, which we exceed. In the last week each
extra day of data adds only ~3% more qualified players, so we are past the point of diminishing
returns for *this* use. More hands would only help through *different* data (other years or player
pools), not more of the same.

Caveat: this is the 2009 online player pool, which was looser and more passive than today's.
The styles we find describe that population.

## Not used (and why)

| Source | Reason |
|---|---|
| ACPC hands (in the same PHH dataset) | Bot-vs-bot competition hands; not human playing styles. May be useful later for testing the baseline. |
| Pluribus hands (PHH dataset) | 10k hands of an AI vs pros; too few and not representative of exploitable players. |
| IRC Poker Database | 1990s play-money/IRC games; play-money behavior differs from real-money play. |

## Opponent code: fullhouse-engine reference bots

| | |
|---|---|
| What | The five reference bots (`template`, `aggressor`, `mathematician`, `shark`, `ref_bot_2`) of the engine behind the e-x.club Showdown (https://www.e-x.club/poker) |
| Source | https://github.com/uzlez/fullhouse-engine, commit `a80b269`, `bots/` |
| Terms | MIT (Fullhouse Hackathon, 2026); license copied to `src/pokerbot/opponents/fullhouse/bots/LICENSE` |
| Use | Out-of-distribution opponents for duplicate benchmarks (`pokerbot eval duplicate`). Vendored verbatim and frozen; not data, no training use. |
