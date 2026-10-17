"""Readable markdown report of a style profile file."""

from __future__ import annotations

from pathlib import Path

from pokerbot.archetypes.profiles import load_profiles

COLUMNS = [
    ("vpip", "VPIP"),
    ("pfr", "PFR"),
    ("limp", "Limp"),
    ("rfi_EP", "Open EP"),
    ("rfi_BTN", "Open BTN"),
    ("3bet", "3-bet"),
    ("fold_to_3bet", "Fold to 3-bet"),
    ("fold_to_steal", "Fold to steal"),
    ("cbet_flop", "C-bet"),
    ("fold_to_cbet_flop", "Fold to c-bet"),
    ("agg_flop", "Flop agg"),
    ("wtsd", "WTSD"),
]
FORMAT_NAMES = {"sh": "Short-handed (3-6 players)", "fr": "Full ring (7-10)", "hu": "Heads-up"}


def render(path: Path) -> str:
    doc = load_profiles(path)
    out = [
        f"# Playing styles ({doc['version']})",
        "",
        f"Generated from `{Path(path).as_posix()}` (run `{doc['run_id']}`;"
        f" source: {doc['source']}).",
        "Rates are pooled over each style's members (weighted by membership probability).",
        "Regenerate with `uv run pokerbot styles report`.",
        "",
    ]
    for fmt in ("sh", "fr", "hu"):
        if fmt not in doc["formats"]:
            continue
        f = doc["formats"][fmt]
        sel = {r["k"]: r for r in f["selection"]}
        k = len(f["styles"])
        out += [
            f"## {FORMAT_NAMES[fmt]}: {k} styles",
            "",
            f"{f['players']:,} player units fitted. K = {k} chosen: BIC {sel[k]['bic']:,.0f}, "
            f"bootstrap ARI {sel[k]['ari']:.2f}.",
            "",
        ]
        cols = [c for c in COLUMNS if c[0] in f["transform"].features]
        out.append("| # | Style | Share | " + " | ".join(h for _, h in cols) + " |")
        out.append("|---|---|---|" + "---|" * len(cols))
        for s in f["styles"]:
            r = s["pooled_rate"]
            cells = " | ".join(f"{r[c]:.0%}" for c, _ in cols)
            out.append(f"| {s['index']} | **{s['name']}** | {s['share']:.1%} | {cells} |")
        out.append("")
        for s in f["styles"]:
            out.append(f"- **{s['name']}**: {s['description']}")
        out += [
            "",
            "<details><summary>Model selection</summary>",
            "",
            "| K | BIC | ARI | Smallest share |",
            "|---|---|---|---|",
        ]
        for r in f["selection"]:
            out.append(f"| {r['k']} | {r['bic']:,.0f} | {r['ari']:.3f} | {r['min_share']:.1%} |")
        out += ["", "</details>", ""]
    return "\n".join(out)
