"""Phase 7 report: ``docs/RESULTS.md`` and plots from :func:`pokerbot.experiment.run` results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokerbot.experiment import GAMES, lineage

TITLES = {"leduc": "Track A: Leduc", "hu": "Track B: heads-up NLHE"}


def _f(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def _cell(e: dict, nd: int = 3) -> str:
    lo, hi = e["ci95"]
    return f"{_f(e['mean'], nd)} [{_f(lo, nd)}, {_f(hi, nd)}]"


def _errorbar(ax, x: float, ci: list[float], y: float, **kw) -> None:
    ax.errorbar(x, y, xerr=[[x - ci[0]], [ci[1] - x]], **kw)


def plots(results: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    games = [g for g in GAMES if g in results["tracks"]]
    if games:
        fig, axes = plt.subplots(1, len(games), figsize=(6 * len(games), 4.4), squeeze=False)
        for ax, game in zip(axes[0], games, strict=True):
            t = results["tracks"][game]
            rows = [(f"evolved, {m}", e) for m, e in t["programs"]["evolved"].items()]
            rows.append(("seed, classifier", t["programs"]["seed"]["classifier"]))
            for y, (_, e) in enumerate(rows):
                for dy, kind, color in ((-0.15, "style", "C0"), (0.15, "switch", "C2")):
                    lab = (
                        {"style": "vs style players", "switch": "vs switchers"}[kind]
                        if y == 0
                        else None
                    )
                    _errorbar(
                        ax,
                        e[kind]["mean"],
                        e[kind]["ci95"],
                        y + dy,
                        fmt="o",
                        color=color,
                        label=lab,
                    )
            s = t["safety"]
            _errorbar(
                ax,
                s["mean"],
                s["ci95"],
                len(rows),
                fmt="s",
                color="C3",
                label="vs the baseline itself",
            )
            ax.axvline(0, color="k", lw=0.8)
            ax.axvline(-s["margin"], color="C3", lw=0.8, ls=":")
            ax.set_yticks(range(len(rows) + 1), [r[0] for r in rows] + ["evolved (safety)"])
            ax.set_xlabel(f"edge over the baseline ({t['unit']})")
            ax.set_title(TITLES[game])
            ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        fig.tight_layout()
        paths.append(out_dir / "exploiter_edges.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(games), figsize=(6 * len(games), 3.4), squeeze=False)
        for ax, game in zip(axes[0], games, strict=True):
            t = results["tracks"][game]
            w = 10 if game == "leduc" else 20
            for key, curve in t["curves"].items():
                c = np.convolve(curve, np.ones(w) / w, mode="valid")
                ax.plot(np.arange(w, len(curve) + 1), c, label=key)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel("hands played against this opponent")
            ax.set_ylabel(f"edge vs style players, {t['unit']}")
            ax.set_title(f"{TITLES[game]} ({w}-hand moving mean)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        paths.append(out_dir / "edge_vs_hands.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)
    if "classifier" in results["tracks"]:
        c = results["tracks"]["classifier"]
        cps = c["checkpoints"]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for i, (name, e) in enumerate(c["programs"].items()):
            test = [_at(e["test"], t) for t in cps]
            ax.plot(
                cps,
                [v["accuracy"] for v in test],
                "o-",
                color=f"C{i}",
                label=f"{name}, synthetic test",
            )
            ax.fill_between(
                cps,
                [v["accuracy_ci"][0] for v in test],
                [v["accuracy_ci"][1] for v in test],
                color=f"C{i}",
                alpha=0.15,
            )
            if "real_test" in e:
                real = [_at(e["real_test"], t) for t in cps]
                ax.plot(
                    cps,
                    [v["accuracy"] for v in real],
                    "x--",
                    color=f"C{i}",
                    label=f"{name}, held-out real",
                )
        ax.set_xscale("log")
        ax.set_xticks(cps, [str(t) for t in cps])
        ax.set_xlabel("hands observed")
        ax.set_ylabel("style accuracy")
        ax.set_title("Track C: 6-max style classifier")
        ax.legend(fontsize=7)
        fig.tight_layout()
        paths.append(out_dir / "classifier_accuracy.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)
    return paths


def _at(d: dict, t: int) -> dict:
    """Checkpoint lookup that works on fresh results (int keys) and reloaded JSON (str keys)."""
    return d[t] if t in d else d[str(t)]


def write_report(results: dict, cfg: dict, run_id: str, out: Path, plot_dir: Path) -> None:
    rel = plot_dir.relative_to(out.parent).as_posix()
    cfg_path = Path(cfg["path"]).as_posix()
    tr = results["tracks"]
    lines = [
        "# Results",
        "",
        f"Phase 7 final experiment `{results['version']}`, pre-registered in "
        f"[`{cfg_path}`](../{cfg_path}) on {cfg['registered']}. Every number below comes from "
        f"run `{run_id}`. Re-run with:",
        "",
        "```bash",
        f"uv run pokerbot final run --config {cfg_path}",
        "```",
        "",
        f"All sessions are fresh test-split seeds from index {results['first_index']:,} on, "
        "never used by evolution, tuning or an earlier evaluation. The programs are the frozen "
        "Phase 5/6 champions and their hand-written seeds. Intervals are 95% bootstrap CIs over "
        "sessions; tests are one-sided, with Holm-Bonferroni over the primary hypotheses at "
        f"family α = {cfg['alpha']}.",
        "",
        "## Primary hypotheses",
        "",
        "| | Claim | Estimate [95% CI] | p | Holm threshold | Result |",
        "|---|---|---|---|---|---|",
    ]
    for h, v in results["hypotheses"].items():
        lo, hi = v["ci95"]
        nd = 1 if h in ("B1", "B2") else 3
        verdict = "**supported**" if v["supported"] else "not supported"
        lines.append(
            f"| {h} | {v['claim']} | {_f(v['estimate'], nd)} [{_f(lo, nd)}, {_f(hi, nd)}] | "
            f"{v['p']:.1e} | {v['threshold']:.4f} | {verdict} |"
        )
    lines += [
        "",
        "Units: A1/A2 chips per hand; B1/B2 bb/100; A3/B3 the MATH §7.3 session score; C1 the "
        "weighted log-loss session score (MATH §6.3). A2/B2 test d > −margin (Leduc "
        f"{cfg['leduc']['margin']} chips/hand, heads-up {cfg['hu']['margin']} bb/100).",
    ]
    names = {
        "leduc": "Track A — Leduc hold'em (exact values, CFR+ Nash baseline)",
        "hu": "Track B — heads-up no-limit hold'em (MCCFR blueprint baseline)",
    }
    for game in GAMES:
        if game not in tr:
            continue
        t = tr[game]
        nd = 3 if game == "leduc" else 1
        hands = t["sessions"] * t["hands_per_session"]
        lines += [
            "",
            f"## {names[game]}",
            "",
            f"{t['sessions']:,} sessions × {t['hands_per_session']} hands ({hands:,} hands per "
            "row; 70% style, 15% baseline, 15% switcher sessions). The edge d over the baseline "
            f"is in {t['unit']}, in duplicate: the same cards and opponents for both.",
            "",
            "| Program, beliefs | vs style players | vs the baseline | vs switchers | KL |",
            "|---|---|---|---|---|",
        ]
        for pname, modes in t["programs"].items():
            for mode, e in modes.items():
                lines.append(
                    f"| {pname}, {mode} | {_cell(e['style'], nd)} | {_cell(e['nash'], nd)} | "
                    f"{_cell(e['switch'], nd)} | {e['kl']:.3f} |"
                )
        s = t["safety"]
        lines += [
            "",
            f"Safety, {s['sessions']:,} baseline-only sessions (indices {s['first_index']:,}–"
            f"{s['last_index']:,}): evolved {_cell(s, nd)} {t['unit']} (margin −{s['margin']}).",
            "",
            f"Evolved − seed, paired session score: {_cell(t['paired_vs_seed'])}.",
            "",
            "Belief ablations: the paired difference in the evolved program's edge vs style "
            "players on the same sessions (oracle = true labels, prior = no classifier, "
            "argmax = the classifier's top style with no confidence):",
            "",
        ]
        for k, a in t["ablations"].items():
            lines.append(f"- {k}: {_cell(a, nd)} (two-sided p {a['p_two_sided']:.2g})")
        ev = t["programs"]["evolved"]["classifier"]
        per = [f"{k[3:]} {_f(v['mean'], nd)}" for k, v in ev.items() if k.startswith("vs ")]
        lines += ["", "Per style (evolved, classifier beliefs): " + ", ".join(per) + "."]
    if "classifier" in tr:
        c = tr["classifier"]
        cps = c["checkpoints"]
        lines += [
            "",
            "## Track C — 6-max style classifier",
            "",
            f"{c['test_sessions']} fresh tables ({c['seats']:,} seats, 200 hands each) and "
            f"{c['real_players'].get('real_test', 0)} held-out real players (used once before, "
            "in Phase 5).",
            "",
            "| Program | " + " | ".join(f"acc@{t}" for t in cps) + f" | log loss@{cps[-1]} |",
            "|---|" + "---|" * (len(cps) + 1),
        ]
        for name, e in c["programs"].items():
            accs = " | ".join(f"{_at(e['test'], t)['accuracy']:.3f}" for t in cps)
            lines.append(f"| {name} | {accs} | {_at(e['test'], cps[-1])['logloss']:.3f} |")
        lines += [
            "",
            "Held-out real players: accuracy against argmax q / cross-entropy H(q, p):",
            "",
            "| Program | " + " | ".join(f"@{t}" for t in cps) + " |",
            "|---|" + "---|" * len(cps),
        ]
        for name, e in c["programs"].items():
            if "real_test" in e:
                cells = [
                    f"{_at(e['real_test'], t)['accuracy']:.3f} / "
                    f"{_at(e['real_test'], t)['xent']:.3f}"
                    for t in cps
                ]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        for k, v in c["paired"].items():
            lines.append(f"- {k}, paired session score: {_cell(v)}, p {v['p']:.1e}")
    lines += ["", "## Plots", ""]
    for name, cap in (
        ("exploiter_edges.png", "Edge per program and belief source"),
        ("edge_vs_hands.png", "Edge vs hands played against one opponent"),
        ("classifier_accuracy.png", "Classifier accuracy vs hands observed"),
    ):
        if (plot_dir / name).exists():
            lines += [f"![{cap}]({rel}/{name})", ""]
    lines += [
        "## Champion lineage",
        "",
        "Each champion's ancestry, newest first, back to its hand-written seed, from the "
        "evolution runs' candidate records.",
        "",
    ]
    for title, key in (
        ("Track A exploiter", "leduc"),
        ("Track B exploiter", "hu"),
        ("Track C classifier", "classifier"),
    ):
        if key not in cfg:
            continue
        spec = cfg[key]["programs"]["evolved"]
        lines += [f"**{title}** (`{spec['path']}`):", ""]
        try:
            chain = lineage(Path(spec["params"]))
        except (OSError, KeyError, ValueError) as e:
            lines += [f"- lineage unavailable ({e})", ""]
            continue
        for step in chain:
            if step.get("seed"):
                lines.append(f"- hand-written seed `{step['candidate']}`")
            elif step.get("seed_of_run"):
                lines.append(f"- run `{step['run']}` started from `{step['file']}`")
            else:
                idea = " ".join(step["idea"].split())
                lines.append(
                    f"- `{step['run']}` {step['candidate']} (gen {step['generation']}): {idea}"
                )
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Opponents are calibrated synthetic bots (Phase 3). Only the classifier is checked on "
        "real players; the exploiters have not played real people.",
        "- The Leduc baseline is an exact equilibrium. The heads-up baseline is an abstracted "
        "MCCFR blueprint, so heads-up edges and margins are relative to that approximation.",
        "- No 6-max (Track C) exploiter was built; Track C covers classification only.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
