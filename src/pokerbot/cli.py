"""Command line entry point: ``uv run pokerbot <command>``."""

from __future__ import annotations

import argparse
import time


def cmd_llm_ping(args: argparse.Namespace) -> None:
    from pokerbot.llm import LLMClient

    client = LLMClient.from_env(cache=False)
    reply = client.complete(
        [{"role": "user", "content": "Reply with exactly: pong"}], temperature=0, max_tokens=16
    )
    print(f"model={reply.model} seconds={reply.seconds:.2f} tokens={reply.completion_tokens}")
    print(reply.text.strip())


def cmd_bench(args: argparse.Namespace) -> None:
    from pokerbot.engine import play_session
    from pokerbot.engine.policies import RandomPolicy

    players = [RandomPolicy(k) for k in range(args.players)]
    start = time.perf_counter()
    play_session(players, args.hands, seed=0)
    seconds = time.perf_counter() - start
    print(f"{args.hands} hands, {args.players} players: {args.hands / seconds:,.0f} hands/sec/core")


HANDHQ_SRC = "data/raw/phh-dataset/data/handhq"
HANDHQ_OUT = "data/processed/handhq"


def cmd_data_convert(args: argparse.Namespace) -> None:
    from pathlib import Path

    from pokerbot.data.handhq import convert_all

    results = convert_all(Path(args.src), Path(args.out), workers=args.workers)
    hands = sum(r.hands for r in results)
    bad = sum(len(r.bad_files) for r in results)
    print(f"total: {hands:,} hands from {len(results)} folders, {bad} unreadable files")


def cmd_data_coverage(args: argparse.Namespace) -> None:
    from pathlib import Path

    import polars as pl

    from pokerbot.data.coverage import breakdown, coverage_by_day

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    print(coverage_by_day(Path(args.data)))
    print(breakdown(Path(args.data), args.min_hands))


def cmd_data_stats(args: argparse.Namespace) -> None:
    import time
    from pathlib import Path

    from pokerbot.stats.aggregate import compute_player_stats
    from pokerbot.track import Run

    start = time.perf_counter()
    config = {k: v for k, v in vars(args).items() if k != "func"}
    with Run.start(
        "player-stats", kind="data", config=config, data={"hands": Path(args.data)}
    ) as run:
        info = compute_player_stats(
            Path(args.data), Path(args.out), workers=args.workers, limit_files=args.limit_files
        )
        seconds = time.perf_counter() - start
        run.finish(**info, seconds_total=seconds)
    print(f"{info['players']:,} players, {info['rows']:,} rows, {seconds:.0f}s")


def cmd_styles_fit(args: argparse.Namespace) -> None:
    from pathlib import Path

    from pokerbot.archetypes.fit import fit_styles

    out = fit_styles(
        Path(args.stats),
        Path(args.out or f"profiles/styles-{args.version}.json"),
        args.version,
        min_hands=args.min_hands,
        ks=range(args.k_min, args.k_max + 1),
        boots=args.boots,
    )
    print(out)


def cmd_styles_report(args: argparse.Namespace) -> None:
    from pathlib import Path

    from pokerbot.archetypes.report import render

    Path(args.out).write_text(render(Path(args.profiles)) + "\n", encoding="utf-8")
    print(args.out)


def cmd_bots_calibrate(args: argparse.Namespace) -> None:
    from pathlib import Path

    from pokerbot.opponents.job import run_calibration

    out = run_calibration(
        Path(args.profiles),
        Path(args.stats),
        Path(args.out or f"profiles/bots-{args.version}.json"),
        args.version,
        formats=tuple(args.format),
        rounds=args.rounds,
        iters=args.iters,
        popsize=args.popsize,
        hands=args.hands,
        workers=args.workers,
        warm_start=Path(args.warm_start) if args.warm_start else None,
        real_showdowns=Path(args.real_showdowns) if args.real_showdowns else None,
    )
    print(out)


def cmd_bots_showdown_real(args: argparse.Namespace) -> None:
    import json
    import time
    from pathlib import Path

    from pokerbot.opponents.showdown import real_tables
    from pokerbot.track import Run

    start = time.perf_counter()
    config = {k: v for k, v in vars(args).items() if k != "func"}
    with Run.start(
        "showdown-ranges", kind="data", config=config,
        data={"hands": Path(args.data), "profiles": Path(args.profiles),
              "player_stats": Path(args.stats)},
    ) as run:  # fmt: skip

        def log(done, total, info):
            print(f"{done}/{total} chunks, " + ", ".join(f"{k} {v:,}" for k, v in info.items()),
                  flush=True)  # fmt: skip

        doc = real_tables(
            Path(args.data), Path(args.profiles), Path(args.stats), tuple(args.format),
            workers=args.workers, limit_files=args.limit_files, log=log,
            parts_dir=Path(args.out).with_name(Path(args.out).stem + "_parts"),
        )  # fmt: skip
        doc["run_id"] = run.dir.name
        out = Path(args.out)
        out.write_text(json.dumps(doc), encoding="utf-8")
        run.save_artifact(out.name, out)
        for fmt, d in doc["formats"].items():
            for name, counts in zip(d["styles"], d["counts"], strict=True):
                run.log_metric(f"{fmt}/{name}/shown_weight", float(sum(map(sum, counts))),
                               style=name)  # fmt: skip
        players = {fmt: d["members"] for fmt, d in doc["formats"].items()}
        run.finish(**doc["info"], players=players, seconds=time.perf_counter() - start)
    print(out)


def cmd_bots_validate(args: argparse.Namespace) -> None:
    from pathlib import Path

    from pokerbot.archetypes.membership import format_stats, hand_quantiles
    from pokerbot.archetypes.profiles import load_profiles
    from pokerbot.opponents.validate import validate
    from pokerbot.track import Run

    real = Path(args.real_showdowns) if args.real_showdowns else None
    features = load_profiles(Path(args.profiles))["formats"][args.format]["transform"].features
    hands_q = hand_quantiles(format_stats(Path(args.stats), args.format, features), args.format)
    data = {"bots": Path(args.bots), "profiles": Path(args.profiles), "stats": Path(args.stats)}
    if real and real.exists():
        data["real_showdowns"] = real
    with Run.start(
        f"bots-validate-{Path(args.bots).stem}-{args.format}", kind="eval",
        config=vars(args) | {"func": None}, data=data,
    ) as run:  # fmt: skip
        report = validate(
            Path(args.bots), Path(args.profiles), args.format, workers=args.workers,
            hands=args.hands, real_showdowns=real, hands_q=hands_q,
        )  # fmt: skip
        run.save_artifact("validation.json", report)
        for name, r in report["styles"].items():
            run.log_metrics(
                {f"{name}/max_abs_center_z": r["max_abs_center_z"],
                 f"{name}/within_spread": r["within_spread"],
                 f"{name}/median_spread_ratio": r["median_spread_ratio"]}, style=name,
            )  # fmt: skip
            worst = sorted(r["stats"].items(), key=lambda kv: -abs(kv[1]["center_z"]))[:3]
            detail = ", ".join(f"{f} z={v['center_z']:+.2f}" for f, v in worst)
            print(
                f"{name:26s} {r['within_spread']}/{r['features']} within 1 sd, "
                f"{r['within_2']} within 2 (legacy {r['legacy_within_spread']}); "
                f"spread {r['median_spread_ratio']:.2f}; worst: {detail}"
            )
        sm = report["summary"]
        run.log_metrics(
            {"pairs_within_1": sm["within_spread"], "pairs_within_2": sm["within_2"],
             "pairs_legacy_within_1": sm["legacy_within_spread"]}
        )  # fmt: skip
        print(
            f"{report['format']}: {sm['within_spread']}/{sm['pairs']} style-stat pairs within 1 sd "
            f"({sm['within_spread'] / sm['pairs']:.0%}), {sm['within_2']} within 2 sd; legacy "
            f"yardstick {sm['legacy_within_spread']}/{sm['pairs']}"
        )
        if "showdowns" in report:
            print("\nshown-hand strength by line (real/sim mean equity, z, TV of buckets):")
            for name, r in report["showdowns"].items():
                run.log_metrics(
                    {f"{name}/sd_lines_tested": r["lines_tested"],
                     f"{name}/sd_lines_within_2se": r["lines_equity_within_2se"],
                     f"{name}/sd_line_share_tv": r["line_share_tv"]}, style=name,
                )  # fmt: skip
                cells = []
                for line, c in r["lines"].items():
                    if "equity_z" in c:
                        cells.append(
                            f"{line} {c['real_equity']:.2f}/{c['sim_equity']:.2f} "
                            f"z={c['equity_z']:+.1f} tv={c['tv']:.2f}"
                        )
                print(
                    f"{name:26s} {r['lines_equity_within_2se']}/{r['lines_tested']} lines within "
                    f"2 se; line-mix TV {r['line_share_tv']:.2f}\n    " + "\n    ".join(cells)
                )
        elif real:
            print(f"(no real showdown tables at {real}: run `pokerbot bots showdown-real`)")


def cmd_bots_tune_spread(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from pokerbot.archetypes.membership import format_stats, hand_quantiles
    from pokerbot.archetypes.profiles import load_profiles
    from pokerbot.opponents.validate import tune_scales
    from pokerbot.track import Run

    bots_path = Path(args.bots)
    features = load_profiles(Path(args.profiles))["formats"][args.format]["transform"].features
    hands_q = hand_quantiles(format_stats(Path(args.stats), args.format, features), args.format)
    with Run.start(
        f"bots-spread-{bots_path.stem}-{args.format}", kind="calibrate",
        config=vars(args) | {"func": None},
        data={"bots": bots_path, "profiles": Path(args.profiles)},
    ) as run:  # fmt: skip

        def log(scale, name, r):
            run.log_metric(f"{name}/median_spread_ratio", r["median_spread_ratio"], style=name,
                           scale=scale)  # fmt: skip
            print(f"scale {scale:.2f} {name:26s} spread ratio {r['median_spread_ratio']:.2f}")

        scales = tune_scales(bots_path, Path(args.profiles), args.format, workers=args.workers,
                             log=log, hands_q=hands_q)  # fmt: skip
        doc = json.loads(bots_path.read_text(encoding="utf-8"))
        for st in doc["formats"][args.format]["styles"]:
            st["x_scale"] = st.get("x_scale", 1.0) * scales[st["name"]]  # scales are relative
        doc.setdefault("spread_run_id", {})
        if isinstance(doc["spread_run_id"], str):  # bots-v1/v2: one format, one run id
            doc["spread_run_id"] = {"sh": doc["spread_run_id"]}
        doc["spread_run_id"][args.format] = run.dir.name
        bots_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        run.save_artifact("scales.json", scales)
        print({k: round(v, 3) for k, v in scales.items()})


def cmd_runs_list(args: argparse.Namespace) -> None:
    from pokerbot.track import list_runs

    _tables()
    print(list_runs())


def cmd_runs_show(args: argparse.Namespace) -> None:
    import json

    from pokerbot.track import find_run, latest_metrics
    from pokerbot.track.read import load_events, load_meta

    _tables()
    d = find_run(args.run)
    meta = load_meta(d)
    keep = ("id", "kind", "status", "started", "ended", "seconds", "git", "seeds", "summary")
    print(json.dumps({k: meta[k] for k in keep}, indent=2))
    print(f"config: {json.dumps(meta['config'])}")
    print(latest_metrics(d))
    events = load_events(d)
    if not events.is_empty():
        print(events.group_by("kind").len().sort("len", descending=True))


def cmd_runs_plot(args: argparse.Namespace) -> None:
    from pokerbot.track import find_run, plot_run

    print(plot_run(find_run(args.run), names=args.metric or None))


def _tables() -> None:
    import polars as pl

    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pokerbot")
    sub = parser.add_subparsers(required=True)

    llm = sub.add_parser("llm", help="LLM utilities").add_subparsers(required=True)
    llm.add_parser("ping", help="check the Luminal endpoint answers").set_defaults(
        func=cmd_llm_ping
    )

    bench = sub.add_parser("bench", help="engine throughput with random players")
    bench.add_argument("--hands", type=int, default=20_000)
    bench.add_argument("--players", type=int, default=6)
    bench.set_defaults(func=cmd_bench)

    data = sub.add_parser("data", help="real hand-history data").add_subparsers(required=True)
    convert = data.add_parser("convert", help="convert HandHQ .phhs files to Parquet")
    convert.add_argument("--src", default=HANDHQ_SRC)
    convert.add_argument("--out", default=HANDHQ_OUT)
    convert.add_argument("--workers", type=int, default=8)
    convert.set_defaults(func=cmd_data_convert)
    coverage = data.add_parser("coverage", help="players with enough hands, by day")
    coverage.add_argument("--data", default=HANDHQ_OUT)
    coverage.add_argument("--min-hands", type=int, default=500)
    coverage.set_defaults(func=cmd_data_coverage)

    stats = data.add_parser("stats", help="per-player stat counts over all real hands")
    stats.add_argument("--data", default=HANDHQ_OUT)
    stats.add_argument("--out", default="data/processed/player_stats.parquet")
    stats.add_argument("--workers", type=int, default=10)
    stats.add_argument("--limit-files", type=int, default=None)
    stats.set_defaults(func=cmd_data_stats)

    styles = sub.add_parser("styles", help="discover playing styles").add_subparsers(required=True)
    sfit = styles.add_parser("fit", help="cluster real players into styles, write profiles")
    sfit.add_argument("--stats", default="data/processed/player_stats.parquet")
    sfit.add_argument("--version", default="v1")
    sfit.add_argument("--out", default=None)
    sfit.add_argument("--min-hands", type=int, default=500)
    sfit.add_argument("--k-min", type=int, default=3)
    sfit.add_argument("--k-max", type=int, default=14)
    sfit.add_argument("--boots", type=int, default=5)
    sfit.set_defaults(func=cmd_styles_fit)
    srep = styles.add_parser("report", help="write a markdown report of a profile file")
    srep.add_argument("--profiles", default="profiles/styles-v1.json")
    srep.add_argument("--out", default="docs/STYLES.md")
    srep.set_defaults(func=cmd_styles_report)

    bots = sub.add_parser("bots", help="synthetic opponents").add_subparsers(required=True)
    cal = bots.add_parser("calibrate", help="fit bot parameters to each style (CMA-ES)")
    cal.add_argument("--profiles", default="profiles/styles-v1.json")
    cal.add_argument("--stats", default="data/processed/player_stats.parquet")
    cal.add_argument("--version", default="v1")
    cal.add_argument("--out", default=None)
    cal.add_argument("--format", action="append", default=None, help="sh, fr, hu (repeatable)")
    cal.add_argument("--rounds", type=int, default=2)
    cal.add_argument("--iters", type=int, default=30)
    cal.add_argument("--popsize", type=int, default=14)
    cal.add_argument("--hands", type=int, default=4000, help="cap on hands per bot evaluation")
    cal.add_argument("--workers", type=int, default=8)
    cal.add_argument("--warm-start", default=None, help="earlier bots file to start from")
    cal.add_argument("--real-showdowns", default="data/processed/showdown_ranges.json",
                     help="real shown-hand tables to match ('' to skip)")  # fmt: skip
    cal.set_defaults(func=cmd_bots_calibrate)
    sdr = bots.add_parser("showdown-real", help="shown-hand strength by line, real players")
    sdr.add_argument("--data", default=HANDHQ_OUT)
    sdr.add_argument("--profiles", default="profiles/styles-v1.json")
    sdr.add_argument("--stats", default="data/processed/player_stats.parquet")
    sdr.add_argument("--out", default="data/processed/showdown_ranges.json")
    sdr.add_argument("--format", action="append", default=None, help="sh, fr, hu (repeatable)")
    sdr.add_argument("--workers", type=int, default=8)
    sdr.add_argument("--limit-files", type=int, default=None)
    sdr.set_defaults(func=cmd_bots_showdown_real)
    val = bots.add_parser("validate", help="compare sampled bots with real players per style")
    val.add_argument("--bots", default="profiles/bots-v1.json")
    val.add_argument("--profiles", default="profiles/styles-v1.json")
    val.add_argument("--format", default="sh")
    val.add_argument("--workers", type=int, default=8)
    val.add_argument("--stats", default="data/processed/player_stats.parquet")
    val.add_argument("--hands", type=int, default=None, help="cap on hands per table")
    val.add_argument("--real-showdowns", default="data/processed/showdown_ranges.json")
    val.set_defaults(func=cmd_bots_validate)
    tune = bots.add_parser("tune-spread", help="measure per-style spread scales (writes x_scale)")
    tune.add_argument("--bots", default="profiles/bots-v1.json")
    tune.add_argument("--profiles", default="profiles/styles-v1.json")
    tune.add_argument("--format", default="sh")
    tune.add_argument("--stats", default="data/processed/player_stats.parquet")
    tune.add_argument("--workers", type=int, default=8)
    tune.set_defaults(func=cmd_bots_tune_spread)

    runs = sub.add_parser("runs", help="tracked experiment runs").add_subparsers(required=True)
    runs.add_parser("list", help="all runs").set_defaults(func=cmd_runs_list)
    show = runs.add_parser("show", help="metadata, metrics summary and events of a run")
    show.add_argument("run", nargs="?", default="latest", help="id, unique substring, or latest")
    show.set_defaults(func=cmd_runs_show)
    plot = runs.add_parser("plot", help="plot a run's metrics to artifacts/metrics.png")
    plot.add_argument("run", nargs="?", default="latest")
    plot.add_argument("--metric", action="append", help="only metrics containing this text")
    plot.set_defaults(func=cmd_runs_plot)

    args = parser.parse_args(argv)
    if getattr(args, "format", "unset") is None:  # calibrate / showdown-real: repeatable --format
        args.format = ["sh", "hu", "fr"] if args.func is cmd_bots_showdown_real else ["sh"]
    args.func(args)


if __name__ == "__main__":
    main()
