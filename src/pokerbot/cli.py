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
    args.func(args)


if __name__ == "__main__":
    main()
