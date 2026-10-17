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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
