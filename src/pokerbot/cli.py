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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
