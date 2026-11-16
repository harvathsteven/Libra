"""CLI entrypoint: extract, pilot, main (Phase 1+)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trait-mtmm")
    parser.add_argument(
        "command",
        choices=["extract", "pilot", "main", "analyze", "figures"],
        help="Pipeline stage (stub until Phase 1–3)",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    raise SystemExit(
        f"Command {args.command!r} is not implemented yet "
        f"(config={args.config}). Complete Phase 0 gate G0 first."
    )


if __name__ == "__main__":
    main()
