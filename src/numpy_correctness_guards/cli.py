"""Shared CLI for consolidated NumPy correctness guards."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .guards.choice_shuffle import (
    detect_shuffle_ignored_bug,
    independent_reference_weighted_sample_without_replacement,
    safe_weighted_choice,
)
from .style import print_fields, resolve_style, status_headline

GUARDS = {
    "choice-shuffle": "Generator.choice(replace=False, p=weights) shuffle probe/workaround",
}


def cmd_list(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(GUARDS, indent=2, sort_keys=True))
        return 0
    for name, description in GUARDS.items():
        print(f"{name:16} {description}")
    return 0


def cmd_choice_shuffle_detect(args: argparse.Namespace) -> int:
    result = detect_shuffle_ignored_bug(
        population_size=args.population_size,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 1 if result.affected else 0

    style = resolve_style(args.no_color)
    level = "fail" if result.affected else "ok"
    print(status_headline(style, level, "numpy Generator.choice shuffle probe"))
    print_fields(
        [
            ("numpy version", result.numpy_version),
            ("affected", "yes" if result.affected else "no"),
            ("detail", result.detail),
        ]
    )
    return 1 if result.affected else 0


def cmd_choice_shuffle_verify(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    n = args.population_size
    k = args.sample_size
    weights = np.linspace(1.0, 2.0, n)
    weights = weights / weights.sum()

    selection_counts_guard = np.zeros(n, dtype=np.int64)
    selection_counts_oracle = np.zeros(n, dtype=np.int64)

    for trial in range(args.trials):
        guard_sample = safe_weighted_choice(
            rng,
            n,
            size=k,
            replace=False,
            p=weights,
            shuffle=True,
            force_workaround=True,
        )
        selection_counts_guard[guard_sample] += 1

        oracle_sample = independent_reference_weighted_sample_without_replacement(
            n,
            k,
            weights,
            seed=args.seed + trial + 1,
        )
        selection_counts_oracle[oracle_sample] += 1

    guard_freq = selection_counts_guard / args.trials
    oracle_freq = selection_counts_oracle / args.trials
    max_abs_diff = float(np.max(np.abs(guard_freq - oracle_freq)))
    mean_abs_diff = float(np.mean(np.abs(guard_freq - oracle_freq)))
    passed = max_abs_diff <= args.tolerance

    payload = {
        "guard": "choice-shuffle",
        "trials": args.trials,
        "population_size": n,
        "sample_size": k,
        "max_abs_freq_diff": max_abs_diff,
        "mean_abs_freq_diff": mean_abs_diff,
        "tolerance": args.tolerance,
        "passed": passed,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1

    style = resolve_style(args.no_color)
    level = "ok" if passed else "fail"
    print(status_headline(style, level, "choice-shuffle selection-frequency check"))
    print_fields(
        [
            ("trials", str(args.trials)),
            ("max abs frequency diff", f"{max_abs_diff:.4f}"),
            ("mean abs frequency diff", f"{mean_abs_diff:.4f}"),
            ("tolerance", f"{args.tolerance:.4f}"),
            ("passed", "yes" if passed else "no"),
        ]
    )
    return 0 if passed else 1


def _add_choice_shuffle_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    choice = sub.add_parser("choice-shuffle", help=GUARDS["choice-shuffle"])
    choice_sub = choice.add_subparsers(dest="choice_shuffle_command", required=True)

    detect = choice_sub.add_parser("detect", help="probe the installed numpy for the bug")
    detect.add_argument("--population-size", type=int, default=500)
    detect.add_argument("--sample-size", type=int, default=100)
    detect.add_argument("--seed", type=int, default=0)
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--no-color", action="store_true")
    detect.set_defaults(func=cmd_choice_shuffle_detect)

    verify = choice_sub.add_parser(
        "verify", help="statistically verify safe_weighted_choice against an independent oracle"
    )
    verify.add_argument("--population-size", type=int, default=200)
    verify.add_argument("--sample-size", type=int, default=40)
    verify.add_argument("--trials", type=int, default=2000)
    verify.add_argument("--tolerance", type=float, default=0.05)
    verify.add_argument("--seed", type=int, default=0)
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--no-color", action="store_true")
    verify.set_defaults(func=cmd_choice_shuffle_verify)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numpy-guard",
        description="Run consolidated NumPy correctness guards.",
    )
    parser.add_argument("--version", action="version", version=f"numpy-correctness-guards {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="list available guards")
    list_cmd.add_argument("--json", action="store_true")
    list_cmd.set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="run a guard by name")
    run_sub = run.add_subparsers(dest="guard", required=True)
    _add_choice_shuffle_commands(run_sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
