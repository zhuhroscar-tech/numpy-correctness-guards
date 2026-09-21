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
from .guards.einsum_newdtype import detect_einsum_newstyle_dtype_bug
from .guards.poisson_variance import diagnose as diagnose_poisson_variance
from .guards.poisson_variance import safe_poisson
from .guards.seedsequence_spawn import detect_spawn_race, verify_guard_eliminates_race
from .guards.timedelta64_floordiv import (
    detect_negative_floordiv_truncation_bug,
    safe_timedelta64_floordiv,
    verify_workaround_against_oracle,
)
from .style import print_fields, resolve_style, status_headline

GUARDS = {
    "choice-shuffle": "Generator.choice(replace=False, p=weights) shuffle probe/workaround",
    "einsum-newdtype": "np.einsum new-style dtype probe/workaround",
    "poisson-variance": "Generator.poisson large-lambda variance probe/workaround",
    "seedsequence-spawn": "SeedSequence.spawn thread-safety race probe/workaround",
    "timedelta64-floordiv": "timedelta64 // int negative floor-division probe/workaround",
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


def cmd_einsum_newdtype_detect(args: argparse.Namespace) -> int:
    try:
        result = detect_einsum_newstyle_dtype_bug()
    except ImportError as exc:
        payload = {
            "error": (
                "numpy_quaddtype is required for the live probe "
                f"(pip install numpy_quaddtype): {exc}"
            )
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            style = resolve_style(args.no_color)
            print(status_headline(style, "warn", "cannot probe: numpy_quaddtype not installed"))
            print_fields([("detail", payload["error"])])
        return 2

    if args.json:
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 1 if result.affected else 0

    style = resolve_style(args.no_color)
    level = "fail" if result.affected else "ok"
    print(status_headline(style, level, "numpy einsum new-style-dtype probe"))
    print_fields(
        [
            ("numpy version", result.numpy_version),
            ("probe dtype", result.dtype_name),
            ("affected", "yes" if result.affected else "no"),
            ("detail", result.detail),
        ]
    )
    return 1 if result.affected else 0


def cmd_poisson_variance_detect(args: argparse.Namespace) -> int:
    result = diagnose_poisson_variance(
        lam=args.lam,
        n_samples=args.samples,
        tolerance=args.tolerance,
    )
    if args.json:
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 1 if result.affected else 0

    style = resolve_style(args.no_color)
    level = "fail" if result.affected else "ok"
    print(status_headline(style, level, "numpy Poisson large-lambda variance probe"))
    print_fields(
        [
            ("numpy version", result.numpy_version),
            ("lam", f"{result.lam:.0e}"),
            ("samples", str(result.n_samples)),
            ("numpy var/lam", f"{result.numpy_var_over_lam:.4f}"),
            ("safe_poisson var/lam", f"{result.safe_var_over_lam:.4f}"),
            ("affected", "yes" if result.affected else "no"),
            ("detail", result.detail),
        ]
    )
    return 1 if result.affected else 0


def cmd_poisson_variance_sample(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    samples = safe_poisson(args.lam, args.size, rng=rng)
    if args.json:
        print(json.dumps({"lam": args.lam, "size": args.size, "samples": samples.tolist()}))
        return 0

    style = resolve_style(args.no_color)
    print(status_headline(style, "info", "safe_poisson sample summary"))
    print_fields(
        [
            ("lam", f"{args.lam:.0e}"),
            ("size", str(args.size)),
            ("empirical mean", f"{samples.mean():.6e}"),
            ("empirical var/lam", f"{samples.astype(float).var() / args.lam:.4f}"),
        ]
    )
    return 0


def cmd_seedsequence_spawn_detect(args: argparse.Namespace) -> int:
    result = detect_spawn_race(
        n_threads=args.threads,
        spawns_per_thread=args.spawns_per_thread,
        seed=args.seed,
        switch_interval=args.switch_interval,
    )
    if args.json:
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 1 if result.affected else 0

    style = resolve_style(args.no_color)
    level = "fail" if result.affected else "ok"
    print(status_headline(style, level, "SeedSequence.spawn() thread-safety probe"))
    print_fields(
        [
            ("numpy version", result.numpy_version),
            ("affected", "yes" if result.affected else "no"),
            ("duplicate children", f"{result.duplicate_count} / {result.total_children}"),
            ("detail", result.detail),
        ]
    )
    return 1 if result.affected else 0


def cmd_seedsequence_spawn_verify(args: argparse.Namespace) -> int:
    payload = verify_guard_eliminates_race(
        n_threads=args.threads,
        spawns_per_thread=args.spawns_per_thread,
        seed=args.seed,
        switch_interval=args.switch_interval,
    )
    passed = payload["guard_fully_eliminates_race"]

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1

    style = resolve_style(args.no_color)
    level = "ok" if passed else "fail"
    print(status_headline(style, level, "guard-eliminates-race verification"))
    print_fields(
        [
            ("bare duplicates", f"{payload['bare_duplicates']} / {payload['bare_total']}"),
            ("guarded duplicates", f"{payload['guarded_duplicates']} / {payload['guarded_total']}"),
            ("guard fully eliminates race", "yes" if passed else "no"),
        ]
    )
    return 0 if passed else 1


def cmd_timedelta64_floordiv_detect(args: argparse.Namespace) -> int:
    result = detect_negative_floordiv_truncation_bug()
    if args.json:
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 1 if result.affected else 0

    style = resolve_style(args.no_color)
    level = "fail" if result.affected else "ok"
    print(status_headline(style, level, "numpy timedelta64 // int floor-division probe"))
    print_fields(
        [
            ("numpy version", result.numpy_version),
            ("affected", "yes" if result.affected else "no"),
            ("mismatches", f"{result.mismatches}/{result.total_checked}"),
            ("detail", result.detail),
        ]
    )
    return 1 if result.affected else 0


def cmd_timedelta64_floordiv_verify(args: argparse.Namespace) -> int:
    payload = verify_workaround_against_oracle()
    passed = payload["passed"]

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1

    style = resolve_style(args.no_color)
    level = "ok" if passed else "fail"
    print(status_headline(style, level, "safe_timedelta64_floordiv correctness check vs oracle"))
    print_fields(
        [
            ("combinations checked", str(payload["checked"])),
            ("failures", str(payload["failures"])),
            ("passed", "yes" if passed else "no"),
        ]
    )
    return 0 if passed else 1


def cmd_timedelta64_floordiv_apply(args: argparse.Namespace) -> int:
    delta = np.timedelta64(args.value, args.unit)
    result = safe_timedelta64_floordiv(delta, args.divisor)
    if args.json:
        print(
            json.dumps(
                {
                    "value": args.value,
                    "unit": args.unit,
                    "divisor": args.divisor,
                    "result": str(result),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    style = resolve_style(args.no_color)
    print(status_headline(style, "ok", "safe timedelta64 floor-division"))
    print_fields(
        [
            ("input", f"np.timedelta64({args.value}, {args.unit!r})"),
            ("divisor", str(args.divisor)),
            ("result", str(result)),
        ]
    )
    return 0


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


def _add_einsum_newdtype_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    einsum = sub.add_parser("einsum-newdtype", help=GUARDS["einsum-newdtype"])
    einsum_sub = einsum.add_subparsers(dest="einsum_newdtype_command", required=True)

    detect = einsum_sub.add_parser("detect", help="probe the installed numpy for the bug")
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--no-color", action="store_true")
    detect.set_defaults(func=cmd_einsum_newdtype_detect)


def _add_poisson_variance_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    poisson = sub.add_parser("poisson-variance", help=GUARDS["poisson-variance"])
    poisson_sub = poisson.add_subparsers(dest="poisson_variance_command", required=True)

    detect = poisson_sub.add_parser("detect", help="empirically probe installed numpy for the bug")
    detect.add_argument("--lam", type=float, default=1e16)
    detect.add_argument("--samples", type=int, default=200_000)
    detect.add_argument("--tolerance", type=float, default=0.05)
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--no-color", action="store_true")
    detect.set_defaults(func=cmd_poisson_variance_detect)

    sample = poisson_sub.add_parser("sample", help="draw corrected Poisson(lam) samples via safe_poisson")
    sample.add_argument("--lam", type=float, required=True)
    sample.add_argument("--size", type=int, required=True)
    sample.add_argument("--seed", type=int, default=None)
    sample.add_argument("--json", action="store_true")
    sample.add_argument("--no-color", action="store_true")
    sample.set_defaults(func=cmd_poisson_variance_sample)


def _add_seedsequence_spawn_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    seedsequence = sub.add_parser("seedsequence-spawn", help=GUARDS["seedsequence-spawn"])
    seedsequence_sub = seedsequence.add_subparsers(dest="seedsequence_spawn_command", required=True)

    detect = seedsequence_sub.add_parser("detect", help="probe the installed numpy for the spawn race")
    detect.add_argument("--threads", type=int, default=4)
    detect.add_argument("--spawns-per-thread", type=int, default=500)
    detect.add_argument("--seed", type=int, default=12345)
    detect.add_argument("--switch-interval", type=float, default=1e-5)
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--no-color", action="store_true")
    detect.set_defaults(func=cmd_seedsequence_spawn_detect)

    verify = seedsequence_sub.add_parser(
        "verify", help="verify GuardedSeedSequence eliminates the race under stress"
    )
    verify.add_argument("--threads", type=int, default=4)
    verify.add_argument("--spawns-per-thread", type=int, default=500)
    verify.add_argument("--seed", type=int, default=12345)
    verify.add_argument("--switch-interval", type=float, default=1e-5)
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--no-color", action="store_true")
    verify.set_defaults(func=cmd_seedsequence_spawn_verify)


def _add_timedelta64_floordiv_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    timedelta = sub.add_parser("timedelta64-floordiv", help=GUARDS["timedelta64-floordiv"])
    timedelta_sub = timedelta.add_subparsers(dest="timedelta64_floordiv_command", required=True)

    detect = timedelta_sub.add_parser("detect", help="probe installed numpy for the bug")
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--no-color", action="store_true")
    detect.set_defaults(func=cmd_timedelta64_floordiv_detect)

    verify = timedelta_sub.add_parser(
        "verify", help="verify safe_timedelta64_floordiv against the independent oracle"
    )
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--no-color", action="store_true")
    verify.set_defaults(func=cmd_timedelta64_floordiv_verify)

    apply = timedelta_sub.add_parser("apply", help="floor-divide one timedelta64 value safely")
    apply.add_argument("--value", type=int, required=True)
    apply.add_argument("--unit", default="us")
    apply.add_argument("--divisor", type=int, required=True)
    apply.add_argument("--json", action="store_true")
    apply.add_argument("--no-color", action="store_true")
    apply.set_defaults(func=cmd_timedelta64_floordiv_apply)


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
    _add_einsum_newdtype_commands(run_sub)
    _add_poisson_variance_commands(run_sub)
    _add_seedsequence_spawn_commands(run_sub)
    _add_timedelta64_floordiv_commands(run_sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
