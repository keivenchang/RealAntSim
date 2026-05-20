#!/usr/bin/env python3
"""Score-driven rollout tuning for the neural worker policy.

This is not differentiable RL: the Rust sim/bench suite is the environment,
so this script evaluates candidate neural runtime gates by running real
rollouts and accepting only candidates that pass the hard realism gates.
Use it after `train_neural_worker.py` has produced warm-start weights.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


FITNESS_RE = re.compile(
    r"FITNESS total=(?P<total>[0-9.]+) pass=(?P<passed>true|false) "
    r"wall=(?P<wall>[0-9.]+) arc=(?P<arc>[0-9.]+) multi=(?P<multi>[0-9.]+) "
    r"loop=(?P<loop>[0-9.]+) cycle=(?P<cycle>[0-9.]+) "
    r"postPickup=(?P<post>[0-9.]+) lostCarrier=(?P<lost>[0-9.]+) "
    r"cluster=(?P<cluster>[0-9.]+) sec=(?P<sec>[0-9.]+)"
)
POST_RE = re.compile(
    r"post_pickup .* score=(?P<score>[0-9.]+) .* "
    r"pDash=(?P<pdash>[0-9.]+) .* pLine=(?P<pline>[0-9.]+)"
)
LOST_RE = re.compile(
    r"lost_carrier .* score=(?P<score>[0-9.]+) .* "
    r"pBack=(?P<pback>[0-9.]+) .* returned=(?P<returned>[0-9]+)"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    blend: float
    min_ticks: int
    min_dist: float
    max_dist: float


@dataclass(frozen=True)
class Fitness:
    candidate: Candidate
    total: float
    passed: bool
    wall: float
    arc: float
    multi: float
    loop: float
    cycle: float
    post: float
    lost: float
    cluster: float
    seconds: float
    output: str


def parse_fitness(candidate: Candidate, output: str) -> Fitness:
    match = FITNESS_RE.search(output)
    if match is None:
        raise RuntimeError(f"could not parse bench output for {candidate.name}\n{output}")
    values = match.groupdict()
    return Fitness(
        candidate=candidate,
        total=float(values["total"]),
        passed=values["passed"] == "true",
        wall=float(values["wall"]),
        arc=float(values["arc"]),
        multi=float(values["multi"]),
        loop=float(values["loop"]),
        cycle=float(values["cycle"]),
        post=float(values["post"]),
        lost=float(values["lost"]),
        cluster=float(values["cluster"]),
        seconds=float(values["sec"]),
        output=output,
    )


def parse_quick_fitness(candidate: Candidate, post_output: str, lost_output: str) -> Fitness:
    post = POST_RE.search(post_output)
    lost = LOST_RE.search(lost_output)
    if post is None or lost is None:
        raise RuntimeError(
            f"could not parse quick bench output for {candidate.name}\n"
            f"{post_output}\n{lost_output}"
        )
    post_score = float(post.group("score"))
    lost_score = float(lost.group("score"))
    passed = (
        float(post.group("pdash")) <= 0.005
        and float(post.group("pline")) <= 0.0
        and float(lost.group("pback")) <= 0.08
        and int(lost.group("returned")) == 0
    )
    return Fitness(
        candidate=candidate,
        total=post_score + lost_score,
        passed=passed,
        wall=0.0,
        arc=0.0,
        multi=0.0,
        loop=0.0,
        cycle=0.0,
        post=post_score,
        lost=lost_score,
        cluster=0.0,
        seconds=0.0,
        output=post_output + lost_output,
    )


def run_command(args: argparse.Namespace, env: dict[str, str], command: str) -> str:
    proc = subprocess.run(
        [str(args.backend), command, "--brain", "neural"],
        cwd=args.root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"{command} failed\n{output}")
    return output


def run_candidate(args: argparse.Namespace, candidate: Candidate) -> Fitness:
    env = os.environ.copy()
    env["REALANTSIM_NEURAL_WORKER_WEIGHTS"] = str(args.weights)
    env["REALANTSIM_NEURAL_BLEND"] = str(candidate.blend)
    env["REALANTSIM_NEURAL_MIN_TICKS"] = str(candidate.min_ticks)
    env["REALANTSIM_NEURAL_MIN_PICKUP_DIST"] = str(candidate.min_dist)
    env["REALANTSIM_NEURAL_MAX_PICKUP_DIST"] = str(candidate.max_dist)
    if args.quick:
        post_output = run_command(args, env, "post_pickup_regression")
        lost_output = run_command(args, env, "lost_carrier_regression")
        fitness = parse_quick_fitness(candidate, post_output, lost_output)
    else:
        output = run_command(args, env, "bench_default")
        fitness = parse_fitness(candidate, output)
    print(
        f"{candidate.name:<18} total={fitness.total:>5.0f} "
        f"pass={str(fitness.passed):<5} post={fitness.post:>4.0f} "
        f"lost={fitness.lost:>4.0f} blend={candidate.blend:.2f} "
        f"minTicks={candidate.min_ticks:>3} "
        f"dist={candidate.min_dist:.0f}-{candidate.max_dist:.0f}",
        flush=True,
    )
    return fitness


def candidate_grid() -> list[Candidate]:
    candidates = [Candidate("current", 1.0, 0, 300.0, 500.0)]
    for blend in [0.4, 0.55, 0.7, 0.85, 1.0]:
        for min_ticks in [0, 30, 90, 180, 300]:
            candidates.append(
                Candidate(
                    f"b{blend:.2f}_t{min_ticks}",
                    blend,
                    min_ticks,
                    300.0,
                    500.0,
                )
            )
    for max_dist in [430.0, 470.0, 540.0, 620.0]:
        candidates.append(Candidate(f"max{max_dist:.0f}", 1.0, 0, 300.0, max_dist))
    return candidates


def sort_key(fitness: Fitness) -> tuple[bool, float]:
    return (fitness.passed, fitness.total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--backend",
        type=Path,
        default=Path("target/release/ant-backend"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("run/neural_worker_weights.json"),
    )
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tune on post_pickup + lost_carrier before confirming with the full bench.",
    )
    args = parser.parse_args()

    candidates = candidate_grid()
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    results = [run_candidate(args, candidate) for candidate in candidates]
    best = max(results, key=sort_key)
    print(
        "\nBEST "
        f"name={best.candidate.name} total={best.total:.0f} pass={best.passed} "
        f"blend={best.candidate.blend:.2f} "
        f"minTicks={best.candidate.min_ticks} "
        f"minDist={best.candidate.min_dist:.0f} "
        f"maxDist={best.candidate.max_dist:.0f}"
    )


if __name__ == "__main__":
    main()
