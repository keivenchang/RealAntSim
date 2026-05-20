#!/usr/bin/env python3
"""Parallel sweep for the weighted worker brain.

The Rust benches are the source of truth. This script runs targeted weighted
rollouts with different env-backed knobs, ranks them by hard-gate risk and
score, then optionally confirms the top candidates with `bench_default`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


WALL_RE = re.compile(
    r"wall_test .* score=(?P<score>[0-9.]+) deliveries=(?P<deliveries>[0-9]+) "
    r"route=(?P<route>[0-9.]+) wallPr=(?P<wall_pr>[0-9.]+) "
    r"bRet=(?P<bret>[0-9.]+) bFast=(?P<bfast>[0-9.]+) "
    r"bPrompt=(?P<bprompt>[0-9.]+).* "
    r"offRoute=(?P<offroute>[0-9.]+) offTrail=(?P<offtrail>[0-9.]+) "
    r"wallDz=(?P<wall_dz>[0-9.]+) offClump=(?P<offclump>[0-9.]+) "
    r"branch=(?P<branch>[0-9]+)"
)
CHECKPOINT_RE = re.compile(r"checkpoints=(?P<values>[0-9]+(?:->[0-9]+)+)")
ARC_RE = re.compile(
    r"arc_to_line .* score=(?P<score>[0-9.]+) deliveries=(?P<deliveries>[0-9]+) "
    r"straight=(?P<straight>[0-9.]+) arc=(?P<arc>[0-9.]+) "
    r"off=(?P<off>[0-9.]+) meanLineDist=(?P<dist>[0-9.]+) "
    r"direct=(?P<direct>[0-9.]+) chord=(?P<chord>[0-9.]+) "
    r"returnDist=(?P<return_dist>[0-9.]+) perfect=(?P<perfect>[0-9.]+) "
    r"scatter=(?P<scatter>[0-9.]+) branch=(?P<branch>[0-9]+)"
)
CLUSTER_RE = re.compile(
    r"cluster_escape .* score=(?P<score>[0-9.]+).* "
    r"sample=(?P<sample>[0-9.]+) final=(?P<final>[0-9.]+) "
    r"bottle=(?P<bottle_workers>[0-9.]+)/(?P<bottle_clump>[0-9.]+)/(?P<bottle_peak>[0-9.]+)"
)
FITNESS_RE = re.compile(
    r"FITNESS total=(?P<total>[0-9.]+) pass=(?P<passed>true|false) "
    r"wall=(?P<wall>[0-9.]+) arc=(?P<arc>[0-9.]+) multi=(?P<multi>[0-9.]+) "
    r"loop=(?P<loop>[0-9.]+) cycle=(?P<cycle>[0-9.]+) "
    r"postPickup=(?P<post>[0-9.]+) lostCarrier=(?P<lost>[0-9.]+) "
    r"cluster=(?P<cluster>[0-9.]+)"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    env: dict[str, str]


@dataclass(frozen=True)
class TargetResult:
    candidate: Candidate
    wall_score: float
    arc_score: float
    cluster_score: float
    gate_failures: tuple[str, ...]
    output: str

    @property
    def targeted_score(self) -> float:
        return self.wall_score + self.arc_score + self.cluster_score


@dataclass(frozen=True)
class FullResult:
    candidate: Candidate
    total: float
    passed: bool
    output: str


def f(match: re.Match[str], name: str) -> float:
    return float(match.group(name))


def run_command(args: argparse.Namespace, candidate: Candidate, command: str) -> str:
    env = os.environ.copy()
    env.update(candidate.env)
    env["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    proc = subprocess.run(
        [str(args.backend), command, "--brain", "weighted"],
        cwd=args.root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"{candidate.name} {command} failed\n{output}")
    return output


def parse_target(candidate: Candidate, wall: str, arc: str, cluster: str) -> TargetResult:
    wall_m = WALL_RE.search(wall)
    arc_m = ARC_RE.search(arc)
    cluster_m = CLUSTER_RE.search(cluster)
    checkpoints_m = CHECKPOINT_RE.search(wall)
    if wall_m is None or arc_m is None or cluster_m is None or checkpoints_m is None:
        raise RuntimeError(f"could not parse output for {candidate.name}\n{wall}\n{arc}\n{cluster}")

    checkpoints = [int(v) for v in checkpoints_m.group("values").split("->")]
    failures: list[str] = []
    if f(wall_m, "score") < 40.0:
        failures.append("wall_score")
    if checkpoints[1] < 120:
        failures.append("wall_8k")
    if checkpoints[2] < 300:
        failures.append("wall_12k")
    if f(wall_m, "wall_pr") > 250.0:
        failures.append("wall_pressure")
    if f(wall_m, "bfast") < 0.30:
        failures.append("wall_fast")
    if f(wall_m, "bprompt") < 0.55:
        failures.append("wall_prompt")
    if f(wall_m, "offroute") > 0.55:
        failures.append("wall_offroute")
    if f(wall_m, "offtrail") > 0.28:
        failures.append("wall_offtrail")
    if f(wall_m, "wall_dz") > 0.35:
        failures.append("wall_deadzone")
    if f(wall_m, "offclump") > 0.12:
        failures.append("wall_offclump")
    if int(wall_m.group("branch")) > 2500:
        failures.append("wall_branch")

    if f(arc_m, "score") < 40.0:
        failures.append("arc_score")
    if f(arc_m, "straight") < 0.75:
        failures.append("arc_straight")
    if f(arc_m, "dist") > 25.0:
        failures.append("arc_dist")
    if f(arc_m, "direct") > 0.08:
        failures.append("arc_direct")
    if f(arc_m, "scatter") > 0.80:
        failures.append("arc_scatter")

    if f(cluster_m, "final") > 0.15:
        failures.append("cluster_final")
    if f(cluster_m, "bottle_clump") > 0.06:
        failures.append("cluster_bottle_clump")
    if f(cluster_m, "bottle_peak") > 0.18:
        failures.append("cluster_bottle_peak")

    return TargetResult(
        candidate=candidate,
        wall_score=f(wall_m, "score"),
        arc_score=f(arc_m, "score"),
        cluster_score=f(cluster_m, "score"),
        gate_failures=tuple(failures),
        output=wall + arc + cluster,
    )


def run_target(args: argparse.Namespace, candidate: Candidate) -> TargetResult:
    wall = run_command(args, candidate, "wall_regression")
    arc = run_command(args, candidate, "arc_regression")
    cluster = run_command(args, candidate, "cluster_regression")
    result = parse_target(candidate, wall, arc, cluster)
    print(
        f"{candidate.name:<22} gates={len(result.gate_failures):>2} "
        f"target={result.targeted_score:>5.1f} wall={result.wall_score:>4.1f} "
        f"arc={result.arc_score:>4.1f} cluster={result.cluster_score:>5.1f} "
        f"{','.join(result.gate_failures[:5])}",
        flush=True,
    )
    return result


def run_full(args: argparse.Namespace, candidate: Candidate) -> FullResult:
    output = run_command(args, candidate, "bench_default")
    match = FITNESS_RE.search(output)
    if match is None:
        raise RuntimeError(f"could not parse full bench for {candidate.name}\n{output}")
    result = FullResult(
        candidate=candidate,
        total=float(match.group("total")),
        passed=match.group("passed") == "true",
        output=output,
    )
    print(
        f"FULL {candidate.name:<22} total={result.total:>5.1f} pass={result.passed}",
        flush=True,
    )
    return result


def candidate_grid() -> list[Candidate]:
    base = {
        "REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT": "2.2",
        "REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE": "1.30",
        "REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE": "0.95",
        "REALANTSIM_WEIGHTED_WALL_WEIGHT": "3.0",
        "REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_NEIGHBOR_AVOID_WEIGHT": "2.0",
        "REALANTSIM_WEIGHTED_BLOCKED_HOME_DEPOSIT_BAND": "62",
        "REALANTSIM_WEIGHTED_OPEN_RETURN_DOT": "0.75",
        "REALANTSIM_WEIGHTED_OPEN_RETURN_WEAVE": "1.25",
        "REALANTSIM_WEIGHTED_OPEN_RETURN_BLEND": "0.85",
        "REALANTSIM_WEIGHTED_OPEN_RETURN_NO_ROUTE_WEAVE": "1.05",
        "REALANTSIM_WEIGHTED_OPEN_RETURN_NO_ROUTE_BLEND": "0.75",
        "REALANTSIM_WEIGHTED_OPEN_LONG_FOOD_LAY_SCALE": "0.40",
        "REALANTSIM_WEIGHTED_LONG_RETURN_DIRECT_DOT": "0.90",
        "REALANTSIM_WEIGHTED_LONG_RETURN_BEND_BLEND": "0.75",
        "REALANTSIM_WEIGHTED_LONG_RETURN_FORWARD": "0.78",
        "REALANTSIM_WEIGHTED_LONG_RETURN_LATERAL": "0.63",
        "REALANTSIM_WEIGHTED_OPEN_ROUTE_DEVIATION_KEEP": "0.16",
        "REALANTSIM_WEIGHTED_OPEN_ROUTE_JITTER": "22.0",
    }

    def c(name: str, **updates: str) -> Candidate:
        env = dict(base)
        env.update(updates)
        return Candidate(name, env)

    candidates = [Candidate("current", base)]
    for band in ["45", "70", "95", "120", "150"]:
        candidates.append(
            c(
                f"best_band{band}",
                REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT="2.2",
                REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE="1.15",
                REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE="1.00",
                REALANTSIM_WEIGHTED_BLOCKED_HOME_DEPOSIT_BAND=band,
            )
        )
    for avoid in ["0.4", "0.8", "1.2", "1.6", "2.0"]:
        candidates.append(
            c(
                f"best_avoid{avoid}",
                REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT="2.2",
                REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE="1.15",
                REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE="1.00",
                REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_NEIGHBOR_AVOID_WEIGHT=avoid,
            )
        )
    for band in ["70", "95", "120"]:
        for avoid in ["0.4", "0.8", "1.2"]:
            candidates.append(
                c(
                    f"best_b{band}_a{avoid}",
                    REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT="2.2",
                    REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE="1.15",
                    REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE="1.00",
                    REALANTSIM_WEIGHTED_BLOCKED_HOME_DEPOSIT_BAND=band,
                    REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_NEIGHBOR_AVOID_WEIGHT=avoid,
                )
            )
    for jitter, route, smell, band, avoid in [
        ("2.0", "1.15", "1.00", "70", "0.8"),
        ("2.2", "1.00", "1.10", "70", "0.8"),
        ("2.2", "1.15", "0.95", "70", "0.8"),
        ("2.4", "1.15", "1.00", "70", "0.8"),
        ("2.2", "1.30", "1.00", "95", "0.8"),
        ("2.0", "1.30", "1.10", "95", "1.2"),
    ]:
        candidates.append(
            c(
                f"focus_j{jitter}_r{route}_s{smell}_b{band}_a{avoid}",
                REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT=jitter,
                REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE=route,
                REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE=smell,
                REALANTSIM_WEIGHTED_BLOCKED_HOME_DEPOSIT_BAND=band,
                REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_NEIGHBOR_AVOID_WEIGHT=avoid,
            )
        )
    for value in ["1.0", "1.2", "1.4", "1.8", "2.0", "2.4"]:
        candidates.append(c(f"jitter{value}", REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT=value))
    for value in ["0.7", "0.85", "1.15", "1.3", "1.5"]:
        candidates.append(c(f"route{value}", REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE=value))
    for value in ["0.85", "0.95", "1.10", "1.25"]:
        candidates.append(c(f"smell{value}", REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE=value))
    for value in ["2.6", "3.4", "3.8"]:
        candidates.append(c(f"wallW{value}", REALANTSIM_WEIGHTED_WALL_WEIGHT=value))
    for lay in ["0.35", "0.40", "0.45", "0.50"]:
        candidates.append(c(f"lay{lay}", REALANTSIM_WEIGHTED_OPEN_LONG_FOOD_LAY_SCALE=lay))
    for direct, blend, lateral in [
        ("0.86", "0.75", "0.63"),
        ("0.90", "0.65", "0.63"),
        ("0.90", "0.85", "0.63"),
        ("0.92", "0.75", "0.70"),
        ("0.88", "0.80", "0.75"),
    ]:
        candidates.append(
            c(
                f"bend{direct}_{blend}_{lateral}",
                REALANTSIM_WEIGHTED_LONG_RETURN_DIRECT_DOT=direct,
                REALANTSIM_WEIGHTED_LONG_RETURN_BEND_BLEND=blend,
                REALANTSIM_WEIGHTED_LONG_RETURN_LATERAL=lateral,
            )
        )
    for jitter, route, smell in [
        ("1.8", "1.15", "1.00"),
        ("2.0", "1.15", "1.00"),
        ("2.0", "1.30", "1.10"),
        ("2.4", "1.30", "1.10"),
        ("1.4", "1.30", "1.10"),
        ("1.8", "0.85", "0.95"),
    ]:
        candidates.append(
            c(
                f"combo_j{jitter}_r{route}_s{smell}",
                REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT=jitter,
                REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE=route,
                REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE=smell,
            )
        )
    for jitter in ["1.8", "2.0", "2.2", "2.4", "2.6", "2.8"]:
        for route in ["1.00", "1.15", "1.30", "1.45", "1.60"]:
            for smell in ["0.95", "1.00", "1.10", "1.20"]:
                candidates.append(
                    c(
                        f"grid_j{jitter}_r{route}_s{smell}",
                        REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT=jitter,
                        REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE=route,
                        REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE=smell,
                    )
                )
    for jitter, route, smell, wall_weight in [
        ("2.2", "1.30", "1.10", "2.7"),
        ("2.4", "1.30", "1.10", "2.7"),
        ("2.6", "1.30", "1.10", "2.7"),
        ("2.2", "1.45", "1.10", "2.8"),
        ("2.4", "1.45", "1.10", "2.8"),
        ("2.6", "1.45", "1.10", "2.8"),
        ("2.2", "1.30", "1.20", "3.2"),
        ("2.4", "1.30", "1.20", "3.2"),
        ("2.6", "1.30", "1.20", "3.2"),
    ]:
        candidates.append(
            c(
                f"wallgrid_j{jitter}_r{route}_s{smell}_w{wall_weight}",
                REALANTSIM_WEIGHTED_WALL_NO_SIGNAL_JITTER_MULT=jitter,
                REALANTSIM_WEIGHTED_WALL_ROUTE_MEMORY_SCALE=route,
                REALANTSIM_WEIGHTED_WALL_FOOD_SMELL_SCALE=smell,
                REALANTSIM_WEIGHTED_WALL_WEIGHT=wall_weight,
            )
        )
    return candidates


def target_sort_key(result: TargetResult) -> tuple[int, float, float]:
    return (len(result.gate_failures), -result.targeted_score, -result.wall_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--backend", type=Path, default=Path("target/release/ant-backend"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--rayon-threads", type=int, default=4)
    parser.add_argument("--top-full", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args()

    candidates = candidate_grid()
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    print(
        f"targeted sweep candidates={len(candidates)} jobs={args.jobs} "
        f"rayon_threads={args.rayon_threads}",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_target, args, candidate) for candidate in candidates]
        targeted = [future.result() for future in concurrent.futures.as_completed(futures)]

    targeted.sort(key=target_sort_key)
    print("\nTOP TARGETED")
    for result in targeted[: args.top_full]:
        print(
            f"{result.candidate.name:<22} gates={len(result.gate_failures):>2} "
            f"target={result.targeted_score:>5.1f} env={result.candidate.env}"
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.jobs, args.top_full)) as pool:
        futures = [
            pool.submit(run_full, args, result.candidate)
            for result in targeted[: args.top_full]
        ]
        full_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    full_results.sort(key=lambda result: (not result.passed, -result.total))
    print("\nTOP FULL")
    for result in full_results:
        print(
            f"{result.candidate.name:<22} total={result.total:>5.1f} "
            f"pass={result.passed} env={result.candidate.env}"
        )


if __name__ == "__main__":
    main()
