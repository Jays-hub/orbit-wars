"""
arena.py — Orbit Wars self-play harness (M2: the iteration engine).

Runs many games between two agents and reports win-rate with a confidence
interval, so M3 parameter tuning is judged on signal, not noise.

Why it's built this way
-----------------------
* Antithetic seat-swap (variance reduction). The board is generated from
  `random.Random(seed)` independently of agent order, and seat 0 starts in Q1
  while seat 1 starts in Q4. So playing the SAME seed once with A in seat 0 and
  once with A in seat 1 is the *same map with swapped homes* — it cancels
  positional bias exactly and reduces board-to-board variance (common random
  numbers). In 4P, A is rotated through all four seats on each seed.
* Defensive outcome reading. A crashed / timed-out agent gets status
  ERROR/TIMEOUT/INVALID and reward None (kaggle_environments core). The arena
  scores that as a LOSS for the failing agent and flags it loudly — on the
  ladder a forfeit is an automatic loss, the only thing that drops rating.
* Reward semantics. The engine sets reward +1 for every agent at the top score
  (>0), else -1. So a 2P score tie shows up as both-+1 and is counted a draw.

Usage
-----
    python arena.py                                  # main.py vs starter, 100 games, 2P
    python arena.py --a main.py --b cand.py --games 300
    python arena.py --players 4 --b starter --games 200
    python arena.py --a main.py --b starter --games 40 --steps 150   # fast smoke

API (for M3)
------------
    from arena import run_arena
    res = run_arena(a="main.py", b="cand.py", games=300)
    # res["score"], res["ci"], res["los"], res["significant"] ...
Agent specs are built-in names ("random"/"starter") or paths to agent .py
files. `_resolve_spec` is the single extension point for M3 parameter
injection (resolve a (path, params) spec to a callable *inside the worker*, so
nothing unpicklable crosses a process boundary).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


@contextlib.contextmanager
def _silence_stdio():
    """Bind stdout/stderr to devnull around the kaggle_environments import so
    its banner (cabt dlopen failure + OpenSpiel INFO logging, both emitted at
    import time) doesn't bury the arena report — including in spawned workers,
    which re-import this module."""
    devnull = open(os.devnull, "w")
    saved = (sys.stdout, sys.stderr)
    sys.stdout = sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved
        devnull.close()


with _silence_stdio():
    from kaggle_environments import make


FAIL_STATUSES = {"ERROR", "TIMEOUT", "INVALID"}
BUILTINS = {"random", "starter"}

# z-scores for common two-sided confidence levels (avoids a scipy dependency).
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}


def _z_for(confidence):
    return _Z.get(round(confidence, 2), 1.9600)


def _resolve_spec(spec):
    """Turn an agent spec into something env.run accepts (a built-in name or a
    file path — both are strings kaggle loads directly). Kept as an explicit
    indirection so M3 can add a parameter-injected variant here, evaluated
    inside the worker process."""
    return spec


# --------------------------------------------------------------------------- #
# One game                                                                     #
# --------------------------------------------------------------------------- #
def play_match(seats, a_seat, seed, episode_steps=None):
    """Play a single game. `seats` lists the agent spec per seat in seat order;
    A occupies index `a_seat`. Returns the raw per-seat rewards/statuses plus
    timing — interpretation happens in `classify`."""
    config = {"seed": seed}
    if episode_steps is not None:
        config["episodeSteps"] = episode_steps
    env = make("orbit_wars", configuration=config, debug=False)
    agents = [_resolve_spec(s) for s in seats]
    t0 = time.perf_counter()
    env.run(agents)
    duration = time.perf_counter() - t0
    final = env.steps[-1]
    n_seats = len(final)

    # Scan EVERY step, not just the last: the engine's terminal block sets all
    # agents to status DONE on game end, overwriting any earlier ERROR/TIMEOUT/
    # INVALID. So a bot that crashed mid-game (and lost its planets to
    # elimination, or even kept a frozen board) looks DONE at the final step.
    # Scanning the whole history is the only reliable way to catch a failure.
    seat_failed = [False] * n_seats
    seat_fail_kind = [None] * n_seats
    for step_state in env.steps:
        for seat in range(n_seats):
            st = getattr(step_state[seat], "status", None)
            if st in FAIL_STATUSES and not seat_failed[seat]:
                seat_failed[seat] = True
                seat_fail_kind[seat] = st

    # Final ship totals per seat (planets + fleets), same as the engine's score.
    # The ladder ignores margin, but it's a low-variance tuning proxy for M3:
    # it gives a gradient even when the win/draw/loss outcome is a tie.
    scores = [0] * n_seats
    try:
        obs = final[0].observation
        for p in getattr(obs, "planets", []) or []:
            if 0 <= p[1] < n_seats:
                scores[p[1]] += p[5]
        for f in getattr(obs, "fleets", []) or []:
            if 0 <= f[1] < n_seats:
                scores[f[1]] += f[6]
    except Exception:
        scores = None

    actual_seed = None
    try:
        actual_seed = env.info.get("seed")
    except Exception:
        pass
    return {
        "seed": seed,
        "actual_seed": actual_seed,
        "a_seat": a_seat,
        "rewards": [s.reward for s in final],
        "statuses": [s.status for s in final],
        "seat_failed": seat_failed,
        "seat_fail_kind": seat_fail_kind,
        "scores": scores,
        "steps": len(env.steps),
        "duration": duration,
    }


def _worker(task):
    """Process-pool entry point. Never raises: a harness-level failure is
    returned as an error record so one bad game can't kill the run."""
    try:
        res = play_match(task["seats"], task["a_seat"], task["seed"], task.get("episode_steps"))
        res["gid"] = task["gid"]
        res["error"] = None
        return res
    except Exception as e:
        return {
            "gid": task["gid"],
            "seed": task["seed"],
            "a_seat": task["a_seat"],
            "error": "".join(traceback.format_exception_only(type(e), e)).strip(),
        }


# --------------------------------------------------------------------------- #
# Outcome classification                                                       #
# --------------------------------------------------------------------------- #
def classify(res, players):
    """Return (outcome, a_failed, b_failed) from A's perspective, where
    outcome is 'win' | 'loss' | 'draw'. An ever-failed bot (crash/timeout at
    any step) is treated as a flagged loss — even if it 'won' on a frozen
    board, it would forfeit on the ladder — so failures never become fake wins."""
    rewards, a = res["rewards"], res["a_seat"]
    sf = res["seat_failed"]
    a_failed = sf[a]

    if players == 2:
        b = 1 - a
        b_failed = sf[b]
        if a_failed and not b_failed:
            return "loss", a_failed, b_failed
        if b_failed and not a_failed:
            return "win", a_failed, b_failed
        if a_failed and b_failed:
            return "draw", a_failed, b_failed
        ra, rb = rewards[a], rewards[b]
        if ra > rb:
            return "win", False, False
        if rb > ra:
            return "loss", False, False
        return "draw", False, False

    # 4 players: A vs a field of B. A "win" is finishing as the sole top scorer.
    b_failed = any(sf[s] for s in range(len(sf)) if s != a)
    if a_failed:
        return "loss", True, b_failed
    if rewards[a] != 1:
        return "loss", a_failed, b_failed
    top = sum(1 for r in rewards if r == 1)
    return ("win" if top == 1 else "draw"), a_failed, b_failed


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion. `successes` may be
    fractional (draws folded in as 0.5), which keeps draws inside the CI."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - margin), min(1.0, center + margin)


def los(wins, losses):
    """Likelihood of Superiority: P(A truly stronger than B) from decisive
    games only. Standard game-testing estimator Phi((W-L)/sqrt(W+L))."""
    d = wins + losses
    if d == 0:
        return 0.5
    return _phi((wins - losses) / math.sqrt(d))


# --------------------------------------------------------------------------- #
# Scheduling + driving                                                         #
# --------------------------------------------------------------------------- #
def build_tasks(a, b, games, players, base_seed, episode_steps):
    """Balanced, seat-swapped schedule. 2P: each seed played twice (A seat 0,
    then A seat 1). 4P: A rotated through all four seats on each seed."""
    tasks = []
    for i in range(games):
        if players == 2:
            seed = base_seed + i // 2
            a_seat = i % 2
            seats = [None, None]
            seats[a_seat] = a
            seats[1 - a_seat] = b
        else:
            seed = base_seed + i // 4
            a_seat = i % 4
            seats = [b, b, b, b]
            seats[a_seat] = a
        tasks.append(
            {"gid": i, "seed": seed, "a_seat": a_seat, "seats": seats, "episode_steps": episode_steps}
        )
    return tasks


def run_arena(a="main.py", b="starter", games=100, players=2, base_seed=12345,
              workers=None, episode_steps=None, confidence=0.95, quiet=False):
    """Run `games` games of A vs B and return an aggregated result dict."""
    if players not in (2, 4):
        raise ValueError("players must be 2 or 4")
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(workers, games))
    z = _z_for(confidence)

    tasks = build_tasks(a, b, players=players, games=games, base_seed=base_seed,
                        episode_steps=episode_steps)

    results, harness_errors = [], []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            (harness_errors if res.get("error") else results).append(res)
            if not quiet:
                print(f"\r  played {done}/{games} games...", end="", flush=True)
    if not quiet:
        print()
    wall = time.perf_counter() - t0

    wins = draws = losses = 0
    a_fail = b_fail = 0
    a_fail_kinds, durations, margins = {}, [], []
    for r in results:
        outcome, af, bf = classify(r, players)
        wins += outcome == "win"
        draws += outcome == "draw"
        losses += outcome == "loss"
        durations.append(r["duration"])
        sc, aseat = r.get("scores"), r["a_seat"]
        if sc is not None:
            others = [sc[s] for s in range(len(sc)) if s != aseat]
            margins.append(sc[aseat] - (max(others) if others else 0))
        if af:
            a_fail += 1
            st = r["seat_fail_kind"][r["a_seat"]] or "FAIL"
            a_fail_kinds[st] = a_fail_kinds.get(st, 0) + 1
        if bf:
            b_fail += 1

    n = wins + draws + losses
    successes = wins + 0.5 * draws
    score, lo, hi = wilson(successes, n, z) if n else (0.0, 0.0, 0.0)
    decisive_los = los(wins, losses)
    if n == 0:
        significant = "no-data"
    elif lo > 0.5:
        significant = "better"
    elif hi < 0.5:
        significant = "worse"
    else:
        significant = "inconclusive"

    return {
        "a": a, "b": b, "players": players, "games_requested": games, "games_scored": n,
        "base_seed": base_seed, "episode_steps": episode_steps, "workers": workers,
        "wins": wins, "draws": draws, "losses": losses,
        "score": score, "ci": (lo, hi), "confidence": confidence, "los": decisive_los,
        "significant": significant,
        "a_failures": a_fail, "a_failure_kinds": a_fail_kinds, "b_failures": b_fail,
        "harness_errors": harness_errors,
        "mean_margin": (sum(margins) / len(margins)) if margins else None,
        "wall": wall, "mean_game_s": (sum(durations) / len(durations)) if durations else 0.0,
        "max_game_s": max(durations) if durations else 0.0,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def print_report(r):
    pct = lambda x: f"{100 * x:5.1f}%"
    lo, hi = r["ci"]
    conf = int(round(r["confidence"] * 100))
    print("=" * 64)
    print(f"  {r['a']}   vs   {r['b']}        ({r['players']}P)")
    print("=" * 64)
    if r["episode_steps"] is not None:
        print(f"  ! episodeSteps={r['episode_steps']} (NOT the ladder's 500 — smoke test only)")
    print(f"  games        : {r['games_scored']} scored / {r['games_requested']} requested"
          f"   (seeds {r['base_seed']}..{r['base_seed'] + r['games_requested'] - 1})")
    print(f"  W / D / L    : {r['wins']} / {r['draws']} / {r['losses']}   (A's perspective)")
    print(f"  win-rate     : {pct(r['score'])}   [{conf}% CI {pct(lo)} .. {pct(hi)}]   (draws = 1/2)")
    print(f"  LOS          : {pct(r['los'])}   P(A truly stronger than B), decisive games")
    if r.get("mean_margin") is not None:
        print(f"  score margin : {r['mean_margin']:+.1f} avg ships (A - best opp; ladder ignores it, M3 tuning proxy)")
    verdict = {
        "better": f"A is significantly BETTER than B (CI lower bound > 50%)",
        "worse": f"A is significantly WORSE than B (CI upper bound < 50%)",
        "inconclusive": "inconclusive — CI straddles 50%, run more games",
        "no-data": "no games scored",
    }[r["significant"]]
    print(f"  verdict      : {verdict}")
    print("-" * 64)
    flag = "  <-- A FAILED, investigate" if r["a_failures"] else ""
    kinds = (" " + ", ".join(f"{k}:{v}" for k, v in r["a_failure_kinds"].items())) if r["a_failure_kinds"] else ""
    print(f"  reliability  : A failures {r['a_failures']}{kinds}{flag}   |   B failures {r['b_failures']}")
    if r["harness_errors"]:
        print(f"  HARNESS ERR  : {len(r['harness_errors'])} game(s) errored in the runner")
        print(f"                 e.g. {r['harness_errors'][0].get('error')}")
    print(f"  timing       : {r['mean_game_s']:.2f}s/game mean, {r['max_game_s']:.2f}s max"
          f"   |   {r['wall']:.1f}s wall, {r['workers']} workers"
          f"   ({r['games_scored'] / r['wall']:.2f} games/s)" if r["wall"] else "")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Orbit Wars self-play arena (M2).")
    ap.add_argument("--a", default="main.py", help="agent A: path to .py or 'random'/'starter' (default main.py)")
    ap.add_argument("--b", default="starter", help="agent B / opponent (default starter)")
    ap.add_argument("--games", type=int, default=100, help="number of games (default 100)")
    ap.add_argument("--players", type=int, default=2, choices=(2, 4), help="2 (A vs B) or 4 (A vs 3xB)")
    ap.add_argument("--seed", type=int, default=12345, help="base seed (default 12345)")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default cpus-1)")
    ap.add_argument("--steps", type=int, default=None, help="episodeSteps override (default 500; use only for fast smoke tests)")
    ap.add_argument("--confidence", type=float, default=0.95, help="CI confidence level (default 0.95)")
    ap.add_argument("--quiet", action="store_true", help="suppress the live progress counter")
    args = ap.parse_args()

    r = run_arena(a=args.a, b=args.b, games=args.games, players=args.players,
                  base_seed=args.seed, workers=args.workers, episode_steps=args.steps,
                  confidence=args.confidence, quiet=args.quiet)
    print_report(r)


if __name__ == "__main__":
    main()
