"""One-game diagnostic: main.py (player 0) vs opp_producer.py (player 1).

Tracks per-player planets / production / ships / fleets over the whole game to
reveal WHERE M1 loses — early expansion, midgame combat, or late collapse.

Usage: python diag_game.py [seed]
"""
import contextlib
import os
import sys


@contextlib.contextmanager
def _silence():
    d = open(os.devnull, "w")
    s = (sys.stdout, sys.stderr)
    sys.stdout = sys.stderr = d
    try:
        yield
    finally:
        sys.stdout, sys.stderr = s
        d.close()


with _silence():
    from kaggle_environments import make

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def stats(obs):
    planets = getattr(obs, "planets", []) or []
    fleets = getattr(obs, "fleets", []) or []
    pl = {0: 0, 1: 0, -1: 0}
    prod = {0: 0, 1: 0}
    ships = {0: 0.0, 1: 0.0}
    fl = {0: 0, 1: 0}
    for p in planets:
        o = p[1]
        pl[o] = pl.get(o, 0) + 1
        if o in (0, 1):
            prod[o] += p[6]
            ships[o] += p[5]
    for f in fleets:
        o = f[1]
        if o in (0, 1):
            ships[o] += f[6]
            fl[o] = fl.get(o, 0) + 1
    return pl, prod, ships, fl


env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
with _silence():
    env.run(["main.py", "opp_producer.py"])

n = len(env.steps)
print(f"seed={SEED}  steps={n}   (P0 = main.py,  P1 = opp_producer.py)")
print(f"{'step':>4} | {'P0pl':>4} {'P1pl':>4} {'neu':>4} | {'P0prod':>6} {'P1prod':>6} | {'P0ship':>7} {'P1ship':>7} | {'P0flt':>5} {'P1flt':>5}")
print("-" * 78)
for k in range(n):
    if k % 10 == 0 or k == n - 1:
        pl, prod, ships, fl = stats(env.steps[k][0].observation)
        print(f"{k:>4} | {pl.get(0,0):>4} {pl.get(1,0):>4} {pl.get(-1,0):>4} | "
              f"{prod[0]:>6} {prod[1]:>6} | {ships[0]:>7.0f} {ships[1]:>7.0f} | {fl[0]:>5} {fl[1]:>5}")

final = env.steps[-1]
print("-" * 78)
print("final rewards:", [s.reward for s in final], " statuses:", [s.status for s in final])
