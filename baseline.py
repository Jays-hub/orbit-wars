"""
Orbit Wars — simulation-guided heuristic agent (M1).

This file IS the submission. Kaggle loads it the same way run.py does, so what
you test locally is what you submit.

Architecture (see CLAUDE.md "winning architecture"):

  1. Parse the observation into planets + in-flight fleets.
  2. Forward-simulate the board (a faithful re-implementation of the engine:
     launch -> production -> fleet movement w/ continuous swept collision ->
     planet rotation & comet motion -> combat). Fleet speed scales with size,
     so arrival turns are emergent, not guessed.
  3. From a no-new-orders baseline sim, derive each owned planet's keep_needed
     (defense reserve) and therefore its dispatchable surplus.
  4. Generate candidate dispatches (captures + reinforcements), lead-targeting
     moving planets and sizing each fleet to the minimum sufficient amount.
  5. Score every candidate by forward simulation and greedily commit the best
     non-conflicting set (the "guided" edge over static rules).
  6. Reliability gate: never crash, never blow the per-turn time budget.

Everything tunable lives in PARAMS (M3 will optimize these via self-play).
"""

import math
import time

# ---------------------------------------------------------------------------
# Engine constants (mirrored from kaggle_environments orbit_wars.py).
# ---------------------------------------------------------------------------
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_MAX_SPEED = 6.0
LOG1000 = math.log(1000.0)

# Planet record indices (we use plain lists for speed in the hot sim loop).
PID, POWN, PX, PY, PRAD, PSHIP, PPROD, PRR, PANG, PROT, PCOM, PCPATH, PCIDX = range(13)
# Fleet record indices.
FOWN, FX, FY, FANG, FSHIP, FSPD = range(6)

# ---------------------------------------------------------------------------
# Tunable parameters. Keep these in one place so the arena (M2) and tuner (M3)
# can sweep them without touching logic.
# ---------------------------------------------------------------------------
PARAMS = {
    "horizon": 55,            # ticks to forward-simulate when scoring
    "time_budget": 0.70,      # seconds; stop expanding the search past this
    "prod_value": 18.0,       # value of 1 production vs 1 ship (production compounds)
    "base_reserve": 1,        # always hold at least this many ships on a planet
    "def_margin": 2,          # extra defenders beyond measured incoming pressure
    "attack_margin": 2,       # extra ships beyond the minimum needed to capture
    "min_gain": 0.5,          # ignore candidate moves whose simulated gain is tiny
    "max_attack_pairs": 14,   # cap (source,target) pairs we pay to aim+score
    "targets_per_source": 6,  # nearest targets considered per owned planet
    "weak_mult": 0.6,         # 4P: how much harder to weight the weakest opponent
    "elim_bonus": 60.0,       # value bonus per opponent reduced to zero
    # Adaptive aggression (Port A): when we fall behind the strongest opponent
    # we must NOT hoard — free defensive reserves and commit lower-gain moves so
    # we keep contesting/expanding instead of being ground down passively.
    "behind_reserve_cut": 0.6,  # when far behind, cut defensive reserves by up to this fraction
    "behind_gain_cut": 0.9,     # when far behind, cut the min-gain commit bar by up to this fraction
    "max_speed": DEFAULT_MAX_SPEED,
    "aim_iters": 4,           # size<->aim refinement iterations
    "aim_scan": (-0.06, -0.03, 0.0, 0.03, 0.06),  # lead-angle search for movers
    "aim_max_ticks": 80,      # give up aiming past this travel time
}


# ---------------------------------------------------------------------------
# Geometry / engine math (copied to match the interpreter exactly).
# ---------------------------------------------------------------------------
def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to segment v-w (sun-collision test)."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return math.hypot(p[0] - v[0], p[1] - v[1])
    t = max(0.0, min(1.0, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    px = v[0] + t * (w[0] - v[0])
    py = v[1] + t * (w[1] - v[1])
    return math.hypot(p[0] - px, p[1] - py)


def swept_pair_hit(A, B, P0, P1, r):
    """True iff a fleet A->B and a planet P0->P1 come within r over t in [0,1].
    Identical to the engine's continuous collision test."""
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def fleet_speed(ships, max_speed):
    """Engine speed curve: 1 ship -> 1.0/turn, ramps to max_speed near 1000."""
    if ships <= 1:
        return 1.0
    s = 1.0 + (max_speed - 1.0) * (math.log(ships) / LOG1000) ** 1.5
    return s if s < max_speed else max_speed


# ---------------------------------------------------------------------------
# Observation parsing.
# ---------------------------------------------------------------------------
def _get(obs, key, default):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def parse_obs(obs, max_speed):
    """Turn the raw observation into sim-ready planet/fleet records.

    Works whether obs is a dict (Kaggle / local file agent) or an attribute
    object. Comet planets carry their full path + current index so the sim can
    advance and expire them exactly like the engine.
    """
    player = int(_get(obs, "player", 0))
    av = float(_get(obs, "angular_velocity", 0.0))
    raw_planets = _get(obs, "planets", []) or []
    raw_fleets = _get(obs, "fleets", []) or []
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
    comets = _get(obs, "comets", []) or []

    # pid -> (path, current_index) for comets. Current position == path[index].
    comet_lookup = {}
    for g in comets:
        idx = g["path_index"] if isinstance(g, dict) else g.path_index
        pids = g["planet_ids"] if isinstance(g, dict) else g.planet_ids
        paths = g["paths"] if isinstance(g, dict) else g.paths
        for i, pid in enumerate(pids):
            comet_lookup[pid] = (paths[i], idx)

    planets = []
    for p in raw_planets:
        pid, own, x, y, rad, ships, prod = p[0], p[1], float(p[2]), float(p[3]), float(p[4]), int(p[5]), p[6]
        if pid in comet_ids:
            cpath, cidx = comet_lookup.get(pid, (None, 0))
            planets.append([pid, own, x, y, rad, ships, prod, 0.0, 0.0, False, True, cpath, cidx])
        else:
            rr = math.hypot(x - CENTER, y - CENTER)
            rotating = (rr + rad) < ROTATION_RADIUS_LIMIT
            ang = math.atan2(y - CENTER, x - CENTER) if rotating else 0.0
            planets.append([pid, own, x, y, rad, ships, prod, rr, ang, rotating, False, None, 0])

    fleets = []
    for f in raw_fleets:
        own, x, y, angle, ships = f[1], float(f[2]), float(f[3]), float(f[4]), int(f[6])
        fleets.append([own, x, y, angle, ships, fleet_speed(ships, max_speed)])

    return player, av, planets, fleets


def clone_planets(planets):
    # Shallow-copy each record; cpath is read-only and safely shared.
    return [p[:] for p in planets]


def clone_fleets(fleets):
    return [f[:] for f in fleets]


# ---------------------------------------------------------------------------
# Forward simulator. One advance() == one engine tick, in engine order. We do
# not model opponents launching *new* fleets (unknowable) — only the fleets
# already in flight, plus our own candidate launch on tick 1.
# ---------------------------------------------------------------------------
# Safe upper bound on how far a planet/comet can move in one tick (rotating
# chord <= r*av <= ~2.5; comet path step ~= cometSpeed ~= 4). 7 is generous and
# only used as a collision pre-filter, so it never skips a real hit.
_PLANET_STEP_BOUND = 7.0


def simulate(planets, fleets, av, me, max_speed, horizon, launch=None, arrivals=None):
    """Advance a *copy* of the board `horizon` ticks. Returns (planets, fleets).

    launch:    optional list of [from_pid, angle, ships] applied on tick 1.
    arrivals:  optional dict pid -> list; if given, every fleet that reaches a
               planet is recorded as (tick, owner, ships) (used for keep_needed).
    """
    planets = clone_planets(planets)
    fleets = clone_fleets(fleets)

    # Tick-1 launch (engine: launch happens before production, fleet then moves
    # this same tick from the planet's current position).
    if launch:
        pmap = {p[PID]: p for p in planets}
        for mv in launch:
            pid, angle, ships = mv[0], mv[1], int(mv[2])
            p = pmap.get(pid)
            if p is None or p[POWN] != me or ships <= 0 or p[PSHIP] < ships:
                continue
            p[PSHIP] -= ships
            sx = p[PX] + math.cos(angle) * (p[PRAD] + 0.1)
            sy = p[PY] + math.sin(angle) * (p[PRAD] + 0.1)
            fleets.append([me, sx, sy, angle, ships, fleet_speed(ships, max_speed)])

    cos, sin, hypot = math.cos, math.sin, math.hypot
    n_comets = 0
    for p in planets:
        if p[PCOM]:
            n_comets += 1
    for t in range(1, horizon + 1):
        # Fast-forward only when nothing can change positions/ownership: no
        # fleets in flight AND no comets left to move/expire. (Comets keep
        # flying off-board and taking their garrison with them, so we must run
        # full ticks while any remain.)
        if not fleets and n_comets == 0 and arrivals is None:
            for p in planets:
                if p[POWN] != -1:
                    p[PSHIP] += p[PPROD]
            continue

        # (1) Production for every owned planet (incl. owned comets).
        for p in planets:
            if p[POWN] != -1:
                p[PSHIP] += p[PPROD]

        # (2) Compute each planet's old/new position for this tick.
        paths = {}
        expiring = set()
        for p in planets:
            old = (p[PX], p[PY])
            if p[PCOM]:
                p[PCIDX] += 1
                cpath = p[PCPATH]
                if cpath is None or p[PCIDX] >= len(cpath):
                    expiring.add(p[PID])
                    paths[p[PID]] = (old, old, True)
                else:
                    np_ = cpath[p[PCIDX]]
                    paths[p[PID]] = (old, (np_[0], np_[1]), old[0] >= 0.0)
            elif p[PROT]:
                p[PANG] += av
                nx = CENTER + p[PRR] * cos(p[PANG])
                ny = CENTER + p[PRR] * sin(p[PANG])
                paths[p[PID]] = (old, (nx, ny), True)
            else:
                paths[p[PID]] = (old, old, True)

        # (3) Move fleets with continuous swept-pair collision detection.
        combat = {}
        survivors = []
        for f in fleets:
            sp = f[FSPD]
            ang = f[FANG]
            ox, oy = f[FX], f[FY]
            nx = ox + cos(ang) * sp
            ny = oy + sin(ang) * sp
            f[FX], f[FY] = nx, ny
            oldf = (ox, oy)
            newf = (nx, ny)

            hit = None
            reach = sp + _PLANET_STEP_BOUND
            for p in planets:
                path = paths[p[PID]]
                if not path[2]:
                    continue
                po = path[0]
                dx = ox - po[0]
                dy = oy - po[1]
                rr = p[PRAD] + reach
                if dx * dx + dy * dy > rr * rr:
                    continue
                if swept_pair_hit(oldf, newf, po, path[1], p[PRAD]):
                    hit = p[PID]
                    break
            if hit is not None:
                combat.setdefault(hit, []).append(f)
                continue
            if not (0.0 <= nx <= BOARD_SIZE and 0.0 <= ny <= BOARD_SIZE):
                continue
            if point_to_segment_distance((CENTER, CENTER), oldf, newf) < SUN_RADIUS:
                continue
            survivors.append(f)
        fleets = survivors

        # (4) Apply planet movement, then drop expired comets.
        for p in planets:
            p[PX], p[PY] = paths[p[PID]][1]
        if expiring:
            planets = [p for p in planets if p[PID] not in expiring]
            n_comets -= len(expiring)

        # (5) Combat resolution (engine-identical).
        if combat:
            pmap = {p[PID]: p for p in planets}
            for pid, flist in combat.items():
                if arrivals is not None:
                    alist = arrivals.get(pid)
                    if alist is None:
                        alist = arrivals[pid] = []
                    for f in flist:
                        alist.append((t, f[FOWN], f[FSHIP]))
                planet = pmap.get(pid)
                if planet is None:  # expired comet -> ships vanish ("black hole")
                    continue
                ps = {}
                for f in flist:
                    ps[f[FOWN]] = ps.get(f[FOWN], 0) + f[FSHIP]
                ordered = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)
                top_owner, top_ships = ordered[0]
                if len(ordered) > 1:
                    second = ordered[1][1]
                    surv = top_ships - second
                    if top_ships == second:
                        surv = 0
                    surv_owner = top_owner if surv > 0 else -1
                else:
                    surv_owner, surv = top_owner, top_ships
                if surv > 0:
                    if planet[POWN] == surv_owner:
                        planet[PSHIP] += surv
                    else:
                        planet[PSHIP] -= surv
                        if planet[PSHIP] < 0:
                            planet[POWN] = surv_owner
                            planet[PSHIP] = -planet[PSHIP]

    return planets, fleets


# ---------------------------------------------------------------------------
# Target-motion prediction + single-fleet aiming (cheap, ignores other planets;
# the full-board sim is the final judge of whether a shot actually lands).
# ---------------------------------------------------------------------------
def predict_pos(T, k, av):
    """Position of planet/comet T k ticks from now, or None if a comet expired."""
    if T[PCOM]:
        cpath = T[PCPATH]
        idx = T[PCIDX] + k
        if cpath is None or idx >= len(cpath):
            return None
        if idx < 0:
            return (T[PX], T[PY])
        return (cpath[idx][0], cpath[idx][1])
    if T[PROT]:
        ang = T[PANG] + av * k
        return (CENTER + T[PRR] * math.cos(ang), CENTER + T[PRR] * math.sin(ang))
    return (T[PX], T[PY])


def single_fleet_arrival(start, srad, angle, speed, T, av, max_ticks):
    """Tick at which a lone fleet from `start` at `angle` would hit T, else None."""
    ca, sa = math.cos(angle), math.sin(angle)
    fx = start[0] + ca * (srad + 0.1)
    fy = start[1] + sa * (srad + 0.1)
    rad = T[PRAD]
    for k in range(1, max_ticks + 1):
        ox, oy = fx, fy
        fx += ca * speed
        fy += sa * speed
        p_old = predict_pos(T, k - 1, av)
        p_new = predict_pos(T, k, av)
        if p_new is None or p_old is None:
            return None
        if swept_pair_hit((ox, oy), (fx, fy), p_old, p_new, rad):
            return k
        if not (0.0 <= fx <= BOARD_SIZE and 0.0 <= fy <= BOARD_SIZE):
            return None
        if point_to_segment_distance((CENTER, CENTER), (ox, oy), (fx, fy)) < SUN_RADIUS:
            return None
    return None


def aim_and_size(S, T, av, max_speed, attack_margin, max_ticks):
    """Return (angle, arrival_tick, size) to capture T from S, or None.

    Jointly refines fleet size (-> speed -> arrival -> garrison-on-arrival ->
    size). For moving targets it scans a small window of lead angles and keeps
    the one that lands soonest.
    """
    sx, sy, srad = S[PX], S[PY], S[PRAD]
    moving = T[PROT] or T[PCOM]
    size = max(1, T[PSHIP] + 1)
    result = None
    for _ in range(PARAMS["aim_iters"]):
        speed = fleet_speed(size, max_speed)
        # Analytic lead: iterate arrival estimate against predicted position.
        k_guess = 1
        for _ in range(5):
            pos = predict_pos(T, k_guess, av) or (T[PX], T[PY])
            d = math.hypot(pos[0] - sx, pos[1] - sy)
            k_guess = max(1, int(round(d / speed)))
        lead_pos = predict_pos(T, k_guess, av) or (T[PX], T[PY])
        lead_ang = math.atan2(lead_pos[1] - sy, lead_pos[0] - sx)

        scan = PARAMS["aim_scan"] if moving else (0.0,)
        best_k = None
        best_ang = None
        for da in scan:
            ang = lead_ang + da
            k = single_fleet_arrival((sx, sy), srad, ang, speed, T, av, max_ticks)
            if k is not None and (best_k is None or k < best_k):
                best_k, best_ang = k, ang
        if best_k is None:
            return None

        garrison = T[PSHIP] + (T[PPROD] * best_k if T[POWN] != -1 else 0)
        needed = garrison + 1 + attack_margin
        result = (best_ang, best_k, needed)
        if needed == size:
            break
        size = max(1, needed)
    return result


# ---------------------------------------------------------------------------
# Defense accounting + board valuation.
# ---------------------------------------------------------------------------
def compute_keep_needed(planets, arrivals, me, horizon, p):
    """Per owned-planet ships we must retain to hold it through the horizon.

    Built from the no-orders baseline: enemy ships arriving at the planet are
    pressure; our own incoming fleets + production before the first threat are
    relief. Conservative (sums all enemy arrivals -> over-defends in 4P), which
    is the right bias on a win/loss ladder.
    """
    base = p["base_reserve"]
    margin = p["def_margin"]
    keep = {}
    for pl in planets:
        if pl[POWN] != me:
            continue
        pid = pl[PID]
        evs = arrivals.get(pid)
        if not evs:
            keep[pid] = base
            continue
        enemy = 0
        friend = 0
        first_enemy = horizon
        for (tk, owner, ships) in evs:
            if owner == me:
                friend += ships
            elif owner != -1:
                enemy += ships
                if tk < first_enemy:
                    first_enemy = tk
        if enemy <= 0:
            keep[pid] = base
            continue
        growth = pl[PPROD] * first_enemy
        need = enemy - friend - growth + margin
        keep[pid] = max(base, need)
    return keep


def board_value(planets, fleets, me, owner_weights, opponents, prod_value, elim_bonus):
    """Scalar score of a simulated end-state from `me`'s perspective.

    My holdings add (ships + prod_value*production); opponents subtract the same
    weighted by owner_weights (4P leans on the weakest). Reducing an opponent to
    zero pays an elimination bonus.
    """
    v = 0.0
    totals = {}
    for pl in planets:
        own = pl[POWN]
        if own == -1:
            continue
        worth = pl[PSHIP] + prod_value * pl[PPROD]
        if own == me:
            v += worth
        else:
            v -= owner_weights.get(own, 1.0) * worth
            totals[own] = totals.get(own, 0) + pl[PSHIP] + pl[PPROD]
    for f in fleets:
        own = f[FOWN]
        if own == me:
            v += f[FSHIP]
        elif own != -1:
            v -= owner_weights.get(own, 1.0) * f[FSHIP]
            totals[own] = totals.get(own, 0) + f[FSHIP]
    for o in opponents:
        if totals.get(o, 0) <= 0:
            v += elim_bonus
    return v


# ---------------------------------------------------------------------------
# Main decision logic.
# ---------------------------------------------------------------------------
def _decide(obs, start_time):
    p = PARAMS
    max_speed = p["max_speed"]
    me, av, planets, fleets = parse_obs(obs, max_speed)

    my_planets = [pl for pl in planets if pl[POWN] == me and not pl[PCOM]]
    if not my_planets:
        return []

    # Who are we fighting, and how strong is each right now?
    opp_strength = {}
    for pl in planets:
        if pl[POWN] != me and pl[POWN] != -1:
            opp_strength[pl[POWN]] = opp_strength.get(pl[POWN], 0) + pl[PSHIP] + pl[PPROD]
    for f in fleets:
        if f[FOWN] != me and f[FOWN] != -1:
            opp_strength[f[FOWN]] = opp_strength.get(f[FOWN], 0) + f[FSHIP]
    opponents = list(opp_strength.keys())
    max_str = max(opp_strength.values()) if opp_strength else 1
    # Weight the weakest opponent highest (prioritize finishing them off).
    owner_weights = {
        o: 1.0 + p["weak_mult"] * (max_str - s) / (max_str + 1.0)
        for o, s in opp_strength.items()
    }

    # Adaptive aggression (Port A). Measure how far behind the strongest
    # opponent we are; when behind, cut hoarded reserves and lower the commit
    # bar so we keep contesting/expanding instead of going passive and being
    # eliminated (the failure mode the arena exposed vs strong agents).
    my_strength = 0.0
    for pl in planets:
        if pl[POWN] == me:
            my_strength += pl[PSHIP] + pl[PPROD]
    for f in fleets:
        if f[FOWN] == me:
            my_strength += f[FSHIP]
    ratio = my_strength / (max_str + 1.0)
    aggr = max(0.0, min(1.0, 1.0 - ratio))          # 0 when even/ahead, ->1 far behind
    reserve_scale = 1.0 - p["behind_reserve_cut"] * aggr
    min_gain_eff = p["min_gain"] * (1.0 - p["behind_gain_cut"] * aggr)

    horizon = p["horizon"]

    # Baseline: no new orders. Drives keep_needed and the value reference point.
    arrivals = {}
    base_planets, base_fleets = simulate(
        planets, fleets, av, me, max_speed, horizon, launch=None, arrivals=arrivals
    )
    v0 = board_value(base_planets, base_fleets, me, owner_weights, opponents,
                     p["prod_value"], p["elim_bonus"])

    keep = compute_keep_needed(planets, arrivals, me, horizon, p)
    surplus = {}
    deficit = {}
    for pl in my_planets:
        k = keep.get(pl[PID], p["base_reserve"])
        k = max(p["base_reserve"], int(k * reserve_scale))  # free reserves when behind
        surplus[pl[PID]] = max(0, pl[PSHIP] - k)
        deficit[pl[PID]] = max(0, k - pl[PSHIP])

    # ---- Candidate generation -------------------------------------------
    # Cheap analytic priority first, then pay for aim+sim only on the best.
    targets = [pl for pl in planets if pl[POWN] != me and not pl[PCOM]]  # comets handled in M4
    pair_prio = []
    for S in my_planets:
        if surplus[S[PID]] <= 0:
            continue
        sx, sy = S[PX], S[PY]
        ranked = sorted(targets, key=lambda t: (t[PX] - sx) ** 2 + (t[PY] - sy) ** 2)
        for T in ranked[: p["targets_per_source"]]:
            d = math.hypot(T[PX] - sx, T[PY] - sy)
            k_rough = max(1, d / 4.0)
            garrison = T[PSHIP] + (T[PPROD] * k_rough if T[POWN] != -1 else 0)
            req_rough = garrison + 1
            if req_rough > surplus[S[PID]]:
                continue
            gain_est = p["prod_value"] * T[PPROD] + (T[PSHIP] if T[POWN] != -1 else 0)
            w = owner_weights.get(T[POWN], 1.0) if T[POWN] != -1 else 1.0
            prio = gain_est * w / (req_rough + 1.0) / (1.0 + 0.05 * d)
            pair_prio.append((prio, S, T))
    pair_prio.sort(key=lambda x: x[0], reverse=True)

    candidates = []  # each: dict(src, dst, move=[pid,angle,ships], size)

    # Reinforcement candidates first (defense before offense).
    needy = [pl for pl in my_planets if deficit[pl[PID]] > 0]
    for D in needy:
        need = deficit[D[PID]] + p["def_margin"]
        helpers = sorted(
            (s for s in my_planets if s[PID] != D[PID] and surplus[s[PID]] > 0),
            key=lambda s: (s[PX] - D[PX]) ** 2 + (s[PY] - D[PY]) ** 2,
        )
        for S in helpers[:2]:
            send = min(surplus[S[PID]], need)
            if send <= 0:
                continue
            aim = aim_and_size_to_friendly(S, D, av, max_speed, p["aim_max_ticks"])
            if aim is None:
                continue
            angle, _k = aim
            candidates.append({
                "src": S[PID], "dst": D[PID],
                "move": [S[PID], angle, int(send)], "size": int(send),
            })
            need -= send
            if need <= 0:
                break

    # Attack candidates (capture neutral/enemy planets).
    for prio, S, T in pair_prio[: p["max_attack_pairs"]]:
        aim = aim_and_size(S, T, av, max_speed, p["attack_margin"], p["aim_max_ticks"])
        if aim is None:
            continue
        angle, _k, size = aim
        if size > surplus[S[PID]] or size <= 0:
            continue
        candidates.append({
            "src": S[PID], "dst": T[PID],
            "move": [S[PID], angle, int(size)], "size": int(size),
        })

    if not candidates:
        return []

    # ---- Score candidates by simulation ---------------------------------
    scored = []
    for c in candidates:
        if time.time() - start_time > p["time_budget"]:
            break
        sp, sf = simulate(planets, fleets, av, me, max_speed, horizon, launch=[c["move"]])
        gain = board_value(sp, sf, me, owner_weights, opponents,
                            p["prod_value"], p["elim_bonus"]) - v0
        scored.append((gain, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    # ---- Greedy commit of best non-conflicting set ----------------------
    avail = dict(surplus)
    taken = set()
    chosen = []
    for gain, c in scored:
        if gain <= min_gain_eff:
            continue
        if c["dst"] in taken:
            continue
        if avail.get(c["src"], 0) < c["size"]:
            continue
        chosen.append(c)
        avail[c["src"]] -= c["size"]
        taken.add(c["dst"])

    if not chosen:
        return []

    moves = [c["move"] for c in chosen]

    # ---- Safety check: the combined plan must beat doing nothing --------
    # Guards against bad interactions between independently-scored moves.
    if len(chosen) > 1 and time.time() - start_time < p["time_budget"]:
        sp, sf = simulate(planets, fleets, av, me, max_speed, horizon, launch=moves)
        combined_gain = board_value(sp, sf, me, owner_weights, opponents,
                                    p["prod_value"], p["elim_bonus"]) - v0
        if combined_gain <= 0:
            # Fall back to just the single best move.
            return [chosen[0]["move"]]

    return moves


def aim_and_size_to_friendly(S, D, av, max_speed, max_ticks):
    """Aim a reinforcement at our own (possibly rotating) planet D. Reinforcing
    fleets are small, so estimate speed from a nominal mid-size fleet."""
    sx, sy, srad = S[PX], S[PY], S[PRAD]
    speed = fleet_speed(50, max_speed)
    moving = D[PROT] or D[PCOM]
    k_guess = 1
    for _ in range(5):
        pos = predict_pos(D, k_guess, av) or (D[PX], D[PY])
        d = math.hypot(pos[0] - sx, pos[1] - sy)
        k_guess = max(1, int(round(d / speed)))
    lead_pos = predict_pos(D, k_guess, av) or (D[PX], D[PY])
    lead_ang = math.atan2(lead_pos[1] - sy, lead_pos[0] - sx)
    scan = PARAMS["aim_scan"] if moving else (0.0,)
    best_k = None
    best_ang = None
    for da in scan:
        ang = lead_ang + da
        k = single_fleet_arrival((sx, sy), srad, ang, speed, D, av, max_ticks)
        if k is not None and (best_k is None or k < best_k):
            best_k, best_ang = k, ang
    if best_k is None:
        return None
    return best_ang, best_k


# ---------------------------------------------------------------------------
# Reliability gate. The agent must NEVER crash and NEVER exceed the time
# budget — a forfeit is an automatic loss, the only thing that drops rating.
# ---------------------------------------------------------------------------
def _safe_fallback(obs):
    """Dependency-free surplus push: from each owned planet send half its ships
    at the nearest non-owned planet. Used only if the main path errors out."""
    try:
        me = int(_get(obs, "player", 0))
        raw = _get(obs, "planets", []) or []
        mine = [pl for pl in raw if pl[1] == me and pl[5] > 2]
        others = [pl for pl in raw if pl[1] != me]
        moves = []
        for s in mine:
            if not others:
                break
            t = min(others, key=lambda o: (o[2] - s[2]) ** 2 + (o[3] - s[3]) ** 2)
            angle = math.atan2(t[3] - s[3], t[2] - s[2])
            ships = int(s[5]) // 2
            if ships >= 1:
                moves.append([s[0], angle, ships])
        return moves
    except Exception:
        return []


def agent(obs):
    start = time.time()
    try:
        return _decide(obs, start)
    except Exception:
        return _safe_fallback(obs)


# Local debugging: `python main.py` prints one decision on a fresh board.
if __name__ == "__main__":
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset()
    ob = env.state[0].observation
    t0 = time.time()
    out = agent(ob)
    print(f"decision in {1000*(time.time()-t0):.1f} ms -> {out}")
