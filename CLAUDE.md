# CLAUDE.md — Orbit Wars (Competitive Playbook)

Project memory for a **competitive** Orbit Wars (Kaggle) push. The goal is ranking,
not learning. Read this fully before assisting, and optimize every suggestion for
expected ladder position under a tight deadline.

---

## Objective

- **Place top 10** in the Orbit Wars competition (~4,210 teams → top ~0.24%, elite).
- **Approach is locked: a simulation-guided heuristic.** Not a static rule set; not
  RL from scratch. The bot generates candidate moves heuristically and chooses among
  them by *forward-simulating* the board. See "Winning architecture" below.
- Honesty over optimism: top 10 in the time remaining is hard and most likely
  requires a team. Don't soften that. If a plan won't realistically reach top 10, say
  so and suggest edits that can reach top 10 instead.

---

## How Claude should work with me

- **Ship, don't teach.** Write code directly. No quizzes, no slow-rolling, no
  "let me give you a chance to attempt it." Hand me working, well-structured code.
- **Be a competitive collaborator.** Always propose the highest expected-value move
  for *ranking*. If I start drifting toward something low-EV for this timeline (e.g.,
  training RL from scratch, gold-plating one module, rebuilding what a public notebook
  already provides), push back hard and redirect.
- **Stay rigorous and honest.** Don't flatter. Flag risks, dead ends, and false
  economies even when unwelcome.
- **Move fast on the clock.** Prefer adapting proven public work over inventing.
  Prioritize by points-per-hour given the days left.
- **Verify live facts.** The figures below are from a live competition and drift —
  re-check dates, field size, leaderboard, and rules before relying on them.

---

## Project facts (VERIFY — live competition, these change)

- **Game:** 2-player 1v1 or 4-player FFA, 100×100 continuous board, central sun, some
  planets rotate. Production compounds (planets generate ships proportional to size).
- **Action format:** each move is `[from_planet_id, angle_in_radians, num_ships]`.
- **Key mechanic — fleet size affects speed:** larger fleets move faster, approaching
  a max speed of 6.0; a ~500-ship fleet moves at ~5, ~1,000 ships hits the cap. Fleets
  travel in straight lines. **Arrival time therefore depends on fleet size** — this is
  why a static heuristic mis-estimates arrivals and a forward simulator wins.
- **Scoring:** TrueSkill-style (μ/σ). **Only win/draw/loss matters — margin of victory
  does NOT affect rating.** Consistency of winning is the entire currency.
- **Field / ceiling:** ~4,210 teams; top ratings ~1,800–2,000. We are targeting the
  strong-to-frontier tier.
- **Submission limits:** up to **5 submissions per team per day**; only the **latest
  2** are tracked for final scoring. Late in the comp, never let an experimental bot
  displace a known-good one.
- **Timeline (IMMINENT — re-check today):** entry & team-merger deadline **~Jun 16,
  2026** (essentially now); final submission deadline **~Jun 23, 2026**; post-deadline
  convergence after that.
- **Team note:** if merged, only the *team's best* bot shows on the public leaderboard
  (a public solo placement requires a team of one). Top 10 realistically needs a team:
  shared labor, pooled 5/day submissions, more compute.
- **Known simulator bugs (exploit these):** e.g., a "black-hole" case where ships
  hitting an expiring comet are deleted from the score — so it can be better to *miss*
  an expiring comet. Read the engine source and the discussion forum for more; one real
  exploit can outweigh days of tuning.

---

## Technical setup (conda)

```bash
conda create -n orbit-wars python=3.11
conda activate orbit-wars
mkdir orbit-wars && cd orbit-wars
pip install "kaggle-environments>=1.28.0"   # Orbit Wars needs >= 1.28.0
```

**Files**
- `main.py` — the bot. This IS the submission; Kaggle loads it the same way the local
  runner does.
- `run.py` — local runner: plays `main.py` vs a built-in agent, prints the result,
  writes `replay.html`.
- `arena.py` — *(to build, M2)* self-play harness: runs many games between two bot
  versions and reports win-rate with statistical confidence. The iteration engine.

Reference material (adapt, don't reinvent): public notebooks — "Robust Agent,"
"Step-by-Step Agent Dev & Ablation," "Orbit Wars 101."

**Caveat (verified 2026-06-16):** the HF `kvatsa5/orbit-wars-agent` `submission.py`
is **non-functional in the current engine** — it returns 0 moves every turn and
loses 0/20 to `starter` (hardcodes owner 0 = neutral *and* its move-gen is fully
passive; fixing the owner bug didn't help). Its README claims (top-5, "~90% vs
starter", "3379 lines") are false — it's 796 lines. Don't re-adapt it. (M1 beats
`starter` 100%, but that proved nothing — see the reality-check below, where M1
loses badly to a *functioning* strong agent.)

---

## The winning architecture: simulation-guided heuristic

Per-turn loop in `main.py`:

1. **Parse** the observation into planets and in-flight fleets. (Handle `obs` as both
   dict and object so one file runs locally and on Kaggle.)
2. **Forward-simulate** the board ~100+ turns ahead, accounting for in-flight fleets,
   per-planet production growth, combat resolution, planet rotation, and
   **fleet-size-dependent speed**. This yields, for each planet, the `keep_needed`
   (ships required to hold it) and the predicted outcome of candidate attacks at their
   *actual arrival turn*.
3. **Generate candidate dispatches** with a heuristic move generator (e.g., for each
   owned planet with surplus, the most valuable reachable targets).
4. **Evaluate candidates by simulation** and pick the best — this is the "guided" part
   and the edge over static rules.
5. **Dispatch:** send only the surplus beyond `keep_needed` (surplus-based dispatch);
   size each fleet to the **minimum sufficient** amount (defenders + growth +
   reinforcements arriving before you + margin); in 4P, prioritize the **weakest
   opponent** (value multiplier) and add an **elimination bonus** to finish weak
   players.
6. **Reliability gate:** wrap everything so the agent NEVER crashes or exceeds the
   per-turn time budget. A forfeit is an automatic loss, and on this ladder losses are
   the only thing that drops rating.

**Tuning is data-assisted, not RL.** Parameterize the heuristic (fleet-size margins,
defense reserves, target-value weights, elimination thresholds) and optimize those
parameters via self-play with CMA-ES or systematic search. This is how compute should
be spent here — amplifying a heuristic, not learning the game from nothing.

---

## Build plan (~1 week, points-per-hour ordered)

- **M0 — Decide the team & get a baseline in (TODAY, before the merger deadline).**
  Merge with a credible team now, or accept a solo run with a reset target. Submit any
  working bot (even the starter) so the pipeline is proven and the ladder slot is live.
- **M1 — Reimplement a strong architecture.** Stand up the forward-simulation core +
  surplus-based dispatch by adapting the documented top agent and public notebooks. Do
  NOT start from a blank file. Goal: reach the strong tier fast, then improve.
- **M2 — Build `arena.py`.** Self-play between two versions, hundreds of games,
  win-rate with confidence intervals. Without this you're tuning on noise.
- **M3 — Tune parameters.** CMA-ES / systematic search over the heuristic's weights
  using the arena. Lock in measurable win-rate gains.
- **M4 — Exploits + 4P logic.** Hunt engine bugs (comet/"black-hole," fleet-speed edge
  cases); add 4P-specific targeting (weakest-enemy, let rivals bleed each other).
- **M5 — Reliability hardening + submission management.** Stress-test for crashes and
  slow turns; submit the best bot; respect the 5/day and latest-2 rules near the end.

---

## Division of labor (if on a team)

- **Simulator/forward-model owner** — the core engine and combat/rotation/speed model.
- **Arena & tuning owner** — `arena.py`, self-play, CMA-ES parameter optimization.
- **Exploits & rules owner** — engine bugs, edge cases, 4P dynamics.
- **Submission manager** — schedules the 5 daily submissions, tracks the latest-2 rule,
  guards the known-good bot.

---

## Explicitly NOT doing (and why)

- **Pure rule-based without simulation:** mis-estimates arrivals because fleet size
  changes speed and planets rotate; loses to simulation-guided bots.
- **RL from scratch:** needs a fast vectorized simulator (the Python engine is slow),
  heavy compute, reward shaping, and weeks of self-play; it is tutorial-only in this
  competition and the worst expected-value bet on this timeline.
- **Gold-plating any one module** while the architecture or reliability lags.

---

## Progress log

### M1 — COMPLETE (2026-06-15)

`main.py` now implements the full simulation-guided architecture:

- **Forward simulator** (`simulate`) — a faithful, list-based re-implementation of
  the engine's turn order (launch → production → fleet move w/ continuous
  swept-pair collision → planet rotation & comet motion → combat). Fleet
  speed scales with size, so arrival turns are emergent. Rotation is propagated
  from the *observed* position by `angular_velocity`/tick (verified exactly
  equal to the engine's absolute-step formula). Comets follow their provided
  `paths`/`path_index` and expire off-board; the "black-hole" deletion of ships
  hitting an expiring comet is reproduced.
- **Surplus dispatch** — `keep_needed` per owned planet is derived from a
  no-orders baseline sim's recorded enemy arrivals (minus growth + friendly
  reinforcements); surplus = ships − keep_needed.
- **Candidate generation** — capture + reinforcement moves, lead-targeting
  moving planets (single-fleet aim search) and sizing each fleet to the
  minimum sufficient amount (garrison-on-arrival + 1 + margin).
- **Sim-guided selection** — every candidate scored by forward sim vs the
  baseline value; greedy commit of the best non-conflicting set, with a
  combined-plan safety re-check.
- **4P-aware** — weakest-opponent value multiplier + elimination bonus.
- **Reliability gate** — try/except + per-turn time budget + dependency-free
  fallback. Never raises, never near the time limit.

All knobs live in `PARAMS` (ready for M3 tuning). The reference HF/notebook
writeups were NOT needed for the architecture (engine source was the spec);
revisit them in M3/M4 for parameter priors and exploits.

**Verified (local, kaggle-environments 1.30.1):**
- Simulator fidelity vs engine ground truth: **28/28 scenarios exact** —
  planet owners/ships match and mid-flight fleet positions match to 1e-4,
  including heavy contested multi-player combat (59 fleets) and live comets.
  (Only unmodelled future comet *spawns* diverge — seed is hidden, unpredictable.)
- 2P: **32–0** vs `random` and `starter` (8 seeds × both seats × both opponents).
- 4P: **16/16** top-score vs 3× `starter` (4 seeds × all 4 seats).
- Timing: max ~50 ms/turn, mean ~5 ms (budget is 1 s + 60 s overage pool).
- Bug found & fixed during M1: a no-fleets "fast-forward" optimization skipped
  comet motion/expiry, freezing owned comets — corrected to run full ticks while
  any comet remains (proven by the fidelity suite going 7/15 → 28/28).

**Next (M2):** build `arena.py` (self-play, hundreds of games, win-rate ± CI)
so M3 parameter tuning isn't done on noise. The simulator being engine-exact
means arena results will transfer to the ladder.

### M2 — COMPLETE (2026-06-16)

`arena.py` is the iteration engine: runs many games between two agents and
reports win-rate with a confidence interval so M3 tunes on signal, not noise.

- **Antithetic seat-swap (variance reduction).** The board is generated from
  `random.Random(seed)` independently of agent order; seat 0 starts Q1, seat 1
  starts Q4. So each seed is played twice — A in seat 0, then A in seat 1 —
  i.e. the *same map with swapped homes*, cancelling positional bias (common
  random numbers). 4P rotates A through all four seats per seed.
- **Defensive outcome reading.** Engine reward is +1 for every agent at the top
  score (>0) else −1, so a 2P score tie is both-+1 → counted a **draw**. A
  crashed/timed-out agent (status ERROR/TIMEOUT/INVALID) is scored a **flagged
  loss**. Crucially the engine's terminal block overwrites all statuses to DONE,
  so the arena scans **every** step (not just the last) to catch a failure that
  was masked — an ever-failed bot is a loss even if it "won" a frozen board.
- **Stats:** W/D/L, win-rate (draws = ½) with **Wilson** CI, **LOS** (P(A truly
  stronger), decisive games), a verdict (CI vs 50%), and a **score-margin** line
  (avg ship differential — ladder ignores margin but it's a low-variance tuning
  proxy). Parallel via `ProcessPoolExecutor`; 2P + 4P; importable `run_arena()`.
- `_resolve_spec()` is the single extension point for **M3 param injection**
  (resolve a (path, params) spec to a callable *inside the worker*, so nothing
  unpicklable crosses processes — sidesteps kaggle's file-agent caching).

**Verified (local, 12-core):**
- **Balance proven, not assumed:** main.py vs itself = exactly **50.0%**, margin
  **+0.0** (0W/10D/0L) — zero seat bias. The high draw rate is real: two
  identical deterministic bots on a 4-fold-symmetric board mirror to ties.
- **Discriminates strength:** vs `starter` **100%** (2P +3173 / 4P +2958 margin
  ships), LOS ~99.8%, verdict "significantly better".
- **Reliability net works:** a deliberately-crashing opponent is flagged
  (B failures = N) and scored a loss; the harness never crashes.
- **Throughput (full 500-step games, 11 workers):** vs starter ~0.7 s/game
  (~2.3 games/s); bot-vs-bot ~8–10 s/game (~0.8 games/s) → ~300 games in ~6 min.

**M3 guidance (from M2 findings):** candidate-vs-baseline self-play is
draw-heavy by symmetry, so win/draw/loss has low signal-per-game near equality —
prefer the **score-margin** as the CMA-ES objective (smooth, low-variance) and
confirm finalists on win-rate; reuse a fixed seed set across candidates (CRN)
for paired comparisons. Inject params via `_resolve_spec` (one process per
candidate to dodge kaggle's agent caching).

### Strength reality-check — M1 has a real weakness (2026-06-16)

M1 dominates `random`/`starter`, but that proved nothing. The only *functioning*
strong public agent found — **`opp_producer`** (Kaggle dataset
`slawekbiel/producer-orbit-wars-utils`; torch + `orbit_lite/`, ~20 ms/turn,
reliable, ladder-legal on speed) — **beats M1 0/8** (seat-balanced seeds, LOS
0.2%, −1677 margin), reproduced with both `diag_game.py` and `arena.py`.

Diagnosis (deterministic seed-1 trajectory, reproduced): M1 is **even through
~step 30** (sim/architecture are fine) — then it (a) **loses the production
snowball** (producer ends ~2× planets / 3–12× ships) and, decisively, (b) **goes
passive when behind** — fleet count collapses to 0 and it *sheds* planets instead
of fighting. The gap is **behavioral, not architectural** — good news: M1's
engine-exact sim + the arena are exactly the tools to close it.

`opp_producer.py` + `orbit_lite/` are kept as the **M3 gauntlet opponent**
(NEVER submit — third-party). Plan: port the IDEAS (not code) into M1, highest
leverage first — **adaptive aggression / anti-hoarding when behind**, then
production-snowball targeting and proactive interception — measuring each change
vs `opp_producer` with `arena.py`, keeping only what moves the needle.

**Next (M3):** execute that port-and-measure loop; tune `PARAMS` with `arena.py`
against the gauntlet (`baseline.py` + `opp_producer.py` + `starter`).

---

*Keep this file current as the bot evolves: update the architecture notes, record
tuned parameters and their measured win-rates, and log any exploits found.*
