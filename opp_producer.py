from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass, replace

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

TOTAL_STEPS = 500


# ---------------------------------------------------------------------------
# Config – extended with dynamic knobs + defense/geometry + comet params
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProducerLiteConfig:
    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 6
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.35
    min_ships_to_launch: float = 4.0
    reinforce_size_beta: float = 2.2
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.20
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    # Dynamic scaling
    min_roi: float = 1.05
    max_roi: float = 1.45
    horizon_min: int = 8
    horizon_max: int = 24
    beta_min: float = 1.2
    beta_max: float = 3.5
    # Proactive defense
    defense_threat_horizon: float = 14.0
    defense_min_intercept_margin: float = 1.05
    defense_max_waves: int = 3
    # Orbital geometry
    geometry_weight: float = 0.35
    # Production snowball
    prod_rush_steps: int = 120
    prod_rush_top_k: int = 3
    prod_rush_roi_discount: float = 0.80
    # Comet hunting
    comet_score_multiplier: float = 2.0
    comet_movement_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Strength proxy
# ---------------------------------------------------------------------------

def _owner_strength(obs, prod: Tensor, player_count: int) -> Tensor:
    dtype = prod.dtype
    device = prod.device
    strength = torch.zeros(int(player_count), dtype=dtype, device=device)
    owner = obs.owner_abs.to(device=device)
    alive = obs.alive.to(device=device)
    ships = obs.ships.to(device=device, dtype=dtype)
    prod_v = prod.to(device=device, dtype=dtype)
    for oid in range(int(player_count)):
        mask = alive & (owner == oid)
        if bool(mask.any()):
            strength[oid] = prod_v[mask].sum() + 0.025 * ships[mask].sum()
    return strength


# ---------------------------------------------------------------------------
# Orbital centrality
# ---------------------------------------------------------------------------

def _orbital_centrality(obs, cache) -> Tensor:
    P = int(obs.P)
    device = obs.device
    if P <= 1:
        return torch.ones(P, device=device)
    d0 = cache.cross_dist[0].clone().float()
    alive = obs.alive.to(device=device)
    d0 = torch.where(alive.view(1, P) & alive.view(P, 1), d0, torch.zeros_like(d0))
    n_alive = alive.float().sum().clamp(min=1.0)
    mean_dist = d0.sum(dim=1) / n_alive
    centrality = 1.0 / (mean_dist + 1.0)
    return centrality.to(obs.ships.dtype)


# ---------------------------------------------------------------------------
# Proactive defense
# ---------------------------------------------------------------------------

def _build_defense_entries(
    *,
    movement: PlanetMovement,
    obs,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
):
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    if P == 0:
        return _empty_entries(device, dtype)

    owned = obs.owned & obs.alive
    if not bool(owned.any()):
        return _empty_entries(device, dtype)

    H = min(int(config.defense_threat_horizon), int(movement.garrison_status(max_horizon=int(config.defense_threat_horizon)).ships.shape[-1]) - 1)
    if H <= 0:
        return _empty_entries(device, dtype)

    status = movement.garrison_status(max_horizon=H)
    ships_at_H = status.ships[:, -1]

    threatened = owned & (ships_at_H < 0)
    if not bool(threatened.any()):
        return _empty_entries(device, dtype)

    tgt_indices = threatened.nonzero(as_tuple=False).squeeze(1)
    src_indices = owned.nonzero(as_tuple=False).squeeze(1)

    if src_indices.numel() == 0 or tgt_indices.numel() == 0:
        return _empty_entries(device, dtype)

    d0 = cache.cross_dist[0].to(dtype)
    src_ships = obs.ships[src_indices].to(dtype)

    all_entries = []
    waves_launched = 0

    for t_i in range(int(tgt_indices.shape[0])):
        if waves_launched >= int(config.defense_max_waves):
            break
        tgt = int(tgt_indices[t_i].item())
        deficit = float(-ships_at_H[tgt].item())
        need = deficit * float(config.defense_min_intercept_margin)

        dists = d0[src_indices, tgt]
        speeds = fleet_speed(src_ships.clamp(min=1.0))
        etas = (dists / speeds.clamp(min=1e-6)).ceil()

        can_arrive = etas <= float(H)
        has_surplus = src_ships > (need + float(config.min_ships_to_launch))
        src_neq_tgt = src_indices != tgt
        valid_src = can_arrive & has_surplus & src_neq_tgt

        if not bool(valid_src.any()):
            continue

        best_src_local = int(torch.where(valid_src, dists, torch.full_like(dists, 1e9)).argmin().item())
        best_src = int(src_indices[best_src_local].item())
        send_ships = min(float(src_ships[best_src_local].item()) * 0.6,
                         need + float(config.min_ships_to_launch))
        send_ships = max(send_ships, float(config.min_ships_to_launch))

        src_t = torch.tensor([[best_src]], dtype=torch.long, device=device)
        tgt_t = torch.tensor([tgt],       dtype=torch.long, device=device)
        send_t = torch.tensor([[send_ships]], dtype=dtype, device=device)
        eta_t = torch.tensor([[float(etas[best_src_local].item())]], dtype=dtype, device=device)
        valid_t = torch.tensor([[True]], dtype=torch.bool, device=device)

        entry = make_launch_set(
            source_slots=src_t,
            target_slots=tgt_t.unsqueeze(-1).expand(1, 1),
            ships=send_t,
            eta=eta_t,
            valid=valid_t,
            player_id=pid,
        )
        all_entries.append(entry)
        waves_launched += 1

    if not all_entries:
        return _empty_entries(device, dtype)
    return concat_launch_entries(all_entries)


# ---------------------------------------------------------------------------
# Comet detection
# ---------------------------------------------------------------------------

def detect_comets(obs, prev_obs, threshold: float = 0.5) -> Tensor:
    P = int(obs.P)
    device = obs.device
    if prev_obs is None or P == 0:
        return torch.zeros(P, dtype=torch.bool, device=device)

    curr_pos = torch.stack([obs.x, obs.y], dim=1)
    prev_pos = torch.stack([prev_obs.x, prev_obs.y], dim=1)
    dist_moved = torch.norm(curr_pos - prev_pos, dim=1)

    is_moving = dist_moved > threshold
    is_neutral = obs.owner_abs < 0
    is_alive = obs.alive
    return is_moving & is_neutral & is_alive


# ---------------------------------------------------------------------------
# Dynamic adjustment
# ---------------------------------------------------------------------------

def _adjust_config(
    config: ProducerLiteConfig,
    *,
    obs,
    prod: Tensor,
    step: int,
    player_count: int,
) -> ProducerLiteConfig:
    pid = int(obs.player_id)
    strength = _owner_strength(obs, prod, int(player_count))
    if pid < 0 or pid >= int(player_count) or strength.numel() == 0:
        return config

    my = float(strength[pid].item())
    leader = float(strength.max().item())
    ratio = my / max(leader, 1e-6)

    remaining = TOTAL_STEPS - int(step)

    if ratio < 1.0:
        deficit = 1.0 - ratio
        roi_drop = 0.30 * (1.0 - torch.exp(torch.tensor(-3.0 * deficit))).item()
        if remaining < 150 and ratio < 0.90:
            time_urgency = (150 - remaining) / 150.0
            roi_drop += 0.12 * time_urgency * deficit
        new_roi = max(config.min_roi, config.max_roi - roi_drop)
        config = replace(config, roi_threshold=new_roi)

    base_horizon = float(config.horizon)
    if ratio > 1.1:
        new_horizon = max(config.horizon_min, int(base_horizon * 0.7))
    elif ratio < 0.8:
        new_horizon = min(config.horizon_max, int(base_horizon * 1.3))
    else:
        new_horizon = int(base_horizon)
    config = replace(config, horizon=new_horizon)

    enemy_total = strength.sum() - my
    if enemy_total > 0:
        beta_scale = 1.0 + 0.3 * (enemy_total / (my + 1e-6)).clamp(0, 2)
        beta = max(config.beta_min, min(config.beta_max, config.reinforce_size_beta * beta_scale))
        config = replace(config, reinforce_size_beta=beta)

    base_waves = int(config.max_waves_per_turn)
    if ratio < 0.70:
        base_waves = min(8, base_waves + 1)
    if remaining < 100 and ratio < 0.95:
        base_waves = min(8, base_waves + 1)
    config = replace(config, max_waves_per_turn=base_waves)

    if remaining < 100:
        min_size = max(6.0, float(config.min_ships_to_launch) + 2.0 * (1.0 - ratio))
    else:
        min_size = float(config.min_ships_to_launch)
    config = replace(config, min_ships_to_launch=min_size)

    if remaining < 200:
        scale = remaining / 200.0
        tighter = max(0.05, config.regroup_pressure_delta_min * scale)
        config = replace(config, regroup_pressure_delta_min=tighter)

    return config


# ---------------------------------------------------------------------------
# Movement helpers
# ---------------------------------------------------------------------------

def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


# ---------------------------------------------------------------------------
# Late-game suppression
# ---------------------------------------------------------------------------

def _suppress_late_candidates(
    *,
    score: Tensor,
    obs,
    target_idx: Tensor,
    cand_tgt_short: Tensor,
    cand_is_def: Tensor,
    cand_eta: Tensor,
    step: int,
    player_id: int,
) -> Tensor:
    remaining = TOTAL_STEPS - int(step)
    if remaining > 120:
        return score
    P = int(obs.P)
    if P <= 0 or score.numel() == 0:
        return score
    device = score.device
    dtype = score.dtype
    pid = int(player_id)
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_owner = obs.owner_abs.to(device=device)[tgt_abs].long()
    eta = cand_eta.reshape(score.shape).to(device=device, dtype=dtype)

    is_neutral = tgt_owner < 0
    is_enemy = (tgt_owner >= 0) & (tgt_owner != pid) & (~cand_is_def)

    neutral_time = (remaining - eta) / max(1.0, 30.0)
    neutral_factor = torch.sigmoid(neutral_time * 0.5)
    score = torch.where(is_neutral, score * neutral_factor, score)

    enemy_time = (remaining - eta) / max(1.0, 20.0)
    enemy_factor = torch.sigmoid(enemy_time * 0.5)
    score = torch.where(is_enemy, score * enemy_factor, score)

    too_late = eta >= remaining
    return torch.where(too_late, torch.full_like(score, float("-inf")), score)


# ---------------------------------------------------------------------------
# Production snowball boost
# ---------------------------------------------------------------------------

def _apply_prod_snowball_boost(
    *,
    score: Tensor,
    obs,
    target_idx: Tensor,
    cand_tgt_short: Tensor,
    prod: Tensor,
    step: int,
    config: ProducerLiteConfig,
) -> Tensor:
    if int(step) > int(config.prod_rush_steps):
        return score

    P = int(obs.P)
    device = score.device
    dtype = score.dtype

    neutral_mask = obs.owner_abs < 0
    if not bool(neutral_mask.any()):
        return score

    prod_neutral = torch.where(neutral_mask & obs.alive, prod.to(dtype), torch.zeros(P, dtype=dtype, device=device))
    if int(prod_neutral.numel()) == 0:
        return score

    top_k = min(int(config.prod_rush_top_k), int(prod_neutral.numel()))
    top_vals = torch.topk(prod_neutral, top_k).values
    if top_vals.numel() == 0:
        return score
    threshold = float(top_vals[-1].item())

    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_prod = prod.to(dtype)[tgt_abs]
    tgt_neutral = (obs.owner_abs[tgt_abs] < 0)
    is_top_prod_neutral = tgt_neutral & (tgt_prod >= threshold - 1e-6)

    boost_factor = 1.0 / float(config.prod_rush_roi_discount)
    score = torch.where(is_top_prod_neutral.reshape(score.shape), score * boost_factor, score)
    return score


# ---------------------------------------------------------------------------
# Comet score boost
# ---------------------------------------------------------------------------

def _apply_comet_boost(
    *,
    score: Tensor,
    obs,
    target_idx: Tensor,
    cand_tgt_short: Tensor,
    comet_mask: Tensor,
    multiplier: float,
) -> Tensor:
    if comet_mask is None or not bool(comet_mask.any()):
        return score
    P = int(obs.P)
    device = score.device
    dtype = score.dtype
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    is_comet_target = comet_mask[tgt_abs]
    boost = torch.where(is_comet_target, torch.tensor(multiplier, dtype=dtype, device=device),
                        torch.tensor(1.0, dtype=dtype, device=device))
    return score * boost.reshape(score.shape)


# ---------------------------------------------------------------------------
# Core planner – with distance penalty & strategic alignment
# ---------------------------------------------------------------------------

def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    memory,
):
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    step = int(obs_tensors["step"].reshape(-1)[0].item())

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    H_eff = torch.full((), float(H), dtype=dtype, device=device)

    ships = obs.ships.to(dtype)
    prod_val = prod.to(dtype)

    # Source scoring with geometry
    centrality = _orbital_centrality(obs, cache)
    geo_w = float(config.geometry_weight)
    source_score = (
        (1.0 - geo_w) * (ships + 0.5 * prod_val * (ships / (ships + 1.0)))
        + geo_w * centrality * ships
    )
    source_mask = obs.owned & obs.alive & (ships >= float(config.min_ships_to_launch))
    source_score = torch.where(source_mask, source_score, torch.tensor(float("-inf"), device=device, dtype=dtype))
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx = torch.topk(source_score, min(S_cap, int(source_score.numel())), dim=0).indices
    source_exists = source_mask[source_idx]

    # Build target shortlist
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )

    # ---- Add comet targets + KNN source supplementation ----
    comet_mask = getattr(memory, "comet_mask", None)
    if comet_mask is not None and bool(comet_mask.any()):
        comet_indices = comet_mask.nonzero(as_tuple=False).squeeze(1)
        all_targets = torch.cat([target_idx, comet_indices])
        unique_targets, inverse = torch.unique(all_targets, return_inverse=True)
        target_idx = unique_targets
        target_exists = torch.ones(target_idx.shape[0], dtype=torch.bool, device=device)
        memory.current_comet_mask = comet_mask

        # KNN source supplementation for comets
        d0 = cache.cross_dist[0].to(dtype)
        owned_mask = obs.owned & obs.alive
        extra_sources = []
        for c in comet_indices.tolist():
            c = int(c)
            dists = d0[c, :]
            valid_sources = owned_mask & (torch.arange(P, device=device) != c)
            if not bool(valid_sources.any()):
                continue
            valid_dists = torch.where(valid_sources, dists, torch.full_like(dists, 1e9))
            k = min(2, int(valid_sources.sum().item()))
            if k > 0:
                nearest = torch.topk(-valid_dists, k).indices
                extra_sources.extend(nearest.tolist())
        if extra_sources:
            extra_src = torch.tensor(extra_sources, dtype=torch.long, device=device).unique()
            all_sources = torch.cat([source_idx, extra_src])
            unique_sources, _ = torch.unique(all_sources, return_inverse=True)
            source_idx = unique_sources
            source_exists = source_mask[source_idx]
            S = int(source_idx.shape[0])
            source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
            drain = safe_drain(
                garrison_status, source_idx=source_idx, source_ships=source_ships,
                H_eff=H_eff, player_id=pid,
            )
    else:
        memory.current_comet_mask = None

    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)

    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]

    if comet_mask is None or not bool(comet_mask.any()):
        source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
        drain = safe_drain(
            garrison_status, source_idx=source_idx, source_ships=source_ships,
            H_eff=H_eff, player_id=pid,
        )

    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)

    beta = float(config.reinforce_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
        if beta > 0.0 or bool(config.enable_regroup) else None
    )

    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange, eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=reinforcement,
    )
    K = int(floor.shape[-1])

    sizes = drain.view(S, 1).expand(S, T).floor().clamp(min=1.0)

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= float(config.min_ships_to_launch))
        & src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
    )

    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )

    # ---- Distance penalty ----
    dist_matrix = cache.cross_dist[0].to(dtype)  # [P, P]
    cand_src_abs = cand_src.squeeze(-1)          # [C]
    cand_tgt_abs = cand_tgt_slot                 # [C]
    dist = dist_matrix[cand_src_abs, cand_tgt_abs]  # [C]
    distance_scale = 1.0 / (1.0 + dist / 15.0)   # tunable
    score = score * distance_scale.reshape(score.shape)

    # ---- Strategic alignment (cosine similarity toward enemy centre) ----
    own_mask = obs.owned & obs.alive
    if own_mask.any():
        pos = torch.stack([obs.x, obs.y], dim=1)  # [P, 2]
        weights = (obs.ships + 0.5 * prod_val).to(dtype)
        centre_own = (pos * own_mask.unsqueeze(-1) * weights.unsqueeze(-1)).sum(dim=0) / (weights[own_mask].sum() + 1e-6)
        enemy_mask = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)
        if enemy_mask.any():
            centre_enemy = (pos * enemy_mask.unsqueeze(-1) * weights.unsqueeze(-1)).sum(dim=0) / (weights[enemy_mask].sum() + 1e-6)
            strategic_vec = centre_enemy - centre_own
            strategic_vec = strategic_vec / (strategic_vec.norm() + 1e-6)
            src_pos = pos[cand_src_abs]  # [C, 2]
            tgt_pos = pos[cand_tgt_abs]  # [C, 2]
            candidate_vec = tgt_pos - src_pos
            candidate_vec_norm = candidate_vec / (candidate_vec.norm(dim=1, keepdim=True) + 1e-6)
            cos_sim = (candidate_vec_norm * strategic_vec).sum(dim=1)
            align_weight = 1.0 + 0.5 * (cos_sim + 1.0)  # between 0.5 and 1.5
            score = score * align_weight.reshape(score.shape)

    # Late-game suppression
    score = _suppress_late_candidates(
        score=score, obs=obs, target_idx=target_idx,
        cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def,
        cand_eta=cand_eta, step=int(step), player_id=pid,
    )

    # Production snowball boost
    score = _apply_prod_snowball_boost(
        score=score, obs=obs, target_idx=target_idx,
        cand_tgt_short=cand_tgt_short, prod=prod,
        step=int(step), config=config,
    )

    # Comet boost
    if memory.current_comet_mask is not None:
        score = _apply_comet_boost(
            score=score,
            obs=obs,
            target_idx=target_idx,
            cand_tgt_short=cand_tgt_short,
            comet_mask=memory.current_comet_mask,
            multiplier=float(config.comet_score_multiplier),
        )

    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries

    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


# ---------------------------------------------------------------------------
# Turn pipeline – runs defense, comet detection, then offense
# ---------------------------------------------------------------------------

def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    step = int(obs_tensors["step"].reshape(-1)[0].item())

    config = _adjust_config(
        config, obs=obs, prod=movement.planet_prod, step=step, player_count=int(player_count)
    )

    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    # Proactive defense
    defense_entries = _build_defense_entries(
        movement=movement, obs=obs, cache=cache,
        config=config, player_count=int(player_count),
    )

    # Comet detection
    prev_obs = getattr(memory, "prev_obs", None)
    comet_mask = detect_comets(obs, prev_obs, threshold=float(config.comet_movement_threshold))
    memory.prev_obs = obs
    memory.comet_mask = comet_mask

    # Offensive wave planning (pass memory)
    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
        memory=memory,
    )

    # Merge defense + offense
    entries = concat_launch_entries([defense_entries, entries])
    entries = disambiguate_duplicate_launches(entries)

    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------

CONFIG_2P = ProducerLiteConfig()

CONFIG_3P = replace(
    ProducerLiteConfig(),
    horizon=15,
    max_sources_per_lane=8,
    max_offensive_targets=10,
    max_defensive_targets=5,
    roi_threshold=1.30,
    prod_rush_steps=100,
)

CONFIG_4P = replace(
    ProducerLiteConfig(),
    horizon=13,
    roi_threshold=1.20,
    max_sources_per_lane=7,
    max_defensive_targets=4,
    max_waves_per_turn=5,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    prod_rush_steps=80,
    geometry_weight=0.45,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    pc = int(player_count)
    if pc >= 4:
        return CONFIG_4P
    elif pc == 3:
        return CONFIG_3P
    return CONFIG_2P


# ---------------------------------------------------------------------------
# Runtime & entry point
# ---------------------------------------------------------------------------

class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        # comet detection state
        self.prev_obs = None
        self.comet_mask = None
        self.current_comet_mask = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.prev_obs = None
        self.comet_mask = None
        self.current_comet_mask = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.prev_obs = None
            mem.comet_mask = None
            mem.current_comet_mask = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
