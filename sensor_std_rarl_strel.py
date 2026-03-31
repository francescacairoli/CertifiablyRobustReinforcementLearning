"""
sensor_rarl_strel.py
====================
Standard and RARL baseline policies for the sensor-noise navigation problem.

Both variants use the same CrossroadSensorSTRELProblem and RolloutModule
defined in certified_sensor_strel.py.  The only difference is how they are
trained:

  Standard — no robustness machinery.  Maximise E[ρ(pos_0, δ=0)], i.e.
             train only on the nominal trajectory (zero sensor bias).

  RARL     — adversarial training.  A neural adversary learns to pick the
             worst constant sensor bias δ ∈ [−ε, ε]² for the current
             protagonist.  Protagonist maximises ρ under adversarial noise.
             Training alternates: adversary minimises ρ.mean(), protagonist
             maximises _agg(ρ, cfg) with a REINFORCE baseline.

Evaluation helper
-----------------
pgd_attack(rollout_mod, pos_0, eps, n_steps)  —  PGD (sign-gradient descent)
to find near-worst-case δ for any policy; used for fair comparison in the
comparison script.

evaluate_policy(problem, rollout_mod, n_eval, seed, device, eps)  —  returns
nominal / random-noise / PGD-adversarial metrics.
"""

import warnings as _warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from certified_sensor_strel import (
    CrossroadSensorSTRELProblem,
    DeterministicPolicy,
    RolloutModule,
    PlanningConfig,
    _set_formula_beta,
    _agg,
)


# ── Adversary policy ──────────────────────────────────────────────────────────

class AdversaryPolicy(nn.Module):
    """
    Neural adversary for sensor-noise RARL.

    Takes the true initial position pos_0 (B, 2) as context — the adversary
    knows where the agent starts and crafts the worst constant sensor bias for
    that starting configuration.

    Output: δ (B, 2) ∈ [−ε, ε]²  (scaled externally by current ε).
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, 2)

    def forward(self, pos_0: torch.Tensor) -> torch.Tensor:
        """Return δ ∈ (−1, 1)²  — caller multiplies by current ε."""
        return torch.tanh(self.mu(self.backbone(pos_0)))


# ── PGD adversarial attack (evaluation) ──────────────────────────────────────

def pgd_attack(
    rollout_mod: RolloutModule,
    pos_0:       torch.Tensor,
    eps:         float,
    n_steps:     int  = 30,
    step_size:   float = None,
) -> torch.Tensor:
    """
    Projected Gradient Descent to find δ* = argmin_δ ρ(pos_0, δ)
                                              s.t.  ‖δ‖_∞ ≤ ε

    Uses the sign of the gradient (PGD-∞); restarts from zero.
    Returned tensor is detached and does not require grad.
    """
    if step_size is None:
        step_size = 2.5 * eps / n_steps

    # Freeze policy weights during attack
    was_training = rollout_mod.training
    rollout_mod.eval()

    delta = torch.zeros_like(pos_0)
    for _ in range(n_steps):
        delta = delta.detach().requires_grad_(True)
        rho   = rollout_mod(pos_0, delta)
        rho.sum().backward()          # minimise ρ: gradient descent on rho
        with torch.no_grad():
            delta = (delta - step_size * delta.grad.sign()).clamp(-eps, eps)

    if was_training:
        rollout_mod.train()
    return delta.detach()


# ── Evaluation (nominal / random / PGD) ──────────────────────────────────────

def evaluate_policy(
    problem:     CrossroadSensorSTRELProblem,
    rollout_mod: RolloutModule,
    n_eval:      int,
    seed:        int,
    device:      torch.device,
    eps:         float,
    n_pgd_steps: int = 30,
) -> Dict:
    gen   = torch.Generator(device=device).manual_seed(seed)
    pos_0 = problem.sample_initial_positions(n_eval, gen)

    # Nominal (δ = 0)
    with torch.no_grad():
        rho_nom = rollout_mod(pos_0, torch.zeros(n_eval, 2, device=device))

    # Random δ ∈ [−ε, ε]²
    gen_r   = torch.Generator(device=device).manual_seed(seed + 1)
    noise_r = ((2 * torch.rand(n_eval, 2, generator=gen_r, device=device) - 1) * eps)
    with torch.no_grad():
        rho_rand = rollout_mod(pos_0, noise_r)

    # PGD worst-case δ
    delta_pgd = pgd_attack(rollout_mod, pos_0, eps, n_steps=n_pgd_steps)
    with torch.no_grad():
        rho_pgd = rollout_mod(pos_0, delta_pgd)

    return {
        "nominal_rho_mean": float(rho_nom.mean()),
        "nominal_sat_rate": float((rho_nom > 0).float().mean()),
        "rand_rho_mean":    float(rho_rand.mean()),
        "rand_sat_rate":    float((rho_rand > 0).float().mean()),
        "pgd_rho_mean":     float(rho_pgd.mean()),
        "pgd_sat_rate":     float((rho_pgd > 0).float().mean()),
        # numpy arrays for histograms
        "nominal_rho_values": rho_nom.detach().cpu().numpy(),
        "rand_rho_values":    rho_rand.detach().cpu().numpy(),
        "pgd_rho_values":     rho_pgd.detach().cpu().numpy(),
        # tensors for plotting
        "_pos_0":       pos_0,
        "_noise_rand":  noise_r,
        "_delta_pgd":   delta_pgd,
    }


# ── Standard training ─────────────────────────────────────────────────────────

def train_standard(
    cfg:    PlanningConfig,
    device: torch.device,
) -> Tuple[DeterministicPolicy, CrossroadSensorSTRELProblem, RolloutModule, Dict]:
    """
    Train a standard STREL policy on the nominal (δ=0) trajectory.
    No adversarial training, no LiRPA bound — the policy only sees the
    unperturbed world during training.
    """
    torch.manual_seed(cfg.seed)
    problem = CrossroadSensorSTRELProblem(cfg, device)

    n_obs   = len(CrossroadSensorSTRELProblem.OBSTACLES)
    obs_dim = 2 + 2 + 2 * n_obs + n_obs
    policy      = DeterministicPolicy(obs_dim, 2, cfg.hidden,
                                      cfg.action_parameterization).to(device)
    rollout_mod = RolloutModule(policy, problem).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train_iters)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    history: Dict[str, List[float]] = {k: [] for k in
        ["iter", "rho_nom_mean", "nom_sat_rate", "goal_dist", "clearance", "beta"]}

    print(f"[Standard] device={device}  obs={obs_dim}  hidden={cfg.hidden}")
    print(f"  {cfg.train_iters} iters · nominal STREL only · "
          f"β: {cfg.strel_beta_min}→{cfg.strel_beta_max}")

    zero_noise = torch.zeros(cfg.batch_size, 2, device=device)

    for it in range(cfg.train_iters):
        alpha = it / max(cfg.train_iters - 1, 1)
        beta  = cfg.strel_beta_min + (cfg.strel_beta_max - cfg.strel_beta_min) * alpha
        cfg.strel_beta = beta
        _set_formula_beta(problem.formula, beta)

        pos_0 = problem.sample_initial_positions(cfg.batch_size, generator)

        rho_nom, min_clr = rollout_mod.forward_full(pos_0, zero_noise)

        loss_main = -_agg(rho_nom, cfg)
        loss_viol = cfg.violation_penalty_weight * torch.relu(-rho_nom).mean()
        loss_safe = cfg.safety_penalty_weight * torch.relu(
            cfg.safety_buffer - min_clr).mean()
        loss = loss_main + loss_viol + loss_safe

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (it + 1) % 25 == 0:
            rho_n   = float(rho_nom.detach().mean())
            nom_sat = float((rho_nom.detach() > 0).float().mean())
            with torch.no_grad():
                step_dt = cfg.dt / float(cfg.integration_substeps)
                pe = pos_0.clone(); ph = []
                for _h in range(cfg.horizon):
                    obs_e = problem.observation(pe)
                    act_e = policy(obs_e)
                    for _s in range(cfg.integration_substeps):
                        pe = pe + step_dt * cfg.max_speed * act_e
                        pe = torch.clamp(pe, cfg.world_min, cfg.world_max)
                    ph.append(pe)
                traj_e  = torch.stack(ph, dim=1)
                goal_d  = float(problem._linf_norm(traj_e[:, -1] - problem.goal).mean())
                clr_val = float(problem.clearance(traj_e).min(dim=-1).values.mean())

            print(f"[Std] iter={it+1:05d} β={beta:.1f} "
                  f"ρ_nom={rho_n:+.3f} nom_sat={nom_sat:.1%} "
                  f"viol={float(loss_viol.detach()):.3f} "
                  f"safe={float(loss_safe.detach()):.3f} "
                  f"goal_d={goal_d:.3f} clr={clr_val:+.3f}")
            history["iter"].append(it + 1)
            history["rho_nom_mean"].append(rho_n)
            history["nom_sat_rate"].append(nom_sat)
            history["goal_dist"].append(goal_d)
            history["clearance"].append(clr_val)
            history["beta"].append(beta)

    return policy, problem, rollout_mod, history


# ── Standard + noise augmentation (domain randomisation) ─────────────────────

def train_standard_aug(
    cfg:    PlanningConfig,
    device: torch.device,
) -> Tuple[DeterministicPolicy, CrossroadSensorSTRELProblem, RolloutModule, Dict]:
    """
    Standard policy gradient with domain randomisation: at every training step a
    random constant sensor bias δ ∼ Uniform([−ε, ε]²) is applied, where ε is
    annealed together with β.  The policy sees noisy observations during training
    but has no adversarial opponent and no certified bound.

    This is the simplest practical approach to noise-robust training and serves
    as the baseline between Standard (no noise) and RARL (adversarial noise).
    """
    torch.manual_seed(cfg.seed)
    problem = CrossroadSensorSTRELProblem(cfg, device)

    n_obs   = len(CrossroadSensorSTRELProblem.OBSTACLES)
    obs_dim = 2 + 2 + 2 * n_obs + n_obs
    policy      = DeterministicPolicy(obs_dim, 2, cfg.hidden,
                                      cfg.action_parameterization).to(device)
    rollout_mod = RolloutModule(policy, problem).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train_iters)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    noise_gen = torch.Generator(device=device).manual_seed(cfg.seed + 2)

    history: Dict[str, List[float]] = {k: [] for k in
        ["iter", "rho_aug_mean", "nom_sat_rate", "aug_sat_rate",
         "goal_dist", "clearance", "eps", "beta"]}

    print(f"[Std+Aug] device={device}  obs={obs_dim}  hidden={cfg.hidden}")
    print(f"  {cfg.train_iters} iters · random δ ∼ Uniform([−ε,ε]²) · "
          f"ε: {cfg.sensor_noise_max_start}→{cfg.sensor_noise_max} · "
          f"β: {cfg.strel_beta_min}→{cfg.strel_beta_max}")

    zero_noise = torch.zeros(cfg.batch_size, 2, device=device)

    for it in range(cfg.train_iters):
        alpha       = it / max(cfg.train_iters - 1, 1)
        current_eps = (cfg.sensor_noise_max_start
                       + (cfg.sensor_noise_max - cfg.sensor_noise_max_start) * alpha)
        beta        = cfg.strel_beta_min + (cfg.strel_beta_max - cfg.strel_beta_min) * alpha
        cfg.strel_beta = beta
        _set_formula_beta(problem.formula, beta)

        pos_0 = problem.sample_initial_positions(cfg.batch_size, generator)

        # Random noise augmentation — new sample every iteration
        noise = ((2 * torch.rand(cfg.batch_size, 2,
                                 generator=noise_gen, device=device) - 1)
                 * current_eps)

        rho_aug, min_clr = rollout_mod.forward_full(pos_0, noise)

        loss_main = -_agg(rho_aug, cfg)
        loss_viol = cfg.violation_penalty_weight * torch.relu(-rho_aug).mean()
        loss_safe = cfg.safety_penalty_weight * torch.relu(
            cfg.safety_buffer - min_clr).mean()
        loss = loss_main + loss_viol + loss_safe

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (it + 1) % 25 == 0:
            rho_a   = float(rho_aug.detach().mean())
            aug_sat = float((rho_aug.detach() > 0).float().mean())
            with torch.no_grad():
                rho_n   = rollout_mod(pos_0, zero_noise)
            nom_sat = float((rho_n > 0).float().mean())

            with torch.no_grad():
                step_dt = cfg.dt / float(cfg.integration_substeps)
                pe = pos_0.clone(); ph = []
                for _h in range(cfg.horizon):
                    obs_e = problem.observation(pe)
                    act_e = policy(obs_e)
                    for _s in range(cfg.integration_substeps):
                        pe = pe + step_dt * cfg.max_speed * act_e
                        pe = torch.clamp(pe, cfg.world_min, cfg.world_max)
                    ph.append(pe)
                traj_e  = torch.stack(ph, dim=1)
                goal_d  = float(problem._linf_norm(traj_e[:, -1] - problem.goal).mean())
                clr_val = float(problem.clearance(traj_e).min(dim=-1).values.mean())

            print(f"[Aug] iter={it+1:05d} β={beta:.1f} ε={current_eps:.4f} "
                  f"ρ_aug={rho_a:+.3f} nom_sat={nom_sat:.1%} aug_sat={aug_sat:.1%} "
                  f"viol={float(loss_viol.detach()):.3f} "
                  f"safe={float(loss_safe.detach()):.3f} "
                  f"goal_d={goal_d:.3f} clr={clr_val:+.3f}")

            history["iter"].append(it + 1)
            history["rho_aug_mean"].append(rho_a)
            history["nom_sat_rate"].append(nom_sat)
            history["aug_sat_rate"].append(aug_sat)
            history["goal_dist"].append(goal_d)
            history["clearance"].append(clr_val)
            history["eps"].append(current_eps)
            history["beta"].append(beta)

    return policy, problem, rollout_mod, history


# ── RARL training ─────────────────────────────────────────────────────────────

def train_rarl(
    cfg:    PlanningConfig,
    device: torch.device,
) -> Tuple[DeterministicPolicy, CrossroadSensorSTRELProblem,
           RolloutModule, AdversaryPolicy, Dict]:
    """
    RARL: protagonist maximises ρ against a learned neural adversary.

    The adversary takes pos_0 and outputs δ ∈ (−1,1)² scaled by current ε.
    Training schedule:
      • one adversary gradient step per iteration (minimise ρ.mean())
      • one protagonist gradient step (maximise _agg(ρ, cfg) with baseline)
      • ε annealed 0 → sensor_noise_max in parallel with β
    """
    torch.manual_seed(cfg.seed)
    problem = CrossroadSensorSTRELProblem(cfg, device)

    n_obs   = len(CrossroadSensorSTRELProblem.OBSTACLES)
    obs_dim = 2 + 2 + 2 * n_obs + n_obs
    policy      = DeterministicPolicy(obs_dim, 2, cfg.hidden,
                                      cfg.action_parameterization).to(device)
    adversary   = AdversaryPolicy(hidden=64).to(device)
    rollout_mod = RolloutModule(policy, problem).to(device)

    pro_opt = optim.Adam(policy.parameters(),    lr=cfg.lr)
    adv_opt = optim.Adam(adversary.parameters(), lr=cfg.lr * 2.0)   # faster adversary
    pro_sched = optim.lr_scheduler.CosineAnnealingLR(pro_opt, T_max=cfg.train_iters)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    mom = 0.9
    baseline = 0.0

    history: Dict[str, List[float]] = {k: [] for k in
        ["iter", "rho_nom_mean", "rho_adv_mean", "nom_sat_rate", "adv_sat_rate",
         "goal_dist", "clearance", "eps", "beta"]}

    print(f"[RARL] device={device}  obs={obs_dim}  hidden={cfg.hidden}")
    print(f"  {cfg.train_iters} iters · neural adversary (δ ∈ [−ε,ε]²) · "
          f"ε: {cfg.sensor_noise_max_start}→{cfg.sensor_noise_max} · "
          f"β: {cfg.strel_beta_min}→{cfg.strel_beta_max}")

    zero_noise = torch.zeros(cfg.batch_size, 2, device=device)

    for it in range(cfg.train_iters):
        alpha       = it / max(cfg.train_iters - 1, 1)
        current_eps = (cfg.sensor_noise_max_start
                       + (cfg.sensor_noise_max - cfg.sensor_noise_max_start) * alpha)
        beta        = cfg.strel_beta_min + (cfg.strel_beta_max - cfg.strel_beta_min) * alpha
        cfg.strel_beta = beta
        _set_formula_beta(problem.formula, beta)

        pos_0 = problem.sample_initial_positions(cfg.batch_size, generator)

        # ── Adversary step: minimise ρ (find δ that causes violations) ─────
        adv_opt.zero_grad(set_to_none=True)
        noise_adv = current_eps * adversary(pos_0)          # (B,2) ∈ [−ε,ε]²
        rho_adv   = rollout_mod(pos_0, noise_adv)
        loss_adv  = rho_adv.mean()                           # minimise → descent
        loss_adv.backward()
        adv_opt.step()

        # ── Protagonist step: maximise ρ under adversarial noise ──────────
        with torch.no_grad():
            noise_det = current_eps * adversary(pos_0)       # detached adversary
        rho_pro   = rollout_mod(pos_0, noise_det)
        rho_pro_agg = _agg(rho_pro, cfg)
        baseline    = mom * baseline + (1.0 - mom) * float(rho_pro_agg.detach())
        loss_rho    = -(rho_pro_agg - baseline)

        # Violation penalty on adversarial robustness
        loss_viol = cfg.violation_penalty_weight * torch.relu(-rho_pro).mean()
        # Safety penalty on nominal clearance (keeps trajectory physically valid)
        with torch.no_grad():
            _, min_clr = rollout_mod.forward_full(pos_0, zero_noise)
        loss_safe = cfg.safety_penalty_weight * torch.relu(
            cfg.safety_buffer - min_clr).mean()

        loss_pro = loss_rho + loss_viol + loss_safe
        pro_opt.zero_grad(set_to_none=True)
        loss_pro.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        pro_opt.step()
        pro_sched.step()

        if (it + 1) % 25 == 0:
            with torch.no_grad():
                rho_n   = rollout_mod(pos_0, zero_noise)
                rho_a   = rollout_mod(pos_0, noise_det)
            rho_n_mean = float(rho_n.mean())
            rho_a_mean = float(rho_a.mean())
            nom_sat    = float((rho_n > 0).float().mean())
            adv_sat    = float((rho_a > 0).float().mean())

            with torch.no_grad():
                step_dt = cfg.dt / float(cfg.integration_substeps)
                pe = pos_0.clone(); ph = []
                for _h in range(cfg.horizon):
                    obs_e = problem.observation(pe)
                    act_e = policy(obs_e)
                    for _s in range(cfg.integration_substeps):
                        pe = pe + step_dt * cfg.max_speed * act_e
                        pe = torch.clamp(pe, cfg.world_min, cfg.world_max)
                    ph.append(pe)
                traj_e  = torch.stack(ph, dim=1)
                goal_d  = float(problem._linf_norm(traj_e[:, -1] - problem.goal).mean())
                clr_val = float(problem.clearance(traj_e).min(dim=-1).values.mean())

            print(f"[RARL] iter={it+1:05d} β={beta:.1f} ε={current_eps:.4f} "
                  f"ρ_nom={rho_n_mean:+.3f} ρ_adv={rho_a_mean:+.3f} "
                  f"nom_sat={nom_sat:.1%} adv_sat={adv_sat:.1%} "
                  f"safe={float(loss_safe.detach()):.3f} "
                  f"goal_d={goal_d:.3f} clr={clr_val:+.3f}")

            history["iter"].append(it + 1)
            history["rho_nom_mean"].append(rho_n_mean)
            history["rho_adv_mean"].append(rho_a_mean)
            history["nom_sat_rate"].append(nom_sat)
            history["adv_sat_rate"].append(adv_sat)
            history["goal_dist"].append(goal_d)
            history["clearance"].append(clr_val)
            history["eps"].append(current_eps)
            history["beta"].append(beta)

    return policy, problem, rollout_mod, adversary, history


# ── Plotting ──────────────────────────────────────────────────────────────────

import os
import dataclasses
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

CKPT_AUG  = "saved_models/sensor_aug_h10_noise005.pt"
CKPT_RARL = "saved_models/sensor_rarl_h10_noise005.pt"
PLOT_OUT  = "sensor_rarl_comparison.png"
N_TRAJ    = 30


def _rollout(rm: RolloutModule,
             pos_0: torch.Tensor,
             noise: torch.Tensor) -> torch.Tensor:
    """(B, H+1, 2) cpu tensor — start + H true steps under constant noise."""
    cfg     = rm.cfg
    step_dt = cfg.dt / float(cfg.integration_substeps)
    pos     = pos_0.clone()
    pts     = [pos.cpu()]
    with torch.no_grad():
        for _ in range(cfg.horizon):
            obs    = rm._observation(pos, noise)   # δ constant — same every step
            action = rm.policy(obs)
            for _ in range(cfg.integration_substeps):
                pos = pos + step_dt * cfg.max_speed * action
                pos = torch.clamp(pos, cfg.world_min, cfg.world_max)
            pts.append(pos.cpu())
    return torch.stack(pts, dim=1)


def _sample_pgd(problem, rm, n, seed, eps, n_pgd_steps, device):
    gen   = torch.Generator(device=device).manual_seed(seed + 77)
    pos_s = problem.sample_initial_positions(n, gen)
    delta = pgd_attack(rm, pos_s, eps, n_steps=n_pgd_steps)
    with torch.no_grad():
        rho_s = rm(pos_s, delta).cpu().numpy()
    return pos_s, rho_s, _rollout(rm, pos_s, delta)


def _draw_env(ax, problem, cfg, title):
    for obs in problem.obstacles.cpu():
        cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
        ax.add_patch(mpatches.Rectangle(
            (cx - r, cy - r), 2 * r, 2 * r,
            color="tab:red", alpha=0.28, zorder=2))
        ax.add_patch(mpatches.Rectangle(
            (cx - r, cy - r), 2 * r, 2 * r,
            fill=False, edgecolor="tab:red", lw=0.9, zorder=2))
    gx, gy = cfg.goal_xy
    gt = cfg.goal_tol
    ax.add_patch(mpatches.Rectangle(
        (gx - gt, gy - gt), 2 * gt, 2 * gt,
        color="limegreen", alpha=0.25, zorder=2))
    ax.scatter(*cfg.goal_xy, marker="*", s=200, color="limegreen", zorder=6)
    ax.set_xlim(-0.2, cfg.world_max + 0.2)
    ax.set_ylim(-0.2, cfg.world_max + 0.2)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.20)


def _draw_trajectories(ax, traj, pos_0_cpu, rho_vals):
    for i in range(traj.shape[0]):
        viol = float(rho_vals[i]) < 0
        c, a = ("tab:red", 0.80) if viol else ("steelblue", 0.42)
        ax.plot(traj[i, :, 0].numpy(), traj[i, :, 1].numpy(),
                "-", lw=1.1, alpha=a, color=c, zorder=3)
        ax.scatter(float(pos_0_cpu[i, 0]), float(pos_0_cpu[i, 1]),
                   s=12, color="black", zorder=5)


def _draw_hist(ax, metrics, label, color, eps):
    nom = metrics["nominal_rho_values"]
    rnd = metrics["rand_rho_values"]
    pgd = metrics["pgd_rho_values"]
    lo  = float(np.concatenate([nom, rnd, pgd]).min()) - 0.05
    hi  = float(np.concatenate([nom, rnd, pgd]).max()) + 0.05
    bins = np.linspace(lo, hi, 40)
    sat  = pgd >= 0
    ax.hist(pgd[sat],  bins=bins, color="steelblue", alpha=0.55,
            label=f"PGD sat  ({sat.mean():.0%})")
    ax.hist(pgd[~sat], bins=bins, color="tab:red",   alpha=0.55,
            label=f"PGD viol ({(~sat).mean():.0%})")
    ax.hist(rnd, bins=bins, histtype="step", lw=1.8, color="tab:orange",
            label=f"Random  ({metrics['rand_sat_rate']:.0%})")
    ax.hist(nom, bins=bins, histtype="step", lw=1.5, color="black", ls="--",
            label=f"Nominal  ({metrics['nominal_sat_rate']:.0%})")
    ax.axvline(0, color="gray", lw=0.9, ls=":")
    ax.axvline(float(pgd.mean()), color="steelblue", lw=1.0, ls="--")
    ax.set_xlabel("ρ", fontsize=8); ax.set_ylabel("Count", fontsize=8)
    ax.set_title(f"{label}  —  ρ distribution  (ε={eps})", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.20)


def plot_results(
    aug_m:  dict, rarl_m: dict,
    aug_h:  dict, rarl_h: dict,
    aug_problem, rarl_problem,
    cfg: PlanningConfig,
    eps: float,
    n_pgd_steps: int,
    device: torch.device,
    aug_rm:  RolloutModule,
    rarl_rm: RolloutModule,
    out: str = PLOT_OUT,
) -> None:
    pos_aug,  rho_aug,  traj_aug  = _sample_pgd(
        aug_problem,  aug_rm,  N_TRAJ, 7, eps, n_pgd_steps, device)
    pos_rarl, rho_rarl, traj_rarl = _sample_pgd(
        rarl_problem, rarl_rm, N_TRAJ, 7, eps, n_pgd_steps, device)

    fig = plt.figure(figsize=(13, 15))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30)

    # Row 0: trajectories under PGD
    for col, (title, problem, pos_s, traj, rho_s) in enumerate([
        ("Std+Aug  (random δ in training)", aug_problem,  pos_aug,  traj_aug,  rho_aug),
        ("RARL  (neural adversary)",        rarl_problem, pos_rarl, traj_rarl, rho_rarl),
    ]):
        ax    = fig.add_subplot(gs[0, col])
        sat_r = (rho_s >= 0).mean()
        _draw_env(ax, problem, cfg,
                  f"{title}\nPGD sat = {sat_r:.0%}  (ε = {eps} m)")
        _draw_trajectories(ax, traj, pos_s.cpu(), rho_s)
        ax.plot([], [], "-", color="steelblue", lw=1.5, label=f"ok ({sat_r:.0%})")
        ax.plot([], [], "-", color="tab:red",   lw=1.5, label=f"viol ({1-sat_r:.0%})")
        ax.legend(fontsize=7, loc="upper right")

    # Row 1: histograms
    for col, (ms, label, color) in enumerate([
        (aug_m,  "Std+Aug", "tab:orange"),
        (rarl_m, "RARL",    "tab:red"),
    ]):
        _draw_hist(fig.add_subplot(gs[1, col]), ms, label, color, eps)

    # Row 2: training curves
    ax_nom  = fig.add_subplot(gs[2, 0])
    ax_qual = fig.add_subplot(gs[2, 1])

    for iters, vals, color, label in [
        (aug_h["iter"],  aug_h["nom_sat_rate"],  "tab:orange", "Std+Aug"),
        (rarl_h["iter"], rarl_h["nom_sat_rate"], "tab:red",    "RARL"),
    ]:
        if iters:
            ax_nom.plot(iters, vals, lw=1.8, color=color, label=label)
    ax_nom.axhline(0, color="gray", lw=0.6, ls=":"); ax_nom.axhline(1, color="gray", lw=0.6, ls=":")
    ax_nom.set_xlabel("Iteration", fontsize=8); ax_nom.set_ylabel("Satisfaction rate", fontsize=8)
    ax_nom.set_title("Nominal satisfaction during training", fontsize=9)
    ax_nom.legend(fontsize=8); ax_nom.grid(alpha=0.20)

    for iters, vals, color, label in [
        (aug_h["iter"],  aug_h.get("rho_aug_mean",  []), "tab:orange", "ρ_aug  (Std+Aug)"),
        (rarl_h["iter"], rarl_h.get("rho_adv_mean", []), "tab:red",    "ρ_adv  (RARL adversary)"),
    ]:
        if iters and vals:
            ax_qual.plot(iters, vals, lw=1.8, color=color, label=label)
    ax_qual.axhline(0, color="gray", lw=0.6, ls=":")
    ax_qual.set_xlabel("Iteration", fontsize=8); ax_qual.set_ylabel("Mean ρ under perturbation", fontsize=8)
    ax_qual.set_title("Robustness quality under perturbation during training", fontsize=9)
    ax_qual.legend(fontsize=8); ax_qual.grid(alpha=0.20)

    fig.suptitle(
        f"Sensor-Noise Navigation: Std+Aug vs RARL\n"
        f"Four-obstacle crossroad corridor · ε = {eps} m · PGD steps = {n_pgd_steps}",
        fontsize=11, y=1.002,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot → {out}")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save(path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(kwargs, path)
    print(f"  saved → {path}")


def _load(path: str, device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


_OBS_DIM = 2 + 2 + 2 * len(CrossroadSensorSTRELProblem.OBSTACLES) + \
           len(CrossroadSensorSTRELProblem.OBSTACLES)   # = 16


def _get_aug(args, device):
    if args.load_aug or args.load_all:
        if not os.path.exists(CKPT_AUG):
            raise FileNotFoundError(
                f"'{CKPT_AUG}' not found — run without --load-aug to train.")
        print(f"[Std+Aug]  loading {CKPT_AUG}")
        ck  = _load(CKPT_AUG, device)
        cfg = PlanningConfig(**ck["cfg"])
        pol = DeterministicPolicy(_OBS_DIM, 2, cfg.hidden,
                                  cfg.action_parameterization).to(device)
        pol.load_state_dict(ck["policy_state"]); pol.eval()
        problem = CrossroadSensorSTRELProblem(cfg, device)
        rm      = RolloutModule(pol, problem).to(device); rm.eval()
        return pol, problem, rm, ck.get("history",
               {"iter": [], "nom_sat_rate": [], "rho_aug_mean": []})
    kw = {}
    if args.aug_iters is not None: kw["train_iters"]      = args.aug_iters
    if args.noise     is not None: kw["sensor_noise_max"] = args.noise
    cfg = PlanningConfig(**kw)
    pol, problem, rm, history = train_standard_aug(cfg, device)
    pol.eval()
    _save(CKPT_AUG, policy_state=pol.state_dict(),
          cfg=dataclasses.asdict(cfg), history=history)
    return pol, problem, rm, history


def _get_rarl(args, device):
    if args.load_rarl or args.load_all:
        if not os.path.exists(CKPT_RARL):
            raise FileNotFoundError(
                f"'{CKPT_RARL}' not found — run without --load-rarl to train.")
        print(f"[RARL]     loading {CKPT_RARL}")
        ck  = _load(CKPT_RARL, device)
        cfg = PlanningConfig(**ck["cfg"])
        pol = DeterministicPolicy(_OBS_DIM, 2, cfg.hidden,
                                  cfg.action_parameterization).to(device)
        pol.load_state_dict(ck["policy_state"]); pol.eval()
        adv = AdversaryPolicy(hidden=64).to(device)
        adv.load_state_dict(ck["adversary_state"]); adv.eval()
        problem = CrossroadSensorSTRELProblem(cfg, device)
        rm      = RolloutModule(pol, problem).to(device); rm.eval()
        return pol, problem, rm, adv, ck.get("history",
               {"iter": [], "nom_sat_rate": [], "rho_adv_mean": []})
    kw = {}
    if args.rarl_iters is not None: kw["train_iters"]      = args.rarl_iters
    if args.noise      is not None: kw["sensor_noise_max"] = args.noise
    cfg = PlanningConfig(**kw)
    pol, problem, rm, adv, history = train_rarl(cfg, device)
    pol.eval(); adv.eval()
    _save(CKPT_RARL, policy_state=pol.state_dict(),
          adversary_state=adv.state_dict(),
          cfg=dataclasses.asdict(cfg), history=history)
    return pol, problem, rm, adv, history


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Train / load Std+Aug and RARL sensor-noise policies and compare.")
    p.add_argument("--aug-iters",  type=int,   default=None, metavar="N",
                   help="train Std+Aug for N iterations")
    p.add_argument("--rarl-iters", type=int,   default=None, metavar="N",
                   help="train RARL for N iterations")
    p.add_argument("--noise",      type=float, default=None, metavar="ε",
                   help=f"sensor noise budget (default: {PlanningConfig().sensor_noise_max})")
    p.add_argument("--pgd-steps",  type=int,   default=30,   metavar="N",
                   help="PGD attack steps for evaluation (default: 30)")
    p.add_argument("--n-eval",     type=int,   default=512,  metavar="N",
                   help="number of evaluation episodes (default: 512)")
    p.add_argument("--out",        type=str,   default=PLOT_OUT, metavar="PATH",
                   help="output plot path")
    p.add_argument("--load-aug",  action="store_true",
                   help=f"load Std+Aug from {CKPT_AUG}")
    p.add_argument("--load-rarl", action="store_true",
                   help=f"load RARL from {CKPT_RARL}")
    p.add_argument("--load-all",  action="store_true",
                   help="load both policies (shorthand)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  [1/2] Std+Aug")
    print("=" * 60)
    aug_pol, aug_problem, aug_rm, aug_hist = _get_aug(args, device)

    print()
    print("=" * 60)
    print("  [2/2] RARL")
    print("=" * 60)
    rarl_pol, rarl_problem, rarl_rm, rarl_adv, rarl_hist = _get_rarl(args, device)

    eps = args.noise if args.noise is not None else PlanningConfig().sensor_noise_max

    print(f"\nEvaluating (n={args.n_eval}, ε={eps}, PGD steps={args.pgd_steps}) …")
    aug_m  = evaluate_policy(aug_problem,  aug_rm,  args.n_eval, 42, device, eps,
                             n_pgd_steps=args.pgd_steps)
    rarl_m = evaluate_policy(rarl_problem, rarl_rm, args.n_eval, 42, device, eps,
                             n_pgd_steps=args.pgd_steps)

    rows = [
        ("Nominal  ρ mean",   "nominal_rho_mean"),
        ("Nominal  sat rate", "nominal_sat_rate"),
        ("Random   ρ mean",   "rand_rho_mean"),
        ("Random   sat rate", "rand_sat_rate"),
        ("PGD      ρ mean",   "pgd_rho_mean"),
        ("PGD      sat rate", "pgd_sat_rate"),
    ]
    print(f"\n{'':=^54}")
    print(f"  {'Metric':<26} {'Std+Aug':>12}  {'RARL':>12}")
    print("-" * 54)
    for label, key in rows:
        print(f"  {label:<26} {aug_m[key]:>12.3f}  {rarl_m[key]:>12.3f}")
    print("=" * 54)

    cfg = PlanningConfig() if args.noise is None else PlanningConfig(sensor_noise_max=args.noise)
    plot_results(
        aug_m, rarl_m, aug_hist, rarl_hist,
        aug_problem, rarl_problem,
        cfg=cfg, eps=eps, n_pgd_steps=args.pgd_steps, device=device,
        aug_rm=aug_rm, rarl_rm=rarl_rm,
        out=args.out,
    )


if __name__ == "__main__":
    main()
