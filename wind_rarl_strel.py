"""
RARL No-Critic STREL Policy Optimization
=========================================
Extends strel_policy_optimization_no_critic.py with a learned adversary
that injects wind at every time step.

Key differences from the base (no-critic) script
--------------------------------------------------
* Adversary policy: deterministic, same observation space as the protagonist,
  outputs a 2-D wind vector scaled to [-wind_max, wind_max]².
* Wind is chosen per time-step by the adversary (not a fixed random episode noise).
* Both policies are trained simultaneously each iteration:
    - Protagonist loss = -(rho - baseline_pro).mean()   [maximize robustness]
    - Adversary  loss =  (rho - baseline_adv).mean()    [minimize robustness]
  with separate Adam optimizers and separate EMA baselines.
* Wind is NOT in the protagonist's observation (policy is wind-agnostic).
* All other settings (horizon, dt, obstacles, STREL formula, …) are identical
  to strel_policy_optimization_no_critic.py.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from diff_certif_strel import (And, Always, AtomicPredicate, Eventually, smooth_min as strel_smooth_min)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanningConfig:
    horizon:                 int   = 10
    dt:                      float = 0.25
    max_speed:               float = 2
    train_iters:             int   = 4000
    batch_size:              int   = 256
    lr:                      float = 2e-3
    grad_clip:               float = 1.0
    hidden:                  int   = 128
    world_min:               float = -4
    world_max:               float = 10
    init_box: Tuple[float, float, float, float] = (0.0, 3.0, 0.0, 3.0)
    goal_xy:  Tuple[float, float]               = (4.00, 3.00)
    goal_tol:                float = 0.45
    init_obstacle_clearance: float = 0.08
    init_goal_clearance:     float = 0.15
    integration_substeps:    int   = 1
    strel_beta_min:          float = 8.0    # beta at iteration 0 (smooth gradients)
    strel_beta_max:          float = 50.0   # beta at final iteration (sharp min/max)
    early_stop_sat:          float = 1.0    # stop when EMA sat_rate >= this threshold
    early_stop_patience:     int   = 20     # … for this many consecutive iters
    early_stop_ema:          float = 1.0    # EMA momentum for sat_rate smoothing
    violation_penalty_weight: float = 1.0   # weight on hinge penalty for violating trajs
    safety_buffer:           float = 0.05  # direct clearance penalty margin (metres)
    safety_penalty_weight:   float = 10.0   # weight on direct obstacle-clearance penalty
    rho_aggregation:         str   = "mean" # "mean" | "percentile"
    rho_percentile:          float = 10.0   # percentile when rho_aggregation="percentile"
    seed:                    int   = 7
    wind_max:                float = 0.01
    wind_max_start:          float = 0.0    # adversary output scaled by curriculum ε
    baseline_momentum:       float = 0.9    # EMA for both baselines
    action_parameterization: str   = "cartesian"  # "cartesian" | "polar" (sin_θ·v, cos_θ·v)
    plot_path:               str   = f"plots_rarl_no_critic_strel_h{horizon}_dt025_trainwind{wind_max}.png"


# ──────────────────────────────────────────────────────────────────────────────
# Policies  (protagonist + adversary share the same 3-layer ReLU architecture)
# ──────────────────────────────────────────────────────────────────────────────

class DeterministicPolicy(nn.Module):
    """
    Shared architecture for protagonist and adversary.

    Protagonist: obs → action ∈ (-1,1)²  via cartesian or polar parameterization.
      - cartesian: (ax, ay) = tanh(linear(h))
      - polar:     (sin_θ·v, cos_θ·v) where (sin_θ, cos_θ, v) = tanh(linear(h))
    Adversary always uses cartesian (wind is a 2-D Cartesian vector by nature).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 parameterization: str = "cartesian"):
        super().__init__()
        self.parameterization = parameterization
        raw_dim = act_dim if parameterization == "cartesian" else act_dim + 1
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, raw_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.mu(self.backbone(obs)))
        if self.parameterization == "polar":
            return raw[:, :2] * raw[:, 2:3]   # (B, 2): direction * speed
        return raw  # (B, 2): cartesian

class AdvPolicy(nn.Module):
    """
    Shared architecture for protagonist and adversary.

    Protagonist: obs → action ∈ (-1,1)²  via cartesian or polar parameterization.
      - cartesian: (ax, ay) = tanh(linear(h))
      - polar:     (sin_θ·v, cos_θ·v) where (sin_θ, cos_θ, v) = tanh(linear(h))
    Adversary always uses cartesian (wind is a 2-D Cartesian vector by nature).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 parameterization: str = "cartesian"):
        super().__init__()
        self.parameterization = parameterization
        raw_dim = act_dim if parameterization == "cartesian" else act_dim + 1
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            #nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, raw_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.mu(self.backbone(obs)))
        if self.parameterization == "polar":
            return raw[:, :2] * raw[:, 2:3]   # (B, 2): direction * speed
        return raw  # (B, 2): cartesian


# ──────────────────────────────────────────────────────────────────────────────
# Navigation problem
# ──────────────────────────────────────────────────────────────────────────────

class ThreeObstacleSTRELProblem:
    def __init__(self, cfg: PlanningConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.goal   = torch.tensor(cfg.goal_xy, dtype=torch.float32, device=device)
        self.obstacles = torch.tensor(
            [
                [1.75, 1.75, 0.38],
                [1.75, 3.75, 0.42],
                [3.75, 2.00, 0.34],
            ],
            dtype=torch.float32, device=device,
        )
        self.spatial_labels = torch.zeros(
            1, 1, cfg.horizon, dtype=torch.long, device=device)
        ego_only = torch.tensor([1.0], dtype=torch.float32, device=device)
        safe  = AtomicPredicate(var_ind=0, threshold=0.0, labels=ego_only, lte=False)
        reach = AtomicPredicate(var_ind=1, threshold=0.0, labels=ego_only, lte=False)
        self.formula = And(
            Always(safe,  beta=cfg.strel_beta_min),
            Eventually(reach, beta=cfg.strel_beta_min),
            beta=cfg.strel_beta_min,
        )

    # ── sampling ─────────────────────────────────────────────────────────────

    def valid_initial_mask(self, pos: torch.Tensor) -> torch.Tensor:
        diff      = pos.unsqueeze(1) - self.obstacles[:, :2].unsqueeze(0)
        dist      = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-9)
        clear     = dist - self.obstacles[:, 2].unsqueeze(0)
        goal_dist = torch.sqrt(
            torch.sum((pos - self.goal.unsqueeze(0)) ** 2, dim=-1) + 1e-9)
        return (torch.all(clear > self.cfg.init_obstacle_clearance, dim=-1)
                & (goal_dist > self.cfg.goal_tol + self.cfg.init_goal_clearance))

    def sample_initial_positions(self, n: int, gen: torch.Generator) -> torch.Tensor:
        xmin, xmax, ymin, ymax = self.cfg.init_box
        samples = torch.zeros(n, 2, dtype=torch.float32, device=self.device)
        filled  = 0
        while filled < n:
            cand = torch.stack([
                xmin + (xmax - xmin) * torch.rand(n, generator=gen, device=self.device),
                ymin + (ymax - ymin) * torch.rand(n, generator=gen, device=self.device),
            ], dim=-1)
            valid = self.valid_initial_mask(cand)
            take  = min(n - filled, int(valid.sum().item()))
            if take > 0:
                samples[filled:filled + take] = cand[valid][:take]
                filled += take
        return samples

    # ── observation (wind-free) ───────────────────────────────────────────────

    def observation(self, pos: torch.Tensor) -> torch.Tensor:
        """Protagonist observation — wind is NOT included."""
        ws        = max(abs(self.cfg.world_min), abs(self.cfg.world_max))
        rel_goal  = (self.goal.unsqueeze(0) - pos) / ws
        rel_obs   = ((self.obstacles[:, :2].unsqueeze(0) - pos.unsqueeze(1))
                     .reshape(pos.shape[0], -1) / ws)
        dist      = torch.sqrt(
            torch.sum((pos.unsqueeze(1) - self.obstacles[:, :2].unsqueeze(0)) ** 2,
                      dim=-1) + 1e-9)
        clearance = dist - self.obstacles[:, 2].unsqueeze(0)
        return torch.cat([pos / ws, rel_goal, rel_obs, clearance], dim=-1)  # (B, 13)

    # ── geometry helpers ──────────────────────────────────────────────────────

    def clearance(self, traj: torch.Tensor) -> torch.Tensor:
        diff = traj.unsqueeze(2) - self.obstacles[:, :2].view(1, 1, -1, 2)
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-9)
        return dist - self.obstacles[:, 2].view(1, 1, -1)          # (B, H, 3)

    def robustness(self, traj: torch.Tensor) -> torch.Tensor:
        safe_margin  = strel_smooth_min(self.clearance(traj), beta=self.formula.beta)#.min(dim=-1).values      # (B, H)
        goal_dist    = torch.sqrt(
            torch.sum((traj - self.goal.view(1, 1, 2)) ** 2, dim=-1) + 1e-9)
        reach_margin = self.cfg.goal_tol - goal_dist                 # (B, H)
        signal = torch.stack([safe_margin, reach_margin], dim=1).unsqueeze(1)
        rho    = self.formula.evaluate(
            signal, self.spatial_labels.expand(traj.shape[0], -1, -1))
        return rho.squeeze(-1).squeeze(-1)                           # (B,)

    # ── adversarial rollout ───────────────────────────────────────────────────

    def rollout(
        self,
        protagonist:       DeterministicPolicy,
        adversary:         AdvPolicy,
        pos_0:             torch.Tensor,
        wind_scale:        float,
        detach_protagonist: bool = False,
        detach_adversary:   bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out protagonist vs adversary for cfg.horizon steps.

        At each step:
          1. Protagonist observes pos (no wind) → action ∈ (-1,1)²
          2. Adversary  observes pos (same obs)  → raw ∈ (-1,1)²
             wind = raw * wind_scale             → bounded in [-wind_scale, wind_scale]²
          3. Integrate: pos += step_dt * (max_speed * action + wind)

        Returns
        -------
        traj  : (batch, horizon, 2)
        winds : (batch, horizon, 2)
        """
        cfg      = self.cfg
        pos      = pos_0
        step_dt  = cfg.dt / float(cfg.integration_substeps)
        pos_hist:  List[torch.Tensor] = []
        wind_hist: List[torch.Tensor] = []

        for _ in range(cfg.horizon):
            obs    = self.observation(pos)                    # (B, 13) — no wind
            action = protagonist(obs)                         # (B, 2) ∈ (-1,1)²
            wind   = adversary(obs) * wind_scale              # (B, 2) ∈ (-ε,ε)²
            if detach_protagonist:
                action = action.detach()
            if detach_adversary:
                wind = wind.detach()

            wind_hist.append(wind)
            for _ in range(cfg.integration_substeps):
                pos = pos + step_dt * (cfg.max_speed * action + wind)
                pos = torch.clamp(pos, cfg.world_min, cfg.world_max)
            pos_hist.append(pos)

        traj  = torch.stack(pos_hist,  dim=1)                 # (B, H, 2)
        winds = torch.stack(wind_hist, dim=1)                 # (B, H, 2)
        return traj, winds

    # ── random-wind rollout ───────────────────────────────────────────────────

    def rollout_random_wind(
        self,
        protagonist: DeterministicPolicy,
        pos_0:       torch.Tensor,
        wind_scale:  float,
        gen:         torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out protagonist with i.i.d. uniform random wind at each step.
        Wind is sampled from U([-wind_scale, wind_scale]²) independently per
        step and episode — no adversary policy involved.

        Returns
        -------
        traj  : (batch, horizon, 2)
        winds : (batch, horizon, 2)
        """
        cfg     = self.cfg
        pos     = pos_0
        step_dt = cfg.dt / float(cfg.integration_substeps)
        pos_hist:  List[torch.Tensor] = []
        wind_hist: List[torch.Tensor] = []

        for _ in range(cfg.horizon):
            obs    = self.observation(pos)
            action = protagonist(obs)
            wind   = (2.0 * torch.rand(pos.shape[0], 2,
                                        generator=gen, device=self.device) - 1.0) * wind_scale
            wind_hist.append(wind)
            for _ in range(cfg.integration_substeps):
                pos = pos + step_dt * (cfg.max_speed * action + wind)
                pos = torch.clamp(pos, cfg.world_min, cfg.world_max)
            pos_hist.append(pos)

        traj  = torch.stack(pos_hist,  dim=1)
        winds = torch.stack(wind_hist, dim=1)
        return traj, winds


# ──────────────────────────────────────────────────────────────────────────────
# Beta annealing helper
# ──────────────────────────────────────────────────────────────────────────────

def _set_formula_beta(formula, beta: float) -> None:
    """Recursively update beta in every node of a STREL formula tree."""
    if hasattr(formula, "beta"):
        formula.beta = beta
    for attr in ("left", "right", "phi", "phi1", "phi2"):
        child = getattr(formula, attr, None)
        if child is not None:
            _set_formula_beta(child, beta)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_rarl(
    cfg:    PlanningConfig,
    device: torch.device,
) -> Tuple[DeterministicPolicy, AdvPolicy,
           ThreeObstacleSTRELProblem, Dict]:
    torch.manual_seed(cfg.seed)
    problem = ThreeObstacleSTRELProblem(cfg, device)

    obs_dim = 2 + 2 + 6 + 3   # pos(2) + rel_goal(2) + rel_obs(3×2) + clearance(3)

    protagonist = DeterministicPolicy(obs_dim, 2, cfg.hidden,
                                      parameterization=cfg.action_parameterization).to(device)
    adversary   = AdvPolicy(obs_dim, 2, cfg.hidden,
                                      parameterization="cartesian").to(device)  # wind is always cartesian

    opt_pro = optim.Adam(protagonist.parameters(), lr=cfg.lr)
    opt_adv = optim.Adam(adversary.parameters(),   lr=cfg.lr/10)

    total_iters = cfg.train_iters
    sched_pro = optim.lr_scheduler.CosineAnnealingLR(opt_pro, T_max=total_iters)
    sched_adv = optim.lr_scheduler.CosineAnnealingLR(opt_adv, T_max=total_iters)

    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    baseline_pro = 0.0
    baseline_adv = 0.0
    mom          = cfg.baseline_momentum

    sat_ema    = None   # EMA of sat_rate for early stopping
    sat_streak = 0      # consecutive iters where EMA sat_rate >= early_stop_sat

    history: Dict[str, List[float]] = {k: [] for k in
        ["iter", "rho_mean", "sat_rate", "goal_dist", "clearance",
         "wind_mag", "wind_scale"]}

    print(f"[RARL-NoCritic] device={device}  obs={obs_dim}  "
          f"horizon={cfg.horizon}  dt={cfg.dt}")
    print(f"  train_iters={cfg.train_iters}  wind_max={cfg.wind_max}  "
          f"wind_max_start={cfg.wind_max_start}")
    print(f"  beta: {cfg.strel_beta_min} → {cfg.strel_beta_max}  "
          f"viol_penalty={cfg.violation_penalty_weight}")

    for it in range(cfg.train_iters):
        # Curriculum: ramp adversary's wind budget and formula beta
        alpha      = it / max(cfg.train_iters - 1, 1)
        wind_scale = cfg.wind_max_start + (cfg.wind_max - cfg.wind_max_start) * alpha
        beta       = cfg.strel_beta_min + (cfg.strel_beta_max - cfg.strel_beta_min) * alpha
        _set_formula_beta(problem.formula, beta)

        pos_0 = problem.sample_initial_positions(cfg.batch_size, generator)

        # ── protagonist update  (adversary detached — fixed opponent) ─────────
        traj_pro, _   = problem.rollout(protagonist, adversary, pos_0, wind_scale,
                                        detach_adversary=True)
        rho_pro       = problem.robustness(traj_pro)

        # Aggregate robustness over the batch
        if cfg.rho_aggregation == "percentile":
            rho_agg = torch.quantile(rho_pro, cfg.rho_percentile / 100.0)
        else:
            rho_agg = rho_pro.mean()

        baseline_pro  = mom * baseline_pro + (1.0 - mom) * float(rho_agg.detach())
        loss_rho      = -(rho_agg - baseline_pro)

        # Violation penalty: hinge on trajectories with negative robustness
        loss_viol = cfg.violation_penalty_weight * torch.relu(-rho_pro).mean()

        # Direct safety penalty: penalise coming within safety_buffer of any
        # obstacle at any timestep — always active, not gated by STREL structure.
        # clearance(traj_pro): (B, H, 3) → min over obstacles → (B, H) → min over time → (B,)
        min_clr_pro = problem.clearance(traj_pro).min(dim=-1).values.min(dim=-1).values
        loss_safe   = cfg.safety_penalty_weight * torch.relu(cfg.safety_buffer - min_clr_pro).mean()

        loss_pro  = loss_rho + loss_viol + loss_safe

        opt_pro.zero_grad(set_to_none=True)
        loss_pro.backward()
        torch.nn.utils.clip_grad_norm_(protagonist.parameters(), cfg.grad_clip)
        opt_pro.step()
        sched_pro.step()

        # ── adversary update  (protagonist detached — fixed opponent) ─────────
        # Rollout always runs (needed for logging); weights updated every 3 steps.
        traj_adv, winds = problem.rollout(protagonist, adversary, pos_0, wind_scale,
                                          detach_protagonist=True)
        rho_adv      = problem.robustness(traj_adv)
        baseline_adv = mom * baseline_adv + (1.0 - mom) * float(rho_adv.mean().detach())

        if it % 2 == 0:
            loss_adv = (rho_adv - baseline_adv).mean()
            opt_adv.zero_grad(set_to_none=True)
            loss_adv.backward()
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), cfg.grad_clip)
            opt_adv.step()
            sched_adv.step()

        # ── logging (use adversary-rollout stats as the "hard" benchmark) ─────
        rho_det  = rho_adv.detach()
        sat_rate = float((rho_det > 0).float().mean())
        goal_d   = float(torch.sqrt(
            torch.sum((traj_adv[:, -1] - problem.goal) ** 2, dim=-1)
            + 1e-9).mean().detach())
        clr_val  = float(problem.clearance(traj_adv).min(dim=-1).values.mean().detach())
        wind_mag = float(winds.norm(dim=-1).mean().detach())

        history["iter"].append(it + 1)
        history["rho_mean"].append(float(rho_det.mean()))
        history["sat_rate"].append(sat_rate)
        history["goal_dist"].append(goal_d)
        history["clearance"].append(clr_val)
        history["wind_mag"].append(wind_mag)
        history["wind_scale"].append(wind_scale)

        # ── early stopping ────────────────────────────────────────────────────
        sat_ema = (sat_rate if sat_ema is None
                   else cfg.early_stop_ema * sat_ema + (1.0 - cfg.early_stop_ema) * sat_rate)
        if sat_ema >= cfg.early_stop_sat:
            sat_streak += 1
        else:
            sat_streak = 0
        if sat_streak >= cfg.early_stop_patience:
            print(f"[RARL] Early stop at iter={it+1} "
                  f"(EMA sat={sat_ema:.2%} ≥ {cfg.early_stop_sat:.0%} "
                  f"for {cfg.early_stop_patience} iters)")
            break

        if (it + 1) % 25 == 0:
            print(f"[RARL] iter={it+1:04d} β={beta:.1f} ε={wind_scale:.3f} "
                  f"rho={float(rho_det.mean()):+.3f} "
                  f"b_pro={baseline_pro:+.3f} b_adv={baseline_adv:+.3f} "
                  f"sat={sat_rate:.2%} (ema={sat_ema:.2%}) "
                  f"viol={float(loss_viol.detach()):.3f} "
                  f"safe={float(loss_safe.detach()):.3f} "
                  f"goal_d={goal_d:.3f} clr={clr_val:+.3f} "
                  f"|w|={wind_mag:.3f}")

    return protagonist, adversary, problem, history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    problem:     ThreeObstacleSTRELProblem,
    protagonist: DeterministicPolicy,
    adversary:   AdvPolicy,
    n_eval:      int,
    seed:        int,
    wind_scale:  float,
) -> Dict:
    """
    Return scalar metrics plus per-episode rho arrays for three conditions:
      - adversarial : learned adversary at wind_scale
      - random      : i.i.d. uniform wind at wind_scale (same magnitude)
      - nominal     : zero wind
    """
    gen   = torch.Generator(device=problem.device).manual_seed(seed)
    pos_0 = problem.sample_initial_positions(n_eval, gen)

    gen_rand = torch.Generator(device=problem.device).manual_seed(seed + 1)

    with torch.no_grad():
        # Vs learned adversary
        traj_adv, _  = problem.rollout(protagonist, adversary, pos_0, wind_scale)
        rho_adv      = problem.robustness(traj_adv)

        # Vs random uniform wind (same magnitude, no adversary policy)
        traj_rand, _ = problem.rollout_random_wind(protagonist, pos_0,
                                                   wind_scale, gen_rand)
        rho_rand     = problem.robustness(traj_rand)

        # Nominal (zero wind)
        traj_nom, _  = problem.rollout(protagonist, adversary, pos_0, 0.0)
        rho_nom      = problem.robustness(traj_nom)

    return {
        "adv_rho_mean":    float(rho_adv.mean()),
        "adv_sat_rate":    float((rho_adv > 0).float().mean()),
        "rand_rho_mean":   float(rho_rand.mean()),
        "rand_sat_rate":   float((rho_rand > 0).float().mean()),
        "nom_rho_mean":    float(rho_nom.mean()),
        "nom_sat_rate":    float((rho_nom > 0).float().mean()),
        "goal_dist_adv":   float(torch.sqrt(
            torch.sum((traj_adv[:, -1] - problem.goal) ** 2, dim=-1)
            + 1e-9).mean()),
        "clearance_adv":   float(
            problem.clearance(traj_adv).min(dim=-1).values.mean()),
        "goal_dist_rand":  float(torch.sqrt(
            torch.sum((traj_rand[:, -1] - problem.goal) ** 2, dim=-1)
            + 1e-9).mean()),
        "clearance_rand":  float(
            problem.clearance(traj_rand).min(dim=-1).values.mean()),
        "adv_rho_values":  rho_adv.cpu().numpy(),
        "rand_rho_values": rho_rand.cpu().numpy(),
        "nom_rho_values":  rho_nom.cpu().numpy(),
        # keep trajectory tensors for plotting (small n_traj subset used)
        "_traj_adv":  traj_adv,
        "_traj_rand": traj_rand,
        "_pos_0":     pos_0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _draw_arena(ax, problem, cfg):
    """Draw obstacles and goal on a trajectory axes."""
    for obs_c in problem.obstacles.cpu():
        ax.add_patch(plt.Circle(
            (float(obs_c[0]), float(obs_c[1])), float(obs_c[2]),
            color="tab:red", alpha=0.28))
    ax.add_patch(plt.Circle(cfg.goal_xy, cfg.goal_tol,
                             color="tab:green", alpha=0.22))
    ax.scatter(*cfg.goal_xy, marker="*", s=150, color="tab:green", zorder=4)
    ax.set_xlim(cfg.world_min, cfg.world_max)
    ax.set_ylim(cfg.world_min, cfg.world_max)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def _draw_trajectories(ax, traj_tensor, pos_0_tensor, color,
                       rho_vals=None, n_show=20):
    """
    Plot the first n_show trajectories.
    Trajectories with rho < 0 (violations) are drawn in red;
    satisfying trajectories use `color`.
    """
    traj  = traj_tensor[:n_show].cpu()
    pos_0 = pos_0_tensor[:n_show].cpu()
    for i in range(traj.shape[0]):
        violated = (rho_vals is not None and float(rho_vals[i]) < 0)
        c     = "tab:red" if violated else color
        alpha = 0.70      if violated else 0.40
        lw    = 1.6       if violated else 1.2
        path  = torch.cat([pos_0[i].unsqueeze(0), traj[i]], dim=0)
        ax.plot(path[:, 0], path[:, 1], 'o-', markersize=2, alpha=alpha, lw=lw, color=c)
        ax.scatter(float(pos_0[i, 0]), float(pos_0[i, 1]),
                   color="black", s=14, zorder=3)


def plot_results(
    cfg:         PlanningConfig,
    problem:     ThreeObstacleSTRELProblem,
    protagonist: DeterministicPolicy,
    adversary:   AdvPolicy,
    history:     Dict[str, List[float]],
    device:      torch.device,
    metrics:     Dict = None,
    eval_wind:   float = None,
) -> str:
    if eval_wind is None:
        eval_wind = cfg.wind_max

    # 5-panel layout: training | wind curriculum | traj (adv) | traj (random) | histogram
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    iters = history["iter"]

    # ── Panel 0: rho & sat-rate training curves ──────────────────────────────
    ax = axes[0]
    ax.plot(iters, history["rho_mean"], lw=2.0, label="mean ρ (vs adversary)")
    ax.plot(iters, history["sat_rate"], lw=2.0, ls="--", label="sat rate (vs adversary)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_title(f"RARL STREL — Training\ntrain wind_max={cfg.wind_max}")
    ax.set_xlabel("Iteration"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 1: adversary wind budget & wind magnitude ──────────────────────
    #ax = axes[1]
    #ax.plot(iters, history["wind_scale"], lw=2.0, color="tab:orange", label="wind budget ε")
    #ax.plot(iters, history["wind_mag"],   lw=1.5, color="tab:red",    label="|wind| mean")
    #ax.set_title("Adversary Wind Curriculum")
    #ax.set_xlabel("Iteration"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panels 2 & 3: trajectories — reuse the eval tensors ──────────────────
    n_show = 20
    has_metrics = metrics is not None and "_traj_adv" in metrics

    if has_metrics:
        traj_adv      = metrics["_traj_adv"]
        traj_rand     = metrics["_traj_rand"]
        pos_0         = metrics["_pos_0"]
        adv_sat       = metrics["adv_sat_rate"]
        rand_sat      = metrics["rand_sat_rate"]
        adv_rho_show  = metrics["adv_rho_values"][:n_show]
        rand_rho_show = metrics["rand_rho_values"][:n_show]
    else:
        # Fallback: generate a small batch
        gen_plot  = torch.Generator(device=device).manual_seed(cfg.seed + 9999)
        pos_0     = problem.sample_initial_positions(n_show, gen_plot)
        with torch.no_grad():
            traj_adv, _  = problem.rollout(protagonist, adversary, pos_0, eval_wind)
            gen_rand_plot = torch.Generator(device=device).manual_seed(cfg.seed + 8888)
            traj_rand, _ = problem.rollout_random_wind(
                protagonist, pos_0, eval_wind, gen_rand_plot)
            rho_adv_fb   = problem.robustness(traj_adv)
            rho_rand_fb  = problem.robustness(traj_rand)
        adv_sat       = float((rho_adv_fb  > 0).float().mean())
        rand_sat      = float((rho_rand_fb > 0).float().mean())
        adv_rho_show  = rho_adv_fb.cpu().numpy()
        rand_rho_show = rho_rand_fb.cpu().numpy()

    # Panel 2 — vs adversary  (violations in red)
    ax = axes[1]
    _draw_arena(ax, problem, cfg)
    _draw_trajectories(ax, traj_adv, pos_0, color="tab:orange",
                       rho_vals=adv_rho_show, n_show=n_show)
    ax.set_title(f"Trajectories vs Adversary (ε={eval_wind})\nsat={adv_sat:.1%}")

    # Panel 3 — vs random wind  (violations in red)
    ax = axes[2]
    _draw_arena(ax, problem, cfg)
    _draw_trajectories(ax, traj_rand, pos_0, color="tab:blue",
                       rho_vals=rand_rho_show, n_show=n_show)
    ax.set_title(f"Trajectories vs Random Wind (ε={eval_wind})\nsat={rand_sat:.1%}")

    # ── Panel 4: robustness histogram (adversarial vs random) ────────────────
    ax = axes[3]
    if has_metrics and "adv_rho_values" in metrics:
        adv_vals  = metrics["adv_rho_values"]
        rand_vals = metrics["rand_rho_values"]
        all_vals  = np.concatenate([adv_vals, rand_vals])
        bins = np.linspace(all_vals.min() - 0.05, all_vals.max() + 0.05, 40)

        ax.hist(rand_vals, bins=bins, alpha=0.55, color="tab:blue",
                label=f"Random wind  sat={metrics['rand_sat_rate']:.1%}")
        ax.hist(adv_vals,  bins=bins, alpha=0.55, color="tab:orange",
                label=f"Adversarial  sat={metrics['adv_sat_rate']:.1%}")
        ax.axvline(0, color="black", lw=1.2, ls="--", label="ρ = 0")
        ax.axvline(metrics["rand_rho_mean"], color="tab:blue",   lw=1.5,
                   label=f"Mean ρ rand = {metrics['rand_rho_mean']:+.3f}")
        ax.axvline(metrics["adv_rho_mean"],  color="tab:orange", lw=1.5,
                   label=f"Mean ρ adv  = {metrics['adv_rho_mean']:+.3f}")
        n = len(adv_vals)
        ax.set_title(f"Robustness Histogram\neval wind={eval_wind}  (n={n})")
        ax.set_xlabel("Robustness ρ"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.set_visible(False)

    fig.tight_layout()
    base, ext = os.path.splitext(cfg.plot_path)
    plot_path = f"{base}_evalwind{eval_wind}{ext}"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"plot → {plot_path}")
    return plot_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse, dataclasses
    parser = argparse.ArgumentParser(description="RARL No-Critic STREL")
    parser.add_argument("--load", metavar="CKPT", default=None,
                        help="checkpoint path; skip training and evaluate only")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="number of episodes for evaluation (default: 1000)")
    parser.add_argument("--wind", type=float, default=None,
                        help="wind magnitude at eval time; defaults to trained wind_max")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = 2 + 2 + 6 + 3   # 13

    if args.load:
        # ── Evaluation-only mode ──────────────────────────────────────────────
        ckpt        = torch.load(args.load, map_location=device)
        cfg         = PlanningConfig(**ckpt["cfg"])
        protagonist = DeterministicPolicy(obs_dim, 2, cfg.hidden,
                                          parameterization=cfg.action_parameterization).to(device)
        adversary   = AdvPolicy(obs_dim, 2, cfg.hidden,
                                          parameterization="cartesian").to(device)
        protagonist.load_state_dict(ckpt["protagonist"])
        adversary.load_state_dict(ckpt["adversary"])
        protagonist.eval(); adversary.eval()
        problem = ThreeObstacleSTRELProblem(cfg, device)
        # Formula initialised with strel_beta_min; set to final trained beta.
        _set_formula_beta(problem.formula, cfg.strel_beta_max)
        eval_wind = args.wind if args.wind is not None else cfg.wind_max
        metrics   = evaluate(problem, protagonist, adversary,
                             n_eval=args.episodes, seed=cfg.seed + 9999,
                             wind_scale=eval_wind)
        history   = {k: [] for k in ["iter", "rho_mean", "sat_rate",
                                      "goal_dist", "clearance", "wind_mag", "wind_scale"]}
        print(f"  trained wind_max : {cfg.wind_max}")
        print(f"  eval wind        : {eval_wind}")
    else:
        # ── Training mode ─────────────────────────────────────────────────────
        cfg = PlanningConfig()
        protagonist, adversary, problem, history = train_rarl(cfg, device)
        eval_wind = cfg.wind_max
        metrics   = evaluate(problem, protagonist, adversary,
                             n_eval=args.episodes, seed=cfg.seed + 9999,
                             wind_scale=eval_wind)

        # Save both protagonist and adversary in a single checkpoint.
        os.makedirs("saved_models", exist_ok=True)
        ckpt_path = f"saved_models/rarl_no_critic_h{cfg.horizon}_wind{cfg.wind_max}.pt"
        torch.save({
            "protagonist":    protagonist.state_dict(),
            "adversary":      adversary.state_dict(),
            "cfg":            dataclasses.asdict(cfg),
            "train_wind_max": cfg.wind_max,
        }, ckpt_path)
        print(f"protagonist + adversary saved → {ckpt_path}")

    # Print evaluation summary
    print("\n=== Evaluation ===")
    print(f"  vs adversary   rho_mean  : {metrics['adv_rho_mean']:+.4f}")
    print(f"  vs adversary   sat_rate  : {metrics['adv_sat_rate']:.2%}")
    print(f"  vs random wind rho_mean  : {metrics['rand_rho_mean']:+.4f}")
    print(f"  vs random wind sat_rate  : {metrics['rand_sat_rate']:.2%}")
    print(f"  nominal (ε=0)  rho_mean  : {metrics['nom_rho_mean']:+.4f}")
    print(f"  nominal (ε=0)  sat_rate  : {metrics['nom_sat_rate']:.2%}")
    print(f"  goal_dist (adv)          : {metrics['goal_dist_adv']:.4f}")
    print(f"  clearance (adv)          : {metrics['clearance_adv']:+.4f}")
    print(f"  goal_dist (rand)         : {metrics['goal_dist_rand']:.4f}")
    print(f"  clearance (rand)         : {metrics['clearance_rand']:+.4f}")

    # Robustness histograms from CLI
    n_bins = 10
    for label, vals in [("Random wind", metrics["rand_rho_values"]),
                        ("Adversarial", metrics["adv_rho_values"])]:
        print(f"\n  {label} ρ histogram  (n={len(vals)}):")
        counts, edges = np.histogram(vals, bins=n_bins)
        max_c = int(counts.max()) if counts.max() > 0 else 1
        for lo, hi, c in zip(edges[:-1], edges[1:], counts):
            bar = "█" * int(round(30 * int(c) / max_c))
            print(f"    [{lo:+.3f}, {hi:+.3f})  {bar}  {c}")

    plot_results(cfg, problem, protagonist, adversary, history, device,
                 metrics=metrics, eval_wind=eval_wind)


if __name__ == "__main__":
    main()
