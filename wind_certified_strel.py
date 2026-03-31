"""
Certified No-Critic STREL Policy Optimization  (v2 — improved)
===============================================================
Improvements over v1
---------------------
1. Single-phase training — ε anneals from 0 → wind_max over all iterations.
   At ε=0 the LiRPA bound equals rho_nom exactly, so the transition is smooth
   and there is no warmup/certified cliff.

2. Mixed loss — certified and nominal losses are blended with a weight α that
   increases from 0→1 in sync with ε:
       loss = −( α·lb  +  (1−α)·rho_nom ) / std(batch)
   At the start α=0 (pure nominal); at the end α=1 (pure certified).

3. Std normalisation replaces the EMA baseline.  Subtracting the batch mean
   and dividing by the batch std gives a proper, zero-mean, unit-variance
   gradient signal.  (The EMA mean was a no-op for gradients; the std scaling
   is the part that actually helps stabilise learning.)

4. ReLU hidden activations instead of Tanh.  Non-crossing ReLUs propagate
   IBP intervals exactly, so the certified bounds are significantly tighter
   during early training when many neurons are in their linear regime.

Training objective
------------------
  loss = −mean( (mixed − mixed.mean()) / (mixed.std() + 1e-8) )
  where  mixed = α(t)·lb  +  (1−α(t))·rho_nom
         α(t)  = t / (T−1)         (linear, 0→1)
         ε(t)  = wind_max · α(t)   (0→wind_max in sync)

Reported metrics
----------------
  rho_lb_mean   : mean certified lower bound  (formal guarantee)
  cert_sat_rate : fraction with rho_lb > 0    (formal certificate)
  nom_sat_rate  : fraction with rho_nom > 0   (nominal performance)
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from diff_certif_strel import (And, Always, AtomicPredicate, Eventually,
                               smooth_min as strel_smooth_min)


# ── LiRPA-compatible predicate ────────────────────────────────────────────────
# AtomicPredicate._mask uses `==` boolean comparisons that export to
# `onnx::If`, which auto_LiRPA cannot trace.  Since we always use a single
# spatial node with label-class 0 and ego-weight 1.0, the mask is identically
# 1.  We override _mask to return ones_like (pure tensor op, no bool/If).

class SimpleAtomicPredicate(AtomicPredicate):
    """AtomicPredicate whose mask is always 1 — safe for LiRPA tracing."""
    def _mask(self, x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(self._predicate(x))   # (batch, N_spatial, T)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanningConfig:
    horizon:                 int   = 10
    dt:                      float = 0.25
    max_speed:               float = 2
    train_iters:             int   = 2500
    batch_size:              int   = 256
    lr:                      float = 2e-3
    hidden:                  int   = 128          # smaller → tighter IBP bounds
    world_min:               float = -4
    world_max:               float = 10
    init_box:    Tuple[float, float, float, float] = (0.0, 3.0, 0.0, 3.0)
    goal_xy:     Tuple[float, float]               = (4.0, 3.00)
    goal_tol:                float = 0.45
    init_obstacle_clearance: float = 0.08
    init_goal_clearance:     float = 0.15
    integration_substeps:    int   = 4
    strel_beta:              float = 50.0   # runtime field — updated each iter
    strel_beta_min:          float = 8.0   # higher → sharper safety gradient from iteration 0
    strel_beta_max:          float = 50.0  # beta at final iteration
    violation_penalty_weight: float = 1.0  # weight on relu(-lb) / relu(-rho_nom) hinge
    safety_buffer:           float = 0.05  # direct clearance penalty margin (metres)
    safety_penalty_weight:   float = 10.0   # weight on direct obstacle-clearance penalty
    cert_warmup_frac:        float = 0.15  # fraction of iters before cert loss kicks in
    seed:                    int   = 7
    wind_max:                float = 0.01
    wind_max_start:          float = 0.0   # epsilon starts here
    rho_aggregation:         str   = "mean"   # "mean" | "percentile"
    rho_percentile:          float = 10.0    # used when rho_aggregation="percentile" (e.g. 10 = worst 10%)
    # lirpa_method: warmup phase method, used for α < lirpa_crown_start_frac.
    #   "IBP"      — fastest, loosest bounds
    #   "CROWN-IBP"— tighter (CROWN for affine, IBP for log/exp/sqrt); recommended
    #   "CROWN"    — tightest, but NaN-prone during early training when untrained
    #                trajectories are far from all obstacles: exp(-β·large_clr)→0
    #                for every obstacle, sum sits at 1e-12 for both IBP bounds,
    #                and the chord formula for log gives 0/0 = NaN.
    # lirpa_crown_start_frac: after this α fraction, switch to full CROWN.
    #   By then the policy avoids obstacles with moderate clearances so at least
    #   one exp term stays above the 1e-12 floor and CROWN is numerically safe.
    lirpa_method:            str   = "CROWN-IBP"
    lirpa_crown_start_frac:  float = 0.9   # α threshold to switch warmup→CROWN
    lirpa_rebuild_interval:  int   = 25   # rebuild BoundedModule every N iters to keep beta current
    action_parameterization: str   = "cartesian"  # "cartesian" | "polar" (sin_θ·v, cos_θ·v)
    plot_path:               str   = f"plots_certified_strel_h{horizon}_dt025_trainwind{wind_max}_{lirpa_method}.png"


# ──────────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────────

class DeterministicPolicy(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 parameterization: str = "cartesian"):
        super().__init__()
        self.parameterization = parameterization
        # Polar: output (sin_θ, cos_θ, v) — 1 extra raw dim
        raw_dim = act_dim if parameterization == "cartesian" else act_dim + 1
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU()
        )
        self.mu = nn.Linear(hidden, raw_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = torch.clamp(self.mu(self.backbone(obs)), -1.0, 1.0)
        if self.parameterization == "polar":
            return raw[:, :2] * raw[:, 2:3]   # (B, 2): direction * speed
        return raw  # (B, 2): cartesian


# ──────────────────────────────────────────────────────────────────────────────
# Navigation problem  (identical to base script)
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
        ego   = torch.tensor([1.0], dtype=torch.float32, device=device)
        # Use SimpleAtomicPredicate so the mask is ones_like (LiRPA-safe).
        safe  = SimpleAtomicPredicate(var_ind=0, threshold=0.0, labels=ego, lte=False)
        reach = SimpleAtomicPredicate(var_ind=1, threshold=0.0, labels=ego, lte=False)
        self.formula = And(
            Always(safe,  beta=cfg.strel_beta_min),
            Eventually(reach, beta=cfg.strel_beta_min),
            beta=cfg.strel_beta_min,
        )

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

    def observation(self, pos: torch.Tensor) -> torch.Tensor:
        """Wind is NOT included in the observation."""
        ws      = max(abs(self.cfg.world_min), abs(self.cfg.world_max))
        rel_goal = (self.goal.unsqueeze(0) - pos) / ws
        rel_obs  = ((self.obstacles[:, :2].unsqueeze(0) - pos.unsqueeze(1))
                    .reshape(pos.shape[0], -1) / ws)
        dist     = torch.sqrt(
            torch.sum((pos.unsqueeze(1) - self.obstacles[:, :2].unsqueeze(0)) ** 2,
                      dim=-1) + 1e-9)
        clearance = dist - self.obstacles[:, 2].unsqueeze(0)
        return torch.cat([pos / ws, rel_goal, rel_obs, clearance], dim=-1)

    def clearance(self, traj: torch.Tensor) -> torch.Tensor:
        diff = traj.unsqueeze(2) - self.obstacles[:, :2].view(1, 1, -1, 2)
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-9)
        return dist - self.obstacles[:, 2].view(1, 1, -1)

    def robustness(self, traj: torch.Tensor) -> torch.Tensor:
        safe_margin  = self.clearance(traj).min(dim=-1).values
        goal_dist    = torch.sqrt(
            torch.sum((traj - self.goal.view(1, 1, 2)) ** 2, dim=-1) + 1e-9)
        reach_margin = self.cfg.goal_tol - goal_dist
        signal = torch.stack([safe_margin, reach_margin], dim=1).unsqueeze(1)
        rho    = self.formula.evaluate(
            signal, self.spatial_labels.expand(traj.shape[0], -1, -1))
        return rho.mean(dim=1).squeeze(-1), rho[:,0]   # (batch,) — avg over time


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
# RolloutModule  — nn.Module whose only perturbed input is the wind vector
# ──────────────────────────────────────────────────────────────────────────────

class RolloutModule(nn.Module):
    """
    forward(pos_0, wind) → rho  (batch,)

    pos_0  (batch, 2): initial positions, treated as exact (eps=0 in LiRPA).
    wind   (batch, 2): constant wind per episode, perturbed in [-ε, ε]².

    Geometry constants (obstacles, goal) are registered as buffers so LiRPA
    treats them as non-perturbed constants throughout the traced graph —
    eliminating "Constant operand has batch dimension" warnings caused by
    accessing plain Python attributes with batch-dependent arithmetic.

    LiRPA propagates interval bounds through the full unrolled rollout and
    returns a certified lower bound on rho over all admissible wind vectors.
    """

    def __init__(self, policy: DeterministicPolicy,
                 problem: ThreeObstacleSTRELProblem):
        super().__init__()
        self.policy  = policy
        self.problem = problem
        self.cfg     = problem.cfg   # shortcut
        # ── geometry: registered so LiRPA sees them as non-perturbed constants ─
        self.register_buffer("obs_xy",   problem.obstacles[:, :2].clone())  # (3,2)
        self.register_buffer("obs_r",    problem.obstacles[:,  2].clone())  # (3,)
        self.register_buffer("goal_buf", problem.goal.clone())              # (2,)
        self.register_buffer("sp_lab",   problem.spatial_labels.clone())    # (1,1,H)

    def _observation(self, pos: torch.Tensor) -> torch.Tensor:
        """Observation built from registered buffers — no plain-attribute access."""
        ws        = max(abs(self.cfg.world_min), abs(self.cfg.world_max))
        rel_goal  = (self.goal_buf.unsqueeze(0) - pos) / ws                 # (B,2)
        # flatten(1): (B,3,2) → (B,6) — avoids dynamic pos.shape[0] in reshape
        rel_obs   = (self.obs_xy.unsqueeze(0) - pos.unsqueeze(1)).flatten(1) / ws
        dist      = torch.sqrt(
            torch.sum((pos.unsqueeze(1) - self.obs_xy.unsqueeze(0)) ** 2,
                      dim=-1) + 1e-9)                                        # (B,3)
        clearance = dist - self.obs_r.unsqueeze(0)                           # (B,3)
        return torch.cat([pos / ws, rel_goal, rel_obs, clearance], dim=-1)   # (B,13)

    def forward(self, pos_0: torch.Tensor, wind: torch.Tensor) -> torch.Tensor:
        """
        Roll out the policy under constant wind and return the STREL robustness.

        smooth_min / smooth_max use log(sum(exp(...))) rather than torch.logsumexp,
        so the ONNX graph only contains ReduceSum + element-wise exp/log — all
        fully supported by auto_LiRPA's IBP propagation.
        """
        cfg     = self.cfg
        pos     = pos_0                                   # (batch, 2)
        step_dt = cfg.dt / float(cfg.integration_substeps)
        pos_hist: List[torch.Tensor] = []

        for _ in range(cfg.horizon):
            obs    = self._observation(pos)               # wind not in obs
            action = self.policy(obs)
            for _ in range(cfg.integration_substeps):
                pos = pos + step_dt * (cfg.max_speed * action + wind)
                pos = torch.clamp(pos, cfg.world_min, cfg.world_max)
            pos_hist.append(pos)

        traj = torch.stack(pos_hist, dim=1)               # (batch, H, 2)

        # ── Safety margin: soft/exact min over obstacles ──────────────────────
        diff_obs  = traj.unsqueeze(2) - self.obs_xy.view(1, 1, -1, 2)
        dist_obs  = torch.sqrt(torch.sum(diff_obs * diff_obs, dim=-1) + 1e-9)
        clearance = dist_obs - self.obs_r.view(1, 1, -1)          # (B, H, 3)
        safe_margin = strel_smooth_min(clearance, beta=cfg.strel_beta)  # (B, H)

        # ── Reach margin ──────────────────────────────────────────────────────
        goal_dist    = torch.sqrt(
            torch.sum((traj - self.goal_buf.view(1, 1, 2)) ** 2, dim=-1) + 1e-9)
        reach_margin = cfg.goal_tol - goal_dist                    # (B, H)

        # ── Evaluate G(safe) ∧ F(reach) via the formula tree ─────────────────
        signal = torch.stack([safe_margin, reach_margin], dim=1).unsqueeze(1)
        lab    = self.sp_lab.expand(traj.shape[0], -1, -1)
        rho    = self.problem.formula.evaluate(signal, lab)         # (B, 1)
        return rho.squeeze(1)                                       # (B,)

    def forward_full(self, pos_0: torch.Tensor,
                     wind: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Same as forward() but also returns the worst-case clearance over the
        trajectory — used for the direct safety penalty in the training loop.

        Returns:
            rho        (B,) — STREL robustness
            min_clr    (B,) — min clearance over all timesteps and obstacles
        """
        cfg     = self.cfg
        pos     = pos_0
        step_dt = cfg.dt / float(cfg.integration_substeps)
        pos_hist: List[torch.Tensor] = []

        for _ in range(cfg.horizon):
            obs    = self._observation(pos)
            action = self.policy(obs)
            for _ in range(cfg.integration_substeps):
                pos = pos + step_dt * (cfg.max_speed * action + wind)
                pos = torch.clamp(pos, cfg.world_min, cfg.world_max)
            pos_hist.append(pos)

        traj = torch.stack(pos_hist, dim=1)                        # (B, H, 2)

        diff_obs  = traj.unsqueeze(2) - self.obs_xy.view(1, 1, -1, 2)
        dist_obs  = torch.sqrt(torch.sum(diff_obs * diff_obs, dim=-1) + 1e-9)
        clearance = dist_obs - self.obs_r.view(1, 1, -1)           # (B, H, 3)

        safe_margin  = strel_smooth_min(clearance, beta=cfg.strel_beta)  # (B, H)
        goal_dist    = torch.sqrt(
            torch.sum((traj - self.goal_buf.view(1, 1, 2)) ** 2, dim=-1) + 1e-9)
        reach_margin = cfg.goal_tol - goal_dist                    # (B, H)

        signal  = torch.stack([safe_margin, reach_margin], dim=1).unsqueeze(1)
        lab     = self.sp_lab.expand(traj.shape[0], -1, -1)
        rho     = self.problem.formula.evaluate(signal, lab).squeeze(1)  # (B,)

        # worst clearance over all timesteps and obstacles — shape (B,)
        min_clr = clearance.min(dim=-1).values.min(dim=-1).values

        return rho, min_clr


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _agg(x: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    """Aggregate a (B,) tensor to a scalar per cfg.rho_aggregation."""
    if cfg.rho_aggregation == "percentile":
        return torch.quantile(x, cfg.rho_percentile / 100.0)
    return x.mean()


def train_certified_policy(
    cfg: PlanningConfig,
    device: torch.device,
) -> Tuple[DeterministicPolicy, ThreeObstacleSTRELProblem,
           RolloutModule, BoundedModule, Dict]:
    import warnings as _warnings
    torch.manual_seed(cfg.seed)
    problem = ThreeObstacleSTRELProblem(cfg, device)

    obs_dim = 2 + 2 + 6 + 3   # pos(2) + rel_goal(2) + rel_obs(3×2) + clearance(3)
    policy      = DeterministicPolicy(obs_dim=obs_dim, act_dim=2, hidden=cfg.hidden,
                                      parameterization=cfg.action_parameterization).to(device)
    rollout_mod = RolloutModule(policy, problem).to(device)

    # ── Build LiRPA wrapper once at the start ────────────────────────────────
    # smooth_min/smooth_max use log(sum(exp(...))) which decomposes into
    # ReduceSum + element-wise exp/log — fully supported by IBP.
    dummy_pos  = torch.zeros(cfg.batch_size, 2, device=device)
    dummy_wind = torch.zeros(cfg.batch_size, 2, device=device)
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*batch dimension.*")
        lirpa_model = BoundedModule(
            rollout_mod, (dummy_pos, dummy_wind),
            device=str(device), verbose=False,
        )

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train_iters)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    history: Dict[str, List[float]] = {k: [] for k in
        ["iter", "rho_lb_mean", "cert_sat_rate", "nom_sat_rate",
         "goal_dist", "clearance", "eps", "alpha", "beta"]}

    print(f"[CertSTREL] device={device}  obs={obs_dim}  "
          f"method={cfg.lirpa_method}  hidden={cfg.hidden} (ReLU)")
    print(f"  {cfg.train_iters} iters · ε: {cfg.wind_max_start}→{cfg.wind_max} · "
          f"β: {cfg.strel_beta_min}→{cfg.strel_beta_max} · "
          f"viol_penalty={cfg.violation_penalty_weight}")

    # ── Single-phase certified training ──────────────────────────────────────
    for it in range(cfg.train_iters):
        # α linearly 0→1; ε, β, and cert-weight all grow together
        alpha       = it / max(cfg.train_iters - 1, 1)
        current_eps = cfg.wind_max_start + (cfg.wind_max - cfg.wind_max_start) * alpha
        beta        = cfg.strel_beta_min + (cfg.strel_beta_max - cfg.strel_beta_min) * alpha
        # Propagate new beta to both the formula tree and the config field used
        # by RolloutModule.forward() for the smooth_min over obstacles.
        cfg.strel_beta = beta
        _set_formula_beta(problem.formula, beta)

        # Rebuild LiRPA model periodically so the traced `beta` constant stays
        # current.  smooth_min/smooth_max capture `beta` as a Python-float constant
        # at trace time; without rebuilding, the model always uses strel_beta_min,
        # causing the 1e-12 stabiliser floor to inflate the certified lb.

        if it % cfg.lirpa_rebuild_interval == 0:
            dummy_pos_rb  = torch.zeros(cfg.batch_size, 2, device=device)
            dummy_wind_rb = torch.zeros(cfg.batch_size, 2, device=device)
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*batch dimension.*")
                lirpa_model = BoundedModule(
                    rollout_mod, (dummy_pos_rb, dummy_wind_rb),
                    device=str(device), verbose=False,
                )


        pos_0 = problem.sample_initial_positions(cfg.batch_size, generator)

        # cert_alpha: separate ramp for the certified weight.
        # Stays 0 for the first cert_warmup_frac of training so the policy
        # first learns good nominal obstacle-avoidance before the IBP bounds
        # (which are loose early on) start influencing the gradient.
        if alpha < cfg.cert_warmup_frac:
            cert_alpha = 0.0
        else:
            cert_alpha = (alpha - cfg.cert_warmup_frac) / max(1.0 - cfg.cert_warmup_frac, 1e-8)

        # ── Nominal robustness + min clearance (plain forward, wind=0) ────────
        # forward_full also returns the minimum obstacle clearance over the
        # trajectory, used for the direct safety penalty below.
        zero_wind = torch.zeros(cfg.batch_size, 2, device=device)
        rho_nom, min_clr_nom = rollout_mod.forward_full(pos_0, zero_wind)

        # ── Certified lower bound via LiRPA ──────────────────────────────────
        # Switch from the warmup method (CROWN-IBP) to full CROWN once the
        # policy is sufficiently trained (α ≥ lirpa_crown_start_frac).  By that
        # point trajectories stay near obstacles with moderate clearances, so the
        # smooth_min exp terms don't fully underflow and CROWN's chord formula
        # for log is numerically safe.
        current_method = (
            "CROWN" if alpha >= cfg.lirpa_crown_start_frac else cfg.lirpa_method
        )
        x_wind = BoundedTensor(
            zero_wind,
            PerturbationLpNorm(norm=float("inf"), eps=current_eps))
        # CROWN: wrap pos_0 with a tiny ε so _observation()'s sqrt nodes get a
        # non-zero interval width at step 0, avoiding the 0/0 chord NaN.
        if current_method == "CROWN":
            x_pos = BoundedTensor(pos_0,
                                  PerturbationLpNorm(norm=float("inf"), eps=1e-7))
        else:
            x_pos = pos_0   # IBP / CROWN-IBP: no wrapping needed
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message=".*batch dimension.*")
            lb, _ = lirpa_model.compute_bounds(
                x=(x_pos, x_wind), method=current_method)
        # lb: (batch,) — worst-case rho over all wind ∈ [-eps, eps]²

        # ── Main loss: cert_α·lb + (1−cert_α)·rho_nom ───────────────────────
        mixed = cert_alpha * lb + (1.0 - cert_alpha) * rho_nom
        if cfg.rho_aggregation == "percentile":
            # Maximise the p-th percentile (worst-case focus)
            loss_main = -_agg(mixed, cfg)
        else:
            # Per-sample standardised mean (default, more stable gradients)
            std       = mixed.detach().std() + 1e-8
            loss_main = -((mixed - mixed.detach().mean()) / std).mean()

        # ── STREL violation hinge ─────────────────────────────────────────────
        loss_viol = (cert_alpha * cfg.violation_penalty_weight * torch.relu(-lb).mean()
                     + (1.0 - cert_alpha) * cfg.violation_penalty_weight * torch.relu(-rho_nom).mean())

        # ── Direct safety penalty ─────────────────────────────────────────────
        # Penalises min clearance < safety_buffer regardless of whether safety
        # is the binding STREL constraint.  This gives an always-active obstacle-
        # avoidance gradient that is not gated by the reach term being satisfied.
        loss_safe = cfg.safety_penalty_weight * torch.relu(cfg.safety_buffer - min_clr_nom).mean()

        loss = loss_main  + loss_safe #+ loss_viol

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (it + 1) % 25 == 0:
            lb_det      = lb.detach()
            rho_lb_mean = float(_agg(lb_det, cfg))   # mean or percentile
            cert_sat    = float((lb_det > 0).float().mean())
            nom_sat     = float((rho_nom.detach() > 0).float().mean())

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
                goal_d  = float(torch.sqrt(
                    torch.sum((traj_e[:, -1] - problem.goal) ** 2,
                              dim=-1) + 1e-9).mean())
                clr_val = float(
                    problem.clearance(traj_e).min(dim=-1).values.mean())

            print(f"[Cert|{current_method}] iter={it+1:05d} β={beta:.1f} ε={current_eps:.4f} "
                  f"α={alpha:.3f} cα={cert_alpha:.3f} "
                  f"lb={rho_lb_mean:+.3f} nom={float(rho_nom.detach().mean()):+.3f} "
                  f"cert_sat={cert_sat:.1%} nom_sat={nom_sat:.1%} "
                  f"safe={float(loss_safe.detach()):.3f} "
                  f"goal_d={goal_d:.3f} clr={clr_val:+.3f}")

            history["iter"].append(it + 1)
            history["rho_lb_mean"].append(rho_lb_mean)
            history["cert_sat_rate"].append(cert_sat)
            history["nom_sat_rate"].append(nom_sat)
            history["goal_dist"].append(goal_d)
            history["clearance"].append(clr_val)
            history["eps"].append(current_eps)
            history["alpha"].append(alpha)
            history["beta"].append(beta)

    return policy, problem, rollout_mod, lirpa_model, history


# ──────────────────────────────────────────────────────────────────────────────
# Final evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_certified(
    problem:     ThreeObstacleSTRELProblem,
    rollout_mod: RolloutModule,
    lirpa_model: BoundedModule,
    n_eval:      int,
    seed:        int,
    device:      torch.device,
    wind_max:    float,
    lirpa_method: str,
) -> Dict:
    """
    Returns scalar metrics and per-episode rho arrays for three conditions:
      - nominal   : wind = 0
      - random    : constant wind per episode drawn from U([-wind_max, wind_max]²)
      - certified : LiRPA worst-case lower bound over all wind ∈ [-wind_max, wind_max]²
    """
    import warnings as _warnings

    gen   = torch.Generator(device=device).manual_seed(seed)
    pos_0 = problem.sample_initial_positions(n_eval, gen)

    # ── Nominal (wind = 0) ────────────────────────────────────────────────────
    with torch.no_grad():
        rho_nom = rollout_mod(pos_0, torch.zeros(n_eval, 2, device=device))

    # ── Random constant wind per episode ─────────────────────────────────────
    gen_rand  = torch.Generator(device=device).manual_seed(seed + 1)
    wind_rand = (2.0 * torch.rand(n_eval, 2, generator=gen_rand, device=device)
                 - 1.0) * wind_max
    with torch.no_grad():
        rho_rand = rollout_mod(pos_0, wind_rand)

    # ── Certified lower bound (LiRPA) ─────────────────────────────────────────
    x_w = BoundedTensor(torch.zeros(n_eval, 2, device=device),
                        PerturbationLpNorm(norm=float("inf"), eps=wind_max))
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*batch dimension.*")
        lb, _ = lirpa_model.compute_bounds(x=(pos_0, x_w), method=lirpa_method)

    return {
        "nominal_rho_mean":  float(rho_nom.mean()),
        "nominal_sat_rate":  float((rho_nom > 0).float().mean()),
        "rand_rho_mean":     float(rho_rand.mean()),
        "rand_sat_rate":     float((rho_rand > 0).float().mean()),
        "cert_lb_mean":      float(lb.mean()),
        "cert_sat_rate":     float((lb > 0).float().mean()),
        # per-episode arrays for histogram (and first n trajectories for plot)
        "nominal_rho_values": rho_nom.detach().cpu().numpy(),
        "rand_rho_values":    rho_rand.detach().cpu().numpy(),
        "cert_lb_values":     lb.detach().cpu().numpy(),
        "_pos_0":             pos_0,
        "_wind_rand":         wind_rand,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _rollout_with_wind(rollout_mod, pos_s, wind_vec, cfg, device):
    """
    Manual trajectory loop so we can collect positions for plotting.
    wind_vec : (B, 2) constant wind per episode.
    Returns traj (B, H, 2) cpu tensor.
    """
    step_dt = cfg.dt / float(cfg.integration_substeps)
    p  = pos_s.clone()
    ph = []
    with torch.no_grad():
        for _ in range(cfg.horizon):
            obs_s = rollout_mod.problem.observation(p)
            act_s = rollout_mod.policy(obs_s)
            for _ in range(cfg.integration_substeps):
                p = p + step_dt * (cfg.max_speed * act_s + wind_vec)
                p = torch.clamp(p, cfg.world_min, cfg.world_max)
            ph.append(p)
    return torch.stack(ph, dim=1).cpu()


def plot_results(
    cfg:         PlanningConfig,
    problem:     ThreeObstacleSTRELProblem,
    history:     Dict[str, List[float]],
    rollout_mod: RolloutModule,
    device:      torch.device,
    metrics:     Dict = None,
) -> str:
    import numpy as np
    n_show = 20

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    iters = history["iter"]

    # ── Panel 0: certified & nominal sat-rate + mean lb ──────────────────────
    ax = axes[0]
    ax.plot(iters, history["cert_sat_rate"], lw=2.0,
            label="certified sat-rate (lb > 0)")
    ax.plot(iters, history["nom_sat_rate"],  lw=2.0, ls="--",
            label="nominal sat-rate (wind=0)")
    ax.plot(iters, history["rho_lb_mean"],   lw=1.5, alpha=0.7,
            label="mean certified lb")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_title(f"STREL Certified Policy\n"
                 f"ε={cfg.wind_max}  method={cfg.lirpa_method}")
    ax.set_xlabel("Iteration"); ax.legend(fontsize=8); ax.grid(alpha=0.3)


    '''
    # ── Panel 1: ε / β / α curriculum ────────────────────────────────────────
    ax = axes[1]
    ax.plot(iters, history["eps"],   lw=2.0, color="tab:orange", label="wind ε")
    ax.plot(iters, history["alpha"], lw=2.0, color="tab:purple", ls="--",
            label="cert weight α")
    if "beta" in history and history["beta"]:
        ax2 = ax.twinx()
        ax2.plot(iters, history["beta"], lw=1.5, color="tab:blue", ls=":",
                 label="β (STREL)")
        ax2.set_ylabel("β", color="tab:blue", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title("ε / α / β Curriculum")
    ax.set_xlabel("Iteration"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    '''
    # ── Panels 2 & 3: trajectories + histogram under random wind ─────────────
    if metrics is not None and "_pos_0" in metrics:
        pos_s     = metrics["_pos_0"][:n_show]
        wind_show = metrics["_wind_rand"][:n_show]
        rho_show  = metrics["rand_rho_values"][:n_show]
    else:
        gen_plot  = torch.Generator(device=device).manual_seed(cfg.seed + 9999)
        pos_s     = problem.sample_initial_positions(n_show, gen_plot)
        gen_w     = torch.Generator(device=device).manual_seed(cfg.seed + 8888)
        wind_show = (2.0 * torch.rand(n_show, 2, generator=gen_w, device=device)
                     - 1.0) * cfg.wind_max
        with torch.no_grad():
            rho_show = rollout_mod(pos_s, wind_show).cpu().numpy()

    traj_s    = _rollout_with_wind(rollout_mod, pos_s, wind_show, cfg, device)
    pos_s_cpu = pos_s.cpu()

    # ── Panel 2: trajectories ──────────────────────────────────────────────
    ax = axes[1]
    for obs_c in problem.obstacles.cpu():
        ax.add_patch(plt.Circle(
            (float(obs_c[0]), float(obs_c[1])), float(obs_c[2]),
            color="tab:red", alpha=0.28))
    ax.add_patch(plt.Circle(cfg.goal_xy, cfg.goal_tol,
                             color="tab:green", alpha=0.22))
    ax.scatter(*cfg.goal_xy, marker="*", s=150, color="tab:green", zorder=4)
    for i in range(traj_s.shape[0]):
        violated = float(rho_show[i]) < 0
        c     = "tab:red"  if violated else "tab:blue"
        alpha = 0.70       if violated else 0.45
        lw    = 1.6        if violated else 1.3
        path  = torch.cat([pos_s_cpu[i].unsqueeze(0), traj_s[i]], dim=0)
        ax.plot(path[:, 0], path[:, 1], 'o-', markersize=2, alpha=alpha, lw=lw, color=c)
        ax.scatter(float(pos_s_cpu[i, 0]), float(pos_s_cpu[i, 1]),
                   color="black", s=14, zorder=3)
    ax.set_xlim(cfg.world_min, cfg.world_max)
    ax.set_ylim(cfg.world_min, cfg.world_max)
    ax.set_aspect("equal")
    rand_sat_str = (f"{metrics['rand_sat_rate']:.1%}" if metrics else "?")
    ax.set_title(f"Trajectories — random wind (ε={cfg.wind_max})\n"
                 f"sat={rand_sat_str}  (red = violation)")
    ax.grid(alpha=0.3)

    # ── Panel 3: robustness histogram ─────────────────────────────────────────
    ax = axes[2]
    if metrics is not None and "rand_rho_values" in metrics:
        rho_vals = metrics["rand_rho_values"]
        mu       = metrics["rand_rho_mean"]
        sat      = metrics["rand_sat_rate"]
        lb_vals  = metrics["cert_lb_values"]
        all_vals = np.concatenate([rho_vals, lb_vals])
        bins     = np.linspace(all_vals.min() - 0.05, all_vals.max() + 0.05, 40)
        sat_mask = rho_vals >= 0
        ax.hist(rho_vals[sat_mask],  bins=bins, color="tab:green",
                alpha=0.65, label="Satisfied (ρ≥0)")
        ax.hist(rho_vals[~sat_mask], bins=bins, color="tab:red",
                alpha=0.65, label="Violated (ρ<0)")
        ax.hist(lb_vals, bins=bins, histtype="step", lw=1.8,
                color="tab:purple", label=f"Certified lb  sat={metrics['cert_sat_rate']:.1%}")
        ax.axvline(0,  color="black",      lw=1.2, ls="--")
        ax.axvline(mu, color="tab:blue",   lw=1.5,
                   label=f"Mean ρ rand = {mu:+.3f}")
        ax.axvline(metrics["cert_lb_mean"], color="tab:purple", lw=1.5, ls="--",
                   label=f"Mean lb = {metrics['cert_lb_mean']:+.3f}")
        n = len(rho_vals)
        ax.set_title(f"Robustness Histogram — random wind (ε={cfg.wind_max})\n"
                     f"sat={sat:.1%}  n={n}")
        ax.set_xlabel("Robustness ρ"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.set_visible(False)

    fig.tight_layout()
    base, ext = os.path.splitext(cfg.plot_path)
    plot_path = f"{base}_wind{cfg.wind_max}{ext}"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"plot → {plot_path}")
    return plot_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import dataclasses, os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = PlanningConfig()

    policy, problem, rollout_mod, lirpa_model, history = \
        train_certified_policy(cfg, device)

    print(f"\n=== Final evaluation  (n=1000, ε={cfg.wind_max}, {cfg.lirpa_method}) ===")
    metrics = evaluate_certified(
        problem, rollout_mod, lirpa_model,
        n_eval=1000, seed=cfg.seed + 9999,
        device=device, wind_max=cfg.wind_max,
        lirpa_method=cfg.lirpa_method,
    )
    print(f"  nominal    rho_mean  : {metrics['nominal_rho_mean']:+.4f}")
    print(f"  nominal    sat_rate  : {metrics['nominal_sat_rate']:.2%}")
    print(f"  random wind rho_mean : {metrics['rand_rho_mean']:+.4f}")
    print(f"  random wind sat_rate : {metrics['rand_sat_rate']:.2%}")
    print(f"  certified  lb_mean   : {metrics['cert_lb_mean']:+.4f}")
    print(f"  certified  sat_rate  : {metrics['cert_sat_rate']:.2%}")

    plot_results(cfg, problem, history, rollout_mod, device, metrics=metrics)

    os.makedirs("saved_models", exist_ok=True)
    ckpt = f"saved_models/cert_strel_h{cfg.horizon}_wind{cfg.wind_max}_{cfg.lirpa_method}_finetuning.pt"
    torch.save({"policy": policy.state_dict(),
                "cfg":    dataclasses.asdict(cfg)}, ckpt)
    print(f"model saved → {ckpt}")


if __name__ == "__main__":
    main()
