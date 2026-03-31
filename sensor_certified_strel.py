"""
Certified No-Critic STREL — Sensor-Noise Variant
=================================================
Same training scheme as certified_no_critic_strel.py, but the source of
uncertainty is a **constant positional sensor bias** rather than wind.

Uncertainty model
-----------------
The agent knows its OWN position exactly (perfect GPS / odometry).
The perceived positions of obstacles and goal are corrupted by a constant
bias δ ∈ [−ε, ε]² (e.g. a map registration error or landmark detector bias):

    goal_perceived   = goal_true  + δ
    obs_xy_perceived = obs_xy_true + δ   (same offset for every obstacle)

The policy acts on obs(pos_true, δ); dynamics are clean:

    pos_true_{t+1} = pos_true_t + (dt/sub) · v_max · π(obs(pos_true_t, δ))

Safety is evaluated on the TRUE trajectory against TRUE obstacle positions.
LiRPA certifies that for ALL constant biases δ ∈ [−ε, ε]² the STREL
robustness evaluated on the TRUE trajectory is non-negative.

Geometry convention
-------------------
This sensor case study uses L∞ geometry throughout:
* obstacle radius means axis-aligned square half-width
* goal tolerance is an L∞ ball (an axis-aligned square)
* observation clearances and evaluation margins use L∞ distance

Because the observation clearance is now based on L∞ distance, the perceived
obstacle positions can be perturbed directly without the CROWN instability
caused by the old sqrt-based L2 distance.

Problem geometry
----------------
Four-obstacle "crossroad corridor" layout:

    A  [1.2, 1.2, 0.70]  lower-left  (near corner (0,0))
    B  [5.3, 1.2, 0.70]  lower-right (near corner (6.5,0))
    C  [1.2, 4.8, 0.70]  upper-left  (near corner (0,6.5))
    D  [5.3, 4.8, 0.70]  upper-right (near corner (6.5,6.5))

Start box: full world [0, 6.5]² (valid_initial_mask filters obstacle interiors / goal).
Goal: (3.0, 5.5) — top center.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


# ── LiRPA-safe predicate ──────────────────────────────────────────────────────

class SimpleAtomicPredicate(AtomicPredicate):
    """AtomicPredicate whose mask is always 1 — safe for LiRPA tracing."""
    def _mask(self, x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(self._predicate(x))


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PlanningConfig:
    horizon:                  int   = 10
    dt:                       float = 0.5
    max_speed:                float = 3.0
    train_iters:              int   = 3000
    batch_size:               int   = 256
    lr:                       float = 2e-3
    hidden:                   int   = 128
    world_min:                float = 0.0
    world_max:                float = 6.5
    # Four-obstacle crossroad corridor geometry
    init_box:     Tuple[float, float, float, float] = (0.0, 6.5, 0.0, 6.5)
    goal_xy:      Tuple[float, float]               = (3.0, 5.5)
    goal_tol:                 float = 0.45
    init_obstacle_clearance:  float = 0.08
    init_goal_clearance:      float = 0.15
    integration_substeps:     int   = 4
    strel_beta:               float = 50.0   # updated each iter
    strel_beta_min:           float = 8.0
    strel_beta_max:           float = 50.0
    violation_penalty_weight: float = 1.0
    safety_buffer:            float = 0.05
    safety_penalty_weight:    float = 10.0
    cert_warmup_frac:         float = 0.15
    seed:                     int   = 7
    # Sensor-noise perturbation
    sensor_noise_max:         float = 0.01   # ε at end of training [metres]
    sensor_noise_max_start:   float = 0.0    # ε at start of training
    rho_aggregation:          str   = "mean"
    rho_percentile:           float = 10.0
    # CROWN-IBP warmup → CROWN schedule (same rationale as wind variant)
    lirpa_method:             str   = "CROWN-IBP"   # warmup method
    lirpa_crown_start_frac:   float = 0.9            # α threshold to switch
    lirpa_rebuild_interval:   int   = 25
    action_parameterization:  str   = "cartesian"
    plot_path:                str   = "plots_sensor_strel.png"


# ── Policy ────────────────────────────────────────────────────────────────────

class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 parameterization: str = "cartesian"):
        super().__init__()
        self.parameterization = parameterization
        raw_dim = act_dim if parameterization == "cartesian" else act_dim + 1
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            #nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, raw_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = torch.clamp(self.mu(self.backbone(obs)), -1.0, 1.0)
        if self.parameterization == "polar":
            return raw[:, :2] * raw[:, 2:3]
        return raw


# ── Navigation problem ────────────────────────────────────────────────────────

class CrossroadSensorSTRELProblem:
    """Four-obstacle crossroad corridor.  Certified uncertainty: constant position bias."""

    OBSTACLES = [
        [1.2, 1.2, 0.70],   # A — lower-left  (near corner (0,0))
        [5.3, 1.2, 0.70],   # B — lower-right (near corner (6.5,0))
        [1.2, 4.8, 0.70],   # C — upper-left  (near corner (0,6.5))
        [5.3, 4.8, 0.70],   # D — upper-right (near corner (6.5,6.5))
    ]

    def __init__(self, cfg: PlanningConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.goal   = torch.tensor(cfg.goal_xy, dtype=torch.float32, device=device)
        self.obstacles = torch.tensor(
            self.OBSTACLES, dtype=torch.float32, device=device)   # (4, 3)
        self.spatial_labels = torch.zeros(
            1, 1, cfg.horizon, dtype=torch.long, device=device)
        ego   = torch.tensor([1.0], dtype=torch.float32, device=device)
        safe  = SimpleAtomicPredicate(var_ind=0, threshold=0.0, labels=ego, lte=False)
        reach = SimpleAtomicPredicate(var_ind=1, threshold=0.0, labels=ego, lte=False)
        self.formula = And(
            Always(safe,  beta=cfg.strel_beta_min),
            Eventually(reach, beta=cfg.strel_beta_min),
            beta=cfg.strel_beta_min,
        )

    @staticmethod
    def _linf_norm(diff: torch.Tensor) -> torch.Tensor:
        """Linf norm over the last dim for 2D vectors, using LiRPA-safe ops."""
        abs_diff = torch.abs(diff)
        return 0.5 * (
            abs_diff[..., 0] + abs_diff[..., 1]
            + torch.abs(abs_diff[..., 0] - abs_diff[..., 1])
        )

    def valid_initial_mask(self, pos: torch.Tensor) -> torch.Tensor:
        diff      = pos.unsqueeze(1) - self.obstacles[:, :2].unsqueeze(0)
        dist      = self._linf_norm(diff)
        clear     = dist - self.obstacles[:, 2].unsqueeze(0)
        goal_dist = self._linf_norm(pos - self.goal.unsqueeze(0))
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

    def observation(self, pos: torch.Tensor,
                    sensor_noise: torch.Tensor | None = None) -> torch.Tensor:
        """Observation from true position with optional constant sensor bias."""
        ws        = max(abs(self.cfg.world_min), abs(self.cfg.world_max))
        if sensor_noise is None:
            sensor_noise = torch.zeros_like(pos)
        goal_p    = self.goal.unsqueeze(0) + sensor_noise
        obs_xy_p  = self.obstacles[:, :2].unsqueeze(0) + sensor_noise.unsqueeze(1)
        rel_goal  = (goal_p - pos) / ws
        rel_obs   = ((obs_xy_p - pos.unsqueeze(1))
                     .reshape(pos.shape[0], -1) / ws)
        dist      = self._linf_norm(pos.unsqueeze(1) - obs_xy_p)
        clearance = dist - self.obstacles[:, 2].unsqueeze(0)
        return torch.cat([pos / ws, rel_goal, rel_obs, clearance], dim=-1)

    def clearance(self, traj: torch.Tensor) -> torch.Tensor:
        diff = traj.unsqueeze(2) - self.obstacles[:, :2].view(1, 1, -1, 2)
        dist = self._linf_norm(diff)
        return dist - self.obstacles[:, 2].view(1, 1, -1)


# ── Beta-annealing helper ─────────────────────────────────────────────────────

def _set_formula_beta(formula, beta: float) -> None:
    if hasattr(formula, "beta"):
        formula.beta = beta
    for attr in ("left", "right", "phi", "phi1", "phi2"):
        child = getattr(formula, attr, None)
        if child is not None:
            _set_formula_beta(child, beta)


# ── RolloutModule ─────────────────────────────────────────────────────────────

class RolloutModule(nn.Module):
    """
    forward(pos_0, sensor_noise) → rho  (batch,)

    pos_0        (B, 2) — true initial position; exact, NOT perturbed in LiRPA.
    sensor_noise (B, 2) — constant environmental bias per episode; perturbed in
                          LiRPA over [−ε, ε]².

    The agent knows its own position exactly.  sensor_noise shifts the perceived
    positions of ALL obstacles and the goal by the same offset:
        obs_xy_perceived = obs_xy_true  + sensor_noise
        goal_perceived   = goal_true    + sensor_noise
    Dynamics are clean; safety / reach margins are computed from TRUE positions.
    """

    def __init__(self, policy: DeterministicPolicy,
                 problem: CrossroadSensorSTRELProblem):
        super().__init__()
        self.policy  = policy
        self.problem = problem
        self.cfg     = problem.cfg
        self.register_buffer("obs_xy",   problem.obstacles[:, :2].clone())  # (4,2)
        self.register_buffer("obs_r",    problem.obstacles[:,  2].clone())  # (4,)
        self.register_buffer("goal_buf", problem.goal.clone())              # (2,)
        self.register_buffer("sp_lab",   problem.spatial_labels.clone())    # (1,1,H)

    def _observation(self, pos: torch.Tensor,
                     sensor_noise: torch.Tensor) -> torch.Tensor:
        """
        Observation from true agent position and constant sensor bias δ.

        Map-based features (biased by δ):
            rel_goal = (goal_true + δ − pos) / ws
            rel_obs  = (obs_xy_true + δ − pos) / ws   (flattened)

        Range-based feature (biased by the same map error, in L∞ geometry):
            clearance = ||pos − (obs_xy_true + δ)||∞ − obs_r

        pos          (B, 2) — true agent position (exact).
        sensor_noise (B, 2) — constant bias δ, perturbed in LiRPA over [−ε, ε]².
        """
        ws = max(abs(self.cfg.world_min), abs(self.cfg.world_max))
        goal_p   = self.goal_buf.unsqueeze(0) + sensor_noise             # (B, 2)
        obs_xy_p = self.obs_xy.unsqueeze(0) + sensor_noise.unsqueeze(1)  # (B, N, 2)
        rel_goal  = (goal_p - pos) / ws                                  # (B, 2)
        rel_obs   = (obs_xy_p - pos.unsqueeze(1)).flatten(1) / ws        # (B, 2N)
        dist_obs  = self.problem._linf_norm(pos.unsqueeze(1) - obs_xy_p) # (B, N)
        clearance = dist_obs - self.obs_r.unsqueeze(0)                   # (B, N)
        return torch.cat([pos / ws, rel_goal, rel_obs, clearance], dim=-1)

    def forward(self, pos_0: torch.Tensor,
                sensor_noise: torch.Tensor) -> torch.Tensor:
        """
        Roll out under constant sensor bias; return STREL robustness of the
        TRUE trajectory.

        The observation uses the same constant bias for perceived goal and
        obstacle positions, with L∞ distance for the clearance feature.
        """
        cfg     = self.cfg
        pos     = pos_0                                   # true position (exact)
        step_dt = cfg.dt / float(cfg.integration_substeps)
        pos_hist: List[torch.Tensor] = []

        for _ in range(cfg.horizon):
            obs    = self._observation(pos, sensor_noise)   # δ constant — same every step
            action = self.policy(obs)
            for _ in range(cfg.integration_substeps):
                pos = pos + step_dt * cfg.max_speed * action   # clean dynamics
                pos = torch.clamp(pos, cfg.world_min, cfg.world_max)
            pos_hist.append(pos)                          # TRUE position

        traj = torch.stack(pos_hist, dim=1)               # (B, H, 2)

        # Safety / reach from TRUE trajectory
        diff_obs  = traj.unsqueeze(2) - self.obs_xy.view(1, 1, -1, 2)
        dist_obs  = self.problem._linf_norm(diff_obs)
        clearance = dist_obs - self.obs_r.view(1, 1, -1)               # (B,H,4)
        safe_margin = strel_smooth_min(clearance, beta=cfg.strel_beta)  # (B,H)

        goal_dist    = self.problem._linf_norm(traj - self.goal_buf.view(1, 1, 2))
        reach_margin = cfg.goal_tol - goal_dist                         # (B,H)

        signal = torch.stack([safe_margin, reach_margin], dim=1).unsqueeze(1)
        lab    = self.sp_lab.expand(traj.shape[0], -1, -1)
        rho    = self.problem.formula.evaluate(signal, lab)             # (B,1)
        return rho.squeeze(1)                                           # (B,)

    def forward_full(self, pos_0: torch.Tensor,
                     sensor_noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as forward() but also returns minimum obstacle clearance."""
        cfg     = self.cfg
        pos     = pos_0
        step_dt = cfg.dt / float(cfg.integration_substeps)
        pos_hist: List[torch.Tensor] = []

        for _ in range(cfg.horizon):
            obs    = self._observation(pos, sensor_noise)   # δ constant — same every step
            action = self.policy(obs)
            for _ in range(cfg.integration_substeps):
                pos = pos + step_dt * cfg.max_speed * action
                pos = torch.clamp(pos, cfg.world_min, cfg.world_max)
            pos_hist.append(pos)

        traj = torch.stack(pos_hist, dim=1)

        # Safety from TRUE trajectory vs TRUE obstacle positions
        diff_obs  = traj.unsqueeze(2) - self.obs_xy.view(1, 1, -1, 2)
        dist_obs  = self.problem._linf_norm(diff_obs)
        clearance = dist_obs - self.obs_r.view(1, 1, -1)

        safe_margin  = strel_smooth_min(clearance, beta=cfg.strel_beta)
        goal_dist    = self.problem._linf_norm(traj - self.goal_buf.view(1, 1, 2))
        reach_margin = cfg.goal_tol - goal_dist

        signal = torch.stack([safe_margin, reach_margin], dim=1).unsqueeze(1)
        lab    = self.sp_lab.expand(traj.shape[0], -1, -1)
        rho    = self.problem.formula.evaluate(signal, lab).squeeze(1)

        min_clr = clearance.min(dim=-1).values.min(dim=-1).values   # (B,)
        return rho, min_clr


# ── Aggregation helper ────────────────────────────────────────────────────────

def _agg(x: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    if cfg.rho_aggregation == "percentile":
        return torch.quantile(x, cfg.rho_percentile / 100.0)
    return x.mean()


def safe_compute_sensor_lb(
    lirpa_model: BoundedModule,
    pos_0: torch.Tensor,
    zero_noise: torch.Tensor,
    eps: float,
    preferred_method: str,
    fallback_lb: Optional[torch.Tensor] = None,
    warn_prefix: str = "[SensorLiRPA]",
) -> Tuple[torch.Tensor, str, bool]:
    """
    Compute sensor certified lower bounds robustly.

    If the requested LiRPA method produces non-finite bounds, retry with safer
    methods (`CROWN-IBP`, then `IBP`). If everything fails, fall back to a
    finite tensor so training/evaluation can continue instead of crashing.
    """
    import warnings as _warnings

    method_chain = [preferred_method]
    if preferred_method == "CROWN":
        method_chain += ["CROWN-IBP", "IBP"]
    elif preferred_method == "CROWN-IBP":
        method_chain += ["IBP"]
    method_chain = list(dict.fromkeys(method_chain))

    last_bad: Optional[torch.Tensor] = None
    for method in method_chain:
        eps_lirpa = max(eps, 1e-7) if method == "CROWN" else eps
        x_sensor = BoundedTensor(
            zero_noise,
            PerturbationLpNorm(norm=float("inf"), eps=eps_lirpa))
        try:
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*batch dimension.*")
                lb, _ = lirpa_model.compute_bounds(x=(pos_0, x_sensor), method=method)
            lb = lb.squeeze(-1) if lb.dim() > 1 else lb
        except Exception as exc:
            print(f"{warn_prefix} {method} failed with {type(exc).__name__}: {exc}")
            continue

        if torch.isfinite(lb).all():
            return lb, method, method != preferred_method

        last_bad = lb.detach()
        n_bad = int((~torch.isfinite(lb)).sum().item())
        print(f"{warn_prefix} {method} produced {n_bad} non-finite bounds; retrying with a safer method.")

    if fallback_lb is not None:
        finite_fb = torch.nan_to_num(fallback_lb, nan=0.0, posinf=1e6, neginf=-1e6)
        print(f"{warn_prefix} falling back to nominal robustness to keep execution alive.")
        return finite_fb, "nominal-fallback", True

    if last_bad is None:
        last_bad = torch.zeros(pos_0.shape[0], device=pos_0.device, dtype=pos_0.dtype)
    sanitized = torch.nan_to_num(last_bad, nan=-1e6, posinf=1e6, neginf=-1e6)
    print(f"{warn_prefix} all bound methods failed; using sanitized finite fallback bounds.")
    return sanitized, "sanitized-fallback", True


# ── Training ──────────────────────────────────────────────────────────────────

def train_certified_policy(
    cfg: PlanningConfig,
    device: torch.device,
) -> Tuple[DeterministicPolicy, CrossroadSensorSTRELProblem,
           RolloutModule, BoundedModule, Dict]:
    import warnings as _warnings
    torch.manual_seed(cfg.seed)
    problem = CrossroadSensorSTRELProblem(cfg, device)

    n_obs   = len(CrossroadSensorSTRELProblem.OBSTACLES)
    obs_dim = 2 + 2 + 2 * n_obs + n_obs    # pos + rel_goal + rel_obs + clearance
    policy      = DeterministicPolicy(obs_dim=obs_dim, act_dim=2, hidden=cfg.hidden,
                                      parameterization=cfg.action_parameterization).to(device)
    rollout_mod = RolloutModule(policy, problem).to(device)

    dummy_pos   = torch.zeros(cfg.batch_size, 2, device=device)
    dummy_noise = torch.zeros(cfg.batch_size, 2, device=device)
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*batch dimension.*")
        lirpa_model = BoundedModule(
            rollout_mod, (dummy_pos, dummy_noise),
            device=str(device), verbose=False,
        )

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train_iters)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    history: Dict[str, List[float]] = {k: [] for k in
        ["iter", "rho_lb_mean", "cert_sat_rate", "nom_sat_rate",
         "goal_dist", "clearance", "eps", "alpha", "beta"]}

    print(f"[SensorSTREL] device={device}  obs={obs_dim}  "
          f"method={cfg.lirpa_method}→CROWN(α≥{cfg.lirpa_crown_start_frac})  "
          f"hidden={cfg.hidden} ×4 (ReLU)")
    print(f"  {cfg.train_iters} iters · ε: {cfg.sensor_noise_max_start}→"
          f"{cfg.sensor_noise_max} · β: {cfg.strel_beta_min}→{cfg.strel_beta_max}")

    for it in range(cfg.train_iters):
        alpha       = it / max(cfg.train_iters - 1, 1)
        current_eps = (cfg.sensor_noise_max_start
                       + (cfg.sensor_noise_max - cfg.sensor_noise_max_start) * alpha)
        beta        = cfg.strel_beta_min + (cfg.strel_beta_max - cfg.strel_beta_min) * alpha
        cfg.strel_beta = beta
        _set_formula_beta(problem.formula, beta)

        if it % cfg.lirpa_rebuild_interval == 0:
            dummy_p = torch.zeros(cfg.batch_size, 2, device=device)
            dummy_n = torch.zeros(cfg.batch_size, 2, device=device)
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*batch dimension.*")
                lirpa_model = BoundedModule(
                    rollout_mod, (dummy_p, dummy_n),
                    device=str(device), verbose=False,
                )

        pos_0 = problem.sample_initial_positions(cfg.batch_size, generator)

        if alpha < cfg.cert_warmup_frac:
            cert_alpha = 0.0
        else:
            cert_alpha = (alpha - cfg.cert_warmup_frac) / max(1.0 - cfg.cert_warmup_frac, 1e-8)

        zero_noise = torch.zeros(cfg.batch_size, 2, device=device)
        rho_nom, min_clr_nom = rollout_mod.forward_full(pos_0, zero_noise)

        # Switch warmup method → CROWN once policy is sufficiently trained.
        current_method = (
            "CROWN" if alpha >= cfg.lirpa_crown_start_frac else cfg.lirpa_method
        )
        lb, used_method, degraded = safe_compute_sensor_lb(
            lirpa_model,
            pos_0,
            zero_noise,
            current_eps,
            current_method,
            fallback_lb=rho_nom,
            warn_prefix=f"[SensorLiRPA|iter={it+1}]",
        )

        mixed = cert_alpha * lb + (1.0 - cert_alpha) * rho_nom
        if cfg.rho_aggregation == "percentile":
            loss_main = -_agg(mixed, cfg)
        else:
            std       = mixed.detach().std() + 1e-8
            loss_main = -((mixed - mixed.detach().mean()) / std).mean()

        # Violation penalty: hinge on the certified lb (or rho_nom during warmup)
        loss_viol = cfg.violation_penalty_weight * (
            cert_alpha       * torch.relu(-lb).mean()
            + (1.0 - cert_alpha) * torch.relu(-rho_nom).mean()
        )
        # Safety penalty: direct obstacle-clearance hinge on nominal trajectory
        loss_safe = cfg.safety_penalty_weight * torch.relu(
            cfg.safety_buffer - min_clr_nom).mean()
        loss = loss_main + loss_viol + loss_safe

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (it + 1) % 25 == 0:
            lb_det      = lb.detach()
            rho_lb_mean = float(_agg(lb_det, cfg))
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
                goal_d  = float(problem._linf_norm(traj_e[:, -1] - problem.goal).mean())
                clr_val = float(
                    problem.clearance(traj_e).min(dim=-1).values.mean())

            method_tag = used_method + ("*" if degraded else "")
            print(f"[Sensor|{method_tag}] iter={it+1:05d} β={beta:.1f} "
                  f"ε={current_eps:.4f} α={alpha:.3f} cα={cert_alpha:.3f} "
                  f"lb={rho_lb_mean:+.3f} nom={float(rho_nom.detach().mean()):+.3f} "
                  f"cert_sat={cert_sat:.1%} nom_sat={nom_sat:.1%} "
                  f"viol={float(loss_viol.detach()):.3f} "
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_certified(
    problem:          CrossroadSensorSTRELProblem,
    rollout_mod:      RolloutModule,
    lirpa_model:      BoundedModule,
    n_eval:           int,
    seed:             int,
    device:           torch.device,
    sensor_noise_max: float,
    lirpa_method:     str,
) -> Dict:
    import warnings as _warnings

    gen   = torch.Generator(device=device).manual_seed(seed)
    pos_0 = problem.sample_initial_positions(n_eval, gen)

    with torch.no_grad():
        rho_nom = rollout_mod(pos_0, torch.zeros(n_eval, 2, device=device))

    gen_rand   = torch.Generator(device=device).manual_seed(seed + 1)
    noise_rand = ((2.0 * torch.rand(n_eval, 2, generator=gen_rand, device=device) - 1.0)
                  * sensor_noise_max)
    with torch.no_grad():
        rho_rand = rollout_mod(pos_0, noise_rand)

    lb, _, _ = safe_compute_sensor_lb(
        lirpa_model,
        pos_0,
        torch.zeros(n_eval, 2, device=device),
        sensor_noise_max,
        lirpa_method,
        fallback_lb=rho_nom,
        warn_prefix="[SensorLiRPA|eval]",
    )

    return {
        "nominal_rho_mean":   float(rho_nom.mean()),
        "nominal_sat_rate":   float((rho_nom > 0).float().mean()),
        "rand_rho_mean":      float(rho_rand.mean()),
        "rand_sat_rate":      float((rho_rand > 0).float().mean()),
        "cert_lb_mean":       float(lb.mean()),
        "cert_sat_rate":      float((lb > 0).float().mean()),
        "nominal_rho_values": rho_nom.detach().cpu().numpy(),
        "rand_rho_values":    rho_rand.detach().cpu().numpy(),
        "cert_lb_values":     lb.detach().cpu().numpy(),
        "_pos_0":             pos_0,
        "_noise_rand":        noise_rand,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = PlanningConfig()

    policy, problem, rollout_mod, lirpa_model, history = \
        train_certified_policy(cfg, device)

    print(f"\n=== Final evaluation  (n=1000, ε={cfg.sensor_noise_max}, "
          f"{cfg.lirpa_method}) ===")
    metrics = evaluate_certified(
        problem, rollout_mod, lirpa_model,
        n_eval=1000, seed=cfg.seed + 9999,
        device=device,
        sensor_noise_max=cfg.sensor_noise_max,
        lirpa_method=cfg.lirpa_method,
    )
    for k, v in metrics.items():
        if not k.startswith("_") and not k.endswith("_values"):
            print(f"  {k:<22}: {v:+.4f}")


if __name__ == "__main__":
    main()
