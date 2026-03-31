"""
compare_policies_strel.py
=========================
Compares Standard, RARL, and Certified STREL policies under three wind
conditions:  no wind  |  random wind  |  adversarial wind.

All nine panels (and the histograms) use the EXACT SAME starting positions,
sampled at least EVAL_MIN_CLEARANCE metres from every obstacle.

First run: trains all three policies and saves checkpoints.
Subsequent runs: loads checkpoints directly — no retraining.

Grid  (3 rows × 6 columns)
──────────────────────────────────────────────────────────────────────────
         │ No-wind traj │ Rand traj │ Adv traj │  hist×3
──────────────────────────────────────────────────────────────────────────
Standard │              │           │          │  ...
RARL     │              │           │          │  ...
Certified│              │           │          │  ...
"""

import os, dataclasses, argparse
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from scipy.stats import beta as _beta_dist


def clopper_pearson_lower(k: int, N: int, delta: float = 0.05) -> float:
    """Clopper-Pearson lower confidence bound p_L for a Binomial proportion.

    Args:
        k:     number of successes  (lb > 0)
        N:     number of trials
        delta: significance level (0.05 → 95 % confidence)

    Returns:
        p_L: lower bound on the true success probability.
    """
    if k == 0:
        return 0.0
    return float(_beta_dist.ppf(delta / 2, k, N - k + 1))

# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────────────
EVAL_WIND_MAX      = 0.01    # wind magnitude for ALL evaluation conditions
EVAL_MIN_CLEARANCE = 0.30   # starting positions ≥ 30 cm from every obstacle
N_EVAL             = 1000   # episodes for histograms / sat-rate
N_SHOW             = 40     # trajectories drawn per panel (same across all panels)
PGD_STEPS          = 30     # PGD iterations for adversarial wind
SEED_EVAL          = 99
DEVICE             = torch.device("cpu")
PLOT_PATH          = "compare_wind_policies_strel.png"
CKPT_STD           = "saved_models/std_no_critic_h10_wind0.05_3000iters.pt"
CKPT_RARL          = "saved_models/rarl_no_critic_h10_wind0.05_3000iters.pt"
CKPT_CERT          = "saved_models/cert_no_critic_h10_wind0.05_3000iters.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Shared geometry — pulled from strel_policy_optimization_no_critic so that
# changing obstacles / goal in that script is automatically reflected here.
# ─────────────────────────────────────────────────────────────────────────────
def _load_geometry():
    from wind_std_strel import PlanningConfig, ThreeObstacleSTRELProblem
    _cfg  = PlanningConfig()
    _prob = ThreeObstacleSTRELProblem(_cfg, torch.device("cpu"))
    return (
        _prob.obstacles.cpu(),                   # (n_obs, 3) tensor [x, y, r]
        tuple(_prob.goal.cpu().tolist()),         # (gx, gy)
        float(_cfg.goal_tol),
        tuple(_cfg.init_box),                    # (xmin, xmax, ymin, ymax)
        float(_cfg.world_min),
        float(_cfg.world_max),
    )

OBS_DATA, GOAL_XY, GOAL_TOL, INIT_BOX, WORLD_MIN, WORLD_MAX = _load_geometry()


# ─────────────────────────────────────────────────────────────────────────────
# Fixed starting positions  (shared by all policies / conditions)
# ─────────────────────────────────────────────────────────────────────────────
def _sample_fixed_positions(n: int, min_clearance: float = EVAL_MIN_CLEARANCE,
                             seed: int = SEED_EVAL) -> torch.Tensor:
    """
    Sample n positions uniformly from INIT_BOX that are at least
    min_clearance metres from every obstacle and at least GOAL_TOL from goal.
    Uses rejection sampling with a fixed seed — reproducible across calls.
    """
    xmin, xmax, ymin, ymax = INIT_BOX
    goal = torch.tensor(GOAL_XY, dtype=torch.float32)
    gen  = torch.Generator().manual_seed(seed)
    out  = []
    while len(out) < n:
        B    = max(n * 4, 512)
        x    = xmin + (xmax - xmin) * torch.rand(B, generator=gen)
        y    = ymin + (ymax - ymin) * torch.rand(B, generator=gen)
        pos  = torch.stack([x, y], dim=1)                         # (B, 2)
        diff = pos.unsqueeze(1) - OBS_DATA[:, :2].unsqueeze(0)    # (B,3,2)
        dist = torch.sqrt((diff**2).sum(-1) + 1e-9)               # (B,3)
        clr  = dist - OBS_DATA[:, 2].unsqueeze(0)                 # (B,3)
        gdist = torch.sqrt(((pos - goal)**2).sum(-1) + 1e-9)      # (B,)
        ok = (clr.min(dim=-1).values > min_clearance) & (gdist > GOAL_TOL + 0.1)
        out.extend(pos[ok].unbind(0))
    return torch.stack(out[:n])                                    # (n, 2)


# ─────────────────────────────────────────────────────────────────────────────
# STREL robustness  G(safe) ∧ F(reach)
#
# _traj_rho_exact  — used only for the RARL state-dependent adversary rollout
#                    (state-dependent wind cannot feed through RolloutModule).
# For all other conditions, robustness is evaluated through the shared
# RolloutModule (smooth_min + formula at beta_max) to match the training
# objective exactly — see _make_eval_rollout_mod and _PolicyWrapper.rho().
# ─────────────────────────────────────────────────────────────────────────────
def _traj_rho_exact(traj: torch.Tensor) -> torch.Tensor:
    """traj (B, H, 2) → rho (B,)  using exact min/max (RARL adversary only)."""
    obs_xy = OBS_DATA[:, :2].to(traj.device)
    obs_r  = OBS_DATA[:,  2].to(traj.device)
    goal   = torch.tensor(GOAL_XY, dtype=torch.float32, device=traj.device)

    diff      = traj.unsqueeze(2) - obs_xy.view(1, 1, -1, 2)
    dist      = torch.sqrt((diff**2).sum(-1) + 1e-9)
    clearance = dist - obs_r.view(1, 1, -1)
    safe_t    = clearance.min(dim=-1).values        # (B, H)

    gdist   = torch.sqrt(((traj - goal.view(1,1,2))**2).sum(-1) + 1e-9)
    reach_t = GOAL_TOL - gdist                      # (B, H)

    G_safe  = safe_t.min(dim=-1).values             # (B,)
    F_reach = reach_t.max(dim=-1).values            # (B,)
    return torch.minimum(G_safe, F_reach)

# Alias used throughout (PGD inner loop + evaluation)
_traj_rho = _traj_rho_exact


# ─────────────────────────────────────────────────────────────────────────────
# Universal rollout
# ─────────────────────────────────────────────────────────────────────────────
def _rollout(policy: nn.Module, pos_0: torch.Tensor, wind: torch.Tensor,
             horizon: int, dt: float, max_speed: float,
             integration_substeps: int, obs_fn,
             no_grad: bool = True) -> torch.Tensor:
    """Returns traj (B, H, 2)."""
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    pos = pos_0.clone()
    step_dt = dt / float(integration_substeps)
    hist: List[torch.Tensor] = []
    with ctx:
        for _ in range(horizon):
            act = policy(obs_fn(pos))
            for _ in range(integration_substeps):
                pos = pos + step_dt * (max_speed * act + wind)
                pos = torch.clamp(pos, WORLD_MIN, WORLD_MAX)
            hist.append(pos)
    return torch.stack(hist, dim=1)


def _rollout_rarl_adv(protagonist: nn.Module, adversary: nn.Module,
                      pos_0: torch.Tensor, wind_scale: float,
                      horizon: int, dt: float, max_speed: float,
                      integration_substeps: int, obs_fn) -> torch.Tensor:
    """Roll out protagonist against state-dependent RARL adversary."""
    pos = pos_0.clone()
    step_dt = dt / float(integration_substeps)
    hist: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(horizon):
            obs  = obs_fn(pos)
            act  = protagonist(obs)
            wind = adversary(obs) * wind_scale
            for _ in range(integration_substeps):
                pos = pos + step_dt * (max_speed * act + wind)
                pos = torch.clamp(pos, WORLD_MIN, WORLD_MAX)
            hist.append(pos)
    return torch.stack(hist, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# PGD adversarial wind  (constant per episode, L∞ ball)
# ─────────────────────────────────────────────────────────────────────────────
def _pgd_wind(policy: nn.Module, pos_0: torch.Tensor, eps: float,
              horizon: int, dt: float, max_speed: float,
              integration_substeps: int, obs_fn,
              n_steps: int = PGD_STEPS) -> torch.Tensor:
    """
    Worst-case constant wind ∈ [-eps,eps]² per episode (PGD, L∞).
    Minimises mean ρ over the batch.
    """
    B    = pos_0.shape[0]
    gen  = torch.Generator().manual_seed(SEED_EVAL + 7)
    wind = (2 * torch.rand(B, 2, generator=gen) - 1) * eps   # random init
    step = eps / n_steps

    for _ in range(n_steps):
        wind = wind.detach().requires_grad_(True)
        traj = _rollout(policy, pos_0, wind, horizon, dt, max_speed,
                        integration_substeps, obs_fn, no_grad=False)
        (-_traj_rho(traj).mean()).backward()
        with torch.no_grad():
            wind = (wind + step * wind.grad.sign()).clamp(-eps, eps)

    return wind.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Certified lower-bound computation  (LiRPA IBP, works for any policy)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_cert_lb(policy: nn.Module, pos_eval: torch.Tensor, eps: float,
                     method: str = "IBP",
                     chunk_size: int = 100) -> torch.Tensor:
    """
    Wrap *policy* in a RolloutModule and compute LiRPA certified lower bounds.

    RolloutModule.forward(pos_0, wind): pos_0 is fixed (no perturbation),
    wind is perturbed in L∞ ball of radius eps.  Policies trained with tanh
    yield looser bounds than the clamp-output certified policy — informative.

    chunk_size controls how many episodes are certified at once.  IBP/CROWN-IBP
    can handle the full batch (N_EVAL=1000) in one pass.  CROWN builds one
    linear-relaxation A-matrix of shape (B, n_perturbed, n_hidden) per node in
    the unrolled graph (hundreds of nodes × hidden=128 × B=1000 ≈ several GB),
    so it must be chunked to avoid OOM-kill.  Default chunk_size=100 keeps
    CROWN well within typical memory budgets.

    Returns lb: (B,) certified lower bound on ρ for each episode.
    """
    import warnings as _w
    from certified_no_critic_strel import (
        PlanningConfig as CertCfg, ThreeObstacleSTRELProblem, RolloutModule)
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

    B    = pos_eval.shape[0]
    cfg  = CertCfg()
    prob = ThreeObstacleSTRELProblem(cfg, DEVICE)

    rollout_mod = RolloutModule(policy, prob)
    rollout_mod.eval()

    pos_0  = pos_eval.to(DEVICE)

    # For CROWN: wrapping pos_0 with ε=1e-7 prevents the 0/0 NaN that arises
    # from the chord formula for sqrt when pos_0 has zero interval width.
    ptb_wind = PerturbationLpNorm(norm=float("inf"), eps=eps)
    ptb_pos  = (PerturbationLpNorm(norm=float("inf"), eps=1e-7)
                if method == "CROWN"
                else PerturbationLpNorm(norm=float("inf"), eps=0.0))

    lb_chunks: List[torch.Tensor] = []
    for start in range(0, B, chunk_size):
        pos_chunk  = pos_0[start:start + chunk_size]
        wind_chunk = torch.zeros_like(pos_chunk)
        n = pos_chunk.shape[0]

        with _w.catch_warnings():
            _w.filterwarnings("ignore")
            lirpa = BoundedModule(rollout_mod, (pos_chunk, wind_chunk),
                                  bound_opts={"conv_mode": "patches"},
                                  device=str(DEVICE))

        x_pos  = BoundedTensor(pos_chunk,  ptb_pos)
        x_wind = BoundedTensor(wind_chunk, ptb_wind)

        with _w.catch_warnings():
            _w.filterwarnings("ignore")
            lb_c, _ = lirpa.compute_bounds(x=(x_pos, x_wind), method=method)
        lb_chunks.append(lb_c.squeeze(-1).detach())

    return torch.cat(lb_chunks, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────
def _save(path: str, **kwargs):
    torch.save(kwargs, path)
    print(f"    saved → {path}")


def _load(path: str) -> dict:
    return torch.load(path, map_location=DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Policy wrappers  (bundle policy + config + obs_fn for clean call sites)
# ─────────────────────────────────────────────────────────────────────────────
class _PolicyWrapper:
    def __init__(self, name, policy, cfg, obs_fn, adversary=None,
                 train_time_s: Optional[float] = None):
        self.name          = name
        self.policy        = policy
        self.cfg           = cfg
        self.obs_fn        = obs_fn
        self.adversary     = adversary        # RARL only
        self.train_time_s  = train_time_s     # None when loaded without timing info

    def rollout(self, pos_0, wind, no_grad=True):
        return _rollout(self.policy, pos_0, wind,
                        self.cfg.horizon, self.cfg.dt, self.cfg.max_speed,
                        self.cfg.integration_substeps, self.obs_fn,
                        no_grad=no_grad)

    def pgd(self, pos_0, eps):
        return _pgd_wind(self.policy, pos_0, eps,
                         self.cfg.horizon, self.cfg.dt, self.cfg.max_speed,
                         self.cfg.integration_substeps, self.obs_fn)

    def rollout_adv(self, pos_0, wind_scale):
        assert self.adversary is not None
        return _rollout_rarl_adv(self.policy, self.adversary, pos_0, wind_scale,
                                 self.cfg.horizon, self.cfg.dt, self.cfg.max_speed,
                                 self.cfg.integration_substeps, self.obs_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Train or load each policy
# ─────────────────────────────────────────────────────────────────────────────
def _get_standard(train_iters: Optional[int] = None,
                   wind_max: Optional[float] = None, load: bool = False) -> _PolicyWrapper:
    from strel_policy_optimization_no_critic import (
        PlanningConfig, train_no_critic_policy, DeterministicPolicy,
        ThreeObstacleSTRELProblem)


    #force = train_iters is not None or wind_max is not None

    train_time_s = None
    if load:
        if not os.path.exists(CKPT_STD):
            raise FileNotFoundError(
                f"Checkpoint '{CKPT_STD}' not found. "
                "Run with --std-iters N to train Standard first.")
        print(f"  [Standard]  loading {CKPT_STD}")
        ck  = _load(CKPT_STD)
        cfg = PlanningConfig(**ck["cfg"])
        pol = DeterministicPolicy(ck["obs_dim"], 2, cfg.hidden).to(DEVICE)
        pol.load_state_dict(ck["policy_state"])
        pol.eval()
        prob = ThreeObstacleSTRELProblem(cfg, DEVICE)
        train_time_s = ck.get("train_time_s")
    else:
        import time as _time
        iters = train_iters or 1000
        kw    = {"train_iters": iters}
        if wind_max is not None:
            kw["wind_max"] = wind_max
        print(f"  [Standard]  training ({iters} iters, wind_max={kw.get('wind_max', 'default')}) …")
        cfg = PlanningConfig(**kw)
        t0 = _time.perf_counter()
        pol, _, _, prob = train_no_critic_policy(cfg, DEVICE)
        train_time_s = _time.perf_counter() - t0
        pol.eval()
        _save(CKPT_STD, policy_state=pol.state_dict(),
              obs_dim=2+2+6+3, cfg=dataclasses.asdict(cfg),
              train_time_s=train_time_s)

    return _PolicyWrapper("Standard", pol, cfg, prob.observation,
                          train_time_s=train_time_s)


def _get_rarl(train_iters: Optional[int] = None,
              wind_max: Optional[float] = None, load: bool = False) -> _PolicyWrapper:
    from wind_rarl_strel import (
        PlanningConfig, train_rarl, DeterministicPolicy, AdvPolicy,
        ThreeObstacleSTRELProblem)

    #force = train_iters is not None or wind_max is not None
    train_time_s = None
    if load:#not force:
        if not os.path.exists(CKPT_RARL):
            raise FileNotFoundError(
                f"Checkpoint '{CKPT_RARL}' not found. "
                "Run with --rarl-iters N to train RARL first.")
        print(f"  [RARL]      loading {CKPT_RARL}")
        ck  = _load(CKPT_RARL)
        cfg = PlanningConfig(**ck["cfg"])
        obs_dim = ck["obs_dim"]
        pol = DeterministicPolicy(obs_dim, 2, cfg.hidden).to(DEVICE)
        pol.load_state_dict(ck["policy_state"])
        pol.eval()
        adv = AdvPolicy(obs_dim, 2, cfg.hidden).to(DEVICE)
        adv.load_state_dict(ck["adversary_state"])
        adv.eval()
        prob = ThreeObstacleSTRELProblem(cfg, DEVICE)
        train_time_s = ck.get("train_time_s")
    else:
        import time as _time
        iters = train_iters or 1000
        kw    = {"train_iters": iters}
        if wind_max is not None:
            kw["wind_max"] = wind_max
        print(f"  [RARL]      training ({iters} iters, wind_max={kw.get('wind_max', 'default')}) …")
        cfg = PlanningConfig(**kw)
        t0 = _time.perf_counter()
        pol, adv, prob, _ = train_rarl(cfg, DEVICE)
        train_time_s = _time.perf_counter() - t0
        pol.eval(); adv.eval()
        _save(CKPT_RARL, policy_state=pol.state_dict(),
              adversary_state=adv.state_dict(),
              obs_dim=2+2+6+3, cfg=dataclasses.asdict(cfg),
              train_time_s=train_time_s)

    return _PolicyWrapper("RARL", pol, cfg, prob.observation, adversary=adv,
                          train_time_s=train_time_s)


def _get_certified(train_iters: Optional[int] = None,
                   wind_max: Optional[float] = None, load: bool = False) -> _PolicyWrapper:
    from wind_certified_strel import (
        PlanningConfig, train_certified_policy, DeterministicPolicy,
        ThreeObstacleSTRELProblem)

    #force = train_iters is not None or wind_max is not None
    train_time_s = None
    if load:#not force:
        if not os.path.exists(CKPT_CERT):
            raise FileNotFoundError(
                f"Checkpoint '{CKPT_CERT}' not found. "
                "Run with --cert-iters N to train Certified first.")
        print(f"  [Certified] loading {CKPT_CERT}")
        ck  = _load(CKPT_CERT)
        cfg = PlanningConfig(**ck["cfg"])
        pol = DeterministicPolicy(ck["obs_dim"], 2, cfg.hidden).to(DEVICE)
        pol.load_state_dict(ck["policy_state"])
        pol.eval()
        prob = ThreeObstacleSTRELProblem(cfg, DEVICE)
        train_time_s = ck.get("train_time_s")
    else:
        import time as _time
        iters = train_iters or 1000
        kw    = {"train_iters": iters}
        if wind_max is not None:
            kw["wind_max"] = wind_max
        print(f"  [Certified] training ({iters} iters, wind_max={kw.get('wind_max', 'default')}) …")
        cfg = PlanningConfig(**kw)
        t0 = _time.perf_counter()
        pol, prob, _, _, _ = train_certified_policy(cfg, DEVICE)
        train_time_s = _time.perf_counter() - t0
        pol.eval()
        _save(CKPT_CERT, policy_state=pol.state_dict(),
              obs_dim=2+2+6+3, cfg=dataclasses.asdict(cfg),
              train_time_s=train_time_s)

    return _PolicyWrapper("Certified", pol, cfg, prob.observation,
                          train_time_s=train_time_s)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
_COLORS  = {"Standard": "tab:blue", "RARL": "tab:orange", "Certified": "tab:green"}
_COL_VIO = "tab:red"
_GRID_KW = dict(color="lightgray", linewidth=0.4, linestyle="-", zorder=0)


def _draw_arena(ax):
    """Arena with light background grid, obstacles, goal."""
    ax.set_xlim(-0.2, 4.6); ax.set_ylim(-0.2, 4.6)
    ax.set_aspect("equal")
    # Background grid (behind everything)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", **_GRID_KW)
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0, 1, 2, 3, 4])
    ax.tick_params(labelsize=10, length=3, pad=2)
    # Obstacles
    for x, y, r in OBS_DATA.tolist():
        ax.add_patch(mpatches.Circle((x, y), r,
                     color="dimgray", alpha=0.55, zorder=2))
    # Goal
    ax.plot(*GOAL_XY, "*", ms=15, color="gold",
            markeredgecolor="darkgoldenrod", zorder=4)
    ax.add_patch(mpatches.Circle(GOAL_XY, GOAL_TOL,
                 color="gold", alpha=0.15, zorder=1))


def _draw_traj_panel(ax, traj: torch.Tensor, rho: torch.Tensor, color: str,
                     pos_0: torch.Tensor = None):
    _draw_arena(ax)
    t  = traj.detach().numpy()                          # (N_SHOW, H, 2)
    r  = rho.detach().numpy()                           # (N_SHOW,)
    p0 = pos_0.detach().numpy() if pos_0 is not None \
         else t[:, 0, :]                                # (N_SHOW, 2) — fallback
    for i in range(len(t)):
        vio = r[i] < 0
        c   = _COL_VIO if vio else color
        al  = 0.78 if vio else 0.45
        lw  = 1.8  if vio else 1.2
        # Full path: initial state prepended to the rollout steps
        xs = [p0[i, 0]] + t[i, :, 0].tolist()
        ys = [p0[i, 1]] + t[i, :, 1].tolist()
        ax.plot(xs, ys, 'o-', markersize=2, color=c, alpha=al, lw=lw, zorder=5 if vio else 3)
        # Starting dot at the true initial position
        ax.plot(p0[i, 0], p0[i, 1], "o", ms=4,
                color=c, alpha=0.90 if vio else 0.65, zorder=6)


def _draw_grouped_hist(ax, rho_dict: Dict[str, torch.Tensor], title: str = ""):
    """Overlay histograms for all three methods on one axis."""
    # Compute shared bin edges from pooled data
    all_vals = torch.cat([v for v in rho_dict.values()]).numpy()
    lo, hi   = float(all_vals.min()), float(all_vals.max())
    bins     = 40
    ax.axvline(0, color="black", lw=1.1, ls="--", zorder=5)
    handles  = []
    for name, rho in rho_dict.items():
        r   = rho.detach().numpy()
        sat = float((r > 0).mean())
        c   = _COLORS[name]
        ax.hist(r, bins=bins, range=(lo, hi),
                color=c, alpha=0.50, edgecolor="none", label=name)
        # Sat-rate as a short vertical annotation
        handles.append(
            mpatches.Patch(color=c, alpha=0.7,
                           label=f"{name}  sat={sat:.0%}"))
    ax.legend(handles=handles, fontsize=11, loc="upper left",
              framealpha=0.7, handlelength=1.2)
    ax.set_xlabel("ρ", fontsize=13)
    ax.set_ylabel("count", fontsize=13)
    ax.tick_params(labelsize=11)
    if title:
        ax.set_title(title, fontsize=13, pad=3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare Standard / RARL / Certified STREL policies.")
    p.add_argument("--std-iters",   type=int,   default=None,
                   metavar="N", help="retrain Standard for N iterations")
    p.add_argument("--rarl-iters",  type=int,   default=None,
                   metavar="N", help="retrain RARL for N iterations")
    p.add_argument("--cert-iters",  type=int,   default=None,
                   metavar="N", help="retrain Certified for N iterations")
    p.add_argument("--train-wind",  type=float, default=None,
                   metavar="ε",
                   help="training wind_max for all methods (forces retrain); "
                        "eval wind defaults to the same value unless --wind is set")
    p.add_argument("--wind",        type=float, default=None,
                   metavar="ε",
                   help="evaluation wind magnitude (defaults to --train-wind "
                        f"if given, else {EVAL_WIND_MAX})")
    p.add_argument("--out",         type=str,   default=PLOT_PATH,
                   metavar="PATH", help="output plot path")
    p.add_argument("--load-std", action="store_true")
    p.add_argument("--load-rarl", action="store_true")
    p.add_argument("--load-cert", action="store_true")
    p.add_argument("--load-all", action="store_true")
    p.add_argument("--no-cert-compare", action="store_true",
                   help="skip the IBP/CROWN-IBP/CROWN comparison and the "
                        "wind-fraction sweep (saves time when not needed)")
    p.add_argument("--load-sweep", action="store_true",
                   help="load CROWN sweep results from checkpoint instead of "
                        "recomputing (implies --no-cert-compare is NOT set)")
    return p.parse_args()


def main():
    args = _parse_args()
    global EVAL_WIND_MAX, PLOT_PATH

    # Eval wind defaults to train-wind when specified, otherwise the constant.
    EVAL_WIND_MAX = args.wind if args.wind is not None \
                    else (args.train_wind if args.train_wind is not None
                          else EVAL_WIND_MAX)
    PLOT_PATH = args.out
    torch.manual_seed(0)

    # ── 1. Load / train policies ─────────────────────────────────────────────
    print("Loading / training policies …")
    wrappers: List[_PolicyWrapper] = [
        _get_standard(args.std_iters,  args.train_wind, args.load_std or args.load_all),
        _get_rarl(args.rarl_iters,     args.train_wind, args.load_rarl or args.load_all),
        _get_certified(args.cert_iters, args.train_wind, args.load_cert or args.load_all),
    ]

    # ── 2. Fixed starting positions ──────────────────────────────────────────
    # pos_show  : exactly N_SHOW positions used for ALL trajectory panels.
    # pos_eval  : N_EVAL positions (superset) used for histogram statistics.
    # Both are drawn from the same deterministic sequence so pos_show is
    # identical to pos_eval[:N_SHOW] — the same physical starting points
    # appear in every panel regardless of policy or wind condition.
    print(f"\nSampling starting positions (≥ {EVAL_MIN_CLEARANCE} m from obstacles) …")
    pos_eval = _sample_fixed_positions(N_EVAL, EVAL_MIN_CLEARANCE, SEED_EVAL)
    pos_show = pos_eval[:N_SHOW]      # first N_SHOW — shared by every traj panel

    # ── 3. Shared random wind (same draws for all policies) ──────────────────
    gen_rand       = torch.Generator().manual_seed(SEED_EVAL + 3)
    rand_wind_eval = (2 * torch.rand(N_EVAL, 2, generator=gen_rand) - 1) * EVAL_WIND_MAX
    rand_wind_show = rand_wind_eval[:N_SHOW]

    # ── 3b. Training-time comparison ─────────────────────────────────────────
    print("\n  Training times:")
    print(f"  {'Policy':<12}  {'Time (s)':>10}  {'Time (min)':>10}  {'Source':>8}")
    print("  " + "─" * 46)
    for w in wrappers:
        if w.train_time_s is not None:
            print(f"  {w.name:<12}  {w.train_time_s:>10.1f}  "
                  f"{w.train_time_s/60:>10.2f}  {'timed':>8}")
        else:
            print(f"  {w.name:<12}  {'N/A':>10}  {'N/A':>10}  "
                  f"{'no info':>8}")

    # Bar chart of training times
    _timed = [(w.name, w.train_time_s)
              for w in wrappers if w.train_time_s is not None]
    if _timed:
        _names, _times = zip(*_timed)
        fig_t, ax_t = plt.subplots(figsize=(5, 3.5))
        bars = ax_t.bar(_names, [t / 60 for t in _times],
                        color=[_COLORS[n] for n in _names],
                        edgecolor="white", linewidth=0.8)
        ax_t.bar_label(bars, fmt="%.1f min", padding=3, fontsize=10)
        ax_t.set_ylabel("Training time (min)", fontsize=12)
        ax_t.set_title("Training time comparison", fontsize=13, fontweight="bold")
        ax_t.tick_params(labelsize=11)
        ax_t.set_ylim(0, max(t / 60 for t in _times) * 1.25)
        ax_t.grid(axis="y", alpha=0.35)
        fig_t.tight_layout()
        t_path = PLOT_PATH.replace(".png", "_train_time.png")
        fig_t.savefig(t_path, dpi=130, bbox_inches="tight")
        print(f"  Saved → {t_path}")
        plt.close(fig_t)

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating …")
    # results[name][cond] = (rho_eval (N_EVAL,), traj_show (N_SHOW, H, 2))
    results: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    for w in wrappers:
        print(f"  {w.name} …", flush=True)
        r: Dict[str, Tuple] = {}

        # No wind ──────────────────────────────────────────────────────────────
        zero_eval = torch.zeros(N_EVAL, 2)
        zero_show = torch.zeros(N_SHOW, 2)
        rho_n  = _traj_rho(w.rollout(pos_eval, zero_eval))
        traj_n = w.rollout(pos_show, zero_show)   # same N_SHOW starts, no wind
        r["none"] = (rho_n, traj_n)

        # Random wind ──────────────────────────────────────────────────────────
        rho_r  = _traj_rho(w.rollout(pos_eval, rand_wind_eval))
        traj_r = w.rollout(pos_show, rand_wind_show)   # same N_SHOW starts
        r["rand"] = (rho_r, traj_r)

        # Adversarial wind ─────────────────────────────────────────────────────
        if w.adversary is not None:
            # RARL: state-dependent adversary (eval + show from same starts)
            rho_a  = _traj_rho(w.rollout_adv(pos_eval, EVAL_WIND_MAX))
            traj_a = w.rollout_adv(pos_show, EVAL_WIND_MAX)
        else:
            # PGD: compute adversarial wind independently for eval and show,
            # both starting from the respective fixed position sets.
            adv_wind_eval = w.pgd(pos_eval, EVAL_WIND_MAX)
            adv_wind_show = adv_wind_eval[:N_SHOW]   # same first N_SHOW positions
            rho_a  = _traj_rho(w.rollout(pos_eval, adv_wind_eval))
            traj_a = w.rollout(pos_show, adv_wind_show)
        r["adv"] = (rho_a, traj_a)

        results[w.name] = r

        for cond, (rho, _) in r.items():
            sat = float((rho > 0).float().mean())
            print(f"    {cond:6s}  ρ̄={float(rho.mean()):+.3f}  sat={sat:.1%}")

    # ── 4b. Certified lower bounds — IBP / CROWN-IBP / CROWN ─────────────────
    # Skipped when --no-cert-compare is set.
    # cert_lb[method][policy_name]  = tensor (N_EVAL,) or None on failure
    # cert_time[method]             = average wall-clock seconds across policies
    import time as _time
    LIRPA_METHODS = ["IBP", "CROWN-IBP", "CROWN"]
    if args.no_cert_compare:
        print("\n[--no-cert-compare] Skipping IBP/CROWN-IBP/CROWN comparison "
              "and wind-fraction sweep.")
        cert_lb   = {m: {w.name: None for w in wrappers} for m in LIRPA_METHODS}
        cert_time = {m: float("nan") for m in LIRPA_METHODS}
        cert_cp   = {m: {w.name: None for w in wrappers} for m in LIRPA_METHODS}
        any_valid = False
    else:
        cert_lb   = {}
        cert_time = {}
        print("\nComputing certified lower bounds (IBP / CROWN-IBP / CROWN) …")
        for method in LIRPA_METHODS:
            cert_lb[method] = {}
            elapsed_list: List[float] = []
            print(f"  [{method}]")
            for w in wrappers:
                print(f"    {w.name} …", flush=True, end=" ")
                try:
                    t0 = _time.perf_counter()
                    lb = _compute_cert_lb(w.policy, pos_eval, EVAL_WIND_MAX, method)
                    elapsed = _time.perf_counter() - t0
                    cert_lb[method][w.name] = lb
                    elapsed_list.append(elapsed)
                    print(f"lb̄={float(lb.mean()):+.3f}  sat={float((lb>0).float().mean()):.1%}"
                          f"  ({elapsed:.1f} s)")
                except Exception as e:
                    cert_lb[method][w.name] = None
                    print(f"[WARN] {e}")
            cert_time[method] = (sum(elapsed_list) / len(elapsed_list)
                                 if elapsed_list else float("nan"))

        # ── 4c. Clopper-Pearson lower bound on P(lb > 0) ─────────────────────
        CP_DELTA = 0.05
        cert_cp = {}
        for method in LIRPA_METHODS:
            cert_cp[method] = {}
            for w in wrappers:
                lb_vals = cert_lb[method][w.name]
                if lb_vals is None:
                    cert_cp[method][w.name] = None
                    continue
                N_cp = len(lb_vals)
                k_cp = int((lb_vals > 0).sum().item())
                cert_cp[method][w.name] = clopper_pearson_lower(k_cp, N_cp, CP_DELTA)
        any_valid = any(v is not None
                        for d in cert_lb.values() for v in d.values())

    CP_DELTA = 0.05  # used in summary table regardless of --no-cert-compare

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    # Layout: rows = wind conditions, cols = [Standard | RARL | Certified | Histogram]
    # The histogram column groups all three methods together per condition.
    CONDITIONS  = ["none",     "rand",        "adv"]
    COND_LABELS = ["No Wind",  "Random Wind", "Adversarial\n(RARL adv / PGD)"]
    METHOD_NAMES = [w.name for w in wrappers]     # Standard, RARL, Certified

    # 3 rows × 4 cols; histogram column a bit wider
    fig, axes = plt.subplots(
        3, 4, figsize=(17, 12),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})

    for row, (cond, cond_lbl) in enumerate(zip(CONDITIONS, COND_LABELS)):

        # ── Trajectory panels (cols 0-2, one per method) ─────────────────────
        for col, w in enumerate(wrappers):
            rho, traj = results[w.name][cond]
            ax = axes[row, col]
            _draw_traj_panel(ax, traj, rho[:N_SHOW], _COLORS[w.name],
                             pos_0=pos_show)
            if row == 0:
                ax.set_title(w.name, fontsize=15, fontweight="bold", pad=7)
            if col == 0:
                ax.set_ylabel(cond_lbl, fontsize=14, fontweight="bold",
                              labelpad=8)

        # ── Grouped histogram (col 3) ─────────────────────────────────────────
        hax = axes[row, 3]
        rho_dict = {w.name: results[w.name][cond][0] for w in wrappers}
        _draw_grouped_hist(hax, rho_dict)
        if row == 0:
            hax.set_title("Robustness ρ", fontsize=15, fontweight="bold", pad=7)

    fig.suptitle(
        f"Standard vs RARL vs Certified — ε = {EVAL_WIND_MAX}  "
        f"({N_EVAL} episodes, same starts ≥ {EVAL_MIN_CLEARANCE} m from obstacles)",
        fontsize=16, fontweight="bold")

    fig.tight_layout()
    plt.savefig(PLOT_PATH, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {PLOT_PATH}")

    # ── 5b. Certified lower-bound figure: 3 vertically stacked subplots ─────
    # One subplot per LiRPA method, each overlaying all three policies.
    if any_valid:
        # Shared bin edges across ALL methods and policies so bin widths are
        # identical and subplots are directly comparable.
        _all_vals = [cert_lb[m][w.name].cpu().numpy()
                     for m in LIRPA_METHODS for w in wrappers
                     if cert_lb[m][w.name] is not None]
        _global_min = float(min(v.min() for v in _all_vals))
        _global_max = float(max(v.max() for v in _all_vals))
        SHARED_BINS = np.linspace(_global_min, _global_max, 41)  # 40 equal bins

        def _plot_cert_lb_fig(log_scale: bool):
            fig_, axes_ = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
            for ax_, method in zip(axes_, LIRPA_METHODS):
                for w in wrappers:
                    lb_vals = cert_lb[method][w.name]
                    if lb_vals is None:
                        continue
                    vals = lb_vals.cpu().numpy()
                    sat  = float((lb_vals > 0).float().mean())
                    p_L  = cert_cp[method][w.name]
                    p_L_str = f"{p_L:.3f}" if p_L is not None else "N/A"
                    ax_.hist(vals, bins=SHARED_BINS, alpha=0.55,
                             color=_COLORS[w.name],
                             label=(f"{w.name}  avg_lb={vals.mean():+.3f}"
                                    f"  sat={sat:.1%}  pL={p_L_str}"),
                             edgecolor="none")
                ax_.axvline(0, color="k", lw=1.2, ls="--")
                if log_scale:
                    ax_.set_yscale("log")
                    ax_.set_ylabel("Count (log)", fontsize=12)
                    ax_.grid(alpha=0.3, which="both")
                else:
                    ax_.set_ylabel("Count", fontsize=12)
                    ax_.grid(alpha=0.3)
                t_avg = cert_time.get(method, float("nan"))
                t_str = f"{t_avg:.1f} s/policy" if not (t_avg != t_avg) else "N/A"
                ax_.set_title(f"{method}   (avg {t_str})",
                              fontsize=13, fontweight="bold", pad=4)
                ax_.legend(fontsize=13, loc="upper left", framealpha=0.7)
                ax_.tick_params(labelsize=10)
            axes_[-1].set_xlabel("Certified lower bound on ρ", fontsize=13)
            scale_tag = " (log scale)" if log_scale else ""
            fig_.suptitle(
                f"Certified robustness lower bounds{scale_tag} — ε = {EVAL_WIND_MAX}  "
                f"({N_EVAL} episodes)",
                fontsize=14, fontweight="bold")
            fig_.tight_layout()
            return fig_

        fig2 = _plot_cert_lb_fig(log_scale=False)
        lb_path = PLOT_PATH.replace(".png", "_cert_lb.png")
        fig2.savefig(lb_path, dpi=130, bbox_inches="tight")
        print(f"Saved → {lb_path}")
        plt.close(fig2)

        fig3 = _plot_cert_lb_fig(log_scale=True)
        lb_log_path = PLOT_PATH.replace(".png", "_cert_lb_log.png")
        fig3.savefig(lb_log_path, dpi=130, bbox_inches="tight")
        print(f"Saved → {lb_log_path}")
        plt.close(fig3)

    # ── 5c. CROWN LB across wind fractions (0.50 / 1.00 / 1.50 / 2.00 × ε_max) ─
    EPS_FRACS  = [0.50, 1.00, 1.50, 2.00]
    eps_levels = [f * EVAL_WIND_MAX for f in EPS_FRACS]
    CKPT_SWEEP = PLOT_PATH.replace(".png", "_crown_sweep.pt")

    # crown_sweep[frac_idx][policy_name] = tensor (N_EVAL,) or None
    crown_sweep: List[Dict[str, Optional[torch.Tensor]]] = []

    if args.load_sweep and os.path.exists(CKPT_SWEEP):
        print(f"\nLoading CROWN sweep from {CKPT_SWEEP} …")
        ck_sw = torch.load(CKPT_SWEEP, map_location=DEVICE, weights_only=False)
        # Verify the checkpoint matches current fracs / eval wind
        if (ck_sw.get("eps_fracs") == EPS_FRACS
                and abs(ck_sw.get("eval_wind_max", -1) - EVAL_WIND_MAX) < 1e-9):
            crown_sweep = ck_sw["crown_sweep"]
            print("  loaded successfully.")
        else:
            print("  [WARN] checkpoint mismatch (fracs or ε_max differ) — recomputing.")
            crown_sweep = []

    if not crown_sweep:
        print("\nComputing CROWN lower bounds at wind fractions "
              f"{EPS_FRACS} × ε_max …")
        for eps_i in eps_levels:
            lb_at_eps: Dict[str, Optional[torch.Tensor]] = {}
            for w in wrappers:
                print(f"  ε={eps_i:.4f}  {w.name} …", flush=True, end=" ")
                try:
                    lb = _compute_cert_lb(w.policy, pos_eval, eps_i, "CROWN")
                    lb_at_eps[w.name] = lb
                    print(f"lb̄={float(lb.mean()):+.3f}  "
                          f"sat={float((lb>0).float().mean()):.1%}")
                except Exception as e:
                    lb_at_eps[w.name] = None
                    print(f"[WARN] {e}")
            crown_sweep.append(lb_at_eps)
        # Save for future --load-sweep runs
        torch.save({"eps_fracs": EPS_FRACS,
                    "eval_wind_max": EVAL_WIND_MAX,
                    "crown_sweep": crown_sweep}, CKPT_SWEEP)
        print(f"  sweep saved → {CKPT_SWEEP}")

    any_sweep = any(v is not None
                    for d in crown_sweep for v in d.values())
    if any_sweep:
        # Shared bins over all fractions and policies
        _sw_vals = [crown_sweep[i][w.name].cpu().numpy()
                    for i in range(len(EPS_FRACS)) for w in wrappers
                    if crown_sweep[i][w.name] is not None]
        _sw_min = float(min(v.min() for v in _sw_vals))
        _sw_max = float(max(v.max() for v in _sw_vals))
        SWEEP_BINS = np.linspace(_sw_min, _sw_max, 41)

        for log_scale in (False, True):
            fig4, axes4 = plt.subplots(len(EPS_FRACS), 1,
                                       figsize=(8, 4 * len(EPS_FRACS)),
                                       sharex=True)
            for ax4, eps_i, frac, lb_at_eps in zip(
                    axes4, eps_levels, EPS_FRACS, crown_sweep):
                for w in wrappers:
                    lb_vals = lb_at_eps[w.name]
                    if lb_vals is None:
                        continue
                    vals = lb_vals.cpu().numpy()
                    sat  = float((lb_vals > 0).float().mean())
                    k_cp = int((lb_vals > 0).sum().item())
                    p_L  = clopper_pearson_lower(k_cp, len(lb_vals), CP_DELTA)
                    ax4.hist(vals, bins=SWEEP_BINS, alpha=0.55,
                             color=_COLORS[w.name],
                             label=(f"{w.name}  avg_lb={vals.mean():+.3f}"
                                    f"  sat={sat:.1%}  pL={p_L:.3f}"),
                             edgecolor="none")
                ax4.axvline(0, color="k", lw=1.2, ls="--")
                if log_scale:
                    ax4.set_yscale("log")
                    ax4.set_ylabel("Count (log)", fontsize=12)
                    ax4.grid(alpha=0.3, which="both")
                else:
                    ax4.set_ylabel("Count", fontsize=12)
                    ax4.grid(alpha=0.3)
                ax4.set_title(f"ε = {frac:.2f} × ε_max  =  {eps_i:.4f}",
                              fontsize=13, fontweight="bold", pad=4)
                ax4.legend(fontsize=13, loc="upper left", framealpha=0.7)
                ax4.tick_params(labelsize=10)

            axes4[-1].set_xlabel("CROWN certified lower bound on ρ", fontsize=13)
            scale_tag = " (log scale)" if log_scale else ""
            fig4.suptitle(
                f"CROWN LB vs wind fraction{scale_tag}  "
                f"(ε_max = {EVAL_WIND_MAX}, {N_EVAL} episodes)",
                fontsize=14, fontweight="bold")
            fig4.tight_layout()
            suffix = "_crown_sweep_log.png" if log_scale else "_crown_sweep.png"
            sweep_path = PLOT_PATH.replace(".png", suffix)
            fig4.savefig(sweep_path, dpi=130, bbox_inches="tight")
            print(f"Saved → {sweep_path}")
            plt.close(fig4)

    # ── 6. Summary table ──────────────────────────────────────────────────────
    print("\n  ┌─────────────┬──────────────┬──────────────┬──────────────┐")
    print(  "  │             │   No Wind    │ Random Wind  │  Adversarial │")
    print(  "  │             │  ρ̄  /  sat   │  ρ̄  /  sat   │  ρ̄  /  sat   │")
    print(  "  ├─────────────┼──────────────┼──────────────┼──────────────┤")
    for w in wrappers:
        parts = []
        for cond in CONDITIONS:
            rho, _ = results[w.name][cond]
            parts.append(f"{float(rho.mean()):+.3f} / {float((rho>0).float().mean()):.1%}")
        print(f"  │ {w.name:<11s} │ {parts[0]:<12s} │ {parts[1]:<12s} │ {parts[2]:<12s} │")
    print(  "  └─────────────┴──────────────┴──────────────┴──────────────┘")

    if any_valid:
        hdr = "  {:<12s}  ".format("Method") + "  ".join(f"{w.name:<24s}" for w in wrappers)
        print(f"\n  Certified lower bounds  (lb̄ / sat / pL @ δ={CP_DELTA}):\n  {hdr}")
        print("  " + "─" * (len(hdr) + 2))
        for method in LIRPA_METHODS:
            row_str = f"  {method:<12s}  "
            for w in wrappers:
                lb = cert_lb[method][w.name]
                p_L = cert_cp[method][w.name]
                if lb is not None:
                    sat_val = float((lb > 0).float().mean())
                    p_L_str = f"{p_L:.3f}" if p_L is not None else "N/A"
                    cell = f"{float(lb.mean()):+.3f} / {sat_val:.0%} / {p_L_str}"
                else:
                    cell = "N/A"
                row_str += f"{cell:<24s}  "
            print(row_str)


if __name__ == "__main__":
    main()
