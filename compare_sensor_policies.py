"""
compare_sensor_policies.py
==========================
Three-way comparison of navigation policies on the sensor-noise problem
(CrossroadSensorSTRELProblem, constant bias δ ∈ [−ε, ε]²):

  PO        — domain randomisation: random δ ∼ Uniform([−ε,ε]²) in training.
  RARL      — adversarial training: neural adversary maximises sensor noise.
  CRRL      — maximises LiRPA CROWN-IBP certified lower bound.

Evaluation
----------
Every policy is evaluated with three perturbation levels:
  • Nominal  : δ = 0
  • Random   : δ ∼ Uniform([−ε, ε]²)
  • PGD      : δ* = argmin ρ  (30 sign-gradient-descent steps)
The certified policy additionally reports the LiRPA worst-case lb.

First run: trains all three policies and saves checkpoints.
Subsequent runs: load checkpoints — no retraining.

Usage
-----
  # Train all three from scratch
  python compare_sensor_policies.py

  # Load all previously saved checkpoints
  python compare_sensor_policies.py --load-all

  # Retrain only the certified policy, load the others
  python compare_sensor_policies.py --load-aug --load-rarl --cert-iters 1200

  # Override sensor noise budget (forces retrain unless --load-* is set)
  python compare_sensor_policies.py --noise 0.08 --aug-iters 1000 --rarl-iters 1000 --cert-iters 1000

  # Custom output path
  python compare_sensor_policies.py --load-all --out my_comparison.png

Output
------
  • Printed metrics table
  • <out>         — 3×3 figure:
      Row 0 : Trajectories under PGD attack (one panel per policy)
      Row 1 : Robustness histograms (nom / rand / PGD / cert-lb)
      Row 2 : Training curves + metrics bar chart
"""

import os
import time as _time
import dataclasses
import argparse
import warnings as _warnings
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from scipy.stats import beta as _beta_dist

import certified_sensor_strel as cert_mod
import sensor_rarl_strel       as rarl_mod

# ── Tunable constants ─────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EVAL      = 1000
N_TRAJ      = 40
N_PGD_STEPS = 30
PLOT_OUT    = "compare_sensor_policies.png"

CKPT_AUG  = "saved_models/sensor_aug_h10_noise005.pt"
CKPT_RARL = "saved_models/sensor_rarl_h10_noise005.pt"
CKPT_CERT = "saved_models/sensor_cert_h10_noise005.pt"

COLORS = {"aug": "tab:blue", "rarl": "tab:orange", "cert": "tab:green"}
LABELS = {"aug": "PO",       "rarl": "RARL",       "cert": "CRRL"}

# obs_dim = 2 (pos) + 2 (rel_goal) + 8 (rel_obs, 4 obstacles×2) + 4 (clearance)
_OBS_DIM = 16


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save(path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(kwargs, path)
    print(f"    saved → {path}")


def _load(path: str) -> dict:
    return torch.load(path, map_location=DEVICE, weights_only=False)


def _empty_history(keys: List[str]) -> Dict[str, List]:
    """Placeholder history for a loaded (not retrained) policy."""
    return {k: [] for k in keys}


# ── Statistical helpers ───────────────────────────────────────────────────────

def clopper_pearson_lower(k: int, N: int, delta: float = 0.05) -> float:
    """Clopper-Pearson lower confidence bound p_L for a Binomial proportion."""
    if k == 0:
        return 0.0
    return float(_beta_dist.ppf(delta / 2, k, N - k + 1))


# ── LiRPA certified lb (any sensor policy, chunked to avoid OOM) ─────────────

def _compute_cert_lb_sensor(
        policy:     cert_mod.DeterministicPolicy,
        problem:    cert_mod.CrossroadSensorSTRELProblem,
        pos_eval:   torch.Tensor,
        eps:        float,
        method:     str  = "IBP",
        chunk_size: int  = 100,
) -> torch.Tensor:
    """
    Wrap *policy* in a fresh RolloutModule and compute LiRPA lower bounds.

    sensor_noise ∈ [−ε, ε]² is the perturbed input; pos_0 is exact.
    Chunked to avoid OOM when method="CROWN" (which stores A-matrices of shape
    (B, n_perturbed, n_hidden) per node — several GB at B=1000).
    """
    cfg = problem.cfg
    rm  = cert_mod.RolloutModule(policy, problem).to(DEVICE)
    rm.eval()

    eps_noise = max(eps, 1e-7) if method == "CROWN" else eps
    ptb_noise = PerturbationLpNorm(norm=float("inf"), eps=eps_noise)

    lb_chunks: List[torch.Tensor] = []
    B = pos_eval.shape[0]
    for start in range(0, B, chunk_size):
        pos_chunk   = pos_eval[start:start + chunk_size]
        noise_chunk = torch.zeros_like(pos_chunk)

        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message=".*batch dimension.*")
            lirpa = BoundedModule(rm, (pos_chunk, noise_chunk),
                                  device=str(DEVICE), verbose=False)

        lb_c, _, _ = cert_mod.safe_compute_sensor_lb(
            lirpa,
            pos_chunk,
            noise_chunk,
            eps,
            method,
            fallback_lb=None,
            warn_prefix=f"[SensorCompare|{method}|chunk={start}:{start + len(pos_chunk)}]",
        )
        lb_chunks.append(lb_c.detach())

    return torch.cat(lb_chunks, dim=0)


# ── Sensor-noise trajectory rollout ──────────────────────────────────────────

def _rollout(rm: cert_mod.RolloutModule,
             pos_0: torch.Tensor,
             noise: torch.Tensor) -> torch.Tensor:
    """Return (B, H+1, 2) cpu tensor — initial position + H true steps."""
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


def _sample_pgd(problem, rm, n, seed, eps):
    """Sample n start positions and compute PGD worst-case δ."""
    gen   = torch.Generator(device=DEVICE).manual_seed(seed + 77)
    pos_s = problem.sample_initial_positions(n, gen)
    delta = rarl_mod.pgd_attack(rm, pos_s, eps, n_steps=N_PGD_STEPS)
    with torch.no_grad():
        rho_s = rm(pos_s, delta).cpu().numpy()
    traj  = _rollout(rm, pos_s, delta)
    return pos_s, delta, rho_s, traj


# ── Load / train helpers ──────────────────────────────────────────────────────

def _rebuild_lirpa(rm: cert_mod.RolloutModule,
                   cfg: cert_mod.PlanningConfig) -> BoundedModule:
    """Rebuild BoundedModule from a (possibly loaded) RolloutModule."""
    dummy_pos   = torch.zeros(cfg.batch_size, 2, device=DEVICE)
    dummy_noise = torch.zeros(cfg.batch_size, 2, device=DEVICE)
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*batch dimension.*")
        lirpa = BoundedModule(
            rm, (dummy_pos, dummy_noise),
            device=str(DEVICE), verbose=False,
        )
    return lirpa


def _get_aug(train_iters: Optional[int],
             noise_max: Optional[float],
             load: bool) -> Tuple:
    """Return (policy, problem, rollout_mod, history)."""
    if load:
        if not os.path.exists(CKPT_AUG):
            raise FileNotFoundError(
                f"Checkpoint '{CKPT_AUG}' not found. "
                "Run without --load-aug to train first.")
        print(f"  [PO]        loading {CKPT_AUG}")
        ck      = _load(CKPT_AUG)
        cfg     = cert_mod.PlanningConfig(**ck["cfg"])
        policy  = cert_mod.DeterministicPolicy(
            _OBS_DIM, 2, cfg.hidden, cfg.action_parameterization).to(DEVICE)
        policy.load_state_dict(ck["policy_state"])
        policy.eval()
        problem = cert_mod.CrossroadSensorSTRELProblem(cfg, DEVICE)
        rm      = cert_mod.RolloutModule(policy, problem).to(DEVICE)
        rm.eval()
        history = ck.get("history",
                         _empty_history(["iter", "nom_sat_rate", "rho_aug_mean"]))
        train_time_s = ck.get("train_time_s")
    else:
        kw = {}
        if train_iters is not None:
            kw["train_iters"] = train_iters
        if noise_max is not None:
            kw["sensor_noise_max"] = noise_max
        cfg = cert_mod.PlanningConfig(**kw)
        print(f"  [PO]        training ({cfg.train_iters} iters, "
              f"ε_max={cfg.sensor_noise_max}) …")
        t0 = _time.perf_counter()
        policy, problem, rm, history = rarl_mod.train_standard_aug(cfg, DEVICE)
        train_time_s = _time.perf_counter() - t0
        policy.eval()
        _save(CKPT_AUG,
              policy_state=policy.state_dict(),
              cfg=dataclasses.asdict(cfg),
              history=history,
              train_time_s=train_time_s)
    return policy, problem, rm, history, train_time_s


def _get_rarl(train_iters: Optional[int],
              noise_max: Optional[float],
              load: bool) -> Tuple:
    """Return (policy, problem, rollout_mod, adversary, history)."""
    if load:
        if not os.path.exists(CKPT_RARL):
            raise FileNotFoundError(
                f"Checkpoint '{CKPT_RARL}' not found. "
                "Run without --load-rarl to train first.")
        print(f"  [RARL]      loading {CKPT_RARL}")
        ck      = _load(CKPT_RARL)
        cfg     = cert_mod.PlanningConfig(**ck["cfg"])
        policy  = cert_mod.DeterministicPolicy(
            _OBS_DIM, 2, cfg.hidden, cfg.action_parameterization).to(DEVICE)
        policy.load_state_dict(ck["policy_state"])
        policy.eval()
        adversary = rarl_mod.AdversaryPolicy(hidden=64).to(DEVICE)
        adversary.load_state_dict(ck["adversary_state"])
        adversary.eval()
        problem = cert_mod.CrossroadSensorSTRELProblem(cfg, DEVICE)
        rm      = cert_mod.RolloutModule(policy, problem).to(DEVICE)
        rm.eval()
        history = ck.get("history",
                         _empty_history(["iter", "nom_sat_rate", "rho_adv_mean"]))
        train_time_s = ck.get("train_time_s")
    else:
        kw = {}
        if train_iters is not None:
            kw["train_iters"] = train_iters
        if noise_max is not None:
            kw["sensor_noise_max"] = noise_max
        cfg = cert_mod.PlanningConfig(**kw)
        print(f"  [RARL]      training ({cfg.train_iters} iters, "
              f"ε_max={cfg.sensor_noise_max}) …")
        t0 = _time.perf_counter()
        policy, problem, rm, adversary, history = rarl_mod.train_rarl(cfg, DEVICE)
        train_time_s = _time.perf_counter() - t0
        policy.eval(); adversary.eval()
        _save(CKPT_RARL,
              policy_state=policy.state_dict(),
              adversary_state=adversary.state_dict(),
              cfg=dataclasses.asdict(cfg),
              history=history,
              train_time_s=train_time_s)
    return policy, problem, rm, adversary, history, train_time_s


def _get_cert(train_iters: Optional[int],
              noise_max: Optional[float],
              load: bool) -> Tuple:
    """Return (policy, problem, rollout_mod, lirpa_model, history)."""
    if load:
        if not os.path.exists(CKPT_CERT):
            raise FileNotFoundError(
                f"Checkpoint '{CKPT_CERT}' not found. "
                "Run without --load-cert to train first.")
        print(f"  [CRRL]      loading {CKPT_CERT}")
        ck      = _load(CKPT_CERT)
        cfg     = cert_mod.PlanningConfig(**ck["cfg"])
        # Restore final beta so the formula and rollout match training end-state
        cfg.strel_beta = cfg.strel_beta_max
        policy  = cert_mod.DeterministicPolicy(
            _OBS_DIM, 2, cfg.hidden, cfg.action_parameterization).to(DEVICE)
        policy.load_state_dict(ck["policy_state"])
        policy.eval()
        problem = cert_mod.CrossroadSensorSTRELProblem(cfg, DEVICE)
        cert_mod._set_formula_beta(problem.formula, cfg.strel_beta_max)  # match training end-state
        rm      = cert_mod.RolloutModule(policy, problem).to(DEVICE)
        rm.eval()
        lirpa   = _rebuild_lirpa(rm, cfg)
        history = ck.get("history",
                         _empty_history(["iter", "nom_sat_rate",
                                         "cert_sat_rate", "rho_lb_mean"]))
        train_time_s = ck.get("train_time_s")
    else:
        kw = {}
        if train_iters is not None:
            kw["train_iters"] = train_iters
        if noise_max is not None:
            kw["sensor_noise_max"] = noise_max
        cfg = cert_mod.PlanningConfig(**kw)
        print(f"  [CRRL]      training ({cfg.train_iters} iters, "
              f"ε_max={cfg.sensor_noise_max}) …")
        t0 = _time.perf_counter()
        policy, problem, rm, lirpa, history = \
            cert_mod.train_certified_policy(cfg, DEVICE)
        train_time_s = _time.perf_counter() - t0
        policy.eval()
        _save(CKPT_CERT,
              policy_state=policy.state_dict(),
              cfg=dataclasses.asdict(cfg),
              history=history,
              train_time_s=train_time_s)
    return policy, problem, rm, lirpa, history, train_time_s


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _draw_env(ax, problem, cfg, title):
    ax.set_axisbelow(True)
    ax.grid(True, color="lightgray", linewidth=0.4, linestyle="-", zorder=0)
    ax.tick_params(labelsize=10, length=3, pad=2)
    for obs in problem.obstacles.cpu():
        cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
        ax.add_patch(mpatches.Rectangle(
            (cx - r, cy - r), 2 * r, 2 * r,
            color="dimgray", alpha=0.55, zorder=2))
    gx, gy = cfg.goal_xy
    gt = cfg.goal_tol
    ax.add_patch(mpatches.Rectangle(
        (gx - gt, gy - gt), 2 * gt, 2 * gt,
        color="gold", alpha=0.15, zorder=1))
    ax.scatter(*cfg.goal_xy, marker="*", s=200, color="gold",
               edgecolors="darkgoldenrod", zorder=6)
    ax.set_xlim(-0.2, cfg.world_max + 0.2)
    ax.set_ylim(-0.2, cfg.world_max + 0.2)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)


def _draw_trajectories(ax, traj, pos_0_cpu, rho_vals, color="tab:blue"):
    for i in range(traj.shape[0]):
        vio = float(rho_vals[i]) < 0
        c = "tab:red" if vio else color
        a = 0.78 if vio else 0.45
        lw = 1.8 if vio else 1.2
        xs = [float(pos_0_cpu[i, 0])] + traj[i, :, 0].numpy().tolist()
        ys = [float(pos_0_cpu[i, 1])] + traj[i, :, 1].numpy().tolist()
        ax.plot(xs, ys, "o-", markersize=2, color=c, alpha=a, lw=lw,
                zorder=5 if vio else 3)
        ax.plot(float(pos_0_cpu[i, 0]), float(pos_0_cpu[i, 1]), "o", ms=4,
                color=c, alpha=0.90 if vio else 0.65, zorder=6)


def _draw_grouped_hist(ax, rho_dict: Dict[str, torch.Tensor], title: str = ""):
    all_vals = torch.cat([v for v in rho_dict.values()]).detach().cpu().numpy()
    lo, hi   = float(all_vals.min()), float(all_vals.max())
    span = max(hi - lo, 1e-6)
    pad = 0.03 * span
    hist_range = (lo - pad, hi + pad)
    ax.axvline(0, color="black", lw=1.1, ls="--", zorder=5)
    handles = []
    for key, rho in rho_dict.items():
        vals = rho.detach().cpu().numpy()
        sat  = float((rho > 0).float().mean())
        ax.hist(vals, bins=60, range=hist_range,
                color=COLORS[key], alpha=0.50, edgecolor="none")
        handles.append(
            mpatches.Patch(color=COLORS[key], alpha=0.7,
                           label=f"{LABELS[key]}  sat={sat:.0%}"))
    ax.legend(handles=handles, fontsize=11, loc="upper left",
              framealpha=0.7, handlelength=1.2)
    ax.set_xlabel("ρ", fontsize=13)
    ax.set_ylabel("count", fontsize=13)
    ax.tick_params(labelsize=11)
    if title:
        ax.set_title(title, fontsize=13, pad=3)


def _draw_training_curves(ax_nom, ax_qual, aug_h, rarl_h, cert_h):
    """Left: nom_sat for all 3.  Right: robustness quality under perturbation."""
    for iters, vals, key in [
        (aug_h["iter"],  aug_h["nom_sat_rate"],  "aug"),
        (rarl_h["iter"], rarl_h["nom_sat_rate"], "rarl"),
        (cert_h["iter"], cert_h["nom_sat_rate"], "cert"),
    ]:
        if iters:
            ax_nom.plot(iters, vals, lw=1.8, color=COLORS[key], label=LABELS[key])
    ax_nom.axhline(0, color="gray", lw=0.6, ls=":")
    ax_nom.axhline(1, color="gray", lw=0.6, ls=":")
    ax_nom.set_xlabel("Iteration", fontsize=8)
    ax_nom.set_ylabel("Satisfaction rate", fontsize=8)
    ax_nom.set_title("Nominal satisfaction rate during training", fontsize=9)
    ax_nom.legend(fontsize=8); ax_nom.grid(alpha=0.20)

    for iters, vals, key, lbl in [
        (aug_h["iter"],  aug_h.get("rho_aug_mean",  []), "aug",  "ρ_aug  (PO)"),
        (rarl_h["iter"], rarl_h.get("rho_adv_mean", []), "rarl", "ρ_adv  (RARL adversary)"),
        (cert_h["iter"], cert_h.get("rho_lb_mean",  []), "cert", "LB     (CROWN-IBP)"),
    ]:
        if iters and vals:
            ax_qual.plot(iters, vals, lw=1.8, color=COLORS[key], label=lbl)
    ax_qual.axhline(0, color="gray", lw=0.6, ls=":")
    ax_qual.set_xlabel("Iteration", fontsize=8)
    ax_qual.set_ylabel("Mean robustness / lb", fontsize=8)
    ax_qual.set_title("Robustness quality under perturbation during training", fontsize=9)
    ax_qual.legend(fontsize=8); ax_qual.grid(alpha=0.20)


def _draw_bar_chart(ax, aug_m, rarl_m, cert_m, eps):
    groups  = ["Nominal\nsat", "Random\nsat", "PGD\nsat"]
    keys    = ["nominal_sat_rate", "rand_sat_rate", "pgd_sat_rate"]
    x       = np.arange(len(groups))
    w       = 0.22
    offsets = np.array([-1, 0, 1]) * w

    for i, (ms, key) in enumerate([(aug_m, "aug"), (rarl_m, "rarl"), (cert_m, "cert")]):
        ax.bar(x + offsets[i], [ms[k] for k in keys], w * 0.90,
               color=COLORS[key], alpha=0.78, label=LABELS[key])

    if "cert_lb_values" in cert_m:
        cert_lb_sat = float((cert_m["cert_lb_values"] > 0).mean())
        ax.axhline(cert_lb_sat, color="tab:purple", lw=1.8, ls="--",
                   label=f"Cert lb sat = {cert_lb_sat:.0%}")

    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Satisfaction rate", fontsize=8)
    ax.set_title(f"Final evaluation metrics  (ε = {eps})", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.20, axis="y")


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare PO / RARL / CRRL sensor-noise policies.")
    p.add_argument("--aug-iters",  type=int,   default=None, metavar="N",
                   help="retrain PO for N iterations")
    p.add_argument("--rarl-iters", type=int,   default=None, metavar="N",
                   help="retrain RARL for N iterations")
    p.add_argument("--cert-iters", type=int,   default=None, metavar="N",
                   help="retrain CRRL for N iterations")
    p.add_argument("--noise",      type=float, default=None, metavar="ε",
                   help="sensor noise budget for training and evaluation "
                        f"(default: PlanningConfig.sensor_noise_max = "
                        f"{cert_mod.PlanningConfig().sensor_noise_max})")
    p.add_argument("--out",        type=str,   default=PLOT_OUT, metavar="PATH",
                   help="output plot path")
    p.add_argument("--load-aug",  action="store_true",
                   help=f"load PO from {CKPT_AUG}")
    p.add_argument("--load-rarl", action="store_true",
                   help=f"load RARL from {CKPT_RARL}")
    p.add_argument("--load-cert", action="store_true",
                   help=f"load CRRL from {CKPT_CERT}")
    p.add_argument("--load-all",  action="store_true",
                   help="load all three policies (shorthand)")
    p.add_argument("--no-cert-compare", action="store_true",
                   help="skip only the IBP/CROWN-IBP/CROWN comparison")
    p.add_argument("--load-sweep", action="store_true",
                   help="load CROWN ε-sweep results from checkpoint")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    global PLOT_OUT
    PLOT_OUT = args.out

    load_aug  = args.load_aug  or args.load_all
    load_rarl = args.load_rarl or args.load_all
    load_cert = args.load_cert or args.load_all

    # ── Load / train ──────────────────────────────────────────────────────────
    print("Loading / training policies …")

    print("=" * 68)
    print("  [1/3] PO — domain randomisation")
    print("=" * 68)
    aug_policy, aug_problem, aug_rm, aug_hist, aug_time = \
        _get_aug(args.aug_iters, args.noise, load_aug)

    print()
    print("=" * 68)
    print("  [2/3] RARL — neural adversary")
    print("=" * 68)
    rarl_policy, rarl_problem, rarl_rm, rarl_adv, rarl_hist, rarl_time = \
        _get_rarl(args.rarl_iters, args.noise, load_rarl)

    print()
    print("=" * 68)
    print("  [3/3] CRRL — CROWN-IBP lower bound")
    print("=" * 68)
    cert_policy, cert_problem, cert_rm, cert_lirpa, cert_hist, cert_time = \
        _get_cert(args.cert_iters, args.noise, load_cert)

    eps = args.noise if args.noise is not None else cert_mod.PlanningConfig().sensor_noise_max

    # ── Training-time comparison ──────────────────────────────────────────────
    _policy_names  = ["PO", "RARL", "CRRL"]
    _policy_colors = [COLORS["aug"], COLORS["rarl"], COLORS["cert"]]
    _policy_times  = [aug_time,  rarl_time,   cert_time]
    print("\n  Training times:")
    print(f"  {'Policy':<12}  {'Time (s)':>10}  {'Time (min)':>10}")
    print("  " + "─" * 38)
    for n, t in zip(_policy_names, _policy_times):
        if t is not None:
            print(f"  {n:<12}  {t:>10.1f}  {t/60:>10.2f}")
        else:
            print(f"  {n:<12}  {'N/A':>10}  {'N/A':>10}")

    _timed = [(n, t) for n, t in zip(_policy_names, _policy_times)
              if t is not None]
    if _timed:
        _tn, _tt = zip(*_timed)
        _tc = [c for n, c in zip(_policy_names, _policy_colors) if n in _tn]
        fig_t, ax_t = plt.subplots(figsize=(5, 3.5))
        bars = ax_t.bar(_tn, [t / 60 for t in _tt], color=_tc,
                        edgecolor="white", linewidth=0.8)
        ax_t.bar_label(bars, fmt="%.1f min", padding=3, fontsize=10)
        ax_t.set_ylabel("Training time (min)", fontsize=12)
        ax_t.set_title("Training time comparison", fontsize=13, fontweight="bold")
        ax_t.tick_params(labelsize=11)
        ax_t.set_ylim(0, max(t / 60 for t in _tt) * 1.25)
        ax_t.grid(axis="y", alpha=0.35)
        fig_t.tight_layout()
        t_path = PLOT_OUT.replace(".png", "_train_time.png")
        fig_t.savefig(t_path, dpi=130, bbox_inches="tight")
        print(f"  Saved → {t_path}")
        plt.close(fig_t)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nEvaluating (n={N_EVAL}, ε={eps}, PGD steps={N_PGD_STEPS}) …")
    aug_m  = rarl_mod.evaluate_policy(aug_problem,  aug_rm,  N_EVAL, 42, DEVICE, eps)
    rarl_m = rarl_mod.evaluate_policy(rarl_problem, rarl_rm, N_EVAL, 42, DEVICE, eps)
    cert_m = rarl_mod.evaluate_policy(cert_problem, cert_rm, N_EVAL, 42, DEVICE, eps)

    print("Computing LiRPA certified lb …")
    cert_lb, _, _ = cert_mod.safe_compute_sensor_lb(
        cert_lirpa,
        cert_m["_pos_0"],
        torch.zeros(N_EVAL, 2, device=DEVICE),
        eps,
        cert_mod.PlanningConfig().lirpa_method,
        fallback_lb=torch.as_tensor(cert_m["nominal_rho_values"], device=DEVICE),
        warn_prefix="[SensorCompare|eval]",
    )
    cert_m["cert_lb_values"] = cert_lb.detach().cpu().numpy()
    cert_m["cert_lb_mean"]   = float(cert_lb.mean())
    cert_m["cert_sat_rate"]  = float((cert_lb > 0).float().mean())

    # ── Metrics table ─────────────────────────────────────────────────────────
    rows = [
        ("Nominal  ρ mean",   "nominal_rho_mean"),
        ("Nominal  sat rate", "nominal_sat_rate"),
        ("Random   ρ mean",   "rand_rho_mean"),
        ("Random   sat rate", "rand_sat_rate"),
        ("PGD      ρ mean",   "pgd_rho_mean"),
        ("PGD      sat rate", "pgd_sat_rate"),
    ]
    print(f"\n{'':=^66}")
    print(f"  {'Metric':<26} {'PO':>12}  {'RARL':>12}  {'CRRL':>12}")
    print("-" * 66)
    for label, key in rows:
        av, rv, cv = aug_m[key], rarl_m[key], cert_m[key]
        print(f"  {label:<26} {av:>12.3f}  {rv:>12.3f}  {cv:>12.3f}")
    print(f"  {'Cert lb  mean':<26} {'—':>12}  {'—':>12}  "
          f"{cert_m['cert_lb_mean']:>12.3f}")
    print(f"  {'Cert lb  sat rate':<26} {'—':>12}  {'—':>12}  "
          f"{cert_m['cert_sat_rate']:>12.3f}")
    print("=" * 66)

    # ── CRRL lb comparison: IBP / CROWN-IBP / CROWN ──────────────────────────
    LIRPA_METHODS = ["IBP", "CROWN-IBP", "CROWN"]
    CP_DELTA      = 0.05
    EPS_FRACS     = [1.00, 3.00, 5.00]
    CKPT_SWEEP    = PLOT_OUT.replace(".png", "_crown_sweep.pt")

    _policies = [
        ("aug",  aug_policy,  aug_problem),
        ("rarl", rarl_policy, rarl_problem),
        ("cert", cert_policy, cert_problem),
    ]
    gen_ev = torch.Generator(device=DEVICE).manual_seed(42 + 1000)
    pos_eval_cert = aug_problem.sample_initial_positions(N_EVAL, gen_ev)

    if args.no_cert_compare:
        print("\n[--no-cert-compare] Skipping IBP/CROWN-IBP/CROWN comparison.")
        cert_lb   = {m: {k: None for k, *_ in _policies} for m in LIRPA_METHODS}
        cert_time_d = {m: float("nan") for m in LIRPA_METHODS}
        cert_cp   = {m: {k: None for k, *_ in _policies} for m in LIRPA_METHODS}
        any_valid = False
        crown_sweep: List[Dict[str, Optional[torch.Tensor]]] = []
    else:
        cert_lb:    Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        cert_time_d: Dict[str, float] = {}
        print("\nComputing certified lower bounds (IBP / CROWN-IBP / CROWN) …")
        for method in LIRPA_METHODS:
            cert_lb[method] = {}
            elapsed_list: List[float] = []
            print(f"  [{method}]")
            for key, pol, prob in _policies:
                print(f"    {LABELS[key]} …", flush=True, end=" ")
                try:
                    t0 = _time.perf_counter()
                    lb = _compute_cert_lb_sensor(pol, prob, pos_eval_cert,
                                                 eps, method)
                    elapsed = _time.perf_counter() - t0
                    cert_lb[method][key] = lb
                    elapsed_list.append(elapsed)
                    print(f"avg_lb={float(lb.mean()):+.3f}  "
                          f"sat={float((lb>0).float().mean()):.1%}  "
                          f"({elapsed:.1f} s)")
                except Exception as e:
                    cert_lb[method][key] = None
                    print(f"[WARN] {e}")
            cert_time_d[method] = (sum(elapsed_list) / len(elapsed_list)
                                   if elapsed_list else float("nan"))

        # Clopper-Pearson
        cert_cp: Dict[str, Dict[str, Optional[float]]] = {}
        for method in LIRPA_METHODS:
            cert_cp[method] = {}
            for key, _, _ in _policies:
                lb_v = cert_lb[method][key]
                if lb_v is None:
                    cert_cp[method][key] = None
                else:
                    k_cp = int((lb_v > 0).sum().item())
                    cert_cp[method][key] = clopper_pearson_lower(
                        k_cp, len(lb_v), CP_DELTA)

        any_valid = any(v is not None
                        for d in cert_lb.values() for v in d.values())

        # ── Cert-lb histogram figures (linear + log) ──────────────────────────
        if any_valid:
            _all_lbv = [cert_lb[m][k].cpu().numpy()
                        for m in LIRPA_METHODS for k, *_ in _policies
                        if cert_lb[m][k] is not None]
            _all_concat = np.concatenate(_all_lbv)
            _gl_lo = float(np.quantile(_all_concat, 0.01))
            _gl_hi = float(np.quantile(_all_concat, 0.99))
            if _gl_hi <= _gl_lo:
                _gl_lo = float(_all_concat.min())
                _gl_hi = float(_all_concat.max())
            _gl_span = max(_gl_hi - _gl_lo, 1e-6)
            _gl_pad = 0.05 * _gl_span
            _gl_range = (_gl_lo - _gl_pad, _gl_hi + _gl_pad)
            SHARED_BINS = np.linspace(_gl_range[0], _gl_range[1], 61)

            def _plot_cert_lb_fig(log_scale: bool):
                fig_, axes_ = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
                for ax_, method in zip(axes_, LIRPA_METHODS):
                    for key, _, _ in _policies:
                        lb_v = cert_lb[method][key]
                        if lb_v is None:
                            continue
                        vals = lb_v.cpu().numpy()
                        sat  = float((lb_v > 0).float().mean())
                        p_L  = cert_cp[method][key]
                        p_L_str = f"{p_L:.3f}" if p_L is not None else "N/A"
                        ax_.hist(vals, bins=SHARED_BINS, alpha=0.55,
                                 color=COLORS[key],
                                 label=(f"{LABELS[key]}  avg_lb={vals.mean():+.3f}"
                                        f"  cert sat={sat:.1%}  pL={p_L_str}"),
                                 edgecolor="none")
                    ax_.axvline(0, color="k", lw=1.2, ls="--")
                    ax_.set_xlim(*_gl_range)
                    if log_scale:
                        ax_.set_yscale("log")
                        ax_.set_ylabel("Count (log)", fontsize=12)
                        ax_.grid(alpha=0.3, which="both")
                    else:
                        ax_.set_ylabel("Count", fontsize=12)
                        ax_.grid(alpha=0.3)
                    t_avg = cert_time_d.get(method, float("nan"))
                    t_str = f"{t_avg:.1f} s/policy" if not np.isnan(t_avg) else "N/A"
                    ax_.set_title(f"{method}   (avg {t_str})",
                                  fontsize=13, fontweight="bold", pad=4)
                    ax_.legend(fontsize=12, loc="upper left", framealpha=0.7)
                    ax_.tick_params(labelsize=10)
                axes_[-1].set_xlabel("Certified lower bound on ρ", fontsize=13)
                scale_tag = " (log scale)" if log_scale else ""
                fig_.suptitle(
                    f"Certified robustness lower bounds{scale_tag} — ε = {eps}  "
                    f"({N_EVAL} episodes)",
                    fontsize=14, fontweight="bold")
                fig_.tight_layout()
                return fig_

            fig_lb = _plot_cert_lb_fig(log_scale=False)
            lb_path = PLOT_OUT.replace(".png", "_cert_lb.png")
            fig_lb.savefig(lb_path, dpi=130, bbox_inches="tight")
            print(f"Saved → {lb_path}")
            plt.close(fig_lb)

            fig_lb_log = _plot_cert_lb_fig(log_scale=True)
            lb_log_path = PLOT_OUT.replace(".png", "_cert_lb_log.png")
            fig_lb_log.savefig(lb_log_path, dpi=130, bbox_inches="tight")
            print(f"Saved → {lb_log_path}")
            plt.close(fig_lb_log)

    # ── CROWN ε-sweep ─────────────────────────────────────────────────────────
    eps_levels = [f * eps for f in EPS_FRACS]
    crown_sweep = []

    if args.load_sweep and os.path.exists(CKPT_SWEEP):
        print(f"\nLoading CROWN sweep from {CKPT_SWEEP} …")
        ck_sw = torch.load(CKPT_SWEEP, map_location=DEVICE,
                           weights_only=False)
        if (ck_sw.get("eps_fracs") == EPS_FRACS
                and abs(ck_sw.get("eval_eps", -1) - eps) < 1e-9):
            crown_sweep = ck_sw["crown_sweep"]
            print("  loaded successfully.")
        else:
            print("  [WARN] mismatch — recomputing.")

    if not crown_sweep:
        print("\nComputing CROWN lower bounds at ε fractions "
              f"{EPS_FRACS} × ε …")
        for eps_i in eps_levels:
            lb_at_eps: Dict[str, Optional[torch.Tensor]] = {}
            for key, pol, prob in _policies:
                print(f"  ε={eps_i:.4f}  {LABELS[key]} …",
                      flush=True, end=" ")
                try:
                    lb = _compute_cert_lb_sensor(pol, prob,
                                                 pos_eval_cert, eps_i,
                                                 "CROWN")
                    lb_at_eps[key] = lb
                    print(f"avg_lb={float(lb.mean()):+.3f}  "
                          f"cert sat={float((lb>0).float().mean()):.1%}")
                except Exception as e:
                    lb_at_eps[key] = None
                    print(f"[WARN] {e}")
            crown_sweep.append(lb_at_eps)
        torch.save({"eps_fracs": EPS_FRACS, "eval_eps": eps,
                    "crown_sweep": crown_sweep}, CKPT_SWEEP)
        print(f"  sweep saved → {CKPT_SWEEP}")

    any_sweep = any(v is not None
                    for d in crown_sweep for v in d.values())
    if any_sweep:
        _sw_vals = [crown_sweep[i][k].cpu().numpy()
                    for i in range(len(EPS_FRACS))
                    for k, *_ in _policies
                    if crown_sweep[i][k] is not None]
        _sw_concat = np.concatenate(_sw_vals)
        _sw_lo = float(np.quantile(_sw_concat, 0.01))
        _sw_hi = float(np.quantile(_sw_concat, 0.99))
        if _sw_hi <= _sw_lo:
            _sw_lo = float(_sw_concat.min())
            _sw_hi = float(_sw_concat.max())
        _sw_span = max(_sw_hi - _sw_lo, 1e-6)
        _sw_pad = 0.05 * _sw_span
        _sw_range = (_sw_lo - _sw_pad, _sw_hi + _sw_pad)
        SWEEP_BINS = np.linspace(_sw_range[0], _sw_range[1], 61)

        for log_scale in (False, True):
            fig4, axes4 = plt.subplots(len(EPS_FRACS), 1,
                                       figsize=(8, 4 * len(EPS_FRACS)),
                                       sharex=True)
            for ax4, eps_i, frac, lb_at_eps in zip(
                    axes4, eps_levels, EPS_FRACS, crown_sweep):
                for key, _, _ in _policies:
                    lb_v = lb_at_eps[key]
                    if lb_v is None:
                        continue
                    vals = lb_v.cpu().numpy()
                    sat  = float((lb_v > 0).float().mean())
                    k_cp = int((lb_v > 0).sum().item())
                    p_L  = clopper_pearson_lower(k_cp, len(lb_v), CP_DELTA)
                    ax4.hist(vals, bins=SWEEP_BINS, alpha=0.55,
                             color=COLORS[key],
                             label=(f"{LABELS[key]}  avg_lb={vals.mean():+.3f}"
                                    f"  cert sat={sat:.1%}  pL={p_L:.3f}"),
                             edgecolor="none")
                ax4.axvline(0, color="k", lw=1.2, ls="--")
                ax4.set_xlim(*_sw_range)
                if log_scale:
                    ax4.set_yscale("log")
                    ax4.set_ylabel("Count (log)", fontsize=12)
                    ax4.grid(alpha=0.3, which="both")
                else:
                    ax4.set_ylabel("Count", fontsize=12)
                    ax4.grid(alpha=0.3)
                ax4.set_title(
                    f"ε = {frac:.2f} × ε_train  =  {eps_i:.5f}",
                    fontsize=13, fontweight="bold", pad=4)
                ax4.legend(fontsize=12, loc="upper left", framealpha=0.7)
                ax4.tick_params(labelsize=10)
            axes4[-1].set_xlabel("CROWN certified lower bound on ρ",
                                 fontsize=13)
            scale_tag = " (log scale)" if log_scale else ""
            fig4.suptitle(
                f"CROWN LB vs ε fraction{scale_tag}  "
                f"(ε_train = {eps}, {N_EVAL} episodes)",
                fontsize=14, fontweight="bold")
            fig4.tight_layout()
            suffix = "_crown_sweep_log.png" if log_scale \
                else "_crown_sweep.png"
            fig4.savefig(PLOT_OUT.replace(".png", suffix),
                         dpi=130, bbox_inches="tight")
            print(f"Saved → {PLOT_OUT.replace('.png', suffix)}")
            plt.close(fig4)

    # ── Shared trajectory / histogram evaluation exactly as compare_policies ─
    CONDITIONS  = ["none", "rand", "adv"]
    COND_LABELS = ["No Noise", "Random Noise", "Adversarial\n(RARL adv / PGD)"]
    pos_eval = aug_problem.sample_initial_positions(
        N_EVAL, torch.Generator(device=DEVICE).manual_seed(42 + 1000))
    pos_show = pos_eval[:N_TRAJ]

    gen_rand = torch.Generator(device=DEVICE).manual_seed(42 + 1003)
    rand_noise_eval = ((2 * torch.rand(N_EVAL, 2, generator=gen_rand, device=DEVICE) - 1) * eps)
    rand_noise_show = rand_noise_eval[:N_TRAJ]

    policies = [
        ("aug",  aug_policy,  aug_problem,  aug_rm),
        ("rarl", rarl_policy, rarl_problem, rarl_rm),
        ("cert", cert_policy, cert_problem, cert_rm),
    ]

    results: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    print("\nEvaluating shared trajectory / histogram panels …")
    for key, pol, prob, rm in policies:
        _ = pol
        rr: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        zero_eval = torch.zeros(N_EVAL, 2, device=DEVICE)
        zero_show = torch.zeros(N_TRAJ, 2, device=DEVICE)
        with torch.no_grad():
            rho_n = rm(pos_eval, zero_eval)
        traj_n = _rollout(rm, pos_show, zero_show)
        rr["none"] = (rho_n, traj_n)

        with torch.no_grad():
            rho_r = rm(pos_eval, rand_noise_eval)
        traj_r = _rollout(rm, pos_show, rand_noise_show)
        rr["rand"] = (rho_r, traj_r)

        if key == "rarl":
            with torch.no_grad():
                adv_eval = eps * rarl_adv(pos_eval)
                adv_show = adv_eval[:N_TRAJ]
                rho_a = rm(pos_eval, adv_eval)
            traj_a = _rollout(rm, pos_show, adv_show)
        else:
            adv_eval = rarl_mod.pgd_attack(rm, pos_eval, eps, n_steps=N_PGD_STEPS)
            adv_show = adv_eval[:N_TRAJ]
            with torch.no_grad():
                rho_a = rm(pos_eval, adv_eval)
            traj_a = _rollout(rm, pos_show, adv_show)
        rr["adv"] = (rho_a, traj_a)
        results[key] = rr

        for cond, (rho, _) in rr.items():
            sat = float((rho > 0).float().mean())
            print(f"  {LABELS[key]:<10} {cond:>4s}  ρ̄={float(rho.mean()):+.3f}  sat={sat:.1%}")

    # ── Figure: exact compare_policies_strel layout (3×4) ───────────────────
    fig, axes = plt.subplots(
        3, 4, figsize=(17, 12),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})

    for row, (cond, cond_lbl) in enumerate(zip(CONDITIONS, COND_LABELS)):
        for col, key in enumerate(["aug", "rarl", "cert"]):
            rho, traj = results[key][cond]
            ax = axes[row, col]
            _draw_env(ax, aug_problem if key == "aug" else rarl_problem if key == "rarl" else cert_problem,
                      cert_mod.PlanningConfig(), "")
            _draw_trajectories(ax, traj, pos_show.cpu(), rho[:N_TRAJ].detach().cpu().numpy(),
                               color=COLORS[key])
            if row == 0:
                ax.set_title(LABELS[key], fontsize=15, fontweight="bold", pad=7)
            if col == 0:
                ax.set_ylabel(cond_lbl, fontsize=14, fontweight="bold", labelpad=8)

        hax = axes[row, 3]
        rho_dict = {key: results[key][cond][0] for key in ["aug", "rarl", "cert"]}
        _draw_grouped_hist(hax, rho_dict)
        if row == 0:
            hax.set_title("Robustness ρ", fontsize=15, fontweight="bold", pad=7)

    fig.suptitle(
        f"PO vs RARL vs CRRL — ε = {eps}  "
        f"({N_EVAL} episodes, same starts from the sensor-noise benchmark)",
        fontsize=16, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {PLOT_OUT}")


if __name__ == "__main__":
    main()
