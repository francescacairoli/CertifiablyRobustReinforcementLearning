import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim

from diff_certif_strel import (And, Always, AtomicPredicate, Eventually, smooth_min as strel_smooth_min)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PlanningConfig:
    horizon: int = 10
    dt: float = 0.25
    max_speed: float = 2
    train_iters: int = 4000
    batch_size: int = 256
    lr: float = 2e-3
    hidden: int = 128
    world_min: float = -4
    world_max: float = 10
    init_box: Tuple[float, float, float, float] = (0.0, 3.0, 0.0, 3.0)
    goal_xy: Tuple[float, float] = (4.00, 3.00)
    goal_tol: float = 0.45
    init_obstacle_clearance: float = 0.08
    init_goal_clearance: float = 0.15
    integration_substeps: int = 4
    collision_eps: float = 1e-3
    strel_beta_min: float = 8.0   # beta at iteration 0 (smooth gradients)
    strel_beta_max: float = 50.0  # beta at final iteration (sharp min/max)
    early_stop_sat: float = 1.  # stop when EMA sat_rate exceeds this threshold …
    early_stop_patience: int = 20 # … for this many consecutive iterations
    early_stop_ema: float = 1.  # EMA momentum for smoothing sat_rate (window ≈ 5 iters)
    violation_penalty_weight: float = 1.0  # weight on hinge penalty for violating trajectories
    safety_buffer:           float = 0.05  # direct clearance penalty margin (metres)
    safety_penalty_weight:   float = 10.0   # weight on direct obstacle-clearance penalty
    rho_aggregation: str = "mean"   # how to aggregate rho over the batch: "mean" | "percentile"
    rho_percentile: float = 10.0    # percentile used when rho_aggregation="percentile" (e.g. 10 = worst-10%)
    seed: int = 7
    wind_max: float = 0.01
    wind_max_start: float = 0.00      # curriculum: starting wind magnitude
    baseline_momentum: float = 0.9    # EMA momentum for rho baseline
    action_parameterization: str = "cartesian"  # "cartesian" | "polar" (sin_θ·v, cos_θ·v)
    plot_path: str = f"plots_strel_no_critic_h{horizon}_dt025_avgrho_trainwind_{wind_max}.png"


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 parameterization: str = "cartesian"):
        super().__init__()
        self.parameterization = parameterization
        # Polar: output (sin_θ, cos_θ, v) — 1 extra raw dim; action = (sin_θ, cos_θ) * v
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


class ThreeObstacleSTRELProblem:
    def __init__(self, cfg: PlanningConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.goal = torch.tensor(cfg.goal_xy, dtype=torch.float32, device=device)
        self.obstacles = torch.tensor(
            [
                [1.75, 1.75, 0.38],
                [1.75, 3.75, 0.42],
                [3.75, 2.00, 0.34],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.spatial_labels = torch.zeros(1, 1, cfg.horizon, dtype=torch.long, device=device)
        ego_only = torch.tensor([1.0], dtype=torch.float32, device=device)
        safe = AtomicPredicate(var_ind=0, threshold=0.0, labels=ego_only, lte=False)
        reach = AtomicPredicate(var_ind=1, threshold=0.0, labels=ego_only, lte=False)
        self.formula = And(Always(safe, beta=cfg.strel_beta_min), Eventually(reach, beta=cfg.strel_beta_min), beta=cfg.strel_beta_min)

    def valid_initial_mask(self, pos: torch.Tensor) -> torch.Tensor:
        diff = pos.unsqueeze(1) - self.obstacles[:, :2].unsqueeze(0)
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-9)
        clear = dist - self.obstacles[:, 2].unsqueeze(0)
        goal_dist = torch.sqrt(torch.sum((pos - self.goal.unsqueeze(0)) ** 2, dim=-1) + 1e-9)
        return torch.all(clear > self.cfg.init_obstacle_clearance, dim=-1) & (
            goal_dist > self.cfg.goal_tol + self.cfg.init_goal_clearance
        )

    def sample_initial_positions(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        xmin, xmax, ymin, ymax = self.cfg.init_box
        samples = torch.zeros(batch_size, 2, dtype=torch.float32, device=self.device)
        filled = 0
        while filled < batch_size:
            cand = torch.stack(
                [
                    xmin + (xmax - xmin) * torch.rand(batch_size, generator=generator, device=self.device),
                    ymin + (ymax - ymin) * torch.rand(batch_size, generator=generator, device=self.device),
                ],
                dim=-1,
            )
            valid = self.valid_initial_mask(cand)
            take = min(batch_size - filled, int(valid.sum().item()))
            if take > 0:
                samples[filled : filled + take] = cand[valid][:take]
                filled += take
        return samples

    def observation(self, pos: torch.Tensor) -> torch.Tensor:
        world_scale = max(abs(self.cfg.world_min), abs(self.cfg.world_max))
        rel_goal = (self.goal.unsqueeze(0) - pos) / world_scale
        rel_obs = (self.obstacles[:, :2].unsqueeze(0) - pos.unsqueeze(1)).reshape(pos.shape[0], -1) / world_scale
        dist = torch.sqrt(torch.sum((pos.unsqueeze(1) - self.obstacles[:, :2].unsqueeze(0)) ** 2, dim=-1) + 1e-9)
        clearance = dist - self.obstacles[:, 2].unsqueeze(0)
        return torch.cat([pos / world_scale, rel_goal, rel_obs, clearance], dim=-1)

    def project_out_of_obstacles(self, pos: torch.Tensor) -> torch.Tensor:
        for j in range(self.obstacles.shape[0]):
            center = self.obstacles[j, :2].unsqueeze(0)
            radius = float(self.obstacles[j, 2].item() + self.cfg.collision_eps)
            diff = pos - center
            dist = torch.sqrt(torch.sum(diff * diff, dim=-1, keepdim=True) + 1e-12)
            inside = dist.squeeze(-1) < radius
            if bool(torch.any(inside).item()):
                direction = diff / dist
                direction = torch.where(
                    inside.unsqueeze(-1),
                    direction,
                    torch.zeros_like(direction),
                )
                pos = torch.where(
                    inside.unsqueeze(-1),
                    center + direction * radius,
                    pos,
                )
        return pos

    def rollout(
        self,
        policy: DeterministicPolicy,
        batch_size: int,
        generator: torch.Generator,
        init_pos: torch.Tensor = None,
        wind_max: float = None,
    ) -> Dict[str, torch.Tensor]:
        if wind_max is None:
            wind_max = self.cfg.wind_max
        pos = self.sample_initial_positions(batch_size, generator) if init_pos is None else init_pos.clone()
        if not bool(torch.all(self.valid_initial_mask(pos)).item()):
            raise ValueError("Initial states must not overlap obstacles or start inside the goal region.")
        init_pos = pos.clone()
        pos_hist: List[torch.Tensor] = []
        act_hist: List[torch.Tensor] = []

        # Sample one wind vector per episode, held constant for the entire rollout.
        #wind = (2.0 * torch.rand(batch_size, 2, generator=generator, device=self.device) - 1.0) * wind_max

        step_dt = self.cfg.dt / float(self.cfg.integration_substeps)
        for _ in range(self.cfg.horizon):
            obs = self.observation(pos)
            wind = (2.0 * torch.rand(batch_size, 2, generator=generator, device=self.device) - 1.0) * wind_max

            action = policy(obs)
            act_hist.append(action)
            for _sub in range(self.cfg.integration_substeps):
                pos = pos + step_dt * (self.cfg.max_speed * action + wind)
                pos = torch.clamp(pos, min=self.cfg.world_min, max=self.cfg.world_max)
            pos_hist.append(pos)

        traj = torch.stack(pos_hist, dim=1)
        actions = torch.stack(act_hist, dim=1)  # (batch, horizon, 2)
        rho_avg, rho_0 = self.robustness(traj)
        clearance = self.clearance(traj)
        goal_dist = torch.sqrt(torch.sum((traj[:, -1] - self.goal.unsqueeze(0)) ** 2, dim=-1) + 1e-9)
        return {
            "init_pos": init_pos,
            "traj": traj,
            "actions": actions,
            "rho_avg": rho_avg,
            "rho_0": rho_0,
            "min_clearance": clearance.min(dim=1).values,
            "final_goal_dist": goal_dist,
        }

    def clearance(self, traj: torch.Tensor) -> torch.Tensor:
        diff = traj.unsqueeze(2) - self.obstacles[:, :2].view(1, 1, -1, 2)
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-9)
        return dist - self.obstacles[:, 2].view(1, 1, -1)

    def robustness(self, traj: torch.Tensor) -> torch.Tensor:
        safe_margin = strel_smooth_min(self.clearance(traj), beta=self.formula.beta)#.min(dim=-1).values
        goal_dist = torch.sqrt(torch.sum((traj - self.goal.view(1, 1, 2)) ** 2, dim=-1) + 1e-9)
        reach_margin = self.cfg.goal_tol - goal_dist
        signal = torch.stack([safe_margin, reach_margin], dim=1).unsqueeze(1)
        rho = self.formula.evaluate(signal, self.spatial_labels.expand(traj.shape[0], -1, -1))
        # Average robustness over the time dimension (all time steps contribute equally).
        return rho.mean(dim=1).squeeze(-1), rho[:,0]  # (batch,)
        #return rho[:,0]

def evaluate_policy(
    problem: ThreeObstacleSTRELProblem,
    policy: DeterministicPolicy,
    episodes: int,
    seed: int,
    wind_max: float = None,
) -> Dict[str, float]:
    if wind_max is None:
        wind_max = problem.cfg.wind_max
    generator = torch.Generator(device=problem.device).manual_seed(seed)
    init_pos = problem.sample_initial_positions(episodes, generator)
    out = problem.rollout(policy, batch_size=episodes, generator=generator,
                          init_pos=init_pos, wind_max=wind_max)
    rho = out["rho_0"]
    return {
        "rho_mean": float(rho.mean().item()),
        "rho_min": float(rho.min().item()),
        "sat_rate": float((rho > 0.0).float().mean().item()),
        "goal_dist_mean": float(out["final_goal_dist"].mean().item()),
        "clearance_mean": float(out["min_clearance"].mean().item()),
        "rho_values": rho.detach().cpu().numpy(),   # raw per-episode values for histogram
    }


def _set_formula_beta(formula, beta: float) -> None:
    """Recursively update beta in every node of a STREL formula tree."""
    if hasattr(formula, "beta"):
        formula.beta = beta
    for attr in ("left", "right", "phi"):
        child = getattr(formula, attr, None)
        if child is not None:
            _set_formula_beta(child, beta)


def train_no_critic_policy(
    cfg: PlanningConfig,
    device: torch.device,
) -> Tuple[DeterministicPolicy, Dict[str, List[float]], Dict[str, torch.Tensor], ThreeObstacleSTRELProblem]:
    torch.manual_seed(cfg.seed)
    problem = ThreeObstacleSTRELProblem(cfg, device)
    obs_dim = 2 + 2 + 6 + 3  # pos + rel_goal + rel_obs(3×2) + clearance(3)
    policy = DeterministicPolicy(obs_dim=obs_dim, act_dim=2, hidden=cfg.hidden,
                                 parameterization=cfg.action_parameterization).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train_iters)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    history: Dict[str, List[float]] = {"iter": [], "rho_mean": [], "rho_std": [], "sat_rate": [], "goal_dist": [], "clearance": [], "wind_max": []}

    # Running baseline for variance reduction (EMA of mean rho).
    rho_baseline = 0.0
    sat_ema = None  # EMA of sat_rate for early stopping; None = not yet initialized
    sat_streak = 0  # consecutive iterations where EMA sat_rate >= early_stop_sat

    for it in range(cfg.train_iters):
        # Curriculum: linearly ramp wind and beta over training.
        alpha = it / max(cfg.train_iters - 1, 1)
        current_wind_max = cfg.wind_max_start + (cfg.wind_max - cfg.wind_max_start) * alpha
        current_beta = cfg.strel_beta_min + (cfg.strel_beta_max - cfg.strel_beta_min) * alpha
        _set_formula_beta(problem.formula, current_beta)

        rollout = problem.rollout(policy, batch_size=cfg.batch_size, generator=generator, wind_max=current_wind_max)
        rho = rollout["rho_0"]

        # Aggregate rho over the batch: mean or a low percentile (CVaR-like).
        if cfg.rho_aggregation == "percentile":
            rho_agg = torch.quantile(rho, cfg.rho_percentile / 100.0)
        else:
            rho_agg = rho.mean()

        # Update EMA baseline and compute advantage.
        rho_baseline = cfg.baseline_momentum * rho_baseline + (1.0 - cfg.baseline_momentum) * float(rho_agg.detach().item())
        loss_rho = -(rho_agg - rho_baseline)

        # Penalty proportional to the fraction of violating trajectories (rho < 0),
        # scaled by their robustness deficit.  relu(-rho) is 0 for satisfied
        # trajectories and |rho| for violated ones; .mean() makes it scale with
        # the violation count.
        loss_violation = cfg.violation_penalty_weight * torch.relu(-rho).mean()

        # Direct safety penalty: penalise coming within safety_buffer of any
        # obstacle at any timestep, regardless of whether safety is the binding
        # STREL constraint.  rollout["min_clearance"] is (B, 3) — min over time
        # per obstacle; take the further min over obstacles to get (B,).
        min_clr    = rollout["min_clearance"].min(dim=-1).values          # (B,)
        loss_safe  = cfg.safety_penalty_weight * torch.relu(cfg.safety_buffer - min_clr).mean()

        loss = loss_rho + loss_violation + loss_safe

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        rho_det = rho.detach()
        sat_rate = float((rho_det > 0.0).float().mean().item())
        history["iter"].append(float(it + 1))
        history["rho_mean"].append(float(rho_det.mean().item()))
        history["rho_std"].append(float(rho_det.std().item()))
        history["sat_rate"].append(sat_rate)
        history["goal_dist"].append(float(rollout["final_goal_dist"].mean().item()))
        history["clearance"].append(float(rollout["min_clearance"].mean().item()))
        history["wind_max"].append(current_wind_max)

        # Early stopping: use EMA of sat_rate to suppress batch noise, then check streak.
        sat_ema = sat_rate if sat_ema is None else cfg.early_stop_ema * sat_ema + (1.0 - cfg.early_stop_ema) * sat_rate
        if sat_ema >= cfg.early_stop_sat:
            sat_streak += 1
        else:
            sat_streak = 0
        if sat_streak >= cfg.early_stop_patience:
            print(f"[NoCritic-STREL-Wind] Early stop at iter={it+1} "
                  f"(EMA sat={sat_ema:.2%} ≥ {cfg.early_stop_sat:.0%} "
                  f"for {cfg.early_stop_patience} iters in a row)")
            break

        if (it + 1) % 25 == 0:
            print(
                f"[NoCritic-STREL-Wind] iter={it+1:04d} "
                f"β={current_beta:.1f} "
                f"wind={current_wind_max:.3f} "
                f"rho={history['rho_mean'][-1]:+.3f} "
                f"sat={sat_rate:.2%} (ema={sat_ema:.2%}) "
                f"viol_pen={float(loss_violation.detach()):.3f} "
                f"safe_pen={float(loss_safe.detach()):.3f} "
                f"goal_d={history['goal_dist'][-1]:.3f} "
                f"clr={history['clearance'][-1]:+.3f}"
            )

    eval_out = problem.rollout(
        policy,
        batch_size=24,
        generator=torch.Generator(device=device).manual_seed(cfg.seed + 999),
        wind_max=cfg.wind_max,
    )
    return policy, history, eval_out, problem


def plot_results(
    cfg: PlanningConfig,
    problem: ThreeObstacleSTRELProblem,
    history: Dict[str, List[float]],
    eval_out: Dict[str, torch.Tensor],
    metrics: Dict = None,
    eval_wind: float = None,
) -> None:
    if eval_wind is None:
        eval_wind = cfg.wind_max
    import math, os
    import numpy as np
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    iters     = history["iter"]
    rho_mean  = history["rho_mean"]
    rho_std   = history["rho_std"]
    sat_rate  = history["sat_rate"]
    n_batch   = cfg.batch_size

    # rho: mean ± 1 std (band shows sample spread, not SE)
    rho_lo = [m - s for m, s in zip(rho_mean, rho_std)]
    rho_hi = [m + s for m, s in zip(rho_mean, rho_std)]
    ax[0].plot(iters, rho_mean, lw=2.0, color="tab:blue", label="Mean ρ")
    ax[0].fill_between(iters, rho_lo, rho_hi, alpha=0.18, color="tab:blue", label="±1 std ρ")

    # sat_rate: Wilson score 95 % CI for a proportion
    sat_lo, sat_hi = [], []
    for p in sat_rate:
        z = 1.96
        lo = (p + z*z/(2*n_batch) - z*math.sqrt(p*(1-p)/n_batch + z*z/(4*n_batch**2))) / (1 + z*z/n_batch)
        hi = (p + z*z/(2*n_batch) + z*math.sqrt(p*(1-p)/n_batch + z*z/(4*n_batch**2))) / (1 + z*z/n_batch)
        sat_lo.append(lo)
        sat_hi.append(hi)
    ax[0].plot(iters, sat_rate, lw=2.0, color="tab:orange", label="Sat. rate")
    ax[0].fill_between(iters, sat_lo, sat_hi, alpha=0.18, color="tab:orange", label="95% CI sat.")
    ax[0].axhline(cfg.early_stop_sat, color="gray", lw=1.0, ls="--", label=f"Stop threshold ({cfg.early_stop_sat:.0%})")

    ax[0].set_title(f"STREL Policy Optimization (train wind={cfg.wind_max})")
    ax[0].set_xlabel("Iteration")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    traj     = eval_out["traj"].detach().cpu()
    init_pos = eval_out["init_pos"].detach().cpu()
    rho_traj = eval_out["rho_0"].detach().cpu()          # per-trajectory robustness
    for obstacle in problem.obstacles.detach().cpu():
        circle = plt.Circle((float(obstacle[0]), float(obstacle[1])), float(obstacle[2]), color="tab:red", alpha=0.28)
        ax[1].add_patch(circle)
    goal = plt.Circle(cfg.goal_xy, cfg.goal_tol, color="tab:green", alpha=0.22)
    ax[1].add_patch(goal)
    for i in range(traj.shape[0]):
        violated = float(rho_traj[i]) < 0
        c     = "tab:red" if violated else "tab:blue"
        alpha = 0.70      if violated else 0.45
        lw    = 1.6       if violated else 1.4
        line = traj[i]
        path = torch.cat([init_pos[i].unsqueeze(0), line], dim=0)
        ax[1].plot(path[:, 0], path[:, 1], 'o-', markersize=2, alpha=alpha, lw=lw, color=c)
        ax[1].scatter(init_pos[i, 0], init_pos[i, 1], color="black", s=14, alpha=0.65)
    ax[1].scatter([cfg.goal_xy[0]], [cfg.goal_xy[1]], marker="*", s=150, color="tab:green")
    ax[1].set_xlim(cfg.world_min, cfg.world_max)
    ax[1].set_ylim(cfg.world_min, cfg.world_max)
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].set_title(f"Rollouts at Deployment (eval wind={eval_wind})")
    ax[1].grid(alpha=0.3)

    # Robustness histogram at deployment time
    if metrics is not None and "rho_values" in metrics:
        rho_vals = metrics["rho_values"]
        sat_rate = metrics["sat_rate"]
        rho_mean = metrics["rho_mean"]
        bins = np.linspace(rho_vals.min() - 0.05, rho_vals.max() + 0.05, 35)
        sat_mask = rho_vals >= 0
        ax[2].hist(rho_vals[sat_mask],  bins=bins, color="tab:green", alpha=0.75, label="Satisfied (ρ≥0)")
        ax[2].hist(rho_vals[~sat_mask], bins=bins, color="tab:red",   alpha=0.75, label="Violated (ρ<0)")
        ax[2].axvline(0,        color="black",      lw=1.2, ls="--")
        ax[2].axvline(rho_mean, color="tab:blue",   lw=1.5, ls="-",  label=f"Mean ρ = {rho_mean:+.3f}")
        ax[2].set_xlabel("Robustness ρ")
        ax[2].set_ylabel("Count")
        ax[2].set_title(f"Deployment Robustness — eval wind={eval_wind}  (sat={sat_rate:.1%}, n={len(rho_vals)})")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)
    else:
        ax[2].set_visible(False)

    fig.tight_layout()
    base, ext = os.path.splitext(cfg.plot_path)
    plot_path = f"{base}_wind{eval_wind}{ext}"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return plot_path


def main() -> None:
    import argparse, dataclasses, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="CKPT", default=None,
                        help="path to a saved checkpoint; skip training and evaluate only")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="number of episodes for evaluation (default: 1000)")
    parser.add_argument("--wind", type=float, default=None,
                        help="wind magnitude at evaluation time; defaults to the trained wind_max")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.load:
        # ── Evaluation-only mode ──────────────────────────────────────────
        ckpt = torch.load(args.load, map_location=device)
        cfg = PlanningConfig(**ckpt["cfg"])
        obs_dim = 2 + 2 + 6 + 3
        policy = DeterministicPolicy(obs_dim=obs_dim, act_dim=2, hidden=cfg.hidden,
                                     parameterization=cfg.action_parameterization).to(device)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        problem = ThreeObstacleSTRELProblem(cfg, device)
        # Formula is built with strel_beta_min; set it to the final trained beta.
        _set_formula_beta(problem.formula, cfg.strel_beta_max)
        eval_wind = args.wind if args.wind is not None else cfg.wind_max
        metrics = evaluate_policy(problem, policy, episodes=args.episodes,
                                   seed=cfg.seed + 1234, wind_max=eval_wind)
        # Generate a plot with only the histogram (no training history)
        generator = torch.Generator(device=device).manual_seed(cfg.seed + 999)
        eval_out = problem.rollout(policy, batch_size=24,
                                   generator=generator, wind_max=eval_wind)
        plot_results(cfg, problem, {"iter": [], "rho_mean": [], "rho_std": [],
                                    "sat_rate": [], "wind_max": []},
                     eval_out, metrics, eval_wind=eval_wind)
        print(f"  trained wind  : {ckpt.get('train_wind_max', cfg.wind_max)}")
        print(f"  eval wind     : {eval_wind}")
    else:
        # ── Training mode ─────────────────────────────────────────────────
        cfg = PlanningConfig()
        policy, history, eval_out, problem = train_no_critic_policy(cfg, device)
        eval_wind = cfg.wind_max
        metrics = evaluate_policy(problem, policy, episodes=args.episodes,
                                   seed=cfg.seed + 1234, wind_max=eval_wind)
        plot_results(cfg, problem, history, eval_out, metrics, eval_wind=eval_wind)
        os.makedirs("saved_models", exist_ok=True)
        ckpt_path = f"saved_models/std_no_critic_h{cfg.horizon}_wind{cfg.wind_max}.pt"
        torch.save({
            "policy":         policy.state_dict(),
            "cfg":            dataclasses.asdict(cfg),
            "train_wind_max": cfg.wind_max,   # disturbance used during training
        }, ckpt_path)
        print(f"model saved → {ckpt_path}")

    print("\nFinal evaluation")
    print(f"rho_mean       : {metrics['rho_mean']:+.4f}")
    print(f"rho_min        : {metrics['rho_min']:+.4f}")
    print(f"sat_rate       : {metrics['sat_rate']:.2%}")
    print(f"goal_dist_mean : {metrics['goal_dist_mean']:.4f}")
    print(f"clearance_mean : {metrics['clearance_mean']:+.4f}")
    base, ext = os.path.splitext(cfg.plot_path)
    print(f"plot           : {base}_wind{eval_wind}{ext}")


if __name__ == "__main__":
    main()
