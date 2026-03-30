import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


# -----------------------------
# Minimal quantitative STREL
# -----------------------------
def smooth_max(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """
    Soft maximum over the last dimension.

    Uses log(sum(exp(beta*x)) + 1e-12) / beta instead of torch.logsumexp so
    that the ONNX graph only contains ReduceSum + element-wise exp/log —
    all of which are supported by auto_LiRPA's IBP propagation.
    (torch.logsumexp exports to ReduceLogSumExp, which auto_LiRPA does not support.)
    """
    xb = beta * x
    return torch.log(torch.sum(torch.exp(xb), dim=-1) + 1e-12) / beta


def smooth_min(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """
    Soft minimum over the last dimension.

    Uses -log(sum(exp(-beta*x)) + 1e-12) / beta — same LiRPA-safe decomposition
    as smooth_max.
    """
    xb = -beta * x
    return -torch.log(torch.sum(torch.exp(xb), dim=-1) + 1e-12) / beta


class STRELFormula:
    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AtomicPredicate(STRELFormula):
    """
    signal: [B, N, F, T]
    lab: [B, N, T] integer class ids
    """

    def __init__(self, var_ind: int, threshold: float, labels: torch.Tensor = None, lte: bool = False):
        self.var_ind = var_ind
        self.sign = -1.0 if lte else 1.0
        self.bias = threshold if lte else -threshold
        self.labels = torch.ones(1) if labels is None else labels.reshape(-1)
        self.neg_inf = -1e2

    def _predicate(self, signal: torch.Tensor) -> torch.Tensor:
        return self.sign * signal[:, :, self.var_ind, :] + self.bias

    def _mask(self, x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        labels = self.labels.to(x.device).view(1, 1, 1, -1)
        class_indices = torch.arange(self.labels.shape[0], device=x.device).view(1, 1, 1, -1)
        one_hot = (lab.unsqueeze(-1) == class_indices).to(x.dtype)
        mask = torch.sum(one_hot * labels, dim=-1)
        return torch.clamp(mask, min=0.0, max=1.0)

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        m = self._mask(signal, lab)
        z = self._predicate(signal)
        return z * m + (1.0 - m) * self.neg_inf


class TrueFormula(STRELFormula):
    """Robust semantics for logical true."""

    def __init__(self, pos_inf: float = 1e2):
        self.pos_inf = pos_inf

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        _ = lab
        return torch.full(
            (signal.shape[0], signal.shape[1], signal.shape[-1]),
            self.pos_inf,
            dtype=signal.dtype,
            device=signal.device,
        )


class Not(STRELFormula):
    def __init__(self, phi: STRELFormula):
        self.phi = phi

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return -self.phi.evaluate(signal, lab)


class Or(STRELFormula):
    def __init__(self, left: STRELFormula, right: STRELFormula,
                 beta: float = 20.0):
        self.left  = left
        self.right = right
        self.beta  = beta

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        l = self.left.evaluate(signal, lab)
        r = self.right.evaluate(signal, lab)
        return smooth_max(torch.stack([l, r], dim=-1), beta=self.beta)


class And(STRELFormula):
    def __init__(self, left: STRELFormula, right: STRELFormula,
                 beta: float = 20.0):
        self.left  = left
        self.right = right
        self.beta  = beta

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        l = self.left.evaluate(signal, lab)
        r = self.right.evaluate(signal, lab)
        return smooth_min(torch.stack([l, r], dim=-1), beta=self.beta)


class Always(STRELFormula):
    def __init__(self, phi: STRELFormula, beta: float = 20.0):
        self.phi  = phi
        self.beta = beta

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return smooth_min(self.phi.evaluate(signal, lab), beta=self.beta)


class Eventually(STRELFormula):
    def __init__(self, phi: STRELFormula, beta: float = 20.0):
        self.phi  = phi
        self.beta = beta

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return smooth_max(self.phi.evaluate(signal, lab), beta=self.beta)


class Reach(STRELFormula):
    """
    Reach(phi1, phi2, d1, d2):
    from source node i, there exists destination j such that
    - path-intermediate nodes satisfy phi1
    - destination satisfies phi2
    - shortest-path spatial distance is in [d1, d2]
    Returns robustness [B, N, T]
    """

    def __init__(
        self,
        phi1: STRELFormula,
        phi2: STRELFormula,
        labels1: Optional[torch.Tensor],
        labels2: Optional[torch.Tensor],
        d1: float,
        d2: float,
        pos_x_ind: int = 2,
        pos_y_ind: int = 3,
    ):
        self.phi1 = phi1
        self.phi2 = phi2
        self.labels1 = None if labels1 is None else labels1.reshape(-1)
        self.labels2 = None if labels2 is None else labels2.reshape(-1)
        self.d1 = d1
        self.d2 = d2
        self.pos_x_ind = pos_x_ind
        self.pos_y_ind = pos_y_ind
        self.eps = 1e-8
        self.neg_inf = -1e2
        self.pos_inf = 1e2

    def _mask(self, x: torch.Tensor, lab: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels is None:
            return torch.ones_like(lab, dtype=x.dtype)
        labels = labels.to(x.device).view(1, 1, 1, -1)
        class_indices = torch.arange(labels.shape[-1], device=x.device).view(1, 1, 1, -1)
        one_hot = (lab.unsqueeze(-1) == class_indices).to(x.dtype)
        mask = torch.sum(one_hot * labels, dim=-1)
        return torch.clamp(mask, min=0.0, max=1.0)

    def _dist_mat(self, signal: torch.Tensor) -> torch.Tensor:
        pos = signal[:, :, [self.pos_x_ind, self.pos_y_ind], :].permute(0, 3, 1, 2)
        diff = pos.unsqueeze(2) - pos.unsqueeze(3)
        return torch.sqrt(torch.sum(diff * diff, dim=-1) + self.eps)

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        b, n, _, t = signal.shape
        dist = self._dist_mat(signal)  # [B,T,N,N]
        adj = dist / (torch.abs(dist) + self.eps)

        idx = torch.arange(n, device=signal.device)
        eye = (idx.unsqueeze(0) == idx.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(signal.dtype)

        m1 = self._mask(signal, lab, self.labels1)
        m2 = self._mask(signal, lab, self.labels2)

        s1 = self.phi1.evaluate(signal, lab)
        s2 = self.phi2.evaluate(signal, lab)
        s1 = s1 * m1 + (1.0 - m1) * self.neg_inf
        s2 = s2 * m2 + (1.0 - m2) * self.neg_inf

        s1_btnt = s1.permute(0, 2, 1)
        s2_btnt = s2.permute(0, 2, 1)

        c = s1_btnt.unsqueeze(-1).expand(b, t, n, n)
        c = c * adj + self.neg_inf * (1.0 - adj)
        c = self.pos_inf * eye + c * (1.0 - eye)

        wcap = c.clone()
        for k in range(n):
            wik = wcap[:, :, :, k].unsqueeze(-1)
            wkj = wcap[:, :, k, :].unsqueeze(-2)
            cand = 0.5 * (wik + wkj - torch.abs(wik - wkj))
            wcap = 0.5 * (wcap + cand + torch.abs(wcap - cand))

        d = dist * adj + self.pos_inf * (1.0 - adj)
        d = d * (1.0 - eye)
        for k in range(n):
            dik = d[:, :, :, k].unsqueeze(-1)
            dkj = d[:, :, k, :].unsqueeze(-2)
            cand = dik + dkj
            d = 0.5 * (d + cand - torch.abs(d - cand))

        low = d - self.d1 + 1e-6
        high = self.d2 - d + 1e-6
        in_range = 0.5 * (low + high - torch.abs(low - high))
        elig = 0.5 * (1.0 + in_range / (torch.abs(in_range) + self.eps))

        s2_dest = s2_btnt.unsqueeze(-2).expand(b, t, n, n)
        pair = 0.5 * (wcap + s2_dest - torch.abs(wcap - s2_dest))
        pair = pair * elig + self.neg_inf * (1.0 - elig)

        # Exclude i→i self-paths: a node should not count as its own "too-close" neighbour
        pair = pair * (1.0 - eye) + self.neg_inf * eye

        best = pair[:, :, :, 0].clone()
        for j in range(1, n):
            best = 0.5 * (best + pair[:, :, :, j] + torch.abs(best - pair[:, :, :, j]))

        return best.permute(0, 2, 1)


class Escape(STRELFormula):
    """
    Escape(phi, [d1, d2]):
    there exists a destination reachable within the distance interval while
    remaining in nodes satisfying phi along the path.
    """

    def __init__(
        self,
        phi: STRELFormula,
        labels: Optional[torch.Tensor],
        d1: float,
        d2: float,
        pos_x_ind: int = 2,
        pos_y_ind: int = 3,
    ):
        self.phi = phi
        self.labels = None if labels is None else labels.reshape(-1)
        self.d1 = d1
        self.d2 = d2
        self.pos_x_ind = pos_x_ind
        self.pos_y_ind = pos_y_ind
        self.eps = 1e-8
        self.neg_inf = -1e2
        self.pos_inf = 1e2

    def _mask(self, x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        if self.labels is None:
            return torch.ones_like(lab, dtype=x.dtype)
        labels = self.labels.to(x.device).view(1, 1, 1, -1)
        class_indices = torch.arange(self.labels.shape[0], device=x.device).view(1, 1, 1, -1)
        one_hot = (lab.unsqueeze(-1) == class_indices).to(x.dtype)
        mask = torch.sum(one_hot * labels, dim=-1)
        return torch.clamp(mask, min=0.0, max=1.0)

    def _dist_mat(self, signal: torch.Tensor) -> torch.Tensor:
        pos = signal[:, :, [self.pos_x_ind, self.pos_y_ind], :].permute(0, 3, 1, 2)
        diff = pos.unsqueeze(2) - pos.unsqueeze(3)
        return torch.sqrt(torch.sum(diff * diff, dim=-1) + self.eps)

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        b, n, _, t = signal.shape
        dist = self._dist_mat(signal)
        adj = dist / (torch.abs(dist) + self.eps)

        idx = torch.arange(n, device=signal.device)
        eye = (idx.unsqueeze(0) == idx.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(signal.dtype)

        m = self._mask(signal, lab)
        s = self.phi.evaluate(signal, lab)
        s = s * m + (1.0 - m) * self.neg_inf
        s_btnt = s.permute(0, 2, 1)

        c = s_btnt.unsqueeze(-1).expand(b, t, n, n)
        c = c * adj + self.neg_inf * (1.0 - adj)
        c = self.pos_inf * eye + c * (1.0 - eye)

        wcap = c.clone()
        for k in range(n):
            wik = wcap[:, :, :, k].unsqueeze(-1)
            wkj = wcap[:, :, k, :].unsqueeze(-2)
            cand = 0.5 * (wik + wkj - torch.abs(wik - wkj))
            wcap = 0.5 * (wcap + cand + torch.abs(wcap - cand))

        d = dist * adj + self.pos_inf * (1.0 - adj)
        d = d * (1.0 - eye)
        for k in range(n):
            dik = d[:, :, :, k].unsqueeze(-1)
            dkj = d[:, :, k, :].unsqueeze(-2)
            cand = dik + dkj
            d = 0.5 * (d + cand - torch.abs(d - cand))

        low = d - self.d1 + 1e-6
        high = self.d2 - d + 1e-6
        in_range = 0.5 * (low + high - torch.abs(low - high))
        elig = 0.5 * (1.0 + in_range / (torch.abs(in_range) + self.eps))

        s_dest = s_btnt.unsqueeze(-2)
        pair = 0.5 * (wcap + s_dest - torch.abs(wcap - s_dest))
        pair = pair * elig + self.neg_inf * (1.0 - elig)

        best = pair[:, :, :, 0].clone()
        for j in range(1, n):
            best = 0.5 * (best + pair[:, :, :, j] + torch.abs(best - pair[:, :, :, j]))

        return best.permute(0, 2, 1)


class Somewhere(STRELFormula):
    """
    Somewhere(phi, [d1, d2]) := Reach(True, phi, [d1, d2]).
    """

    def __init__(
        self,
        phi: STRELFormula,
        labels: torch.Tensor,
        d2: float,
        d1: float = 0.0,
        pos_x_ind: int = 2,
        pos_y_ind: int = 3,
    ):
        self.phi = phi
        self.labels = labels.reshape(-1)
        self.d1 = d1
        self.d2 = d2
        self.reach = Reach(
            phi1=TrueFormula(),
            phi2=phi,
            labels1=None,
            labels2=self.labels,
            d1=d1,
            d2=d2,
            pos_x_ind=pos_x_ind,
            pos_y_ind=pos_y_ind,
        )

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return self.reach.evaluate(signal, lab)


class Everywhere(STRELFormula):
    """
    Everywhere(phi, [d1, d2]) := Not(Somewhere(Not(phi), [d1, d2])).
    """

    def __init__(
        self,
        phi: STRELFormula,
        labels: torch.Tensor,
        d2: float,
        d1: float = 0.0,
        pos_x_ind: int = 2,
        pos_y_ind: int = 3,
    ):
        self.formula = Not(
            Somewhere(
                phi=Not(phi),
                labels=labels,
                d1=d1,
                d2=d2,
                pos_x_ind=pos_x_ind,
                pos_y_ind=pos_y_ind,
            )
        )

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return self.formula.evaluate(signal, lab)


class Surround(STRELFormula):
    """
    Surround(left, right, [0, d2]):
    left holds at the source, the region cannot reach a complementary node
    before hitting right, and it cannot escape beyond d2 while remaining in left.
    """

    def __init__(
        self,
        left: STRELFormula,
        right: STRELFormula,
        left_labels: torch.Tensor,
        right_labels: torch.Tensor,
        all_labels: torch.Tensor,
        d2: float,
        d1: float = 0.0,
        pos_x_ind: int = 2,
        pos_y_ind: int = 3,
        beta: float = 20.0,
        max_escape_distance: float = 1e6,
    ):
        self.left = left
        self.right = right
        self.left_labels = left_labels.reshape(-1)
        self.right_labels = right_labels.reshape(-1)
        self.all_labels = all_labels.reshape(-1)
        self.complementary_labels = torch.clamp(
            self.all_labels - torch.clamp(self.left_labels + self.right_labels, max=1.0),
            min=0.0,
            max=1.0,
        )

        blocked_region = Not(Or(left, right, beta=beta))
        self.formula = And(
            And(
                left,
                Not(
                    Reach(
                        phi1=left,
                        phi2=blocked_region,
                        labels1=self.left_labels,
                        labels2=self.complementary_labels,
                        d1=d1,
                        d2=d2,
                        pos_x_ind=pos_x_ind,
                        pos_y_ind=pos_y_ind,
                    ),
                ),
                beta=beta,
            ),
            Not(
                Escape(
                    phi=left,
                    labels=self.left_labels,
                    d1=d2,
                    d2=max_escape_distance,
                    pos_x_ind=pos_x_ind,
                    pos_y_ind=pos_y_ind,
                )
            ),
            beta=beta,
        )

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return self.formula.evaluate(signal, lab)