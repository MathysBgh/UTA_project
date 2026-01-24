"""
uta_core.py — UTA with pairs (monotonic + non-monotonic GT + inference)
Clean refactor:
- Control number of non-monotonic criteria (n_non_monotonic) OR explicit list
- Control peak location: center/left/right/random with configurable ranges
- Pair sampling by indices (dominance-filtered) + optional label flipping noise
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gurobipy import GRB, LinExpr, Model


# ============================================================
# Scaling utilities
# ============================================================

def scale_matrix(X: np.ndarray, mode: Optional[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Column-wise scaling.
    mode in {None, 'minmax', 'sym', 'zscore'}:
      - minmax: [0,1]
      - sym:    [-1,1]
      - zscore: (x-mean)/std
    """
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    meta: Dict[str, Any] = {"mode": mode, "params": []}

    if mode is None:
        return X.copy(), meta

    Xs = np.empty_like(X)
    for j in range(m):
        col = X[:, j].astype(float)

        if mode in ("minmax", "unitrange"):
            mn, mx = float(col.min()), float(col.max())
            rng = (mx - mn) if mx > mn else 1.0
            Xs[:, j] = (col - mn) / rng
            meta["params"].append({"min": mn, "max": mx})

        elif mode == "sym":
            mn, mx = float(col.min()), float(col.max())
            rng = (mx - mn) if mx > mn else 1.0
            Xs[:, j] = 2.0 * (col - mn) / rng - 1.0
            meta["params"].append({"min": mn, "max": mx})

        elif mode == "zscore":
            mu, sd = float(col.mean()), float(col.std(ddof=0))
            sd = sd if sd > 0 else 1.0
            Xs[:, j] = (col - mu) / sd
            meta["params"].append({"mean": mu, "std": sd})

        else:
            raise ValueError("Unknown scaling mode.")
    return Xs, meta


def scale_with_info(X: np.ndarray, scale_info: Dict[str, Any]) -> np.ndarray:
    """Apply scaling using stored scale_info parameters."""
    mode = scale_info.get("mode")
    if not mode:
        return np.asarray(X, dtype=float)

    X = np.asarray(X, dtype=float).copy()
    params = scale_info["params"]

    for j in range(X.shape[1]):
        if mode in ("minmax", "unitrange"):
            mn, mx = params[j]["min"], params[j]["max"]
            rng = (mx - mn) if mx > mn else 1.0
            X[:, j] = (X[:, j] - mn) / rng

        elif mode == "sym":
            mn, mx = params[j]["min"], params[j]["max"]
            rng = (mx - mn) if mx > mn else 1.0
            X[:, j] = 2.0 * (X[:, j] - mn) / rng - 1.0

        elif mode == "zscore":
            mu, sd = params[j]["mean"], params[j]["std"]
            sd = sd if sd > 0 else 1.0
            X[:, j] = (X[:, j] - mu) / sd

        else:
            raise ValueError("Unknown scaling mode in scale_info.")
    return X


# ============================================================
# Breakpoints & interpolation helpers
# ============================================================

def build_breaks(minv: float, maxv: float, L: int) -> np.ndarray:
    """Build L equidistant breakpoints between minv and maxv."""
    if L < 2:
        raise ValueError("L must be >= 2.")
    return np.linspace(minv, maxv, L)


def interp_expr(m: Model, x: float, brk: np.ndarray, uvars: List[Any]) -> LinExpr:
    """
    Linear interpolation of u(x) over breakpoints brk and decision values uvars.
    Returns a Gurobi affine expression.
    """
    if x <= brk[0]:
        return LinExpr(uvars[0])
    if x >= brk[-1]:
        return LinExpr(uvars[-1])

    j = np.searchsorted(brk, x) - 1
    j = max(0, min(j, len(brk) - 2))

    x0, x1 = float(brk[j]), float(brk[j + 1])
    w1 = (x - x0) / (x1 - x0)
    w0 = 1.0 - w1

    expr = LinExpr()
    expr.addTerms([w0, w1], [uvars[j], uvars[j + 1]])
    return expr


# ============================================================
# Non-monotonic controls
# ============================================================

@dataclass(frozen=True)
class PeakConfig:
    """
    Controls peak location (t in [0,1]).
    - center: sample peak in [center_low, center_high]
    - left:   sample peak in [left_low, left_high]
    - right:  sample peak in [right_low, right_high]
    - random: sample peak in [0,1]
    """
    mode: str = "center"
    center_low: float = 0.4
    center_high: float = 0.6
    left_low: float = 0.0
    left_high: float = 0.3
    right_low: float = 0.7
    right_high: float = 1.0

    def sample_peak(self, rng: np.random.Generator) -> float:
        mode = self.mode.lower()
        if mode == "center":
            return float(rng.uniform(self.center_low, self.center_high))
        if mode == "left":
            return float(rng.uniform(self.left_low, self.left_high))
        if mode == "right":
            return float(rng.uniform(self.right_low, self.right_high))
        if mode == "random":
            return float(rng.uniform(0.0, 1.0))
        raise ValueError(f"Unknown peak mode: {self.mode}")


def choose_non_monotonic_criteria(
    m: int,
    rng: np.random.Generator,
    non_monotonic_criteria: Optional[List[int]] = None,
    n_non_monotonic: Optional[int] = None,
) -> List[int]:
    """
    Priority:
      1) if non_monotonic_criteria provided -> validate + return
      2) else if n_non_monotonic provided   -> sample exactly that many indices
      3) else                               -> default = 50% (legacy behavior)
    """
    if non_monotonic_criteria is not None:
        idx = sorted(set(int(i) for i in non_monotonic_criteria))
        if any(i < 0 or i >= m for i in idx):
            raise ValueError(f"non_monotonic_criteria must be within [0, {m-1}]")
        return idx

    if n_non_monotonic is not None:
        k = int(n_non_monotonic)
        if k < 0 or k > m:
            raise ValueError(f"n_non_monotonic must be in [0, {m}]")
        if k == 0:
            return []
        return sorted(rng.choice(np.arange(m), size=k, replace=False).tolist())

    # Legacy fallback: 1/2 chance each criterion
    return [j for j in range(m) if rng.random() < 0.5]


# ============================================================
# Ground truth generator
# ============================================================

def uta_gt(
    X: np.ndarray,
    L: int,
    seed: Optional[int] = None,
    weights_dirichlet_alpha: float = 1.0,
    increments_dirichlet_alpha: float = 1.0,
    scale: Optional[str] = "minmax",
    # --- non-monotonic selection ---
    non_monotonic_criteria: Optional[List[int]] = None,
    n_non_monotonic: Optional[int] = None,
    # --- non-monotonic shape ---
    peak_cfg: PeakConfig = PeakConfig(),
    # --- noise ---
    noise_level: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate additive GT (monotonic + single-peaked non-monotonic marginals).

    Monotonic criteria:
      - u_k is PWL monotone via Dirichlet increments (u[0]=0, u[-1]=1)
    Non-monotonic criteria:
      - single-peaked "bump" (concave parabola) with configurable peak position

    Returns dict with:
      scale_info, breaks, u_values (m x L), weights, scores, ranking,
      is_monotonic (bool per criterion),
      non_monotonic_criteria (list)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, m = X.shape

    nm_idx = choose_non_monotonic_criteria(
        m=m,
        rng=rng,
        non_monotonic_criteria=non_monotonic_criteria,
        n_non_monotonic=n_non_monotonic,
    )

    Xs, scale_info = scale_matrix(X, scale)
    breaks = [build_breaks(Xs[:, j].min(), Xs[:, j].max(), L) for j in range(m)]
    w = rng.dirichlet(np.full(m, weights_dirichlet_alpha))

    u_values = np.zeros((m, L), dtype=float)
    is_monotonic = np.ones(m, dtype=bool)

    tgrid = np.linspace(0.0, 1.0, L)

    for j in range(m):
        if j in nm_idx:
            # single-peaked concave parabola
            is_monotonic[j] = False
            peak = peak_cfg.sample_peak(rng)
            raw = -((tgrid - peak) ** 2)
            raw -= raw.min()
            if raw.max() > 0:
                raw /= raw.max()
            u_values[j, :] = raw
        else:
            # monotone PWL via positive increments
            inc = rng.dirichlet(np.full(L - 1, increments_dirichlet_alpha))
            u_values[j, 0] = 0.0
            u_values[j, 1:] = np.cumsum(inc)

    # Optional perturbation of marginal values (kept in [0,1])
    if noise_level and noise_level > 0:
        u_values = np.clip(u_values + rng.normal(0.0, noise_level, size=u_values.shape), 0.0, 1.0)

    # Scores + ranking on X
    scores = np.zeros(n, dtype=float)
    for a in range(n):
        s = 0.0
        for j in range(m):
            x = float(Xs[a, j])
            brk = breaks[j]
            uv = u_values[j]

            if x <= brk[0]:
                u = uv[0]
            elif x >= brk[-1]:
                u = uv[-1]
            else:
                k = np.searchsorted(brk, x) - 1
                k = max(0, min(k, L - 2))
                lam = (x - brk[k]) / (brk[k + 1] - brk[k])
                u = (1.0 - lam) * uv[k] + lam * uv[k + 1]

            s += w[j] * u
        scores[a] = s

    ranking = np.argsort(-scores).tolist()

    return {
        "scale_info": scale_info,
        "breaks": breaks,
        "u_values": u_values,
        "weights": w,
        "scores": scores,
        "ranking": ranking,
        "is_monotonic": is_monotonic,
        "non_monotonic_criteria": nm_idx,
        "peak_cfg": peak_cfg.__dict__,
    }


# ============================================================
# Scoring function (from model dict)
# ============================================================

def make_scoring_fn(model_dict: Dict[str, Any]):
    """
    Build U(X) from dict {breaks, u_values, weights, scale_info}.
    Works for GT or inferred model.
    """
    breaks = model_dict["breaks"]
    u_vals = np.asarray(model_dict["u_values"], float)
    w = np.asarray(model_dict["weights"], float)
    scale_info = model_dict["scale_info"]

    def U(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        Xs = scale_with_info(X, scale_info)
        n = Xs.shape[0]
        scores = np.zeros(n, dtype=float)

        for a in range(n):
            s = 0.0
            for j in range(Xs.shape[1]):
                x = float(Xs[a, j])
                brk = breaks[j]
                uv = u_vals[j]

                if x <= brk[0]:
                    u = uv[0]
                elif x >= brk[-1]:
                    u = uv[-1]
                else:
                    k = np.searchsorted(brk, x) - 1
                    k = max(0, min(k, len(brk) - 2))
                    lam = (x - brk[k]) / (brk[k + 1] - brk[k])
                    u = (1.0 - lam) * uv[k] + lam * uv[k + 1]

                s += w[j] * u

            scores[a] = s
        return scores

    return U


# ============================================================
# Pair sampling (dominance filtered) — BY INDICES
# ============================================================

def dominates(x: np.ndarray, y: np.ndarray) -> bool:
    """x dominates y iff x_j >= y_j for all j and strict > for at least one j."""
    ge = np.all(x >= y)
    gt = np.any(x > y)
    return bool(ge and gt)


def sample_pairs_indices(
    X: np.ndarray,
    U_fn,
    n_pairs: int,
    seed: int = 0,
    flip_prob: float = 0.0,
    max_tries: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample (i,j) pairs on X WITHOUT dominance, label by sign(U(i)-U(j)).
    labels: +1 means i preferred to j, -1 otherwise.
    flip_prob: probability to flip label (preference noise).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    n = X.shape[0]

    pairs: List[Tuple[int, int]] = []
    labels: List[int] = []

    tries = 0
    while len(pairs) < n_pairs and tries < max_tries:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            tries += 1
            continue

        xi, xj = X[i], X[j]
        if dominates(xi, xj) or dominates(xj, xi):
            tries += 1
            continue

        ui = float(U_fn(xi.reshape(1, -1))[0])
        uj = float(U_fn(xj.reshape(1, -1))[0])
        if ui == uj:
            tries += 1
            continue

        y = 1 if ui > uj else -1
        if flip_prob > 0 and rng.random() < flip_prob:
            y *= -1

        pairs.append((int(i), int(j)))
        labels.append(int(y))
        tries += 1

    if len(pairs) < n_pairs:
        warnings.warn(f"Only generated {len(pairs)}/{n_pairs} pairs (max_tries hit).")

    return np.array(pairs, dtype=int), np.array(labels, dtype=int)


# ============================================================
# UTA inference from PAIRS (Gurobi)
# ============================================================

def uta_inf_pairs(
    X: np.ndarray,
    pairs_idx: np.ndarray,   # (K,2)
    labels: np.ndarray,      # (K,) +1 if i ≻ j, -1 otherwise
    L: int,
    scale: Optional[str] = "minmax",
    use_non_monotonic: bool = False,
    # Non-monotone objective controls
    gamma_weight: float = 0.5,
    gamma_upper_bound: float = 2.0,
    epsilon_lb: float = 0.01,
    epsilon_ub: float = 0.5,
    # Slack (noise handling)
    slack_weight: float = 1.0,
    # Misc
    gurobi_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Learn UTA from pairwise comparisons.
    - use_non_monotonic=False: classic monotone UTA (u nondecreasing + weights via normalization)
    - use_non_monotonic=True: allow non-monotonic u and add slope-variation penalty via gamma,
                              plus epsilon variable in [epsilon_lb, epsilon_ub]

    NOTE: This is consistent with your current structure; main changes are cleanliness + robustness.
    """
    t0 = time.perf_counter()

    X = np.asarray(X, float)
    pairs_idx = np.asarray(pairs_idx, int)
    labels = np.asarray(labels, int)

    n, m = X.shape
    K = pairs_idx.shape[0]

    Xs, scale_info = scale_matrix(X, scale)
    breaks = [build_breaks(Xs[:, j].min(), Xs[:, j].max(), L) for j in range(m)]

    model = Model("UTA_Pairs")
    if not verbose:
        model.setParam("OutputFlag", 0)
    if gurobi_params:
        for k, v in gurobi_params.items():
            model.setParam(k, v)

    # u_{j,l}
    u: List[List[Any]] = []
    for j in range(m):
        if use_non_monotonic:
            # bounded [0,1] for stability (matches typical bounded formulations)
            uj = [model.addVar(lb=0.0, ub=1.0, name=f"u_{j}_{l}") for l in range(L)]
        else:
            uj = [model.addVar(lb=0.0, name=f"u_{j}_{l}") for l in range(L)]
        u.append(uj)

    # Monotonicity constraints (only in monotone setting)
    if not use_non_monotonic:
        for j in range(m):
            for l in range(1, L):
                model.addConstr(u[j][l] >= u[j][l - 1], name=f"mono_{j}_{l}")

    # Normalize: u[j][0]=0 and sum_j u[j][-1]=1 (weights encoded in last point)
    for j in range(m):
        model.addConstr(u[j][0] == 0.0, name=f"norm0_{j}")
    model.addConstr(sum(u[j][-1] for j in range(m)) == 1.0, name="norm_sum_weights")

    # Slope-variation gamma for non-monotone (penalize curvature / changes)
    gamma = None
    if use_non_monotonic and L > 2:
        gamma = []
        for j in range(m):
            gj = [model.addVar(lb=0.0, ub=gamma_upper_bound, name=f"gamma_{j}_{k}") for k in range(L - 2)]
            gamma.append(gj)

        # |(u[l+1]-u[l]) - (u[l]-u[l-1])| <= gamma
        for j in range(m):
            for l in range(1, L - 1):
                d1 = u[j][l] - u[j][l - 1]
                d2 = u[j][l + 1] - u[j][l]
                model.addConstr(d2 - d1 <= gamma[j][l - 1], name=f"gpos_{j}_{l}")
                model.addConstr(d1 - d2 <= gamma[j][l - 1], name=f"gneg_{j}_{l}")

    # epsilon (discriminatory power)
    epsilon_var = None
    if use_non_monotonic:
        epsilon_var = model.addVar(lb=epsilon_lb, ub=epsilon_ub, name="epsilon")
    else:
        # fixed epsilon in monotone case (you can also make it a var if you prefer)
        epsilon_var = None

    # slack for each constraint (noise)
    slacks = [model.addVar(lb=0.0, name=f"slack_{k}") for k in range(K)]

    # Pairwise preference constraints:
    # If labels[k] = +1 => V(i) - V(j) >= epsilon - slack
    # If labels[k] = -1 => V(j) - V(i) >= epsilon - slack
    for k in range(K):
        i, j = int(pairs_idx[k, 0]), int(pairs_idx[k, 1])
        y = int(labels[k])

        Vi = LinExpr()
        Vj = LinExpr()

        for crit in range(m):
            Vi += interp_expr(model, float(Xs[i, crit]), breaks[crit], u[crit])
            Vj += interp_expr(model, float(Xs[j, crit]), breaks[crit], u[crit])

        eps_term = epsilon_var if epsilon_var is not None else 1e-2  # default tiny margin
        if y == 1:
            model.addConstr(Vi - Vj >= eps_term - slacks[k], name=f"pref_{k}")
        else:
            model.addConstr(Vj - Vi >= eps_term - slacks[k], name=f"pref_{k}")

    # Objective:
    # - non-monotone: maximize epsilon - gamma_weight*sum(gamma) - slack_weight*sum(slacks)
    # - monotone:     minimize slacks (or maximize margin if you convert epsilon into var)
    if use_non_monotonic:
        gamma_sum = 0.0 if gamma is None else sum(gamma[j][l] for j in range(m) for l in range(L - 2))
        model.setObjective(
            epsilon_var - gamma_weight * gamma_sum - slack_weight * sum(slacks),
            GRB.MAXIMIZE
        )
    else:
        model.setObjective(-slack_weight * sum(slacks), GRB.MAXIMIZE)

    model.optimize()

    status = model.Status
    if status != GRB.OPTIMAL:
        warnings.warn(f"Gurobi ended with status={status}")

    # Extract u-values
    u_values = np.zeros((m, L), dtype=float)
    for j in range(m):
        for l in range(L):
            u_values[j, l] = float(u[j][l].X)

    weights = u_values[:, -1].copy()

    gamma_values = None
    if use_non_monotonic and gamma is not None:
        gamma_values = np.zeros((m, L - 2), dtype=float)
        for j in range(m):
            for l in range(L - 2):
                gamma_values[j, l] = float(gamma[j][l].X)

    # Scores & ranking on X
    def _score_one(xrow: np.ndarray) -> float:
        s = 0.0
        for j in range(m):
            brk = breaks[j]
            uv = u_values[j]
            x = float(xrow[j])
            if x <= brk[0]:
                uval = uv[0]
            elif x >= brk[-1]:
                uval = uv[-1]
            else:
                k = np.searchsorted(brk, x) - 1
                k = max(0, min(k, L - 2))
                lam = (x - brk[k]) / (brk[k + 1] - brk[k])
                uval = (1.0 - lam) * uv[k] + lam * uv[k + 1]
            s += weights[j] * uval
        return float(s)

    scores = np.array([_score_one(Xs[a]) for a in range(n)], dtype=float)
    ranking = np.argsort(-scores).tolist()

    t1 = time.perf_counter()
    return {
        "status": status,
        "obj": float(model.ObjVal) if status == GRB.OPTIMAL else None,
        "time_sec": float(t1 - t0),
        "scores": scores,
        "ranking": ranking,
        "breaks": breaks,
        "u_values": u_values,
        "weights": weights,
        "gamma_values": gamma_values,
        "epsilon": float(epsilon_var.X) if (use_non_monotonic and epsilon_var is not None and status == GRB.OPTIMAL) else None,
        "method_used": "non-monotonic" if use_non_monotonic else "monotonic",
        "scale_info": scale_info,
    }


# ============================================================
# Simple plotting helpers (optional for notebook)
# ============================================================

def detect_monotonic_flags(u_values: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Criterion is monotonic if all diffs >= -tol OR all diffs <= tol."""
    u_values = np.asarray(u_values, float)
    m, _ = u_values.shape
    flags = np.zeros(m, dtype=bool)
    for j in range(m):
        diffs = np.diff(u_values[j])
        inc = np.all(diffs >= -tol)
        dec = np.all(diffs <= tol)
        flags[j] = bool(inc or dec)
    return flags