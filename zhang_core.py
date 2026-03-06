"""zhang_core.py — Zhang et al. (2025) representative sorting model (lexicographic optimization)

Implements the core optimization logic from:
  Z. Zhang et al., "Lexicographic optimization-based approaches to learning a representative
  sorting model for MCS problems with non-monotonic criteria", Computers & Operations Research,
  175 (2025) 106917.

What this module covers
-----------------------
- Constraint sets used in the paper to learn additive sorting models with potentially non-monotonic
  marginal value functions:
  * E_AR   (Eq. 12): assignment examples constraints
  * E_Sort (Eq. 13): additive global value + piecewise-linear interpolation + threshold spacing
  * E_Bound (Eq. 14): bounds on marginal values at breakpoints
  * E_Slope (Prop. 4): linearization of gamma_{l,j} = |slope_left - slope_right|

- Consistency check model (M-1)
- Minimum adjustment model (M-2) via Prop. 3 (Eq. 17)
- Representative learning via lexicographic optimization:
  * Approach 1 (M-4) then (M-6): minimize complexity then maximize discriminative power
  * Approach 2 (M-7) then (M-8): maximize discriminative power then minimize complexity

Design choices
--------------
- Breakpoints beta_{l,j} are built from data min/max with equal-width sub-intervals, like in UTA.
  (Paper assumes breakpoints are given; this is the standard practical choice.)
- We omit b0 and bq from the optimization models, as done in the paper, and compute them post-solve
  from min/max of marginal values + epsilon.
- We keep the implementation dependency-light by reusing a few helpers from uta_core.py:
  scale_matrix, scale_with_info, build_breaks, and interp_expr.

Expected workflow
-----------------
- In a notebook, you will:
  1) Prepare X (n x m), and a set of reference indices ar_idx with their assigned categories B (1..q).
  2) Call fit_zhang_representative(..., approach=1 or 2) to learn v_j and thresholds.
  3) Use predict_categories(...) to assign categories for new alternatives.

"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# NOTE
# ----
# We intentionally avoid importing gurobipy at module import time.
# Your local environment (where you run the notebooks) already uses Gurobi,
# but this container may not ship gurobipy. We therefore import gurobipy lazily
# inside the optimization routines.

def scale_matrix(X: np.ndarray, mode: Optional[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Column-wise scaling (same API as your uta_core.py).

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
            raise ValueError("Unknown scaling mode.")

    return X


def build_breaks(minv: float, maxv: float, L: int) -> np.ndarray:
    """Equally spaced breakpoints between minv and maxv (inclusive)."""
    if L < 2:
        raise ValueError("L must be >= 2")
    if maxv <= minv:
        return np.array([minv for _ in range(L)], dtype=float)
    return np.linspace(float(minv), float(maxv), int(L), dtype=float)


def interp_expr(mdl: Any, x: float, brk: np.ndarray, uvars: List[Any]) -> Any:
    """Piecewise-linear interpolation expression in terms of breakpoint variables.

    This mirrors uta_core.interp_expr but is kept local to avoid importing gurobipy
    at module import time.
    """
    # Lazy import so the module can be imported without gurobipy present.
    try:
        from gurobipy import LinExpr
    except Exception as e:  # pragma: no cover
        raise ImportError("gurobipy is required to build optimization expressions.") from e

    brk = np.asarray(brk, float)
    L = len(brk)

    if x <= brk[0]:
        return LinExpr(uvars[0])
    if x >= brk[-1]:
        return LinExpr(uvars[-1])

    k = int(np.searchsorted(brk, x) - 1)
    k = max(0, min(k, L - 2))
    denom = float(brk[k + 1] - brk[k])
    lam = (float(x) - float(brk[k])) / denom if denom != 0 else 0.0

    # (1-lam)*u_k + lam*u_{k+1}
    expr = LinExpr()
    expr += (1.0 - lam) * uvars[k]
    expr += lam * uvars[k + 1]
    return expr


# ============================================================
# Data structures
# ============================================================

@dataclass
class ZhangFitResult:
    """Container for a fitted sorting model."""

    status: int
    obj: Optional[float]
    time_sec: float

    # Learned objects
    breaks: List[np.ndarray]          # per criterion, shape (L,)
    v_values: np.ndarray              # shape (m, L)
    thresholds: np.ndarray            # b_1..b_{q-1}, shape (q-1,)
    epsilon: float

    # Derived thresholds including b0 and bq
    b0: float
    bq: float

    # Metadata
    q: int
    approach: int
    scale_info: Dict[str, Any]

    # Convenience
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": int(self.status),
            "obj": None if self.obj is None else float(self.obj),
            "time_sec": float(self.time_sec),
            "breaks": self.breaks,
            "v_values": self.v_values,
            "thresholds": self.thresholds,
            "epsilon": float(self.epsilon),
            "b0": float(self.b0),
            "bq": float(self.bq),
            "q": int(self.q),
            "approach": int(self.approach),
            "scale_info": self.scale_info,
        }


# ============================================================
# Internal helpers
# ============================================================

def _validate_inputs(
    X: np.ndarray,
    ar_idx: np.ndarray,
    B: np.ndarray,
    q: int,
    L: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    ar_idx = np.asarray(ar_idx, int).ravel()
    B = np.asarray(B, int).ravel()

    if X.ndim != 2:
        raise ValueError("X must be 2D (n x m).")
    n, _m = X.shape

    if ar_idx.size != B.size:
        raise ValueError("ar_idx and B must have the same length.")
    if np.any(ar_idx < 0) or np.any(ar_idx >= n):
        raise ValueError("ar_idx contains out-of-range indices.")

    if q < 2:
        raise ValueError("q must be >= 2.")
    if np.any(B < 1) or np.any(B > q):
        raise ValueError("B categories must be in {1..q}.")

    if L < 2:
        raise ValueError("L (number of breakpoints per criterion) must be >= 2.")

    return X, ar_idx, B


def _compute_b0_bq(v_values: np.ndarray, epsilon: float) -> Tuple[float, float]:
    """Compute b0 and bq after solving, per Remark 2 (paper page 7)."""
    v_values = np.asarray(v_values, float)
    mins = np.min(v_values, axis=1)
    maxs = np.max(v_values, axis=1)
    b0 = float(np.sum(mins))
    bq = float(np.sum(maxs) + float(epsilon))
    return b0, bq


def _score_from_v(
    X: np.ndarray,
    breaks: List[np.ndarray],
    v_values: np.ndarray,
    scale_info: Dict[str, Any],
) -> np.ndarray:
    """Compute V(a_i)=sum_j v_j(x_ij) for rows of X."""
    X = np.asarray(X, float)
    Xs = scale_with_info(X, scale_info)
    v_values = np.asarray(v_values, float)

    n, m = Xs.shape
    L = v_values.shape[1]
    if v_values.shape[0] != m:
        raise ValueError("v_values shape mismatch with X.")

    scores = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(m):
            brk = breaks[j]
            vv = v_values[j]
            x = float(Xs[i, j])

            if x <= brk[0]:
                v = vv[0]
            elif x >= brk[-1]:
                v = vv[-1]
            else:
                k = np.searchsorted(brk, x) - 1
                k = max(0, min(k, L - 2))
                lam = (x - brk[k]) / (brk[k + 1] - brk[k])
                v = (1.0 - lam) * vv[k] + lam * vv[k + 1]
            s += float(v)
        scores[i] = float(s)
    return scores


def predict_categories(
    X: np.ndarray,
    fit: Dict[str, Any] | ZhangFitResult,
) -> np.ndarray:
    """Assign categories using the threshold-based value-driven sorting procedure.

    Uses b0 and bq (computed post-fit) and thresholds b1..b_{q-1}.
    Rule (paper Eq. 3): if b_{h-1} <= V(a) < b_h => assign C_h.

    Returns
    -------
    cats : np.ndarray of shape (n,), values in {1..q}
    """
    d = fit.to_dict() if isinstance(fit, ZhangFitResult) else fit

    breaks = d["breaks"]
    v_values = np.asarray(d["v_values"], float)
    thresholds = np.asarray(d["thresholds"], float)  # length q-1
    q = int(d["q"])
    b0 = float(d["b0"])
    bq = float(d["bq"])
    scale_info = d["scale_info"]

    V = _score_from_v(X, breaks, v_values, scale_info)

    # Full threshold vector: b0, b1..b_{q-1}, bq
    b_full = np.concatenate([[b0], thresholds, [bq]]).astype(float)

    cats = np.empty(V.shape[0], dtype=int)
    for i, val in enumerate(V):
        # Find largest h such that val >= b_{h-1} and val < b_h.
        # Use searchsorted on upper bounds.
        h = int(np.searchsorted(b_full[1:], val, side="right") + 1)
        h = max(1, min(h, q))
        cats[i] = h
    return cats


def make_scoring_fn(
    fit: Dict[str, Any] | ZhangFitResult,
):
    """Return a callable scoring function V(X) for a fitted model."""
    d = fit.to_dict() if isinstance(fit, ZhangFitResult) else fit
    breaks = d["breaks"]
    v_values = np.asarray(d["v_values"], float)
    scale_info = d["scale_info"]

    def V_fn(X: np.ndarray) -> np.ndarray:
        return _score_from_v(X, breaks, v_values, scale_info)

    return V_fn


# ============================================================
# Model builders
# ============================================================

def _add_EBound(model: Model, v: List[List[Any]]) -> None:
    """E_Bound (Eq. 14): 0 <= v_j(beta_lj) <= 1."""
    m = len(v)
    for j in range(m):
        for l, var in enumerate(v[j]):
            model.addConstr(var >= 0.0, name=f"bound_lb_{j}_{l}")
            model.addConstr(var <= 1.0, name=f"bound_ub_{j}_{l}")

def _add_ESort(
    model: Model,
    Xs: np.ndarray,
    breaks: List[np.ndarray],
    v: List[List[Any]],
    b: List[Any],
    epsilon: Any,
    ar_idx: np.ndarray,
    q: int,
) -> Tuple[Dict[int, Any], Dict[Tuple[int, int], Any]]:
    """E_Sort (Eq. 13): additive V for reference alternatives + threshold spacing.

    Returns
    -------
    V_vars : dict i -> Var for V(a_i) for i in ar_idx
    v_interp_vars : dict (i,j) -> Var for v_j(x_ij) for i in ar_idx

    Notes
    -----
    We create explicit vars for V(a_i) and v_j(x_ij) so constraints are transparent.
    v_j(x_ij) is enforced by a standard piecewise-linear interpolation using interp_expr.
    """
    from gurobipy import GRB, LinExpr  # lazy import
    m = Xs.shape[1]
    V_vars: Dict[int, Any] = {}
    v_interp_vars: Dict[Tuple[int, int], Any] = {}

    # For each reference alternative, enforce V(ai)=sum_j v_j(xij)
    for i in ar_idx.tolist():
        Vi = model.addVar(lb=-GRB.INFINITY, name=f"V_{i}")
        V_vars[i] = Vi

        expr = LinExpr()
        for j in range(m):
            # Create a helper var for marginal value at x_ij
            vij = model.addVar(lb=-GRB.INFINITY, name=f"v_{i}_{j}")
            v_interp_vars[(i, j)] = vij

            # Enforce vij == interpolation(v_j(beta), x_ij)
            # interp_expr returns LinExpr in terms of v[j][l] variables.
            model.addConstr(
                vij == interp_expr(model, float(Xs[i, j]), breaks[j], v[j]),
                name=f"interp_{i}_{j}",
            )
            expr += vij

        model.addConstr(Vi == expr, name=f"Vsum_{i}")

    # Threshold spacing constraints: b_h - b_{h-1} >= epsilon for h=2..q-1
    # Here b list is length (q-1): b[0]=b1, b[q-2]=b_{q-1}
    if q >= 4:
        for h in range(2, q):  # h=2..q-1
            model.addConstr(b[h - 1 - 1] <= b[h - 1], name=f"b_order_{h}")
            model.addConstr(b[h - 1] - b[h - 2] >= epsilon, name=f"b_gap_{h}")
    elif q == 3:
        # No (h=2..q-1) constraints since only b1,b2? Actually q=3 => b1,b2? Wait q-1=2 => h=2..2, yes one.
        model.addConstr(b[1] - b[0] >= epsilon, name="b_gap_2")
        model.addConstr(b[0] <= b[1], name="b_order_2")
    else:
        # q=2 => no internal spacing constraints
        pass

    return V_vars, v_interp_vars


def _add_EAR(
    model: Model,
    V_vars: Dict[int, Any],
    b: List[Any],
    epsilon: Any,
    ar_idx: np.ndarray,
    B: np.ndarray,
    q: int,
) -> None:
    """E_AR (Eq. 12): assignment examples constraints."""
    # b list is length (q-1) : b1..b_{q-1}
    for idx_pos, i in enumerate(ar_idx.tolist()):
        Bi = int(B[idx_pos])
        Vi = V_vars[i]

        if Bi == 1:
            # V(ai) <= b1 - epsilon
            model.addConstr(Vi <= b[0] - epsilon, name=f"AR_up_{i}")
        elif Bi == q:
            # V(ai) >= b_{q-1}
            model.addConstr(Vi >= b[q - 2], name=f"AR_low_{i}")
        else:
            # b_{Bi-1} <= V(ai) <= b_{Bi} - epsilon
            model.addConstr(Vi >= b[Bi - 2], name=f"AR_low_{i}")
            model.addConstr(Vi <= b[Bi - 1] - epsilon, name=f"AR_up_{i}")


def _add_ESlope(
    model: Model,
    breaks: List[np.ndarray],
    v: List[List[Any]],
    gamma: List[List[Any]],
) -> None:
    """E_Slope (Prop. 4): linearize gamma_{l,j} = | slope_left - slope_right |.

    In paper notation:
      gamma_{l,j} = | (v(beta_l)-v(beta_{l-1}))/(beta_l-beta_{l-1}) -
                    (v(beta_{l+1})-v(beta_l))/(beta_{l+1}-beta_l) |
      for l = 2..s_j and j in M.

    Here:
      - breaks[j] is beta_{1..L}
      - v[j][t] is v(beta_{t+1}) (0-indexed)
      - gamma[j][k] corresponds to l = k+1 (since k=1..L-2)

    We add:
      gamma >= left-right
      gamma >= -(left-right)
    """
    m = len(v)
    for j in range(m):
        beta = np.asarray(breaks[j], float)
        L = len(beta)
        if L <= 2:
            continue

        for k in range(1, L - 1):  # k=1..L-2 (0-based internal points)
            left = (v[j][k] - v[j][k - 1]) / float(beta[k] - beta[k - 1])
            right = (v[j][k + 1] - v[j][k]) / float(beta[k + 1] - beta[k])
            diff = left - right

            g = gamma[j][k - 1]  # gamma index 0..L-3
            model.addConstr(g >= diff, name=f"slope_pos_{j}_{k}")
            model.addConstr(g >= -diff, name=f"slope_neg_{j}_{k}")


# ============================================================
# Public optimization routines
# ============================================================

def check_consistency(
    X: np.ndarray,
    ar_idx: np.ndarray,
    B: np.ndarray,
    q: int,
    L: int,
    scale: Optional[str] = "minmax",
    epsilon_fixed: float = 1e-3,
    gurobi_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Consistency check model (M-1).

    If optimal objective is 0, assignment examples are consistent.

    Notes
    -----
    The paper uses epsilon inside the constraints; for a pure feasibility/consistency test,
    it can be fixed to an arbitrarily small positive number. We expose epsilon_fixed.
    """
    from gurobipy import GRB, Model  # lazy import
    t0 = time.perf_counter()

    X, ar_idx, B = _validate_inputs(X, ar_idx, B, q, L)
    Xs, scale_info = scale_matrix(X, scale)
    n, m = Xs.shape
    breaks = [build_breaks(Xs[:, j].min(), Xs[:, j].max(), L) for j in range(m)]

    model = Model("Zhang_Consistency_M1")
    if not verbose:
        model.setParam("OutputFlag", 0)
    if gurobi_params:
        for k, v_ in gurobi_params.items():
            model.setParam(k, v_)

    # Decision variables: v_j(beta_lj), thresholds b1..b_{q-1}, deltas per reference alt
    v_vars: List[List[Any]] = []
    for j in range(m):
        vj = [model.addVar(lb=0.0, ub=1.0, name=f"v_{j}_{l}") for l in range(L)]
        v_vars.append(vj)

    b_vars = [model.addVar(lb=-GRB.INFINITY, name=f"b_{h}") for h in range(1, q)]  # 1..q-1

    # E_Sort (V vars + interp) + spacing constraints (Eq. 13)
    eps = float(epsilon_fixed)
    eps_var = model.addVar(lb=eps, ub=eps, name="epsilon_fixed")
    V_vars, _ = _add_ESort(model, Xs, breaks, v_vars, b_vars, eps_var, ar_idx, q)

    # E_Bound (already in v_vars bounds) but keep explicitly for readability
    _add_EBound(model, v_vars)

    # Deltas
    delta_plus = {}
    delta_minus = {}
    for i in ar_idx.tolist():
        delta_plus[i] = model.addVar(lb=0.0, name=f"dplus_{i}")
        delta_minus[i] = model.addVar(lb=0.0, name=f"dminus_{i}")

    # (M-1) constraints (paper section 4.3.1)
    # b_{Bi-1} - d+ <= V <= b_{Bi} + d- - epsilon
    # with special cases Bi=1 or Bi=q
    for pos, i in enumerate(ar_idx.tolist()):
        Bi = int(B[pos])
        Vi = V_vars[i]

        if Bi in range(2, q):
            model.addConstr(b_vars[Bi - 2] - delta_plus[i] <= Vi, name=f"M1_low_{i}")
            model.addConstr(Vi <= b_vars[Bi - 1] + delta_minus[i] - eps_var, name=f"M1_up_{i}")
        elif Bi == 1:
            model.addConstr(Vi <= b_vars[0] + delta_minus[i] - eps_var, name=f"M1_up_{i}")
        else:  # Bi == q
            model.addConstr(b_vars[q - 2] - delta_plus[i] <= Vi, name=f"M1_low_{i}")

    model.setObjective(
        sum(delta_plus[i] + delta_minus[i] for i in ar_idx.tolist()),
        GRB.MINIMIZE,
    )

    model.optimize()
    status = model.Status

    t1 = time.perf_counter()
    obj = float(model.ObjVal) if status == GRB.OPTIMAL else None

    return {
        "status": int(status),
        "obj": obj,
        "time_sec": float(t1 - t0),
        "is_consistent": bool(obj is not None and abs(obj) <= 1e-8),
        "scale_info": scale_info,
    }


def adjust_preferences_minimum(
    X: np.ndarray,
    ar_idx: np.ndarray,
    B: np.ndarray,
    q: int,
    L: int,
    scale: Optional[str] = "minmax",
    epsilon_fixed: float = 1e-3,
    big_m: float = 10.0,
    gurobi_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Minimum adjustment optimization model (M-2), using Prop. 3 (Eq. 17).

    Output is adjusted category assignments B_star for the same ar_idx.

    Notes
    -----
    This is the 'repair' step if check_consistency reports inconsistency.
    The objective (15) is min sum |Bi - Bi'|. We linearize absolute values with
    auxiliary variables d_i >= Bi - Bi', d_i >= -(Bi - Bi').
    """
    from gurobipy import GRB, Model  # lazy import
    t0 = time.perf_counter()

    X, ar_idx, B = _validate_inputs(X, ar_idx, B, q, L)
    Xs, scale_info = scale_matrix(X, scale)
    n, m = Xs.shape
    breaks = [build_breaks(Xs[:, j].min(), Xs[:, j].max(), L) for j in range(m)]

    model = Model("Zhang_Adjust_M2")
    if not verbose:
        model.setParam("OutputFlag", 0)
    if gurobi_params:
        for k, v_ in gurobi_params.items():
            model.setParam(k, v_)

    # v and thresholds
    v_vars: List[List[Any]] = []
    for j in range(m):
        vj = [model.addVar(lb=0.0, ub=1.0, name=f"v_{j}_{l}") for l in range(L)]
        v_vars.append(vj)

    b_vars = [model.addVar(lb=-GRB.INFINITY, name=f"b_{h}") for h in range(1, q)]
    eps = float(epsilon_fixed)
    eps_var = model.addVar(lb=eps, ub=eps, name="epsilon_fixed")

    V_vars, _ = _add_ESort(model, Xs, breaks, v_vars, b_vars, eps_var, ar_idx, q)
    _add_EBound(model, v_vars)

    # Prop. 3: t_{h,i} binary, sum_h t_{h,i}=1, Bi' = sum h*t_{h,i}
    t_hi: Dict[Tuple[int, int], Any] = {}
    B_adj: Dict[int, Any] = {}
    for i in ar_idx.tolist():
        # binaries for each category
        for h in range(1, q + 1):
            t_hi[(h, i)] = model.addVar(vtype=GRB.BINARY, name=f"t_{h}_{i}")

        model.addConstr(sum(t_hi[(h, i)] for h in range(1, q + 1)) == 1, name=f"onecat_{i}")

        # adjusted category integer (continuous is fine as convex comb of ints)
        B_adj[i] = model.addVar(lb=1.0, ub=float(q), name=f"B_adj_{i}")
        model.addConstr(
            B_adj[i] == sum(float(h) * t_hi[(h, i)] for h in range(1, q + 1)),
            name=f"Bdef_{i}",
        )

        Vi = V_vars[i]

        # Big-M constraints (Eq. 17)
        for h in range(2, q + 1):
            model.addConstr(
                Vi >= b_vars[h - 2] + big_m * (t_hi[(h, i)] - 1.0),
                name=f"ARbig_low_{h}_{i}",
            )
        for h in range(1, q):
            model.addConstr(
                Vi <= b_vars[h - 1] - eps_var + big_m * (1.0 - t_hi[(h, i)]),
                name=f"ARbig_up_{h}_{i}",
            )

    # Objective: min sum |Bi - Bi'|
    d_abs: Dict[int, Any] = {}
    for pos, i in enumerate(ar_idx.tolist()):
        Bi = float(B[pos])
        d_abs[i] = model.addVar(lb=0.0, name=f"dabs_{i}")
        model.addConstr(d_abs[i] >= Bi - B_adj[i], name=f"abs_pos_{i}")
        model.addConstr(d_abs[i] >= -(Bi - B_adj[i]), name=f"abs_neg_{i}")

    model.setObjective(sum(d_abs[i] for i in ar_idx.tolist()), GRB.MINIMIZE)
    model.optimize()

    status = model.Status
    if status != GRB.OPTIMAL:
        warnings.warn(f"Gurobi ended with status={status} in M-2")

    B_star = np.array([int(round(float(B_adj[i].X))) for i in ar_idx.tolist()], dtype=int) if status == GRB.OPTIMAL else None

    t1 = time.perf_counter()
    return {
        "status": int(status),
        "obj": float(model.ObjVal) if status == GRB.OPTIMAL else None,
        "time_sec": float(t1 - t0),
        "B_star": B_star,
        "scale_info": scale_info,
    }


def fit_zhang_representative(
    X: np.ndarray,
    ar_idx: np.ndarray,
    B: np.ndarray,
    q: int,
    L: int,
    approach: int = 2,
    scale: Optional[str] = "minmax",
    epsilon_lb: float = 1e-3,
    epsilon_ub: float = 0.5,
    gurobi_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Learn the representative sorting model via Zhang et al. lexicographic approaches.

    Parameters
    ----------
    X : (n,m) array
    ar_idx : indices of reference alternatives A_R
    B : (|A_R|,) categories in {1..q} for each reference alternative
    q : number of ordered categories
    L : number of breakpoints per criterion (paper uses s_j subintervals => L=s_j+1)
    approach : 1 or 2
      - 1: prioritize complexity (min sum gamma) then maximize epsilon (M-4 then M-6)
      - 2: prioritize discriminative power (max epsilon) then minimize sum gamma (M-7 then M-8)

    Returns
    -------
    fit_dict : dict (same style as uta_core results)
    """
    from gurobipy import GRB, Model  # lazy import
    t0 = time.perf_counter()

    X, ar_idx, B = _validate_inputs(X, ar_idx, B, q, L)
    if approach not in (1, 2):
        raise ValueError("approach must be 1 or 2.")

    Xs, scale_info = scale_matrix(X, scale)
    n, m = Xs.shape
    breaks = [build_breaks(Xs[:, j].min(), Xs[:, j].max(), L) for j in range(m)]

    def _build_base_model(name: str) -> Tuple[Model, List[List[Any]], List[Any], Any, Dict[int, Any], List[List[Any]]]:
        """Build common vars + constraints (E_AR, E_Sort, E_Bound) plus gamma vars (no ESlope yet)."""
        mdl = Model(name)
        if not verbose:
            mdl.setParam("OutputFlag", 0)
        if gurobi_params:
            for k, v_ in gurobi_params.items():
                mdl.setParam(k, v_)

        # v_j(beta_lj)
        v_vars: List[List[Any]] = []
        for j in range(m):
            vj = [mdl.addVar(lb=0.0, ub=1.0, name=f"v_{j}_{l}") for l in range(L)]
            v_vars.append(vj)

        # thresholds b1..b_{q-1}
        b_vars = [mdl.addVar(lb=-GRB.INFINITY, name=f"b_{h}") for h in range(1, q)]

        # epsilon
        eps_var = mdl.addVar(lb=epsilon_lb, ub=epsilon_ub, name="epsilon")

        # E_Sort
        V_vars, _ = _add_ESort(mdl, Xs, breaks, v_vars, b_vars, eps_var, ar_idx, q)

        # E_AR
        _add_EAR(mdl, V_vars, b_vars, eps_var, ar_idx, B, q)

        # E_Bound (redundant with var bounds, but explicit)
        _add_EBound(mdl, v_vars)

        # gamma vars: for each criterion, L-2 internal points (if L>=3)
        gamma_vars: List[List[Any]] = []
        if L >= 3:
            for j in range(m):
                gj = [mdl.addVar(lb=0.0, name=f"gamma_{j}_{k}") for k in range(L - 2)]
                gamma_vars.append(gj)
        else:
            gamma_vars = [[ ]] * m  # unused

        return mdl, v_vars, b_vars, eps_var, V_vars, gamma_vars

    def _extract_solution(
        mdl: Model,
        v_vars: List[List[Any]],
        b_vars: List[Any],
        eps_var: Any,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        v_values = np.zeros((m, L), dtype=float)
        for j in range(m):
            for l in range(L):
                v_values[j, l] = float(v_vars[j][l].X)
        thresholds = np.array([float(bv.X) for bv in b_vars], dtype=float)
        eps_val = float(eps_var.X)
        return v_values, thresholds, eps_val

    # --------------------- Approach 1 ---------------------
    if approach == 1:
        # Stage 1: min sum gamma (M-4), with ESlope linearization
        mdl1, v1, b1, eps1, V1, g1 = _build_base_model("Zhang_A1_stage1_M4")
        if L >= 3:
            _add_ESlope(mdl1, breaks, v1, g1)
            gamma_sum1 = sum(g1[j][k] for j in range(m) for k in range(L - 2))
        else:
            gamma_sum1 = 0.0
        mdl1.setObjective(gamma_sum1, GRB.MINIMIZE)
        mdl1.optimize()

        status1 = mdl1.Status
        if status1 != GRB.OPTIMAL:
            warnings.warn(f"Approach 1 stage 1 ended with status={status1}")
            t1 = time.perf_counter()
            return {
                "status": int(status1),
                "obj": None,
                "time_sec": float(t1 - t0),
                "approach": 1,
            }

        gamma_star = float(mdl1.ObjVal)

        # Stage 2: max epsilon s.t. gamma_sum == gamma_star (M-6)
        mdl2, v2, b2, eps2, V2, g2 = _build_base_model("Zhang_A1_stage2_M6")
        if L >= 3:
            _add_ESlope(mdl2, breaks, v2, g2)
            gamma_sum2 = sum(g2[j][k] for j in range(m) for k in range(L - 2))
            mdl2.addConstr(gamma_sum2 == gamma_star, name="gamma_star")
        mdl2.setObjective(eps2, GRB.MAXIMIZE)
        mdl2.optimize()

        status = mdl2.Status
        obj = float(mdl2.ObjVal) if status == GRB.OPTIMAL else None
        if status != GRB.OPTIMAL:
            warnings.warn(f"Approach 1 stage 2 ended with status={status}")

        if status == GRB.OPTIMAL:
            v_values, thresholds, eps_val = _extract_solution(mdl2, v2, b2, eps2)
        else:
            v_values, thresholds, eps_val = np.empty((m, L)), np.empty((q - 1,)), float("nan")

    # --------------------- Approach 2 ---------------------
    else:
        # Stage 1: max epsilon (M-7)
        mdl1, v1, b1, eps1, V1, g1 = _build_base_model("Zhang_A2_stage1_M7")
        mdl1.setObjective(eps1, GRB.MAXIMIZE)
        mdl1.optimize()

        status1 = mdl1.Status
        if status1 != GRB.OPTIMAL:
            warnings.warn(f"Approach 2 stage 1 ended with status={status1}")
            t1 = time.perf_counter()
            return {
                "status": int(status1),
                "obj": None,
                "time_sec": float(t1 - t0),
                "approach": 2,
            }

        eps_star = float(eps1.X)

        # Stage 2: min sum gamma with epsilon fixed (M-8)
        mdl2, v2, b2, eps2, V2, g2 = _build_base_model("Zhang_A2_stage2_M8")
        mdl2.addConstr(eps2 == eps_star, name="epsilon_star")

        if L >= 3:
            _add_ESlope(mdl2, breaks, v2, g2)
            gamma_sum2 = sum(g2[j][k] for j in range(m) for k in range(L - 2))
        else:
            gamma_sum2 = 0.0

        mdl2.setObjective(gamma_sum2, GRB.MINIMIZE)
        mdl2.optimize()

        status = mdl2.Status
        obj = float(mdl2.ObjVal) if status == GRB.OPTIMAL else None
        if status != GRB.OPTIMAL:
            warnings.warn(f"Approach 2 stage 2 ended with status={status}")

        if status == GRB.OPTIMAL:
            v_values, thresholds, eps_val = _extract_solution(mdl2, v2, b2, eps2)
        else:
            v_values, thresholds, eps_val = np.empty((m, L)), np.empty((q - 1,)), float("nan")

    # Derived b0 and bq per paper
    b0, bq = _compute_b0_bq(v_values, eps_val) if status == GRB.OPTIMAL else (float("nan"), float("nan"))

    t2 = time.perf_counter()
    fit = ZhangFitResult(
        status=int(status),
        obj=obj,
        time_sec=float(t2 - t0),
        breaks=breaks,
        v_values=v_values,
        thresholds=thresholds,
        epsilon=float(eps_val),
        b0=float(b0),
        bq=float(bq),
        q=int(q),
        approach=int(approach),
        scale_info=scale_info,
    )

    # Also compute scores for all X (handy for debugging / plotting)
    scores = _score_from_v(X, breaks, v_values, scale_info) if status == GRB.OPTIMAL else None

    out = fit.to_dict()
    out["scores"] = scores
    out["method_used"] = f"zhang_approach_{approach}"
    return out
