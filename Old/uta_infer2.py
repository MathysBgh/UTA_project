
"""
uta_pipeline.py — UTA (UTilités Additives) factorized pipeline.

Objectif
--------
Factoriser l'inférence de fonctions d'utilité partielles (piecewise linear) et des poids
dans un modèle additif de type UTA à partir d'alternatives *déjà classées* (meilleur → pire).
L'optimisation est un LP (minimisation de slack sur les contraintes d'ordre).
Aucune dépendance obligatoire à Gurobi : backend par défaut = SciPy linprog.
Si Gurobi est dispo, vous pouvez passer backend="gurobi" pour accélérer.

Modèle
------
Score(a) = sum_i w_i * u_i(x_{ai})
- u_i : fonction d'utilité croissante, linéaire par morceaux sur L breakpoints par critère i
- w_i ≥ 0 et sum_i w_i = 1

Pour linéariser, on travaille avec m_i[k] = w_i * u_i(b_i[k]) (valeur pondérée aux nœuds).
On fixe m_i[0] = 0 (identifiabilité) et impose m_i[k+1] ≥ m_i[k].
On normalise via sum_i m_i[L-1] = 1 (équivaut à sum_i w_i = 1 puisque u_i(L-1)=1).

Le score d'une alternative a se calcule comme combinaison linéaire des m_i[k] avec des
coefficients α_i,a[k] (interpolation barycentrique) pré-calculés à partir des breakpoints.

Contrôles
---------
- Contraintes d'ordre consécutives : S[a_j] ≥ S[a_{j+1}] - ξ_j  (margin facultatif)
- Objectif : min sum_j ξ_j  (ξ_j ≥ 0)
- Monotonicité : m_i[k+1] - m_i[k] ≥ 0
- Identifiabilité : m_i[0] = 0  ∀i
- Normalisation : sum_i m_i[L-1] = 1

API rapide
----------
from uta_pipeline import UTAPipeline

pipe = UTAPipeline(num_breakpoints=4, scale="minmax", pairs="consecutive", backend="auto")
pipe.fit(X, ordered=True)     # X trié du meilleur au pire
scores = pipe.transform(X)    # re-calcul des scores
details = pipe.get_details()  # poids, utilités, breakpoints, slacks, etc.

# Prédire un nouvel ensemble :
scores_new = pipe.transform(X_new)

Auteur : ChatGPT (2025-10-24)
Licence : MIT
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional, Dict, Any, List, Tuple

# Backends optionnels
_BACKEND_SCIPY_OK = False
try:
    from scipy.optimize import linprog
    _BACKEND_SCIPY_OK = True
except Exception:
    _BACKEND_SCIPY_OK = False

_BACKEND_GUROBI_OK = False
try:
    import gurobipy as gp  # type: ignore
    from gurobipy import GRB  # type: ignore
    _BACKEND_GUROBI_OK = True
except Exception:
    _BACKEND_GUROBI_OK = False


def _minmax_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    span = np.where(xmax > xmin, xmax - xmin, 1.0)
    Xs = (X - xmin) / span
    return Xs, xmin, span


def _quantile_breakpoints(x: np.ndarray, L: int) -> np.ndarray:
    """Breakpoints réguliers en quantiles sur [0,1] (suppose x déjà ∈ [0,1])."""
    if L < 2:
        raise ValueError("num_breakpoints L must be >= 2")
    # Utilise des quantiles empiriques pour couvrir la distribution
    qs = np.linspace(0.0, 1.0, L)
    b = np.quantile(x, qs)
    # Eviter les segments de longueur nulle (fusion légère vers croissante)
    for k in range(1, L):
        if b[k] <= b[k-1]:
            b[k] = min(1.0, b[k-1] + 1e-6)
    b[0] = 0.0
    b[-1] = 1.0
    return b


def _precompute_alphas(Xs: np.ndarray, B: List[np.ndarray]) -> List[np.ndarray]:
    """
    Pour chaque critère i, pré-calculer α_i,a[k] pour toutes les alternatives a et breakpoints k.
    Retourne une liste de matrices [n x L] par critère.
    """
    n, d = Xs.shape
    L = B[0].shape[0]
    alphas: List[np.ndarray] = []
    for i in range(d):
        bi = B[i]
        Ai = np.zeros((n, L), dtype=float)
        for a in range(n):
            x = Xs[a, i]
            # Trouver le segment s tel que bi[s] <= x <= bi[s+1]
            s = np.searchsorted(bi, x, side="right") - 1
            s = max(0, min(s, L - 2))
            left, right = bi[s], bi[s+1]
            if right <= left + 1e-12:
                lam = 0.0
            else:
                lam = (x - left) / (right - left)
            Ai[a, s] += (1.0 - lam)
            Ai[a, s + 1] += lam
        alphas.append(Ai)
    return alphas


@dataclass
class UTAResult:
    scores: np.ndarray          # [n]
    ranking: np.ndarray         # indices best→worst
    weights: np.ndarray         # [d]
    utilities: List[np.ndarray] # d listes de [L] (u_i aux breakpoints)
    breakpoints: List[np.ndarray]  # d listes de [L] (sur l'échelle *standardisée* 0..1)
    m_vars: List[np.ndarray]    # d listes de [L] (m_i[k] = w_i * u_i(b_i[k]))
    slacks: np.ndarray          # [n-1] (si pairs="consecutive")
    scale_params: Dict[str, np.ndarray]  # xmin, span pour inversions éventuelles


class UTAPipeline:
    def __init__(
        self,
        num_breakpoints: int = 4,
        scale: Literal["minmax", "none"] = "minmax",
        pairs: Literal["consecutive"] = "consecutive",
        backend: Literal["auto", "scipy", "gurobi"] = "auto",
        margin: float = 0.0,
    ) -> None:
        self.L = int(num_breakpoints)
        if self.L < 2:
            raise ValueError("num_breakpoints must be >= 2")
        self.scale = scale
        self.pairs = pairs
        self.margin = float(margin)
        # backend resolution
        if backend == "auto":
            if _BACKEND_GUROBI_OK:
                self.backend = "gurobi"
            elif _BACKEND_SCIPY_OK:
                self.backend = "scipy"
            else:
                raise RuntimeError("No LP backend available (install SciPy or Gurobi).")
        else:
            self.backend = backend
            if backend == "gurobi" and not _BACKEND_GUROBI_OK:
                raise RuntimeError("Gurobi backend requested but not available.")
            if backend == "scipy" and not _BACKEND_SCIPY_OK:
                raise RuntimeError("SciPy backend requested but not available.")
        # Fitted artifacts
        self._fitted: bool = False
        self._B: List[np.ndarray] = []
        self._alphas: List[np.ndarray] = []
        self._m: List[np.ndarray] = []
        self._weights: Optional[np.ndarray] = None
        self._utilities: List[np.ndarray] = []
        self._scores: Optional[np.ndarray] = None
        self._slacks: Optional[np.ndarray] = None
        self._scale_params: Dict[str, np.ndarray] = {}

    # -------------------------- FIT ---------------------------------
    def fit(self, X: np.ndarray, ordered: bool = True) -> "UTAPipeline":
        """
        X : [n x d] alternatives. Si ordered=True, X est trié (meilleur → pire).
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if self.scale == "minmax":
            Xs, xmin, span = _minmax_scale(X)
            self._scale_params = {"xmin": xmin, "span": span}
        else:
            Xs = X.copy()
            self._scale_params = {"xmin": np.zeros(d), "span": np.ones(d)}

        # Breakpoints par critère (quantiles sur X standardisé)
        B: List[np.ndarray] = []
        for i in range(d):
            bi = _quantile_breakpoints(Xs[:, i], self.L)
            B.append(bi)

        # Coefficients d'interpolation α
        alphas = _precompute_alphas(Xs, B)  # liste de d matrices [n x L]

        # LP : variables = concat_i m_i[k] (d*L) + slacks ξ_j (n-1)
        num_m = d * self.L
        num_slack = (n - 1) if ordered else 0
        var_dim = num_m + num_slack

        # Helper pour indexer m_i[k]
        def idx_m(i: int, k: int) -> int:
            return i * self.L + k

        # Construction A_ub x ≤ b_ub, A_eq x = b_eq, bounds
        A_ub = []
        b_ub = []

        # Contraintes d'ordre consécutives : S[j] - S[j+1] ≥ margin - ξ_j
        # <=> -S[j] + S[j+1] + ξ_j ≤ -margin
        # Où S[j] = sum_i sum_k α[i][j,k] * m_i[k]
        if ordered and n >= 2:
            for j in range(n - 1):
                row = np.zeros(var_dim)
                # -S[j]
                for i in range(d):
                    row[idx_m(i, 0):idx_m(i, self.L)] -= alphas[i][j, :]
                # +S[j+1]
                for i in range(d):
                    row[idx_m(i, 0):idx_m(i, self.L)] += alphas[i][j + 1, :]
                # +ξ_j
                row[num_m + j] = 1.0
                A_ub.append(row)
                b_ub.append(-self.margin)

            # bornes ξ_j ≥ 0
            bounds_slack = [(0.0, None) for _ in range(num_slack)]
        else:
            bounds_slack = []

        # Monotonicité m_i[k+1] - m_i[k] ≥ 0  <=> -(m_i[k+1]-m_i[k]) ≤ 0
        for i in range(d):
            for k in range(self.L - 1):
                row = np.zeros(var_dim)
                row[idx_m(i, k + 1)] -= 1.0
                row[idx_m(i, k)] += 1.0
                A_ub.append(row)
                b_ub.append(0.0)

        # Egalités : m_i[0] = 0  ∀i
        A_eq = []
        b_eq = []
        for i in range(d):
            row = np.zeros(var_dim)
            row[idx_m(i, 0)] = 1.0
            A_eq.append(row)
            b_eq.append(0.0)

        # Normalisation : sum_i m_i[L-1] = 1
        row = np.zeros(var_dim)
        for i in range(d):
            row[idx_m(i, self.L - 1)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

        # Bornes m_i[k] : non borné sup, mais on sait m_i[k] ≥ 0 par m_i[0]=0 et monotonicité
        bounds_m = [(0.0, None) for _ in range(num_m)]
        bounds = bounds_m + bounds_slack

        # Objectif : min sum ξ_j
        c = np.zeros(var_dim)
        if num_slack > 0:
            c[num_m:] = 1.0

        # Résolution
        if self.backend == "scipy":
            res = linprog(
                c=c,
                A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                A_eq=np.array(A_eq) if A_eq else None,
                b_eq=np.array(b_eq) if b_eq else None,
                bounds=bounds,
                method="highs",
            )
            if not res.success:
                raise RuntimeError(f"LP infeasible/failed: {res.message}")
            x = res.x
        elif self.backend == "gurobi":
            m = gp.Model("uta")
            m.Params.OutputFlag = 0
            # variables
            vars_m = m.addVars(num_m, lb=0.0, name="m")
            vars_s = m.addVars(num_slack, lb=0.0, name="xi")
            # monotonicité
            for i in range(d):
                for k in range(self.L - 1):
                    m.addConstr(vars_m[idx_m(i, k + 1)] - vars_m[idx_m(i, k)] >= 0.0)
            # m_i[0]=0
            for i in range(d):
                m.addConstr(vars_m[idx_m(i, 0)] == 0.0)
            # normalisation
            m.addConstr(gp.quicksum(vars_m[idx_m(i, self.L - 1)] for i in range(d)) == 1.0)
            # ordre
            if ordered and n >= 2:
                for j in range(n - 1):
                    expr = 0.0
                    for i in range(d):
                        a_j = alphas[i][j, :]
                        a_j1 = alphas[i][j + 1, :]
                        for k in range(self.L):
                            expr += (a_j1[k] - a_j[k]) * vars_m[idx_m(i, k)]
                    expr += vars_s[j]
                    m.addConstr(expr >= self.margin)
            # objectif
            obj = gp.quicksum(vars_s[j] for j in range(num_slack))
            m.setObjective(obj, GRB.MINIMIZE)
            m.optimize()
            if m.status != GRB.OPTIMAL:
                raise RuntimeError("Gurobi did not find an optimal solution.")
            x = np.zeros(var_dim)
            for t in range(num_m):
                x[t] = vars_m[t].X
            for j in range(num_slack):
                x[num_m + j] = vars_s[j].X
        else:
            raise RuntimeError("Unknown backend")

        # Dépaqueter solution
        m_vars = []
        for i in range(d):
            m_i = x[idx_m(i, 0):idx_m(i, self.L)]
            m_vars.append(m_i.copy())
        slacks = x[num_m:] if num_slack > 0 else np.zeros(0)

        # Poids : w_i = m_i[L-1] - m_i[0] = m_i[L-1]
        weights = np.array([m_vars[i][-1] - m_vars[i][0] for i in range(d)])
        # Utilités aux breakpoints : u_i[k] = m_i[k] / w_i (si w_i>0) sinon 0
        utilities = []
        for i in range(d):
            wi = weights[i]
            if wi > 1e-12:
                ui = m_vars[i] / wi
            else:
                ui = np.zeros_like(m_vars[i])
            # Clamp [0,1] pour propreté numérique
            ui = np.clip(ui, 0.0, 1.0)
            utilities.append(ui)

        # Scores des n alternatives apprises
        S = self._compute_scores_from_m(alphas, m_vars)

        # Sauvegarde pour transform()
        self._B = B
        self._alphas = alphas
        self._m = m_vars
        self._weights = weights
        self._utilities = utilities
        self._scores = S
        self._slacks = slacks
        self._fitted = True
        return self

    # ------------------------ TRANSFORM ------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Calcule les scores d'un nouvel ensemble d'alternatives via les m_i[k] appris."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        if d != len(self._B):
            raise ValueError(f"X has {d} criteria but model has {len(self._B)}")
        # Standardise comme à l'entraînement
        if self.scale == "minmax":
            xmin = self._scale_params["xmin"]
            span = self._scale_params["span"]
            Xs = (X - xmin) / np.where(span > 0, span, 1.0)
        else:
            Xs = X.copy()
        # Clip dans [0,1] pour stabilité
        Xs = np.clip(Xs, 0.0, 1.0)
        # Recalcule des α pour ce X
        alphas_new = _precompute_alphas(Xs, self._B)
        scores = self._compute_scores_from_m(alphas_new, self._m)
        return scores

    def fit_transform(self, X: np.ndarray, ordered: bool = True) -> UTAResult:
        self.fit(X, ordered=ordered)
        ranking = np.argsort(-self._scores)  # best → worst
        return UTAResult(
            scores=self._scores.copy(),
            ranking=ranking,
            weights=self._weights.copy() if self._weights is not None else np.array([]),
            utilities=[u.copy() for u in self._utilities],
            breakpoints=[b.copy() for b in self._B],
            m_vars=[m.copy() for m in self._m],
            slacks=self._slacks.copy() if self._slacks is not None else np.array([]),
            scale_params={k: v.copy() for k, v in self._scale_params.items()},
        )

    # ------------------------- HELPERS -------------------------------
    @staticmethod
    def _compute_scores_from_m(alphas: List[np.ndarray], m_vars: List[np.ndarray]) -> np.ndarray:
        """S[a] = sum_i sum_k α_i,a[k] * m_i[k]."""
        n = alphas[0].shape[0]
        d = len(alphas)
        L = m_vars[0].shape[0]
        S = np.zeros(n, dtype=float)
        for a in range(n):
            acc = 0.0
            for i in range(d):
                Ai = alphas[i][a, :]   # [L]
                mi = m_vars[i]         # [L]
                acc += float(np.dot(Ai, mi))
            S[a] = acc
        return S

    def get_details(self) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return {
            "weights": self._weights.copy() if self._weights is not None else None,
            "utilities": [u.copy() for u in self._utilities],
            "breakpoints": [b.copy() for b in self._B],
            "m_vars": [m.copy() for m in self._m],
            "scores": self._scores.copy() if self._scores is not None else None,
            "slacks": self._slacks.copy() if self._slacks is not None else None,
            "scale_params": {k: v.copy() for k, v in self._scale_params.items()},
            "backend": self.backend,
            "num_breakpoints": self.L,
            "scale": self.scale,
        }


# ----------------------------- Facade --------------------------------

def uta_rank(
    X: np.ndarray,
    L: int = 4,
    ordered: bool = True,
    scale: Literal["minmax", "none"] = "minmax",
    pairs: Literal["consecutive"] = "consecutive",
    backend: Literal["auto", "scipy", "gurobi"] = "auto",
    margin: float = 0.0,
    return_details: bool = True,
) -> Dict[str, Any]:
    """
    Facade rapide : entraîne + renvoie scores/ranking (+ détails si souhaité).

    Parameters
    ----------
    X : ndarray shape [n, d]
        Alternatives. Si `ordered=True`, X est déjà trié (meilleur→pire).
    L : int
        Nombre de breakpoints (>=2) par critère.
    scale : {"minmax","none"}
        Standardisation préalable sur [0,1].
    pairs : (réservé pour extensions; aujourd'hui "consecutive" uniquement)
    backend : {"auto","scipy","gurobi"}
    margin : float
        Marge optionnelle sur les contraintes d'ordre.
    return_details : bool

    Returns
    -------
    dict : {
        "scores", "ranking", "weights", "utilities", "breakpoints",
        "m_vars", "slacks", "model"
    }
    """
    pipe = UTAPipeline(num_breakpoints=L, scale=scale, pairs=pairs, backend=backend, margin=margin)
    res = pipe.fit_transform(np.asarray(X, dtype=float), ordered=ordered)
    out = {
        "scores": res.scores,
        "ranking": res.ranking,
        "weights": res.weights,
        "utilities": res.utilities,
        "breakpoints": res.breakpoints,
        "m_vars": res.m_vars,
        "slacks": res.slacks,
        "model": pipe,  # pour réutiliser transform() ensuite
    }
    return out
