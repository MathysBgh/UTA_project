import numpy as np
from gurobipy import Model, GRB
from typing import List, Optional, Literal, Tuple


def uta_rank(
    X: np.ndarray,
    order: Optional[np.ndarray] = None,      # if None → assume already best→worst = [0,1,2,...]
    L: int | List[int] = 4,
    scale: Literal["minmax","minmax_robust"]="minmax",
    pairs: Literal["consecutive","all_pairs"]="consecutive",
    eps: float=1e-6,
    return_details: bool=False,
    output_flag: int=0
):
    """
    UTA inference with all criteria treated as benefit (monotone increasing).
    X: (n,m) raw (not necessarily scaled).
    order: permutation best→worst. If None → assume X is already sorted → order=np.arange(n).
    L: segments per criterion (int or list of length m).
    """

    X = np.asarray(X, float)
    n, m = X.shape

    # ----- If order not provided → assume X already sorted -----
    if order is None:
        order = np.arange(n)
    else:
        order = np.asarray(order, int)
        assert set(order.tolist())==set(range(n)), "order must be permutation of 0..n-1"

    # ----- Standardisation → [0,100] -----
    if scale=="minmax":
        lo = np.nanmin(X, axis=0)
        hi = np.nanmax(X, axis=0)
    else:
        lo = np.quantile(X, 0.01, axis=0)
        hi = np.quantile(X, 0.99, axis=0)

    span = np.where(hi>lo, hi-lo, 1.)
    Z = 100. * np.clip((X - lo)/span, 0., 1.)

    # ----- Breakpoints -----
    if isinstance(L, int):
        L_list = [L]*m
    else:
        assert len(L)==m
        L_list = list(L)

    z_breaks = [np.linspace(0,100,L_list[i]+1) for i in range(m)]

    # Precompute (k, λ)
    def locate_segment_and_lambda(z_value, z_points):
        if z_value<=z_points[0]: return 1,0.
        if z_value>=z_points[-1]: return len(z_points)-1,1.
        k = np.searchsorted(z_points, z_value, side="right")
        k = min(max(1,k),len(z_points)-1)
        lam = (z_value - z_points[k-1])/(z_points[k]-z_points[k-1])
        return k, lam

    seg_lambda = [[locate_segment_and_lambda(Z[j,i], z_breaks[i]) for j in range(n)] for i in range(m)]

    # Ranking pairs
    if pairs=="consecutive":
        pairs_idx = [(order[t], order[t+1]) for t in range(len(order)-1)]
    else: # all pairs
        pairs_idx = [(order[a], order[b]) for a in range(len(order)) for b in range(a+1,len(order))]

    # ----- LP -----
    model = Model("UTA_inference")
    model.Params.OutputFlag = output_flag

    u = [[model.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,name=f"u_{i}_{k}")
          for k in range(L_list[i]+1)] for i in range(m)]

    for i in range(m):
        model.addConstr(u[i][0] == 0.)
    model.addConstr(sum(u[i][-1] for i in range(m)) == 1.0)

    for i in range(m):
        for k in range(1,L_list[i]+1):
            model.addConstr(u[i][k] >= u[i][k-1])

    sigma_p = [model.addVar(lb=0,vtype=GRB.CONTINUOUS) for _ in range(n)]
    sigma_m = [model.addVar(lb=0,vtype=GRB.CONTINUOUS) for _ in range(n)]

    s_hat = []
    for j in range(n):
        expr = 0.
        for i in range(m):
            k,lam = seg_lambda[i][j]
            expr += (1-lam)*u[i][k-1] + lam*u[i][k]
        expr = expr - sigma_p[j] + sigma_m[j]
        s_hat.append(expr)

    for (a,b) in pairs_idx:
        model.addConstr(s_hat[a] >= s_hat[b] + eps)

    model.setObjective(sum(sigma_p)+sum(sigma_m), GRB.MINIMIZE)
    model.optimize()

    # ----- Read solution -----
    u_breaks = [np.array([u[i][k].X for k in range(L_list[i]+1)]) for i in range(m)]
    weights = np.array([ub[-1] for ub in u_breaks])

    def pwl_eval(z_points, u_points, z):
        return np.interp(np.clip(z,z_points[0],z_points[-1]), z_points, u_points)

    scores = np.array([
        sum(pwl_eval(z_breaks[i], u_breaks[i], Z[j,i]) for i in range(m))
        for j in range(n)
    ])
    ranking = np.argsort(-scores)

    if not return_details:
        return ranking

    return {
        "ranking": ranking,
        "scores": scores,
        "weights": weights,
        "z_breaks": z_breaks,
        "u_breaks": u_breaks,
        "obj_slack": float(model.ObjVal),
        "status": model.Status
    }
