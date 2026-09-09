"""
cohalign_bench.py -- empirical validation machinery for CohAlign.

Provides:
  - activity_preflight / pick_parameters: locate readout-visible (active)
    parameters via the noiseless parameter-shift gradient map, and select
    validation parameters spanning the alignment range
  - paired_degradation: common-random-numbers paired measurement of the
    relative squared-gradient degradation Delta_tilde = 1 - M2(gamma)/M2(0)
    with bootstrap confidence intervals (shared theta draws between the
    noiseless and noisy passes, and across channels)
  - omega_hat: empirical alignment ratio Delta_tilde / (2 gamma L lambda_coh)
  - ols_loglog: ordinary least squares on log-log design matrices with R^2,
    RMSE and coefficient table (used for the predictor-comparison phase)
"""
import numpy as np

# ------------------------------------------------------ activity preflight
def parameter_shift_grad(model, theta, beta, ell, q, channel=None,
                         gamma=0.0, shift=np.pi / 2):
    tp, tm = theta.copy(), theta.copy()
    tp[ell, q] += shift
    tm[ell, q] -= shift
    fp = model.output(model.evolve_dm(tp, beta, channel, gamma))
    fm = model.output(model.evolve_dm(tm, beta, channel, gamma))
    return 0.5 * (fp - fm)

def activity_preflight(model, beta, n_draw=4, delta_init=0.05, seed=11):
    """Mean squared noiseless gradient for every (ell, q). O(n L) circuit pairs
    per draw; identifies the backward light cone of the readout."""
    act = np.zeros((model.L, model.n))
    rng = np.random.default_rng(seed)
    for _ in range(n_draw):
        theta = rng.uniform(-delta_init, delta_init, size=(model.L, model.n))
        for ell in range(model.L):
            for q in range(model.n):
                g = parameter_shift_grad(model, theta, beta, ell, q)
                act[ell, q] += g * g
    return act / n_draw

def pick_parameters(act, k=2, floor_frac=1e-4, abs_floor=1e-20,
                    distinct_layers=True):
    """Top-k active parameters by mean squared gradient (descending).

    With distinct_layers=True (default) the selected parameters are drawn
    from k different layers, so validation covers genuinely distinct (not
    symmetry-equivalent) parameter locations.

    Raises if the whole map is numerically zero -- the signature of an input
    state that carries no trainable-phase response for this readout at this
    geometry (observed for the localised basis-state input at r = 2, L = 3;
    the inactivity is geometry-specific, not universal for r >= 2)."""
    if act.max() < abs_floor:
        raise RuntimeError(
            "activity preflight found no readout-visible parameters "
            f"(max activity {act.max():.2e}); the input state carries no "
            "trainable-phase response for this readout at this geometry -- "
            "for higher sectors consider Brickwork(..., init_state='spread')")
    flat = [(-act[ell, q], ell, q)
            for ell in range(act.shape[0]) for q in range(act.shape[1])
            if act[ell, q] > floor_frac * act.max()]
    flat.sort()
    if not distinct_layers:
        return [(ell, q) for (_, ell, q) in flat[:k]]
    out, used_layers = [], set()
    for (_, ell, q) in flat:
        if ell in used_layers:
            continue
        out.append((ell, q)); used_layers.add(ell)
        if len(out) == k:
            break
    # fall back to plain top-k if fewer than k layers are active
    if len(out) < k:
        out = [(ell, q) for (_, ell, q) in flat[:k]]
    return out

# ------------------------------------------------- CRN paired degradation
def paired_degradation(model, beta, ell, q, channel, gamma,
                       n_theta=8, delta_init=0.05, seed=1234,
                       shift=np.pi / 2, bootstrap_B=200, boot_seed=7,
                       g0_floor=1e-12, return_draws=False,
                       theta_draws=None):
    """CRN paired estimate of Delta_tilde = 1 - M2(gamma)/M2(0).

    The SAME theta draws feed the noiseless and noisy passes (and, because
    the seed is caller-fixed, the same draws are shared across channels and
    gamma values).  M2 is the mean squared parameter-shift derivative over
    the retained draws; the bootstrap resamples draw indices.
    Returns dict with m2_zero, m2_noise, delta_tilde, ci_lo, ci_hi, n_used;
    with return_draws=True also g0_sq and gn_sq (per-draw squared gradients,
    aligned across gamma values by the shared CRN seed) for joint bootstraps.
    """
    rng = np.random.default_rng(seed)
    g0_sq, gn_sq = [], []
    if theta_draws is not None:
        n_theta = len(theta_draws)
    for j in range(n_theta):
        theta = (theta_draws[j] if theta_draws is not None
                 else rng.uniform(-delta_init, delta_init, size=(model.L, model.n)))
        g0 = parameter_shift_grad(model, theta, beta, ell, q, None, 0.0, shift)
        if g0 * g0 < g0_floor:
            continue
        gn = parameter_shift_grad(model, theta, beta, ell, q, channel, gamma, shift)
        g0_sq.append(g0 * g0)
        gn_sq.append(gn * gn)
    g0_sq, gn_sq = np.asarray(g0_sq), np.asarray(gn_sq)
    if len(g0_sq) == 0:
        return {"m2_zero": np.nan, "m2_noise": np.nan, "delta_tilde": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "n_used": 0}
    m2_0, m2_g = float(np.mean(g0_sq)), float(np.mean(gn_sq))
    delta = 1.0 - m2_g / m2_0
    brng = np.random.default_rng(boot_seed)
    stats = []
    for _ in range(bootstrap_B):
        idx = brng.integers(0, len(g0_sq), size=len(g0_sq))
        stats.append(1.0 - np.mean(gn_sq[idx]) / np.mean(g0_sq[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    out = {"m2_zero": m2_0, "m2_noise": m2_g, "delta_tilde": float(delta),
           "ci_lo": float(lo), "ci_hi": float(hi), "n_used": int(len(g0_sq))}
    if return_draws:
        out["g0_sq"] = g0_sq
        out["gn_sq"] = gn_sq
    return out

def omega_hat(delta_tilde, gamma, L, lambda_coh):
    """Empirical alignment ratio Delta_tilde / (2 gamma L lambda_coh)."""
    den = 2.0 * gamma * L * lambda_coh
    return float(delta_tilde / den) if den > 0 else float("nan")

# --------------------------------------------------------------- regression
def ols_loglog(y, X_cols, names):
    """OLS of y on [1, X_cols...]. Returns dict with R2, RMSE, coefficients."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float)
                                             for c in X_cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "RMSE": float(np.sqrt(ss_res / len(y))),
            "coef": dict(zip(["intercept"] + list(names), [float(v) for v in b])),
            "n": int(len(y))}
