"""
cohalign_rates.py -- generator-level coherence-rate estimators (the CohAlign
contribution).

Implements three objects on the in-sector off-diagonal block:

1.  lambda_coh  (worst-case sector coherence rate)
        The largest eigenvalue of the Hermitian part of -L_r, where
        L_r = (S - I)/gamma_probe and S is the channel superoperator
        restricted to the off-diagonal pair basis.  This is the
        variational worst case  sup_A -Re<A, L_r A> / <A, A>  and is
        computed here for EVERY channel, structured channels included
        (no analytic labels).

2.  lambda_vis_op  (operator-level aligned rate; quadratic estimator)
        The ensemble-averaged Rayleigh quotient of -L_r on the gradient
        mode  G_i(theta) = Pi_off,r [Z_qi, O_B(ins; theta)], where
        O_B(ins) is the readout Heisenberg-evolved back to the Rz
        insertion point (through all later layers AND the same layer's
        XY block), and the slot-ell mode is the forward conjugation of
        G_i to noise slot ell.  State-independent; this is the literal
        computable form of the aligned-rate Rayleigh quotient.

3.  response_rates  (response-weighted, layer-resolved aligned rates;
        the sharp a-priori predictor)
        Exact first-order degradation rates of the readout derivative,
        one per noise slot:
          post-insertion slots:  r_ell = -Re Tr(O_B^(ell) L_ch(D^(ell))) / df0
          pre-insertion  slots:  r_ell = -Re Tr(F^(ell)  L_ch(rho_ell)) / df0
        where D^(ell) is the derivative-carrying operator propagated
        forward from the insertion, F^(ell) is the adjoint response
        functional propagated backward, rho_ell is the noiseless state
        at slot ell, and df0 = Tr(O_B^(ell) D^(ell)) is the noiseless
        derivative (slot-invariant; used as an internal frame check).
        First-order prediction:  Delta_tilde ~= 2 * gamma * sum_ell r_ell.

The predicted alignment ratio reported by the audit is
    omega_pred = sum_ell r_ell / (L * lambda_coh),
directly comparable to the empirical  omega_hat = Delta_tilde / (2 gamma L
lambda_coh)  extracted from paired degradation measurements.
"""
import time
import numpy as np
from cohalign_core import (pair_basis, embed_one_qubit, PAULI_Z)

# ------------------------------------------------- restricted superoperator
def restricted_offdiag_superoperator(channel, n: int, r: int,
                                     gamma_probe: float = 1e-3) -> tuple:
    """Full matrix S of the channel on the in-sector off-diagonal pair basis.

    S[(c,d),(a,b)] = (Phi_gamma(|a><b|))[c,d]; off-diagonal-in-pair-basis
    leakage is captured exactly.  Returns (S, pairs).
    """
    pairs = pair_basis(n, r)
    K = len(pairs)
    dim = 2**n
    S = np.zeros((K, K), dtype=complex)
    for col, (a, b) in enumerate(pairs):
        E = np.zeros((dim, dim), dtype=complex)
        E[a, b] = 1.0
        Eo = channel.apply(E, gamma_probe, n)
        for row, (c, d) in enumerate(pairs):
            S[row, col] = Eo[c, d]
    return S, pairs

def restricted_generator(channel, n: int, r: int,
                         gamma_probe: float = 1e-3) -> tuple:
    """L_r = (S - I)/gamma_probe on the pair basis. Returns (L_r, pairs)."""
    S, pairs = restricted_offdiag_superoperator(channel, n, r, gamma_probe)
    return (S - np.eye(len(pairs))) / gamma_probe, pairs

def lambda_coh_worst(L_r: np.ndarray) -> float:
    """sup_{A != 0} -Re<A, L_r A>/<A, A> = max eig of the Hermitian part of -L_r."""
    H = -0.5 * (L_r + L_r.conj().T)
    return float(np.linalg.eigvalsh(H)[-1])

def lambda_coh_typical(L_r: np.ndarray) -> float:
    """Mean of the Hermitian-part spectrum (average contraction rate)."""
    H = -0.5 * (L_r + L_r.conj().T)
    return float(np.mean(np.linalg.eigvalsh(H)))

def rayleigh_rate(L_r: np.ndarray, g: np.ndarray) -> float:
    """-Re<g, L_r g> / <g, g> for a pair-basis vector g."""
    den = float(np.real(np.vdot(g, g)))
    if den < 1e-28:
        return float("nan")
    return float(-np.real(np.vdot(g, L_r @ g)) / den)

def vec_offdiag(M: np.ndarray, pairs: list) -> np.ndarray:
    return np.array([M[a, b] for (a, b) in pairs], dtype=complex)

# --------------------------------------------------- gradient-mode geometry
def insertion_readout(model, theta: np.ndarray, beta: np.ndarray,
                      ell_i: int) -> np.ndarray:
    """Heisenberg readout at the Rz insertion point of layer ell_i:
    evolved back through all layers > ell_i and the XY block of layer ell_i."""
    U_back = model.xy_layer(beta[ell_i], ell_i)
    for ell in range(ell_i + 1, model.L):
        U_back = model.layer_unitary(theta, beta, ell) @ U_back
    return U_back.conj().T @ model.readout @ U_back

def slot_modes(model, theta: np.ndarray, beta: np.ndarray,
               ell_i: int, q_i: int) -> dict:
    """Gradient mode G = [Z_qi, O_B(ins)] forward-conjugated to each noise
    slot ell in [ell_i, L-1]. Returns {ell: mode matrix}."""
    Zq = embed_one_qubit(PAULI_Z, q_i, model.n)
    OB_ins = insertion_readout(model, theta, beta, ell_i)
    G = Zq @ OB_ins - OB_ins @ Zq
    W = model.xy_layer(beta[ell_i], ell_i)
    M = W @ G @ W.conj().T
    modes = {ell_i: M}
    for ell in range(ell_i + 1, model.L):
        Ul = model.layer_unitary(theta, beta, ell)
        M = Ul @ M @ Ul.conj().T
        modes[ell] = M
    return modes

# ------------------------------------- estimator 1: quadratic (Eq.-7 literal)
def lambda_mode(model, beta: np.ndarray, channel, ell_i: int, q_i: int,
                  n_theta: int = 20, delta_init: float = 0.05,
                  seed: int = 1, gamma_probe: float = 1e-3,
                  L_r=None, pairs=None, theta_draws=None) -> dict:
    """Operator mode contraction diagnostic. Ensemble-averaged Rayleigh
    quotient of -L_r on the slot modes of the derivative carrying operator.
    This is a state-independent operator diagnostic, not a visibility
    measure, and it does not by itself predict the measured response.

    Returns {"lambda_vis_op", "per_slot", "lambda_coh"}; per_slot maps each
    noise slot ell >= ell_i to its mean quadratic rate.
    """
    if L_r is None or pairs is None:
        L_r, pairs = restricted_generator(channel, model.n, model.r,
                                          gamma_probe)
    lam_coh = lambda_coh_worst(L_r)
    rng = np.random.default_rng(seed)
    per = {ell: [] for ell in range(ell_i, model.L)}
    if theta_draws is not None:
        n_theta = len(theta_draws)
    for j in range(n_theta):
        theta = (theta_draws[j] if theta_draws is not None
                 else rng.uniform(-delta_init, delta_init,
                                  size=(model.L, model.n)))
        for ell, M in slot_modes(model, theta, beta, ell_i, q_i).items():
            g = vec_offdiag(M, pairs)
            v = rayleigh_rate(L_r, g)
            if np.isfinite(v):
                per[ell].append(v)
    per_slot = {ell: (float(np.mean(v)) if v else float("nan"))
                for ell, v in per.items()}
    vals = [v for v in per_slot.values() if np.isfinite(v)]
    return {"lambda_vis_op": float(np.mean(vals)) if vals else float("nan"),
            "per_slot": per_slot, "lambda_coh": lam_coh}

# ------------------- estimator 2: response-weighted, layer-resolved (sharp)
def lambda_vis_op(*args, **kwargs):
    """Deprecated alias for lambda_mode, retained for backward
    compatibility with earlier releases."""
    import warnings
    warnings.warn("lambda_vis_op is deprecated; use lambda_mode",
                  DeprecationWarning, stacklevel=2)
    return lambda_mode(*args, **kwargs)

def response_rates(model, theta: np.ndarray, beta: np.ndarray, channel,
                   ell_i: int, q_i: int, gamma_probe: float = 1e-3,
                   frame_check: bool = False) -> dict:
    """Exact first-order aligned rates, one per noise slot (all L slots).

    Returns {"rates": {ell: r_ell}, "df0": noiseless derivative,
             "frame_defect": worst frame-invariance violation if checked}.
    Delta_tilde_pred = 2 * gamma * sum(rates.values()).
    """
    n, L = model.n, model.L
    Zq = embed_one_qubit(PAULI_Z, q_i, n)
    # noiseless states after each layer; state at the insertion point
    psi = model.initial_state()
    rho = np.outer(psi, psi.conj())
    rho_slot = []
    rho_pre_ins = None
    for ell in range(L):
        Uz = model.z_layer(theta[ell])
        Ux = model.xy_layer(beta[ell], ell)
        rho_z = Uz @ rho @ Uz.conj().T
        if ell == ell_i:
            rho_pre_ins = rho_z
        rho = Ux @ rho_z @ Ux.conj().T
        rho_slot.append(rho)
    # derivative-carrying operator, forward from the insertion
    C = -0.5j * (Zq @ rho_pre_ins - rho_pre_ins @ Zq)
    Ux_i = model.xy_layer(beta[ell_i], ell_i)
    D = Ux_i @ C @ Ux_i.conj().T
    D_slot = {ell_i: D}
    for ell in range(ell_i + 1, L):
        Ul = model.layer_unitary(theta, beta, ell)
        D = Ul @ D @ Ul.conj().T
        D_slot[ell] = D
    # Heisenberg readout at each slot
    OB_slot = {L - 1: model.readout}
    for ell in range(L - 2, -1, -1):
        Ul = model.layer_unitary(theta, beta, ell + 1)
        OB_slot[ell] = Ul.conj().T @ OB_slot[ell + 1] @ Ul
    df0 = float(np.real(np.trace(OB_slot[ell_i] @ D_slot[ell_i])))
    # adjoint response functional for pre-insertion slots
    OB_ins = Ux_i.conj().T @ OB_slot[ell_i] @ Ux_i
    F = -0.5j * (OB_ins @ Zq - Zq @ OB_ins)
    Uz_i = model.z_layer(theta[ell_i])
    F = Uz_i.conj().T @ F @ Uz_i
    F_slot = {}
    if ell_i >= 1:
        F_slot[ell_i - 1] = F
        for ell in range(ell_i - 2, -1, -1):
            Ul = model.layer_unitary(theta, beta, ell + 1)
            F_slot[ell] = Ul.conj().T @ F_slot[ell + 1] @ Ul
    frame_defect = 0.0
    if frame_check and abs(df0) > 1e-16:
        for ell in range(ell_i, L):
            v = float(np.real(np.trace(OB_slot[ell] @ D_slot[ell])))
            frame_defect = max(frame_defect, abs(v - df0) / abs(df0))
        for ell in F_slot:
            v = float(np.real(np.trace(F_slot[ell] @ rho_slot[ell])))
            frame_defect = max(frame_defect, abs(v - df0) / abs(df0))
    Lch = lambda X: (channel.apply(X, gamma_probe, n) - X) / gamma_probe
    # raw first-order response terms h_ell (no division by the derivative)
    h_slots = {}
    for ell in range(ell_i, L):
        h_slots[ell] = float(np.real(np.trace(OB_slot[ell] @ Lch(D_slot[ell]))))
    for ell in F_slot:
        h_slots[ell] = float(np.real(np.trace(F_slot[ell] @ Lch(rho_slot[ell]))))
    # per-draw ratios r_ell = -h_ell / g0 as an OPTIONAL diagnostic only
    if abs(df0) > 1e-14:
        rates = {ell: float(-h / df0) for ell, h in h_slots.items()}
    else:
        rates = {ell: float("nan") for ell in h_slots}
    return {"rates": rates, "h_slots": h_slots, "df0": df0,
            "frame_defect": frame_defect}

def response_terms(model, theta, beta, channel, ell_i, q_i,
                   gamma_probe=1e-3, frame_check=False):
    """Raw first-order response terms for one draw. Returns
    {"g0": noiseless derivative, "h_slots": {ell: h_ell}, "frame_defect"}.
    The cross-moment estimator aggregates these without ever dividing by
    an individual draw."""
    out = response_rates(model, theta, beta, channel, ell_i, q_i,
                         gamma_probe, frame_check)
    return {"g0": out["df0"], "h_slots": out["h_slots"],
            "frame_defect": out["frame_defect"]}

def lambda_vis_resp(model, beta: np.ndarray, channel, ell_i: int, q_i: int,
                    n_theta: int = 20, delta_init: float = 0.05,
                    seed: int = 1, gamma_probe: float = 1e-3,
                    bootstrap_B: int = 200, bootstrap_seed=None,
                    theta_draws=None) -> dict:
    """Ensemble response-weighted rates over the small-box prior.

    Aggregation is the derivative-squared-weighted mean,
        rate_ell = sum_draws df0^2 r_ell / sum_draws df0^2,
    which is the exact first-order object matched by the paired M2 ratio:
        M2(gamma)/M2(0) = <df0^2 (1 - 2 gamma sum_ell r_ell)> / <df0^2>
                        = 1 - 2 gamma sum_ell rate_ell + O(gamma^2).
    A plain mean of per-draw rate ratios is a mean-of-ratios and becomes
    unstable whenever df0 varies strongly across the prior (deep circuits);
    the weighted form is identical where df0 is stable and remains finite
    everywhere.  The effective sample size ESS = (sum w)^2 / sum w^2 and a
    weighted-bootstrap 95% CI on rate_sum quantify ensemble uncertainty.

    Returns {"rate_sum", "rate_sum_ci": (lo, hi), "per_slot", "n_used",
             "ess"}.
    """
    if theta_draws is not None:
        n_theta = len(theta_draws)
    rng = np.random.default_rng(seed)
    # Direct cross-moment accumulation in a SINGLE pass over the draws.
    # If theta_draws is supplied it is used verbatim, which lets a matched
    # finite-difference reference share EXACTLY the same parameter draws.
    # For each slot, N_ell = sum_j (-g0_j * h_ell_j) and D = sum_j g0_j^2,
    # with Lambda_ell = N_ell / D. No per-draw division is performed, every
    # sampled draw enters, and per-draw numerator and denominator arrays are
    # retained so the bootstrap is a direct ratio-of-sums resample with no
    # second execution of the response calculation.
    Lp = model.L
    num_draw = np.zeros((n_theta, Lp))
    den_draw = np.zeros(n_theta)
    k = 0
    for j in range(n_theta):
        theta = (theta_draws[j] if theta_draws is not None
                 else rng.uniform(-delta_init, delta_init,
                                  size=(Lp, model.n)))
        t = response_terms(model, theta, beta, channel, ell_i, q_i,
                           gamma_probe)
        g0 = t["g0"]
        if not np.isfinite(g0):
            continue
        for ell, h in t["h_slots"].items():
            num_draw[k, ell] = -g0 * h
        den_draw[k] = g0 * g0
        k += 1
    num_draw, den_draw = num_draw[:k], den_draw[:k]
    if k == 0 or den_draw.sum() <= 0.0:
        return {"rate_sum": float("nan"), "rate_sum_ci": (float("nan"),) * 2,
                "per_slot": {ell: float("nan") for ell in range(Lp)},
                "n_used": 0, "ess": 0.0,
                "bootstrap_B": int(bootstrap_B),
                "bootstrap_seed": int(bootstrap_seed
                                      if bootstrap_seed is not None
                                      else seed + 10_000),
                "num_draw": num_draw, "den_draw": den_draw}
    D = float(den_draw.sum())
    per_slot = {ell: float(num_draw[:, ell].sum() / D) for ell in range(Lp)}
    rate_sum = float(num_draw.sum() / D)
    ess = float(D ** 2 / np.sum(den_draw ** 2))  # weight concentration
    if bootstrap_seed is None:
        bootstrap_seed = seed + 10_000
    brng = np.random.default_rng(bootstrap_seed)
    tot = num_draw.sum(axis=1)
    stats = []
    for _ in range(bootstrap_B):
        idx = brng.integers(0, k, size=k)
        dd = den_draw[idx].sum()
        if dd > 0:
            stats.append(float(tot[idx].sum() / dd))
    lo, hi = (np.percentile(stats, [2.5, 97.5]) if stats
              else (float("nan"), float("nan")))
    return {"rate_sum": rate_sum, "rate_sum_ci": (float(lo), float(hi)),
            "per_slot": per_slot, "n_used": int(k), "ess": ess,
            "bootstrap_B": int(bootstrap_B),
            "bootstrap_seed": int(bootstrap_seed),
            "num_draw": num_draw, "den_draw": den_draw}

def prepare_channel_context(channel, n, r, gamma_probe=1e-3):
    """Build the restricted generator ONCE and derive everything that does
    not depend on the parameter. Returns a dict context consumed by
    audit_parameter, so auditing many parameters of one channel reuses the
    expensive construction. The normalisation status is classified from the
    relative Frobenius norms of the Hermitian and anti-Hermitian parts of
    the restricted generator rather than from an absolute rate threshold."""
    L_r, pairs = restricted_generator(channel, n, r, gamma_probe)
    H = -0.5 * (L_r + L_r.conj().T)
    A = 0.5 * (L_r - L_r.conj().T)
    nH = float(np.linalg.norm(H)); nA = float(np.linalg.norm(A))
    if nH >= 10.0 * nA or nA == 0.0:
        status = "contractive"
    elif nA >= 10.0 * nH:
        status = "coherent_dominated"
    else:
        status = "mixed"
    return {"channel": channel, "n": n, "r": r,
            "gamma_probe": gamma_probe, "L_r": L_r, "pairs": pairs,
            "lambda_coh_worst": lambda_coh_worst(L_r),
            "lambda_coh_typical": lambda_coh_typical(L_r),
            "herm_norm": nH, "antiherm_norm": nA,
            "normalisation_status": status}

def lambda_mode_from_context(ctx, model, beta, ell_i, q_i, n_theta=20,
                             delta_init=0.05, seed=1, theta_draws=None):
    """Operator mode contraction diagnostic computed from a prepared channel
    context; the restricted generator is never rebuilt here."""
    return lambda_mode(model, beta, ctx["channel"], ell_i, q_i,
                         n_theta=n_theta, delta_init=delta_init, seed=seed,
                         gamma_probe=ctx["gamma_probe"],
                         L_r=ctx["L_r"], pairs=ctx["pairs"],
                         theta_draws=theta_draws)

def audit_parameter(ctx, model, beta, ell_i, q_i, n_theta=20,
                    delta_init=0.05, seed=1, theta_draws=None,
                    bootstrap_B=200, bootstrap_seed=None):
    """Audit one parameter using a prepared channel context. The generator
    is not rebuilt. omega values are reported only when the context status
    is contractive; otherwise the signed susceptibility is the output."""
    t0 = time.time()
    channel = ctx["channel"]; gamma_probe = ctx["gamma_probe"]
    lam_w = ctx["lambda_coh_worst"]; lam_t = ctx["lambda_coh_typical"]
    op = lambda_mode_from_context(ctx, model, beta, ell_i, q_i,
                                  n_theta=n_theta, delta_init=delta_init,
                                  seed=seed, theta_draws=theta_draws)
    rp = lambda_vis_resp(model, beta, channel, ell_i, q_i,
                         n_theta=n_theta, delta_init=delta_init,
                         seed=seed, gamma_probe=gamma_probe,
                         bootstrap_B=bootstrap_B,
                         bootstrap_seed=bootstrap_seed,
                         theta_draws=theta_draws)
    contractive = ctx["normalisation_status"] == "contractive" and lam_w > 0
    omega_op = op["lambda_vis_op"] / lam_w if contractive else float("nan")
    omega_resp = (rp["rate_sum"] / (model.L * lam_w)
                  if contractive else float("nan"))
    ci = rp["rate_sum_ci"]
    omega_resp_ci = (tuple(c / (model.L * lam_w) for c in ci)
                     if contractive else (float("nan"),) * 2)
    out = {"channel": getattr(channel, "name", type(channel).__name__),
           "n": model.n, "L": model.L, "r": model.r,
           "ell_i": ell_i, "q_i": q_i,
           "lambda_coh_worst": lam_w, "lambda_coh_typical": lam_t,
           "normalisation_status": ctx["normalisation_status"],
           "lambda_mode": op["lambda_vis_op"], "omega_mode": omega_op,
           "lambda_response": rp["rate_sum"],
           "lambda_response_ci": rp["rate_sum_ci"],
           "omega_response": omega_resp, "omega_response_ci": omega_resp_ci,
           "per_slot_response": rp["per_slot"],
           "signed_susceptibility_only": not contractive,
           "ess": rp["ess"], "n_theta_used": rp["n_used"],
           "bootstrap_B": rp["bootstrap_B"],
           "bootstrap_seed": rp["bootstrap_seed"],
           "wall_seconds": time.time() - t0,
           # deprecated aliases retained for backward compatibility
           "lambda_vis_op": op["lambda_vis_op"], "omega_op": omega_op,
           "rate_sum_resp": rp["rate_sum"], "omega_resp": omega_resp,
           "omega_resp_ci": omega_resp_ci,
           "per_slot_resp": rp["per_slot"], "per_slot_op": op["per_slot"]}
    if not contractive:
        out["note"] = ("normalisation status is "
                       f"{ctx['normalisation_status']}; the audit reports "
                       "the signed response susceptibility and no "
                       "normalised alignment ratio")
    return out

def audit_parameters(ctx, model, beta, params, n_theta=20,
                     delta_init=0.05, seed=1):
    """Audit a list of (ell, q) parameters with one shared context."""
    return [audit_parameter(ctx, model, beta, e, q, n_theta=n_theta,
                            delta_init=delta_init, seed=seed)
            for (e, q) in params]

def audit_channel(model, beta: np.ndarray, channel, ell_i: int, q_i: int,
                  n_theta: int = 20, delta_init: float = 0.05,
                  seed: int = 1, gamma_probe: float = 1e-3) -> dict:
    """One-call audit of a (channel, architecture, parameter) triple.
    Equivalent to prepare_channel_context followed by audit_parameter; the
    restricted generator is built exactly once."""
    t0 = time.time()
    ctx = prepare_channel_context(channel, model.n, model.r, gamma_probe)
    out = audit_parameter(ctx, model, beta, ell_i, q_i, n_theta=n_theta,
                          delta_init=delta_init, seed=seed)
    out["wall_seconds"] = time.time() - t0
    return out


