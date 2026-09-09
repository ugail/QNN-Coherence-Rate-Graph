"""
cgl_graph.py -- the coherence-graph layer (the new contribution of this
pipeline). Builds on cohalign_core / cohalign_rates, which are materialised
verbatim from the CohAlign v1.8.1 release so that every rate, mode and
response quantity used here is bit-identical to the published pipeline.

Objects implemented:

1.  Analytic pairwise decay rates (per unit gamma, first order) for every
    phase-covariant channel of the CohAlign suite, in any charge sector:
      dephasing family   rate(a,b) = 2 sum_q w_q [bit_q differs]
      correlated ZZ      rate(a,b) = 2 sum_e   [edge product differs]
      damping family     rate(a,b) = phi(a) + phi(b),
                         phi(a) = (1/2) sum_{q in exc(a)} w_q
      depolarising       rate(a,b) = (2/3)(n - d) + (4/3) d,  d = Hamming
      biased Pauli       rate(a,b) = n (rX + rY) + 2 rZ d
      X error            rate(a,b) = n            (leakage probe; in-sector)
    These are exact first-order values; the numeric finite-probe generator
    carries the documented O(gamma_probe) bias on top.

2.  The structural decomposition certificates.
      difference form  (graph-Laplacian / squared-distance):
        rate(a,b) = || v(a) - v(b) ||^2  with  v_k(a) = sqrt(c_k) l_k(a)
        over the jump spectra l_k, for the Z-diagonal (dephasing) family.
      potential form (Schroedinger):
        rate(a,b) = phi(a) + phi(b)  for the damping family.
    Both are verified to machine precision channel by channel.

3.  The protection graph and its kernel: pairs (a,b) with rate < tol form
    the zero-rate edge set; its ordered-pair count must equal the kernel
    dimension of the Hermitian part of -L_r, and its connected components
    organise the protected subspace.

4.  The combinatorial cut-crossing predictor: pure edge-set reachability
    (no linear algebra) computing, per noise slot, the visible interaction
    cone I(ell) and asking whether any unprotected 2-subset of I(ell) is
    present. "All slots protected" is a sufficient condition for zero
    response; its negation is the exposure prediction tested in Phase C.

5.  Visible pair weights: the pairwise decomposition of the first-order
    response pairing, giving the fraction of visible weight on unprotected
    coherence pairs (the quantitative companion of the cone predictor).

6.  The support-cut (Cheeger-type) lower bound on the mode rate for
    diagonal-generator channels, and helpers for the adversarial sweep.

7.  Slot-resolved noise (independent per-slot strengths) for the
    second-order / biharmonic phase, plus Fiedler-value helpers for the
    higher-sector spectral analysis.
"""
import numpy as np
from itertools import combinations
from cohalign_core import (sector_basis, pair_basis, embed_one_qubit,
                           PAULI_Z)

TOL_RATE_ZERO = 1e-9


# ------------------------------------------------------------- bit helpers
def _bits(a, n):
    return [(a >> q) & 1 for q in range(n)]

def _z(a, q):
    return -1.0 if ((a >> q) & 1) else 1.0

def _exc(a, n):
    return [q for q in range(n) if (a >> q) & 1]

def _hamming(a, b):
    return bin(a ^ b).count("1")


# --------------------------------------------- analytic first-order rates
def analytic_rate_pair(channel, a, b, n):
    """Exact first-order decay rate (per unit gamma) of the coherence
    |a><b| under the named CohAlign channel, restricted to the in-sector
    block. Real part only; the null coherent-dissipative control maps to
    its amplitude-damping dissipative part. Raises for channels without a
    phase-covariant diagonal form (none in the shipped suite except the
    pure coherent over-rotation, whose Hermitian rate is zero)."""
    name = channel.name
    if name in ("dephase", "inhom_dephase"):
        w = getattr(channel, "weights", np.ones(n))
        return 2.0 * sum(float(w[q]) for q in range(n)
                         if ((a >> q) & 1) != ((b >> q) & 1))
    if name == "corr_dephase":
        tot = 0.0
        for (i, j) in channel.edges:
            pa = _z(a, i) * _z(a, j)
            pb = _z(b, i) * _z(b, j)
            if pa * pb < 0:
                tot += 2.0
        return tot
    if name in ("amp_damp", "site_amp_damp", "coh_diss_mix"):
        w = getattr(channel, "weights", np.ones(n))
        return _damping_potential(a, n, w) + _damping_potential(b, n, w)
    if name == "depol":
        d = _hamming(a, b)
        return (2.0 / 3.0) * (n - d) + (4.0 / 3.0) * d
    if name == "biased_pauli":
        rX, rY, rZ = channel.ratios
        return n * (rX + rY) + 2.0 * rZ * _hamming(a, b)
    if name == "x_err":
        return float(n)
    if name == "site_z_overrotation":
        return 0.0
    raise ValueError(f"no analytic diagonal form for channel '{name}'")

def _damping_potential(a, n, w):
    return 0.5 * sum(float(w[q]) for q in _exc(a, n))

def analytic_rate_vector(channel, n, r):
    pairs = pair_basis(n, r)
    return np.array([analytic_rate_pair(channel, a, b, n)
                     for (a, b) in pairs]), pairs


# ------------------------------------------ structural form certificates
def difference_form_embedding(channel, n):
    """Jump-spectrum embedding v(a) for the Z-diagonal family, such that
    rate(a,b) = ||v(a) - v(b)||^2 exactly. Returns a callable a -> vector.
    """
    name = channel.name
    if name in ("dephase", "inhom_dephase"):
        w = getattr(channel, "weights", np.ones(n))
        def v(a):
            return np.array([np.sqrt(w[q] / 2.0) * _z(a, q)
                             for q in range(n)])
        return v
    if name == "corr_dephase":
        edges = channel.edges
        def v(a):
            return np.array([np.sqrt(0.5) * _z(a, i) * _z(a, j)
                             for (i, j) in edges])
        return v
    raise ValueError(f"'{name}' is not in the Z-diagonal difference family")

def check_difference_form(channel, n, r):
    """max_ab | rate_analytic(a,b) - ||v(a)-v(b)||^2 | over the sector.
    An algebraic identity; the return value should be at machine precision.
    """
    v = difference_form_embedding(channel, n)
    worst = 0.0
    for (a, b) in pair_basis(n, r):
        pred = analytic_rate_pair(channel, a, b, n)
        emb = float(np.sum((v(a) - v(b)) ** 2))
        worst = max(worst, abs(pred - emb))
    return worst

def check_potential_form(channel, n, r):
    """max_ab | rate_analytic(a,b) - (phi(a)+phi(b)) | for the damping
    family (Schroedinger / vertex-potential form)."""
    w = getattr(channel, "weights", np.ones(n))
    worst = 0.0
    for (a, b) in pair_basis(n, r):
        pred = analytic_rate_pair(channel, a, b, n)
        pot = _damping_potential(a, n, w) + _damping_potential(b, n, w)
        worst = max(worst, abs(pred - pot))
    return worst


# ----------------------------------------------- numeric generator probes
def offdiag_mass(L_r):
    """Relative off-pair-diagonal mass of the restricted generator
    (diagonality certificate for the phase-covariant diagonal families)."""
    A = np.abs(L_r)
    d = float(np.trace(A))
    tot = float(A.sum())
    off = tot - d
    return off / max(tot, 1e-30)

def numeric_diag_rates(L_r):
    return -np.real(np.diag(L_r))

def hermitian_rate_spectrum(L_r):
    H = -0.5 * (L_r + L_r.conj().T)
    return np.linalg.eigvalsh(H)


# ------------------------------------------------ protection graph, kernel
def protected_mask(rates, tol=TOL_RATE_ZERO):
    return np.asarray(rates) < tol

def protection_components(n, r, mask, pairs):
    """Connected components of the protection graph on sector basis states
    (edges = zero-rate coherence pairs). Returns (n_components, sizes)."""
    sec = sector_basis(n, r)
    idx = {s: k for k, s in enumerate(sec)}
    parent = list(range(len(sec)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for m, (a, b) in zip(mask, pairs):
        if m:
            union(idx[a], idx[b])
    roots = {}
    for k in range(len(sec)):
        roots.setdefault(find(k), 0)
        roots[find(k)] += 1
    sizes = sorted(roots.values(), reverse=True)
    return len(sizes), sizes

def kernel_dimension(L_r, tol=1e-6):
    ev = hermitian_rate_spectrum(L_r)
    return int(np.sum(np.abs(ev) < tol))


# ---------------------------------------- combinatorial cut-crossing cones
def _grow(S, edge_sites, n):
    """One layer of support growth through the edge set
    {(j, j+1 mod n) : j in edge_sites} (same both directions)."""
    S = set(S)
    out = set(S)
    for j in edge_sites:
        k = (j + 1) % n
        if j in S:
            out.add(k)
        if k in S:
            out.add(j)
    return out

def cones(n, L, r, ell_i, q_i, edge_sets):
    """All cones needed by the cut predictor. edge_sets[ell] is the list of
    left sites of layer ell's hopping edges (Brickwork.E)."""
    state = {}
    S = set(range(r))                    # localised input, excitations 0..r-1
    for ell in range(L):
        S = _grow(S, edge_sets[ell], n)
        state[ell] = set(S)              # support after layer ell (slot ell)
    ob = {L - 1: {0}}
    for ell in range(L - 2, -1, -1):
        ob[ell] = _grow(ob[ell + 1], edge_sets[ell + 1], n)
    f = {}
    if ell_i >= 1:
        f_start = _grow(ob[ell_i], edge_sets[ell_i], n) | {q_i}
        f[ell_i - 1] = f_start
        for ell in range(ell_i - 2, -1, -1):
            f[ell] = _grow(f[ell + 1], edge_sets[ell + 1], n)
    return state, ob, f

def cut_predictor(n, L, r, ell_i, q_i, edge_sets, protected_edges):
    """Pure graph-reachability exposure predictor (r = 1 geometry).

    For each noise slot ell, the visible interaction cone is
      I(ell) = ob_cone(ell)  cap state_cone(ell)      (post-insertion)
      I(ell) = f_cone(ell)   cap state_cone(ell)      (pre-insertion)
    The slot is combinatorially protected when every 2-subset of I(ell) is
    a protected pair. 'All slots protected' is a sufficient condition for a
    vanishing response; the predictor reports exposure otherwise, which is
    a necessary condition only (cancellations can still protect).
    Returns {"protected": bool, "per_slot": {ell: (sorted I, exposed)}}.
    """
    if r != 1:
        raise ValueError("cut_predictor implemented for r = 1")
    prot = {frozenset(e) for e in protected_edges}
    state, ob, f = cones(n, L, r, ell_i, q_i, edge_sets)
    per_slot, all_ok = {}, True
    for ell in range(L):
        if ell >= ell_i:
            I = ob[ell] & state[ell]
        elif ell in f:
            I = f[ell] & state[ell]
        else:
            I = set()
        exposed = any(frozenset(p) not in prot
                      for p in combinations(sorted(I), 2))
        per_slot[ell] = (sorted(I), bool(exposed))
        all_ok = all_ok and not exposed
    return {"protected": bool(all_ok), "per_slot": per_slot}


# --------------------------------------------------- visible pair weights
def visible_pair_weights(model, theta, beta, ell_i, q_i, pairs):
    """Pairwise decomposition of the first-order response pairing.

    Post-insertion slots pair OB_slot with the derivative operator D_slot;
    pre-insertion slots pair the adjoint functional F_slot with the state.
    The weight of coherence pair p = (a, b) is the summed magnitude of the
    two factors' product on p, so a pair carries weight only where BOTH the
    functional and the propagated operator are supported: the quantitative
    form of the interaction cone I(ell). (Plumbing mirrors
    cohalign_rates.response_rates so the two share slot conventions.)
    """
    n, L = model.n, model.L
    Zq = embed_one_qubit(PAULI_Z, q_i, n)
    psi = model.initial_state()
    rho = np.outer(psi, psi.conj())
    rho_slot, rho_pre_ins = [], None
    for ell in range(L):
        Uz = model.z_layer(theta[ell])
        Ux = model.xy_layer(beta[ell], ell)
        rho_z = Uz @ rho @ Uz.conj().T
        if ell == ell_i:
            rho_pre_ins = rho_z
        rho = Ux @ rho_z @ Ux.conj().T
        rho_slot.append(rho)
    C = -0.5j * (Zq @ rho_pre_ins - rho_pre_ins @ Zq)
    Ux_i = model.xy_layer(beta[ell_i], ell_i)
    D = Ux_i @ C @ Ux_i.conj().T
    D_slot = {ell_i: D}
    for ell in range(ell_i + 1, L):
        Ul = model.layer_unitary(theta, beta, ell)
        D = Ul @ D @ Ul.conj().T
        D_slot[ell] = D
    OB_slot = {L - 1: model.readout}
    for ell in range(L - 2, -1, -1):
        Ul = model.layer_unitary(theta, beta, ell + 1)
        OB_slot[ell] = Ul.conj().T @ OB_slot[ell + 1] @ Ul
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
    w = np.zeros(len(pairs))
    for k, (a, b) in enumerate(pairs):
        tot = 0.0
        for ell in range(ell_i, L):
            tot += abs(OB_slot[ell][b, a] * D_slot[ell][a, b])
        for ell in F_slot:
            tot += abs(F_slot[ell][b, a] * rho_slot[ell][a, b])
        w[k] = tot
    return w

def unprotected_fraction(weights, mask_protected):
    tot = float(np.sum(weights))
    if tot <= 0:
        return float("nan")
    return float(np.sum(weights[~mask_protected]) / tot)


# ----------------------------------- support-cut (Cheeger-type) mode bound
def support_cut_bound(g_pairvec, rates, mask_protected,
                      tol=TOL_RATE_ZERO):
    """For a diagonal restricted generator, the mode rate is the
    rate-weighted mean over the mode's support,
        lambda_mode = sum_p rates_p |g_p|^2 / sum_p |g_p|^2,
    so it is bounded below by phi * r_min, where phi is the mode's
    Frobenius-weight fraction on unprotected pairs and r_min the smallest
    nonzero unprotected rate. Returns (bound, phi, r_min, lam_exact)."""
    g2 = np.abs(np.asarray(g_pairvec)) ** 2
    tot = float(g2.sum())
    if tot <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rates = np.asarray(rates, dtype=float)
    phi = float(g2[~mask_protected].sum() / tot)
    un = rates[~mask_protected]
    un = un[un > tol]
    r_min = float(un.min()) if un.size else 0.0
    lam_exact = float(np.sum(rates * g2) / tot)
    return phi * r_min, phi, r_min, lam_exact


# ------------------------------------------------ slot-resolved evolution
def evolve_dm_slotwise(model, theta, beta, channel, gvec):
    """Density-matrix evolution with an independent noise strength per slot
    (gvec[ell] after layer ell). gvec = gamma * ones reproduces
    Brickwork.evolve_dm exactly."""
    psi = model.initial_state()
    rho = np.outer(psi, psi.conj())
    for ell in range(model.L):
        U = model.layer_unitary(theta, beta, ell)
        rho = U @ rho @ U.conj().T
        g = float(gvec[ell])
        if channel is not None and g > 0:
            rho = channel.apply(rho, g, model.n)
    return rho

def grad_slotwise(model, theta, beta, channel, gvec, ell, q,
                  shift=np.pi / 2):
    tp, tm = theta.copy(), theta.copy()
    tp[ell, q] += shift
    tm[ell, q] -= shift
    fp = model.output(evolve_dm_slotwise(model, tp, beta, channel, gvec))
    fm = model.output(evolve_dm_slotwise(model, tm, beta, channel, gvec))
    return 0.5 * (fp - fm)

def m2_slotwise(model, beta, channel, gvec, ell, q, theta_draws,
                shift=np.pi / 2):
    vals = [grad_slotwise(model, th, beta, channel, gvec, ell, q, shift) ** 2
            for th in theta_draws]
    return float(np.mean(vals))


# ------------------------------------------------------- spectral helpers
def vertex_graph_laplacian(rates, pairs, n, r):
    """Weighted graph on the sector basis states with edge weight equal to
    the pairwise decay rate; returns the ordinary graph Laplacian D - W.
    (An analysis object for clustering, distinct from the restricted
    generator itself, which is diagonal on ordered pairs.)"""
    sec = sector_basis(n, r)
    idx = {s: k for k, s in enumerate(sec)}
    K = len(sec)
    W = np.zeros((K, K))
    for rate, (a, b) in zip(rates, pairs):
        W[idx[a], idx[b]] = rate
    W = 0.5 * (W + W.T)
    return np.diag(W.sum(axis=1)) - W

def fiedler_value(Lap):
    ev = np.linalg.eigvalsh(Lap)
    return float(ev[1]) if len(ev) > 1 else float("nan")
