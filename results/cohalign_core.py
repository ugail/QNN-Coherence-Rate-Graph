"""
cohalign_core.py -- model and noise-channel library for CohAlign.

Standalone implementation of:
  - charge-sector bookkeeping for the U(1) symmetry on n qubits
  - the U(1)-equivariant brickwork ansatz on the cycle C_n
    (trainable Rz layer followed by fixed XY hopping on alternating edges,
     single-qubit Markovian noise applied once per layer)
  - nine reference noise channels: four restricted-isotropic
    (amplitude damping, dephasing, depolarising, X-error) and five
    structured (inhomogeneous dephasing, site-dependent amplitude damping,
    biased Pauli, coherent-dissipative mix, correlated two-site dephasing)

Every channel exposes  apply(M, gamma, n)  acting on an arbitrary
2^n x 2^n matrix, so the same code path serves density-matrix evolution
and the linear-map probes used by cohalign_rates.
"""
import numpy as np
from itertools import combinations

# ---------------------------------------------------------------- Pauli
I2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# ------------------------------------------------------ sector bookkeeping
def sector_basis(n: int, r: int) -> list:
    """Computational-basis indices of Hamming weight r (bit q = site q)."""
    out = []
    for pos in combinations(range(n), r):
        s = 0
        for p in pos:
            s |= (1 << p)
        out.append(s)
    return sorted(out)

def sector_projector(n: int, r: int) -> np.ndarray:
    P = np.zeros((2**n, 2**n), dtype=complex)
    for s in sector_basis(n, r):
        P[s, s] = 1.0
    return P

def pair_basis(n: int, r: int) -> list:
    """Ordered pairs (a, b), a != b, spanning the in-sector off-diagonal block."""
    sec = sector_basis(n, r)
    return [(a, b) for a in sec for b in sec if a != b]

# ---------------------------------------------------------- embeddings
def embed_one_qubit(op: np.ndarray, q: int, n: int) -> np.ndarray:
    """Embed a 2x2 operator on qubit q (bit q of the integer index)."""
    mats = [I2] * n
    mats[n - 1 - q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def xy_unitary(beta: float, i: int, j: int, n: int) -> np.ndarray:
    """exp(-i beta (X_i X_j + Y_i Y_j)) as a full 2^n unitary.

    Acts as identity on aligned bit pairs and as a cos/sin block on the
    swap-coupled pair, so it preserves Hamming weight exactly.
    """
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    c, s = np.cos(2 * beta), np.sin(2 * beta)
    for st in range(dim):
        if ((st >> i) & 1) == ((st >> j) & 1):
            continue
        sw = st ^ (1 << i) ^ (1 << j)
        if st < sw:
            U[st, st] = c
            U[sw, sw] = c
            U[st, sw] = -1j * s
            U[sw, st] = -1j * s
    return U

# ----------------------------------------------------------- the ansatz
class Brickwork:
    """U(1)-equivariant brickwork ansatz on the cycle C_n.

    Layer ell applies trainable Rz(theta[ell, q]) on every site, then fixed
    XY hopping exp(-i beta[ell, j] (XX + YY)) on the alternating edge set
    E_ell (even edges for even ell, odd edges for odd ell, cycle-closing
    edge included). Noise, when present, acts once after each layer.
    """
    INIT_SPREAD_SEED = 202   # fixed seed of the delocalised sector input

    def __init__(self, n: int, L: int, r: int = 1, init_state: str = "localized",
                 spread_seed: int = None):
        if n < 2:
            raise ValueError("need n >= 2")
        if init_state not in ("localized", "spread"):
            raise ValueError("init_state must be 'localized' or 'spread'")
        self.n, self.L, self.r = n, L, r
        self.init_state_mode = init_state
        self.spread_seed = int(spread_seed) if spread_seed is not None \
            else self.INIT_SPREAD_SEED
        self.dim = 2**n
        self.E = [[j for j in range(n) if j % 2 == ell % 2] for ell in range(L)]
        self.P_r = sector_projector(n, r)
        self.readout = self.P_r @ embed_one_qubit(PAULI_Z, 0, n) @ self.P_r

    # -- unitaries -----------------------------------------------------
    def z_layer(self, theta_ell: np.ndarray) -> np.ndarray:
        d = np.ones(self.dim, dtype=complex)
        for q in range(self.n):
            ph_m = np.exp(-1j * theta_ell[q] / 2)
            ph_p = np.exp(+1j * theta_ell[q] / 2)
            for st in range(self.dim):
                d[st] *= ph_m if ((st >> q) & 1) == 0 else ph_p
        return np.diag(d)

    def xy_layer(self, beta_ell: np.ndarray, ell: int) -> np.ndarray:
        U = np.eye(self.dim, dtype=complex)
        for j in self.E[ell]:
            U = xy_unitary(beta_ell[j], j, (j + 1) % self.n, self.n) @ U
        return U

    def layer_unitary(self, theta: np.ndarray, beta: np.ndarray, ell: int) -> np.ndarray:
        return self.xy_layer(beta[ell], ell) @ self.z_layer(theta[ell])

    # -- states --------------------------------------------------------
    def initial_state(self) -> np.ndarray:
        """Sector-r input state.

        "localized": the basis state |1^r 0^{n-r}> (excitations on sites
        0..r-1).  NOTE: at r = 2 with the primary shallow geometry (L = 3)
        this input carries no trainable-phase response for the Z_0 readout
        to numerical precision (the activity preflight returns a map at the
        numerical floor).  The inactivity is geometry-specific -- residual
        activity of order 1e-4 reappears at greater depth -- so this is a
        guarded configuration, not a universal r >= 2 statement.
        "spread": a fixed, seeded delocalised superposition over the full
        sector basis (seed INIT_SPREAD_SEED), which restores generic
        theta-response; higher-sector validation uses this input to avoid
        the inactive configuration.
        """
        psi = np.zeros(self.dim, dtype=complex)
        if self.init_state_mode == "localized":
            psi[(1 << self.r) - 1] = 1.0
            return psi
        sec = sector_basis(self.n, self.r)
        rng = np.random.default_rng(self.spread_seed)
        amps = rng.normal(size=len(sec)) + 1j * rng.normal(size=len(sec))
        amps = amps / np.linalg.norm(amps)
        for a, s in zip(amps, sec):
            psi[s] = a
        return psi

    def evolve_dm(self, theta: np.ndarray, beta: np.ndarray,
                  channel=None, gamma: float = 0.0) -> np.ndarray:
        """rho after L layers with per-layer noise (noiseless when gamma=0)."""
        psi = self.initial_state()
        rho = np.outer(psi, psi.conj())
        for ell in range(self.L):
            U = self.layer_unitary(theta, beta, ell)
            rho = U @ rho @ U.conj().T
            if channel is not None and gamma > 0:
                rho = channel.apply(rho, gamma, self.n)
        return rho

    def output(self, rho: np.ndarray) -> float:
        return float(np.real(np.trace(self.readout @ rho)))

def teacher_background(seed: int, L: int, n: int, scale: float = 0.5) -> np.ndarray:
    """Fixed random hopping background beta (part of the architecture)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(L, n))

# ------------------------------------------------------------- channels
def apply_single_qubit_kraus(M: np.ndarray, Ks: list, q: int, n: int) -> np.ndarray:
    """Public helper. Applies a single-qubit Kraus map on site q of an
    n-qubit operator M and returns the result."""
    return _apply_1q_kraus(M, Ks, q, n)

def _apply_1q_kraus(M: np.ndarray, Ks: list, q: int, n: int) -> np.ndarray:
    """sum_K K_q M K_q^dag on an arbitrary 2^n x 2^n matrix via tensor reshape."""
    t = M.reshape((2,) * (2 * n))
    aL, aR = n - 1 - q, 2 * n - 1 - q
    out = np.zeros_like(t)
    for K in Ks:
        tmp = np.tensordot(K, t, axes=([1], [aL]))
        tmp = np.moveaxis(tmp, 0, aL)
        tmp = np.tensordot(K.conj(), tmp, axes=([1], [aR]))
        tmp = np.moveaxis(tmp, 0, aR)
        out = out + tmp
    return out.reshape(2**n, 2**n)

class NoiseChannel:
    """Base class. Subclasses either provide kraus_single(gamma) for uniform
    per-qubit action, or override apply() for structured action."""
    name = "base"
    def kraus_single(self, gamma: float) -> list:
        raise NotImplementedError
    def apply(self, M: np.ndarray, gamma: float, n: int) -> np.ndarray:
        Ks = self.kraus_single(gamma)
        for q in range(n):
            M = _apply_1q_kraus(M, Ks, q, n)
        return M

class AmplitudeDamping(NoiseChannel):
    name = "amp_damp"
    def kraus_single(self, g):
        return [np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex),
                np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)]

class Dephasing(NoiseChannel):
    name = "dephase"
    def kraus_single(self, g):
        return [np.sqrt(1 - g) * I2, np.sqrt(g) * PAULI_Z]

class Depolarising(NoiseChannel):
    name = "depol"
    def kraus_single(self, g):
        return [np.sqrt(1 - g) * I2, np.sqrt(g / 3) * PAULI_X,
                np.sqrt(g / 3) * PAULI_Y, np.sqrt(g / 3) * PAULI_Z]

class XError(NoiseChannel):
    """Bit-flip channel; breaks U(1). Retained as the symmetry-breaking probe."""
    name = "x_err"
    def kraus_single(self, g):
        return [np.sqrt(1 - g) * I2, np.sqrt(g) * PAULI_X]

class InhomogeneousDephasing(NoiseChannel):
    """Per-qubit dephasing rates gamma * w_q, profile normalised to mean 1."""
    name = "inhom_dephase"
    def __init__(self, weights):
        w = np.asarray(weights, dtype=float)
        self.weights = w / np.mean(w)
    def apply(self, M, gamma, n):
        for q in range(n):
            gq = float(gamma * self.weights[q])
            if gq <= 0:
                continue
            Ks = [np.sqrt(1 - gq) * I2, np.sqrt(gq) * PAULI_Z]
            M = _apply_1q_kraus(M, Ks, q, n)
        return M

class SiteDependentAmpDamp(NoiseChannel):
    """Per-qubit amplitude-damping rates gamma * w_q, mean-normalised."""
    name = "site_amp_damp"
    def __init__(self, weights):
        w = np.asarray(weights, dtype=float)
        self.weights = w / np.mean(w)
    def apply(self, M, gamma, n):
        for q in range(n):
            gq = float(gamma * self.weights[q])
            if gq <= 0:
                continue
            Ks = [np.array([[1, 0], [0, np.sqrt(1 - gq)]], dtype=complex),
                  np.array([[0, np.sqrt(gq)], [0, 0]], dtype=complex)]
            M = _apply_1q_kraus(M, Ks, q, n)
        return M

class BiasedPauli(NoiseChannel):
    """Pauli noise with error ratios (r_X, r_Y, r_Z) summing to one."""
    # Convention. The probability tuple is ordered (p_X, p_Y, p_Z) and the
    # shipped benchmark uses (0.6, 0.2, 0.2), an X-dominated channel with
    # p_X != p_Y, which deliberately breaks phase covariance.
    name = "biased_pauli"
    def __init__(self, ratios=(0.6, 0.2, 0.2)):
        s = float(sum(ratios))
        self.ratios = tuple(x / s for x in ratios)
    def kraus_single(self, g):
        rX, rY, rZ = self.ratios
        return [np.sqrt(max(1 - g, 0.0)) * I2, np.sqrt(g * rX) * PAULI_X,
                np.sqrt(g * rY) * PAULI_Y, np.sqrt(g * rZ) * PAULI_Z]

class CoherentDissipativeMix(NoiseChannel):
    """Null coherent-component control. A uniform global Rz over-rotation
    (eps = eps_ratio * gamma, applied before amplitude damping at gamma)
    acts as a global phase within any fixed charge sector, so the coherent
    component is operationally invisible in-sector by construction and the
    channel must audit identically to amplitude damping. A genuinely active
    coherent perturbation is provided by SiteZOverRotation below."""
    name = "coh_diss_mix"
    def __init__(self, epsilon_ratio=0.5):
        self.epsilon_ratio = float(epsilon_ratio)
    def apply(self, M, gamma, n):
        eps = self.epsilon_ratio * gamma
        Urot = np.array([[np.exp(-1j * eps / 2), 0],
                         [0, np.exp(+1j * eps / 2)]], dtype=complex)
        for q in range(n):
            M = _apply_1q_kraus(M, [Urot], q, n)
        Ks = [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
              np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]
        for q in range(n):
            M = _apply_1q_kraus(M, Ks, q, n)
        return M

class CorrelatedDephasing(NoiseChannel):
    """Two-site ZZ dephasing on a fixed edge set:
        M -> (1 - gamma) M + gamma (Z_i Z_j) M (Z_i Z_j)   per edge.

    On the disjoint even-edge pairing {(2k, 2k+1)} in the single-excitation
    sector, coherences between sites of the SAME edge are exactly protected
    while cross-edge coherences decay -- the alignment-structured control.
    """
    name = "corr_dephase"
    def __init__(self, n, edges=None):
        if edges is None:
            edges = [(2 * k, 2 * k + 1) for k in range(n // 2)]
        self.edges = list(edges)
    def apply(self, M, gamma, n):
        for (i, j) in self.edges:
            t = M.reshape((2,) * (2 * n))
            tmp = t
            for q in (i, j):
                aL, aR = n - 1 - q, 2 * n - 1 - q
                tmp = np.tensordot(PAULI_Z, tmp, axes=([1], [aL]))
                tmp = np.moveaxis(tmp, 0, aL)
                tmp = np.tensordot(PAULI_Z.conj(), tmp, axes=([1], [aR]))
                tmp = np.moveaxis(tmp, 0, aR)
            M = ((1 - gamma) * t + gamma * tmp).reshape(2**n, 2**n)
        return M

class SiteZOverRotation(NoiseChannel):
    """Pure coherent site-dependent Z over-rotation (no dissipation).

    Applies exp(-i * gamma * c_q * Z_q / 2) on every site with the fixed
    non-uniform profile c_q = q / (n - 1). Because the profile is not
    proportional to the conserved total charge, the rotation acts
    non-trivially on in-sector coherences. Its restricted generator is
    anti-Hermitian in the Frobenius geometry, so lambda_coh vanishes and
    the audit reports a *signed* response susceptibility without a
    normalised alignment interpretation."""
    name = "site_z_overrotation"
    def apply(self, M, gamma, n):
        for q in range(n):
            c = q / (n - 1) if n > 1 else 0.0
            eps = gamma * c
            Urot = np.array([[np.exp(-1j * eps / 2), 0],
                             [0, np.exp(+1j * eps / 2)]], dtype=complex)
            M = _apply_1q_kraus(M, [Urot], q, n)
        return M

def build_channel_suite(n: int) -> dict:
    """The nine reference channels at system size n, isotropic first."""
    w = np.linspace(0.5, 1.5, n)
    return {
        "amp_damp":      AmplitudeDamping(),
        "dephase":       Dephasing(),
        "depol":         Depolarising(),
        "x_err":         XError(),
        "inhom_dephase": InhomogeneousDephasing(w),
        "site_amp_damp": SiteDependentAmpDamp(w),
        "biased_pauli":  BiasedPauli((0.6, 0.2, 0.2)),
        "coh_diss_mix":  CoherentDissipativeMix(0.5),
        "corr_dephase":  CorrelatedDephasing(n),
    }

ISOTROPIC = ("amp_damp", "dephase", "depol", "x_err")
STRUCTURED = ("inhom_dephase", "site_amp_damp", "biased_pauli",
              "coh_diss_mix", "corr_dephase")

def check_trace_preserving(channel, n: int, gamma: float = 0.05, seed: int = 0,
                           tol: float = 1e-10) -> float:
    """Max |Tr Phi(rho) - 1| over a few random density matrices."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(3):
        A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)
        worst = max(worst, abs(float(np.real(np.trace(channel.apply(rho, gamma, n)))) - 1.0))
    return worst
