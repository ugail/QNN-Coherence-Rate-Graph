# A Coherence-Rate Diagnostic for Trainability Under Noise in Equivariant Quantum Neural Networks

Reference implementation, data, and reproducibility package for

> **A Coherence-Rate Diagnostic for Trainability Under Noise in Equivariant Quantum Neural Networks** by H. Ugail and N. Howard

This repository provides the **coherence-rate graph** diagnostic for noisy
equivariant quantum neural networks (QNNs). For the noise-channel families
that dominate practical device models, the restricted generator acts
diagonally on the coherence pairs of a charge sector, and its pairwise decay
rates define a weighted graph on the sector basis. The package computes
closed-form decay rates for a reference suite of nine channels, combinatorial
certificates of parameter protection and exposure depth (Boolean
reachability, no linear algebra, no simulation), a path-support certificate
implying exact gradient invariance at every noise strength, response
diagnostics including exact per-slot product laws, and a deterministic
prior-integrated population benchmark that supplies exact targets against
which finite-ensemble estimates are judged.


<img width="2794" height="1144" alt="fig4_slots_second_order" src="https://github.com/user-attachments/assets/cd19b67f-b4c9-409d-9089-95bff7292919" />

---

## What's in this repository

```
.
├── coherence_graph_pipeline_.ipynb   # main experiments (Phases A–P)
├── coherence_graph_figures_.ipynb    # figure generation
├── results/                          # all CSV outputs, provenance sidecars,
│                                     #   module sources, MANIFEST, figures
├── requirements.txt                  # environment specification
└── README.md
```

All committed numerical outputs — the phase CSVs, their `.meta.json`
provenance sidecars, the module sources exactly as executed, and the run
manifest — live in a single `results/` directory. The two notebooks are
intentionally separated so that the simulation step and the figure step can
each be run, re-run, or inspected independently of the other.

The notebooks are written for Google Colab and mount Google Drive when run
there. Near the top of each notebook a configuration cell sets the
input/output locations; point these at wherever you place the `results/`
directory. They run equally well on a local machine, where the pipeline
writes to a local output folder and the figures notebook reads
`results/` in the working directory by default. Everything runs on a
single CPU; no GPU or quantum hardware is required.

### `coherence_graph_pipeline_.ipynb` — main experiments

Runs the validation phases A through P on the U(1)-equivariant brickwork
ansatz (n = 8, primary depth L = 3): exact self-tests, coherence-graph
recovery for all nine channels, the cut-crossing exposure predictor, the
support bound and a 200-generator adversarial sweep, slot closed forms,
two-excitation sector spectra, high-ensemble deep-circuit measurements with
difference-based equivalence analysis and replication, pair-diagonality
scope, exact slot product laws with derived exponents, visible-functional
protection and the path-support certificate, the sector-versus-input
factorial, and the deterministic population benchmark. Each phase writes
one or two CSVs with a provenance sidecar and is cached: a phase whose
output matches the recorded configuration fingerprint and source hash is
skipped on re-run, and a cached output produced by a different phase
implementation is recomputed rather than silently reused.

`CGL_MODE` selects the run: `paper` (full ensembles; the committed
results), `smoke` (identical code paths at reduced ensembles, minutes on a
laptop; statistical columns indicative only) and `figures` (read-only; never
computes a phase). All seeds are fixed, so runs are exactly reproducible at
their respective ensemble sizes.

### `coherence_graph_figures_.ipynb` — figure generation

Builds the seven figures in the paper. It reads the CSVs produced by the
pipeline notebook from the results directory set in its configuration cell
and writes the PNGs to a `figures/` subdirectory. All figures use
two-column journal styling at 300 dpi.

| Figure | Content | Reads from `results/` |
| --- | --- | --- |
| `fig1_graph_recovery.png` | Coherence-rate graph recovery | `phaseB_pair_rates.csv` |
| `fig2_cut_crossing.png` | Cut-crossing predictor vs measurement | `phaseC_cut_crossing.csv` |
| `fig3_bound.png` | Support bound and adversarial sweep | `phaseD_bound.csv`, `phaseD_sweep.csv` |
| `fig4_slots_second_order.png` | Slot closed forms and the two-term predictor | `phaseE_slot_forms.csv` |
| `fig5_r2_and_residuals.png` | Higher-sector protection collapse and residual scaling | `phaseG_r2_graph.csv` |
| `fig6_deep_stress.png` | Deep-circuit stress test | `phaseH_deep_stress.csv` |
| `fig7_population.png` | Factorial and population benchmark | `phaseK2_factorial.csv`, `phaseP_benchmark.csv`, `phaseP_estimator_stats.csv` |

---

## Reproducing the paper's numerical results

Run the pipeline notebook first (set `CGL_MODE=paper` for the full run, or
`smoke` to exercise every code path quickly); it regenerates every CSV in
`results/`. The mapping from manuscript content to file is:

| Manuscript content | File in `results/` |
| --- | --- |
| Exact identity self-tests (trace, sector, diagonality, forms) | `phaseA_selftests.csv` |
| Closed-form rates, kernels, components, form certificates (Table 2) | `phaseB_graph_recovery.csv`, `phaseB_pair_rates.csv` |
| Exposure predictor vs CRN measurement across depths | `phaseC_cut_crossing.csv` |
| Support bound audits | `phaseD_bound.csv` |
| Adversarial sweep over 200 random covariant generators (18 counterexamples) | `phaseD_sweep.csv` |
| Slot closed forms (2/3 at L = 3, 1/2 at L = 4, final slot zero) | `phaseE_slot_forms.csv` |
| Exact per-slot product laws and closed-form Λ₁, K₂ | `phaseF2_product_law.csv`, `phaseF2_uniform_grid.csv` |
| Finite-difference convergence to the exact K₂ | `phaseF2_fd_convergence.csv` |
| Slot exponents derived first, then tested | `phaseF2_derived.csv` |
| Two-excitation sector graph spectra and protected fractions | `phaseG_r2_graph.csv` |
| Deep-circuit measurements and three-strength extrapolation | `phaseH_deep_stress.csv` |
| Difference-based equivalence verdicts (depth five and six) | `phaseH2_equivalence.csv` |
| Replicated ensemble-size study | `phaseH2_replication.csv` |
| Pair-diagonality scope by channel and sector | `phaseI2_diagonality_scope.csv` |
| Readout-visible protection and the coherent control | `phaseJ_visible_protection.csv` |
| Exact fixed-point witness | `phaseJ2_fixed_points.csv` |
| Path-support certificate and slot pair sets | `phaseJ3_path_certificate.csv` |
| Response-weight overlap at r = 2 | `phaseK_r2_overlap.csv` |
| Sector-versus-input 2×2 factorial | `phaseK2_factorial.csv` |
| Exact population targets and estimator statistics | `phaseP_benchmark.csv`, `phaseP_population.csv`, `phaseP_estimator_stats.csv`, `phaseP_replication_stats.csv` |

Then run the figures notebook to regenerate the seven PNGs from those CSVs.

Every CSV carries a `.meta.json` sidecar recording the mode, configuration
fingerprint, seeds and, for execution-time records, the hash of the phase
source that produced it; `MANIFEST.json` records the configuration, seeds,
per-phase wall times and package versions of the run. The `results/`
directory also contains the four Python modules exactly as executed
(`cohalign_core.py`, `cohalign_rates.py`, `cohalign_bench.py`,
`cgl_graph.py`); these are written by the notebook itself, and the
configuration fingerprint covers their source hashes, so the committed
outputs are bound to the committed code.

---

## Environment

Python 3.10 or later with NumPy, SciPy, pandas and Matplotlib; see
`requirements.txt`. The full `paper` run completes on a single CPU core;
`smoke` mode finishes in minutes.

---

## Citation

If you use this diagnostic, any part of the code in this repository, or any results, tables, or figures, please cite:

> H. Ugail and N. Howard, *A Coherence-Rate Diagnostic for Trainability Under Noise in Equivariant Quantum Neural Networks*, 2026. Under review.

---

## License

This repository is released under the MIT License.
