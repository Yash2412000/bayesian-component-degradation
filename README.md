# Hierarchical Bayesian Modelling of Mechanical Component Degradation

A fully Bayesian hierarchical model for predicting mechanical component integrity over time, built for a consultancy-style engineering client problem. The goal: predict when components degrade below 30% integrity, and identify which physical attributes drive faster degradation.

Built as part of the MSc Data Science programme (CM52060 – Bayesian Data Science) at the University of Bath (2025–2026).

---

## Problem

An engineering client monitors 75 mechanical components via integrity readings (0–100%) over time. Components degrade following an exponential decay pattern. Two modelling challenges:

1. **Sparse data** — 25 test components have only 1–4 observations each, all before day 10. Standard ML methods fail here due to identifiability issues
2. **Feature attribution** — quantify which of 5 physical attributes (X1–X5) drive the decay rate, to inform maintenance scheduling

879 total readings across 75 components over 45 days.

---

## Approach

Two hierarchical Bayesian models implemented in **NumPyro** with **HMC/NUTS** sampling:

### Baseline Model
Individual exponential decay per component with shared population-level priors:

```
f_i(t) = u_i * exp(-v_i * t / 100)
y_i(t) ~ Normal(f_i(t), σ_y²)
```

Per-component parameters `u_i` (initial integrity) and `v_i` (decay rate) drawn from learned population distributions. Non-centred reparameterisation used throughout for efficient HMC geometry.

### Enhanced Model
Extends the baseline by incorporating 5 physical covariates as shared predictors of effective decay rate:

```
f_i(t) = u_i * exp(-(v_i + w^T * x_i) * t / 100)
```

Shrinkage prior on weights: `w_j ~ Normal(0, σ_w²)` with `σ_w ~ HalfNormal(2)` to prevent overfitting on small datasets.

---

## Sampler Configuration

- 4 parallel chains, 2,000 warm-up + 2,000 sampling iterations = **8,000 posterior samples**
- Convergence criteria: zero divergent transitions, R̂ ≤ 1.05, n_eff > 10% of total samples

| Model | Divergences | Max R̂ | Min n_eff |
|---|---|---|---|
| Baseline | 0 | 1.00 | 1,353 |
| Enhanced | 0 | 1.00 | 2,607 |

Both models achieve perfect convergence.

---

## Results

### Baseline Model — Posterior Hyperparameters

| Parameter | Mean | Std | 5% | 95% |
|---|---|---|---|---|
| µ_u (mean initial integrity) | 89.70 | 0.88 | 88.27 | 91.15 |
| σ_u | 5.28 | 0.81 | 3.96 | 6.60 |
| µ_v (mean decay rate) | 3.92 | 0.21 | 3.58 | 4.28 |
| σ_v | 1.58 | 0.17 | 1.31 | 1.86 |
| σ_y (observation noise) | 5.11 | 0.13 | 4.88 | 5.32 |

### Enhanced Model — Feature Weights

| Feature | Mean | Std | 95% CI | Credibly ≠ 0? |
|---|---|---|---|---|
| X1 | −0.087 | 0.051 | [−0.188, +0.011] | No |
| X2 | +0.169 | 0.044 | [+0.082, +0.253] | Yes |
| X3 | **+1.066** | 0.064 | [+0.938, +1.193] | **Yes** |
| X4 | +0.227 | 0.027 | [+0.175, +0.281] | Yes |
| X5 | −0.029 | 0.041 | [−0.110, +0.050] | No |

**X3 is the dominant degradation driver** — posterior mean 16+ standard deviations from zero. The enhanced model reduces σ_v from 1.58 → 0.34, confirming the covariates explain a large portion of the variance in decay rates that the baseline attributed to random component-level variation.

---

## Key Findings

- Hierarchical partial pooling enables sensible predictions even for components with a single observation — uncertainty bands widen appropriately for sparse components rather than collapsing to arbitrary values
- X3 is the primary driver of degradation rate (w = +1.07 ± 0.06); components at the high end of the X3 range (up to 6.82) receive ~6.74 additional decay rate units on top of the baseline, reaching the 30% overhaul threshold significantly faster
- X4 and X2 show credible but weaker positive effects; X1 and X5 are indistinguishable from zero
- The enhanced model's σ_v drops from 1.58 to 0.34 — a 78% reduction in unexplained component-level decay rate variability, absorbed by the covariate structure
- Observation noise σ_y remains stable at ~5.1% across both models, confirming structural improvements are not artefacts of noise absorption
- Blind test predictions for 25 held-out components span the full probability range (0.000–1.000), consistent with X3's strong influence

---

## Visualisations

| File | Description |
|---|---|
| `fits_baseline.png` | Baseline posterior predictive fits — well-sampled and sparse components |
| `fits_enhanced.png` | Enhanced model posterior predictive fits |
| `fits_comparison.png` | Side-by-side baseline vs enhanced on K#0052 (4 obs) |
| `weights_violin.png` | Posterior distributions of feature weights w_j with 95% credible intervals |
| `trace_enh_weights.png` | Enhanced model trace — feature weights and σ_w |
| `trace_enh_hyper.png` | Enhanced model trace — hyperparameters |
| `trace_enh_uv.png` | Enhanced model trace — per-component u_i and v_i |
| `trace_base_uv.png` | Baseline model trace — per-component u_i and v_i |
| `Baseline_Model_Trace_Hyper-parameters.png` | Baseline model trace — hyperparameters |

---

## Stack

Python · NumPyro · JAX · ArviZ · NumPy · Matplotlib

---

## Files

- `Assessment-PartA.ipynb` — Part A implementation
- `Assessment-PartB.ipynb` — Part B implementation and evaluation
- `deployable.ipynb` — Deployable inference notebook for blind test predictions
- `predictions.csv` — P(integrity ≤ 30% at t=30) for 25 held-out test components
- `Bayesian_Data_Science_Part_B_Report.pdf` — Full consultancy report
