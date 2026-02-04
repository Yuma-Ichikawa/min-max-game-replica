# RS saddle-point vs finite-size numerics (two-temperature replica)

This repository contains **research-grade, reproducible Python code** to compare:

1) **Replica-symmetric (RS) saddle-point predictions** for the two-temperature free energy density
\(v(\beta_{\max},\beta_{\min}) = \lim_{L\to\infty} \Phi(\beta_{\max},\beta_{\min})/L\),

and

2) **Finite-size numerical estimates** at:
   - **finite temperature** (Monte Carlo integration using an exact inner-simplex integral via divided differences), and
   - **zero temperature** (exact minimax value via linear programming).

The code is designed to accompany the manuscript `two_temperature_replica.tex` and mirrors its notation as closely as possible.

---

## Installation

Create a fresh environment (recommended) and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

### (A) Zero temperature: RS theory vs LP numerics

Run a small demo sweep over \(\gamma=N/M\):

```bash
python -m experiments.run_zeroT_curve --gammas 0.3 0.5 0.8 1.0 1.3 --trials 20 --base_M 80 --seed 0
```

This will:
- solve the RS zero-temperature equations for each \(\gamma\),
- solve the finite-size minimax value via LP for `trials` random matrices,
- print a comparison table,
- optionally save plots if `--save_figs` is provided.

### (B) Finite temperature: RS theory vs Monte Carlo numerics

Example at \(\beta_{\max}=\beta_{\min}=1\):

```bash
python -m experiments.run_finiteT_check --gamma 0.5 --beta_max 1.0 --beta_min 1.0 --sigma 1.0 \
  --N 40 --M 80 --x_samples 2000 --trials 10 --seed 0
```

This will:
- solve the RS saddle equations (fixed-point + 1D bracketing for multipliers),
- compute an RS prediction for \(\Phi/L\),
- compute finite-size Monte Carlo estimates for the same \(N\times M\) sizes using the exact inner integral.

> **Note**: Finite-temperature Monte Carlo is expensive. Start small and increase `--x_samples` / `--trials` as needed.

---

## What is implemented

### RS finite-temperature saddle (1RSB structure in y-sector)

Implemented in `src/rs_finiteT.py`.

Unknowns (matching the manuscript):
- \(Q_x,q_x\) for the minimizer sector,
- \(Q_y,q_1,q_0\) for the maximizer sector (1RSB blocks),
- multipliers \(m_x,m_y\).

The solver uses:
- Gauss–Hermite quadrature for Gaussian integrals over \(z,\eta\),
- a robust 1D bracketing root solve for enforcing \(\langle x\rangle=1\) and \(\langle y\rangle=1\),
- damped fixed-point iteration for overlaps.

### Finite-temperature finite-size numerics

Implemented in `src/finiteT_mc.py`.

We compute (for each random matrix C):
\[
\Phi = -\frac{1}{\beta_{\min}}\log \int_{\mathcal{X}} \mathrm{d}\mathbf{x}\; \Bigl[ Z_y(\mathbf{x}) \Bigr]^k,
\quad
Z_y(\mathbf{x})=\int_{\mathcal{Y}} \mathrm{d}\mathbf{y}\; e^{\beta_{\max} V(\mathbf{x},\mathbf{y})},
\quad
k=-\beta_{\min}/\beta_{\max}.
\]

Key ingredient: **exact** evaluation of \(Z_y(\mathbf{x})\) for simplex constraints via **divided differences**.

### Zero-temperature finite-size numerics (LP)

Implemented in `src/zeroT_lp.py`.

We solve the standard primal LP:
\[
t(C) = \min_{\mathbf{p}\in\Delta_N}\;\max_j (C^T\mathbf{p})_j
\]
and its dual, using `scipy.optimize.linprog` (HiGHS).

We report:
- \(f_L=(NM)^{1/4} t(C)\),
- support fractions,
- second moments \(q_x,q_y\).

### Zero-temperature RS theory

Implemented in `src/rs_zeroT.py`.

It solves the scalar RS system for \((\alpha_x,\alpha_y)\) and returns:
- \(f(\gamma)\),
- \(\rho_x,\rho_y\),
- \(q_x,q_y\).

---

## Reproducibility notes

- All scripts accept `--seed`.
- Results fluctuate with finite-size and Monte Carlo error; increase `--trials` and `--x_samples` to reduce variance.
- For finite temperature, the inner integral is exact but the outer integral is Monte Carlo.

---

## License

MIT (see `LICENSE`).
