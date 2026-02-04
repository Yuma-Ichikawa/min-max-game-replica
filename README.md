# RS saddle-point vs finite-size numerics (two-temperature replica)

<p align="center">
  <img src="assets/nash_equilibrium_gamma_sweep.gif" alt="Nash Equilibrium Animation" width="800">
</p>

<p align="center">
  <em>Animation: Nash equilibrium value and mixed strategies as γ = N/M varies (zero-temperature limit)</em>
</p>

---

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

## Saddle-point equations

### Zero-temperature RS saddle (`src/rs_zeroT.py`)

For the minimax game \(t(C) = \min_{\mathbf{p}\in\Delta_N} \max_{\mathbf{q}\in\Delta_M} \mathbf{p}^\top C\, \mathbf{q}\) with i.i.d. \(\mathcal{N}(0,1)\) entries and \(\gamma = N/M\), the RS ansatz yields a **2-variable system** for \((\alpha_x, \alpha_y)\):

$$\Phi(\alpha_y) = \gamma\,\Phi(\alpha_x), \qquad \sqrt{\gamma}\,\alpha_x\sqrt{q(\alpha_y)} + \alpha_y\sqrt{q(\alpha_x)} = 0$$

where \(\Phi(x)\) is the standard normal CDF, \(\phi(x)\) the PDF, and

$$A(\alpha) = \alpha\Phi(\alpha) + \phi(\alpha), \quad B(\alpha) = (\alpha^2+1)\Phi(\alpha) + \alpha\phi(\alpha), \quad q(\alpha) = \frac{B(\alpha)}{A(\alpha)^2}.$$

**Outputs**: scaled game value \(f(\gamma) = \tfrac{1}{2}\bigl(\gamma^{-1/4}\alpha_x\sqrt{q_y} - \gamma^{1/4}\alpha_y\sqrt{q_x}\bigr)\), support fractions \(\rho_x = \Phi(\alpha_x)\), \(\rho_y = \Phi(\alpha_y)\), and overlaps \(q_x, q_y\).

### Finite-temperature RS/1RSB saddle (`src/rs_finiteT.py`)

For the two-temperature free energy with \(k = -\beta_{\min}/\beta_{\max}\), we solve for **7 unknowns**: \((Q_x, q_x)\) (minimizer), \((Q_y, q_1, q_0)\) (maximizer, 1RSB), and multipliers \((m_x, m_y)\).

**Conjugate variables** (from stationarity):

$$\chi_x = \frac{\sigma\beta_{\max}^2}{\sqrt{\gamma}} k^2 q_0, \quad \chi_0 = \sqrt{\gamma}\sigma\beta_{\max}^2 q_x, \quad \chi_1 = \sqrt{\gamma}\sigma\beta_{\max}^2 (Q_x - q_x)$$

$$\hat{Q}_x = \chi_x - \frac{\sigma\beta_{\max}^2}{\sqrt{\gamma}}\bigl[kQ_y + k(k-1)q_1\bigr]$$

**Site measures**:
- x-sector: truncated Gaussian \(\propto e^{-\hat{Q}_x x^2/2 + (m_x + \sqrt{\chi_x}z)x}\) on \(x \geq 0\)
- y-sector: truncated exponential \(\propto e^{(m_y + \sqrt{\chi_0}z + \sqrt{\chi_1}\eta)y}\) on \([0, y_{\max}]\), reweighted by \([Z_y]^k\)

**Self-consistency**: \(\langle x\rangle = 1\), \(\langle y\rangle = 1\) (1D root solve); overlaps match moments (damped fixed-point). Output: \(v = -g_*/\beta_{\min}\).

---

## What is implemented

| Module | Description |
|--------|-------------|
| `src/rs_zeroT.py` | Zero-T RS saddle: solves \((\alpha_x,\alpha_y)\) → \(f(\gamma), \rho_x, \rho_y, q_x, q_y\) |
| `src/rs_finiteT.py` | Finite-T RS/1RSB saddle: solves 7 order parameters → \(v(\beta_{\max},\beta_{\min})\) |
| `src/zeroT_lp.py` | LP solver for \(t(C) = \min_{\mathbf{p}}\max_j (C^\top\mathbf{p})_j\) via HiGHS |
| `src/finiteT_mc.py` | Monte Carlo + divided differences for \(\Phi = -\beta_{\min}^{-1}\log\int [Z_y(\mathbf{x})]^k\,d\mathbf{x}\) |
| `src/divided_differences.py` | Exact simplex integral \(\int \delta(m-\sum y_j)e^{\sum a_j y_j}dy\) via DD recursion |
| `src/quadrature.py` | Gauss–Hermite nodes/weights for \(\int Dz\, f(z)\) |
| `src/truncated_distributions.py` | Moments of truncated Gaussian/exponential |

---

## Reproducibility notes

- All scripts accept `--seed`.
- Results fluctuate with finite-size and Monte Carlo error; increase `--trials` and `--x_samples` to reduce variance.
- For finite temperature, the inner integral is exact but the outer integral is Monte Carlo.

---

## License

MIT (see `LICENSE`).
