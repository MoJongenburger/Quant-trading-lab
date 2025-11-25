# SABR Stochastic Volatility Model

This module implements the **SABR stochastic volatility model** and its approximate
implied volatility formula (Hagan et al.), together with basic calibration scaffolding.
It is designed to work on **implied volatility smiles** (e.g. from listed equity options,
FX options, or swaptions).

It is part of the broader `quant-trading-lab` project and focuses on three layers:

- **Model** – SABR dynamics for the forward and its volatility
- **Implied volatility** – Hagan-style approximation for Black–Scholes vol
- **Calibration** – fitting $(\alpha, \beta, \rho, \nu)$ to market smiles

Dedicated notebooks (`sabr_smile_trading_demo.ipynb`, `sabr_surface_3d.ipynb`)
show how to use the model for calibration, trading ideas and visualisation.

---

## 1. Model overview

### 1.1 Motivation

Market-implied volatilities are not flat:

- Smiles / skews across strike
- Term structures across maturity

The **SABR model** (Stochastic Alpha, Beta, Rho) is widely used to describe and fit
these smiles, especially in rates and FX markets, and can also be applied to equity options.

Key goals:

- Capture smile / skew in a **parsimonious** way
- Provide a relatively simple **four-parameter** model
- Maintain a (semi-)analytic link to **Black–Scholes implied volatility**

### 1.2 SABR dynamics

In its classic (lognormal) form, for a forward price $F_t$ under the risk-neutral measure:

$$
\begin{aligned}
dF_t &= \alpha_t F_t^\beta \, dW_t^{(1)}, \\\\
d\alpha_t &= \nu \alpha_t \, dW_t^{(2)}, \\\\
dW_t^{(1)} dW_t^{(2)} &= \rho \, dt.
\end{aligned}
$$

where:

- $F_t$ – forward price at time $t$ (e.g. forward rate / forward stock)
- $\alpha_t$ – stochastic volatility level at time $t$
- $\beta \in [0, 1]$ – elasticity (controls how vol depends on the level of $F_t$)
- $\nu > 0$ – volatility of volatility (**vol-of-vol**)
- $\rho \in [-1, 1]$ – correlation between forward and volatility
- $\alpha_0 = \alpha > 0$ – initial volatility level

Typically we treat $(\alpha, \beta, \rho, \nu)$ as parameters to be calibrated **per maturity**.

Special cases:

- $\beta = 1$ – lognormal SABR (proportional volatility)
- $\beta = 0$ – normal SABR (additive volatility)

---

## 2. Intuition

### 2.1 Forward dynamics

The forward evolves with level-dependent volatility:

$$
\mathrm{vol}[F_t] \approx \alpha_t F_t^\beta.
$$

- For $\beta = 1$: volatility is proportional to $F_t$ (like Black–Scholes).
- For $\beta < 1$: volatility grows **sub-linearly** with $F_t$
  (often used in rates to control skew).

### 2.2 Volatility dynamics

The volatility $\alpha_t$ follows a lognormal-type process:

$$
d\alpha_t = \nu \alpha_t \, dW_t^{(2)}.
$$

So:

- $\alpha_t$ stays positive.
- $\nu$ controls how “noisy” volatility itself is (**vol-of-vol**).

### 2.3 Correlation $\rho$

The Brownian motions are correlated:

$$
dW_t^{(1)} dW_t^{(2)} = \rho \, dt.
$$

- $\rho < 0$ typically produces **downside skew** (common in equities / FX).
- $\rho > 0$ produces **upside skew**.

Together, $(\beta, \rho, \nu)$ shape the **smile and skew**.

---

## 3. Implied volatility (Hagan et al.)

SABR is usually used via an approximate formula for the **Black–Scholes implied volatility**
$\sigma_{\text{BS}}(F_0, K, T)$ for a European option with:

- forward $F_0$ at time 0,
- strike $K$,
- maturity $T$,
- SABR parameters $(\alpha, \beta, \rho, \nu)$.

In the (lognormal) SABR case, Hagan et al. derived a widely used approximation. Denoting

- $z$ – a transformed log-moneyness term
- $x(z)$ – a correction factor
- and including a time correction $1 + T \cdot (\dots)$,

we obtain a closed-form expression for $\sigma_{\text{BS}}$ which:

- behaves well in both ATM and OTM regimes,
- has a separate **ATM limit** when $F_0 \to K$,
- is fast enough for calibration and surface construction.

In this module we implement the **Hagan lognormal SABR formula** with:

- support for scalar and vectorised strikes,
- a robust ATM limit,
- simple input validation.

This links the SABR parameters to a full **Black–Scholes implied vol smile**, enabling:

- fast evaluation of $\sigma_{\text{BS}}(K)$ across strikes,
- efficient calibration to market implied volatilities.

---

## 4. Calibration

For a given maturity $T$ we observe a set of implied volatilities:

$\{(K_i, \sigma_{\text{mkt}}(K_i))\}_{i=1}^N$

We want to find SABR parameters $(\alpha, \beta, \rho, \nu)$ such that
the SABR implied vols $\sigma_{\text{SABR}}(K_i)$ fit the market smile.

A common approach:

1. **Fix $\beta$** based on convention or prior (e.g. $\beta = 0.5$ or $0.7$).
2. Minimise a least-squares error over $(\alpha, \rho, \nu)$:

   $\text{Error}(\alpha, \rho, \nu)
   = \sum_{i=1}^N w_i \Big(
   \sigma_{\text{SABR}}(K_i; \alpha, \beta, \rho, \nu)
   - \sigma_{\text{mkt}}(K_i)
   \Big)^2$

   where $w_i$ are optional weights (e.g. volume-based or uniform).

In this repo we provide:

- A SABR implied vol function (`sabr_implied_vol_hagan`)
- A simple least-squares **grid-search** calibration routine (`calibrate_sabr_grid`)
  that, given strikes and market vols, returns calibrated $(\alpha, \rho, \nu)$
  for a fixed $\beta$, without requiring SciPy.

The calibration logic is primarily used in:

- `sabr_smile_trading_demo.ipynb` – synthetic smile, calibration, and a toy trading strategy.
- `test_sabr_model.py` – basic unit tests to verify consistency and calibration behaviour.

---

## 5. Implementation layout

Current SABR implementation:

```text
src/quant_trading_lab/models/sabr/
├─ __init__.py
├─ sabr.py          # SABR params + Hagan implied vol
└─ calibration.py   # least-squares grid-search calibration
````

Key components:

* **`SabrParams`** dataclass
  Encapsulates $(\alpha, \beta, \rho, \nu)$ and provides type safety.

* **`sabr_implied_vol_hagan`**
  Vectorised Hagan lognormal implied vol approximation with ATM handling.

* **`calibrate_sabr_grid`**
  Lightweight least-squares calibration via grid search, robust and dependency-light.

Associated notebooks:

* `notebooks/sabr_smile_trading_demo.ipynb`
  Synthetic smile generation, SABR calibration, mispricing-based options strategy, and PnL analysis.

* `notebooks/sabr_surface_3d.ipynb`
  3D SABR surface visualisation + noisy, “market-like” smiles vs smooth model.

Unit tests:

* `tests/test_sabr_model.py`
  Ensures implied vols are positive, skew behaves as expected for negative $\rho$,
  and calibration approximately recovers known parameters on synthetic data.

---

## 6. References

* Hagan, P., Kumar, D., Lesniewski, A., and Woodward, D. (2002),
  *Managing Smile Risk*, Wilmott Magazine.

* Hagan, P. and Woodward, D. (1999),
  *Equivalent Black Volatilities*.

````

