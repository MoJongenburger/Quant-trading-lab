# SABR Stochastic Volatility Model

This module implements the **SABR stochastic volatility model** and its
approximate implied volatility formula (Hagan et al.), together with basic
calibration scaffolding. It is designed to be used on **implied volatility
smiles** (e.g. from listed equity options, FX options, or swaptions).

It is part of the broader `quant-trading-lab` project and focuses on three layers:

1. **Model** – SABR dynamics for the forward and its volatility  
2. **Implied volatility** – Hagan-style approximation for Black–Scholes vol  
3. **Calibration** – fitting $(\alpha, \beta, \rho, \nu)$ to market smiles

Later, notebooks will connect this to real option data (e.g. from `yfinance`)
to calibrate SABR to equity option smiles.

---

## 1. Model overview

### 1.1 Motivation

Market-implied volatilities are not flat:

- Smiles / skews across **strike**  
- Term structures across **maturity**

The **SABR model** (Stochastic Alpha, Beta, Rho) is widely used to describe and
fit these smiles, especially in **interest rate** and **FX** markets, and can
also be applied to equity options.

Key goals:

- Capture **smile/skew** in a parsimonious way  
- Provide a relatively simple **four-parameter** model  
- Keep a (semi-)analytic connection to Black–Scholes implied volatility

---

### 1.2 SABR dynamics

In its classic form, for a forward price $F_t$ (e.g. forward rate or forward
underlying), the SABR model under the risk-neutral measure is:

```math
\begin{aligned}
dF_t &= \alpha_t F_t^{\beta} \, dW_t^{(1)}, \\
d\alpha_t &= \nu \alpha_t \, dW_t^{(2)}, \\
dW_t^{(1)} dW_t^{(2)} &= \rho \, dt.
\end{aligned}
````

where:

* $F_t$ – forward price at time $t$ (e.g. forward rate / forward stock)
* $\alpha_t$ – stochastic volatility level at time $t$
* $\beta \in [0, 1]$ – elasticity (controls how vol depends on the level of $F_t$)
* $\nu > 0$ – volatility of volatility
* $\rho \in [-1, 1]$ – correlation between forward and volatility
* $\alpha_0 = \alpha > 0$ – initial volatility level

Typically we treat $(\alpha, \beta, \rho, \nu)$ as **parameters to be calibrated**
for each maturity.

Special cases:

* $\beta = 1$ – lognormal SABR (proportional volatility)
* $\beta = 0$ – normal SABR (additive volatility)

---

## 2. Intuition

### 2.1 Forward dynamics

The forward evolves with **level-dependent** volatility:

```math
\text{Vol}[F_t] \approx \alpha_t F_t^{\beta}.
```

* For $\beta = 1$: volatility is proportional to $F_t$ (like Black–Scholes).
* For $\beta < 1$: volatility grows sub-linearly with $F_t$ (often used in
  rates to control skew).

### 2.2 Volatility dynamics

The volatility $\alpha_t$ follows a **lognormal**-type process:

```math
d\alpha_t = \nu \alpha_t \, dW_t^{(2)}.
```

So:

* $\alpha_t$ stays positive
* $\nu$ controls how “noisy” the volatility is (vol-of-vol)

### 2.3 Correlation $\rho$

The Brownian motions are correlated:

```math
dW_t^{(1)} dW_t^{(2)} = \rho \, dt.
```

* $\rho < 0$ produces **downside skew** (common in equities / FX)
* $\rho > 0$ produces **upside skew**

Together, $(\beta, \rho, \nu)$ shape the smile.

---

## 3. Implied volatility (Hagan et al.)

SABR is usually used via an **approximate formula** for the
Black–Scholes implied volatility $\sigma_{\text{BS}}(F_0, K, T)$ for a
European option with:

* forward $F_0$ at time 0,
* strike $K$,
* maturity $T$,
* SABR parameters $(\alpha, \beta, \rho, \nu)$.

In the (lognormal) SABR case, Hagan et al. derived an approximation of the form:

```math
\sigma_{\text{BS}}(F_0, K, T)
\approx
\frac{\alpha}{(F_0 K)^{(1 - \beta)/2}}
\cdot
\frac{z}{x(z)}
\cdot
\left[
1
+
T \cdot \left(
\frac{(1 - \beta)^2}{24} \frac{\alpha^2}{(F_0 K)^{1 - \beta}}
+
\frac{1}{4}\frac{\rho \beta \nu \alpha}{(F_0 K)^{(1 - \beta)/2}}
+
\frac{2 - 3\rho^2}{24}\nu^2
\right)
\right],
```

where

```math
z = \frac{\nu}{\alpha} (F_0 K)^{(1 - \beta)/2} \ln\left(\frac{F_0}{K}\right),
```

and

```math
x(z) = \ln \left(
\frac{
\sqrt{1 - 2\rho z + z^2} + z - \rho
}{
1 - \rho
}
\right).
```

For $F_0 \to K$ (at-the-money), a well-defined ATM limit is used instead of the
direct formula.

This approximation links the **SABR parameters** to a **Black–Scholes implied
vol smile**, enabling:

* fast evaluation of $\sigma_{\text{BS}}(K)$ across strikes $K$
* efficient **calibration** to market implied volatilities

---

## 4. Calibration

In practice, for a given maturity $T$ we observe a set of implied volatilities:

```math
\{ (K_i, \sigma_{\text{mkt}}(K_i)) \}_{i=1}^N
```

We want to find SABR parameters $(\alpha, \beta, \rho, \nu)$ such that the
Hagan SABR implied vols $\sigma_{\text{SABR}}(K_i)$ fit the market smile.

Typical calibration approach:

1. Fix $\beta$ based on convention or prior (e.g. $\beta = 0.5$ or $0.7$).
2. Minimize an error function over $(\alpha, \rho, \nu)$, e.g.

   ```math
   \text{Error}(\alpha, \rho, \nu)
   = \sum_{i=1}^N w_i \left(
   \sigma_{\text{SABR}}(K_i; \alpha, \beta, \rho, \nu)
   - \sigma_{\text{mkt}}(K_i)
   \right)^2.
   ```

   where $w_i$ are optional weights (e.g. volume-based or uniform).

In this module we provide:

* a **SABR implied vol function** (Hagan lognormal formula), and
* a simple **least-squares calibration routine** that, given strikes and
  market vols, returns calibrated $(\alpha, \rho, \nu)$ for a fixed $\beta$.

Later, a notebook will show how to use:

* equity or ETF option chains (via `yfinance`),
* convert option prices to implied vols (if needed, using the Black–Scholes
  implied vol solver from the `black_scholes` module),
* calibrate SABR to the observed smile.

---

## 5. Implementation layout (in this repo)

The SABR code in `quant-trading-lab` is planned to be organized as:

* `sabr.py`

  * Core SABR functions:

    * Hagan lognormal implied volatility approximation
    * ATM SABR vol limit
    * Helpers for $z$ and $x(z)$

* `calibration.py`

  * Objective functions for least-squares calibration
  * Simple optimizer wrapper (e.g. using `scipy` if available, or a custom
    grid / local search as a fallback)

* `README.md` (this file)

  * Mathematical background and modelling choices

Notebooks in the root `notebooks/` folder will demonstrate:

* Calibrating SABR to a synthetic smile generated from known parameters
* Calibrating SABR to a **real equity options smile** derived from option
  chains (e.g. SPY, AAPL, PLTR), using either:

  * implied vols directly from the data source, or
  * implied vols computed via the Black–Scholes module.

---

## 6. References

* Hagan, P., Kumar, D., Lesniewski, A., and Woodward, D. (2002),
  *“Managing Smile Risk”*, Wilmott Magazine.

* Hagan, P. and Woodward, D. (1999),
  *“Equivalent Black Volatilities”*.

````

---

Next step (when you’re ready): we’ll add

- `__init__.py`
- `sabr.py` with the Hagan implied vol function
- `calibration.py` with a least-squares calibrator

so you can later plug in option data (either synthetic or from `yfinance`) and say:

```python
params = calibrate_sabr(F0, T, strikes, market_vols, beta=0.5)
````
