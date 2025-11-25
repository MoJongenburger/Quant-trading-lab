# Quant Trading Lab

A personal quant research lab containing mathematical finance models, trading strategies, and reproducible experiments in Python.

The goal of this repository is to bridge the gap between theory and practice:

- Implement well-known models from quantitative finance (Heston, Black–Scholes, SABR, CRR, etc.)
- Turn them into systematic trading or hedging strategies
- Apply them to both simulated and real market data
- Keep the code clean and modular, in a style a quant researcher / developer would recognize

---

## Current components

### 1. Heston stochastic volatility model

Implemented under `src/quant_trading_lab/models/heston/`.

**Model**

- Stochastic variance (CIR-type) with mean reversion and leverage effect  
- Negative correlation between price and variance (leverage effect / skew)

**Pricing**

- Monte Carlo pricing of European calls under Heston dynamics  
- Black–Scholes pricing as a benchmark using a constant “market” volatility  

**Trading**

- Mispricing metric  

  $$
  m = \frac{C_{\text{Heston}} - C_{\text{BS}}}{C_{\text{BS}}}
  $$

- Simple buy / sell / flat rule based on a relative mispricing threshold  
- PnL simulation along simulated Heston paths  

The Heston module is documented in detail in  
`src/quant_trading_lab/models/heston/README.md`.

---

### 2. Black–Scholes model

Implemented under `src/quant_trading_lab/models/black_scholes/`.

**Pricing**

- Closed-form pricing of European calls and puts  
- Put–call parity helpers  
- Implied volatility solvers for calls and puts (bisection-based)

**Greeks**

- $\Delta$ (delta), $\Gamma$ (gamma), vega, $\Theta$ (theta), $\rho$  
- Designed for use in hedging simulations and risk analysis  

**Strategy / Hedging**

- Discrete-time delta-hedging engine for a European call:
  - Short 1 call at $t = 0$ at its Black–Scholes price  
  - Hedge with the underlying using the Black–Scholes delta  
  - Track the replicating portfolio value vs theoretical option price  
  - Analyse the resulting hedging error  

The Black–Scholes module is documented in  
`src/quant_trading_lab/models/black_scholes/README.md`.

---

### 3. SABR stochastic volatility smile model

Implemented under `src/quant_trading_lab/models/sabr/`.

**Model**

- Lognormal SABR dynamics for the forward $F_t$ and volatility $\alpha_t$:

  $$
  \begin{aligned}
  dF_t &= \alpha_t F_t^\beta \, dW_t^{(1)}, \\
  d\alpha_t &= \nu \alpha_t \, dW_t^{(2)}, \\
  dW_t^{(1)} dW_t^{(2)} &= \rho \, dt,
  \end{aligned}
  $$

- Parameters $(\alpha, \beta, \rho, \nu)$ shape the overall level, skew and smile curvature

**Implied volatility**

- Hagan lognormal SABR implied volatility approximation  
- Works for single strikes or full smiles across strikes  

**Calibration**

- Simple least-squares calibration via grid search:
  - Fixes $\beta$ and fits $(\alpha, \rho, \nu)$ to a given smile  
  - Suitable for synthetic experiments and pedagogy (no SciPy required)

The SABR module is documented in  
`src/quant_trading_lab/models/sabr/README.md`.

---

### 4. Cox–Ross–Rubinstein (CRR) binomial model

Implemented under `src/quant_trading_lab/models/crr/`.

**Model**

- Discrete-time recombining binomial tree for the underlying price  
- Standard CRR parameterisation from volatility:

  $$
  \Delta t = \frac{T}{N}, \quad
  u = e^{\sigma \sqrt{\Delta t}}, \quad
  d = \frac{1}{u}, \quad
  p = \frac{e^{r \Delta t} - d}{u - d}
  $$

- Risk-neutral probability $p$ checked to lie in $[0,1]$

**Pricing**

- European call and put pricing via backward induction under the risk-neutral measure  
- American call and put pricing with early exercise at each node  

**Theoretical link**

- European CRR prices converge to Black–Scholes prices as $N \to \infty$  
- American call on a non-dividend-paying stock $\approx$ European call (no early exercise premium)  
- American put is never worth less than the European put

The CRR module is documented in  
`src/quant_trading_lab/models/crr/README.md`.

---

## Real-data and synthetic notebooks

In `notebooks/` you can find research-style notebooks that connect:

> data → model → pricing → strategy / hedge → PnL → plots

### Current notebooks

- **`heston_model_demo.ipynb`**  
  Simulates and plots Heston price and variance paths, shows how stochastic variance evolves over time, and compares terminal price distributions under Heston vs a simple lognormal model.

- **`black_scholes_delta_hedge.ipynb`**  
  Downloads real equity data (e.g. `AAPL`, `PLTR`) via `yfinance`, estimates a constant volatility from historical returns, sets up a 1-year at-the-money call, and runs a discrete-time Black–Scholes delta-hedging strategy.  
  At the bottom you can use:

  ```python
  results = run_bs_delta_hedge_for_ticker("AAPL", period="1y")
  plot_bs_hedging_results(results)
````

and simply change `"AAPL"` to another ticker.

* **`sabr_smile_trading_demo.ipynb`**
  Synthetic SABR smile construction, noisy “market” vols, SABR calibration, a simple mispricing-based options strategy, and PnL distribution plots.

* **`sabr_surface_3d.ipynb`**
  3D SABR implied volatility surface visualisation and 2D smiles for multiple maturities, with optional noise to mimic real-world quoted implied vol surfaces.

* **`crr_model_trading_demo.ipynb`**
  CRR binomial pricing and trading demo:

  * Convergence of CRR European call prices to Black–Scholes as the number of steps increases
  * American vs European prices for calls and puts
  * A toy cross-sectional mispricing strategy where:

    * the “market” uses a coarse CRR tree,
    * the “model” uses a fine CRR tree,
    * and PnL is simulated under the fine model.

See `notebooks/README.md` for more detail.

---

## Planned modules

Planned additions to this lab include:

**Black–Scholes utilities (extensions)**

* Implied volatility term structures and basic vol surfaces
* More structured Greeks-based risk reports

**Mean-reversion / pairs trading**

* OU-type processes and statistical arbitrage toy examples

**Portfolio & risk models**

* Markowitz mean–variance optimisation
* CAPM-style factor overlays and simple risk attribution

**Volatility trading ideas (simplified)**

* Variance-swap approximations
* Gamma / vega-driven strategies in a toy setting

Each model will live in its own subdirectory with:

* A model-specific `README.md` (math + intuition)
* Python implementation (simulation / pricing / strategy logic)
* Optionally a notebook and/or example script

---

## Repository structure

High-level layout (simplified):

```text
quant-trading-lab/
├─ README.md
├─ LICENSE
├─ src/
│  └─ quant_trading_lab/
│     ├─ __init__.py
│     └─ models/
│        ├─ heston/
│        │  ├─ README.md
│        │  ├─ params.py
│        │  ├─ simulation.py
│        │  ├─ pricing.py
│        │  └─ strategy.py
│        ├─ black_scholes/
│        │  ├─ README.md
│        │  ├─ __init__.py
│        │  ├─ pricing.py
│        │  ├─ greeks.py
│        │  └─ strategy.py
│        ├─ sabr/
│        │  ├─ README.md
│        │  ├─ __init__.py
│        │  ├─ sabr.py
│        │  └─ calibration.py
│        └─ crr/
│           ├─ README.md
│           ├─ __init__.py
│           └─ pricing.py
├─ notebooks/
│  ├─ README.md
│  ├─ heston_model_demo.ipynb
│  ├─ black_scholes_delta_hedge.ipynb
│  ├─ sabr_smile_trading_demo.ipynb
│  ├─ sabr_surface_3d.ipynb
│  └─ crr_model_trading_demo.ipynb
└─ tests/
   ├─ README.md
   ├─ test_heston_pricing.py
   ├─ test_black_scholes_pricing.py
   ├─ test_black_scholes_greeks.py
   ├─ test_sabr_model.py
   └─ test_crr_pricing.py
```


