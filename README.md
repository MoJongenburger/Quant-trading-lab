# Quant Trading Lab

A personal quant research lab containing mathematical finance models, trading strategies, and reproducible experiments in Python.

The goal of this repository is to **bridge the gap between theory and practice**:

- Implement well-known models from quantitative finance (Heston, Black–Scholes, etc.)
- Turn them into systematic trading or hedging strategies
- Apply them to both **simulated** and **real** market data
- Keep the code clean and modular, in a style a quant researcher / developer would recognize

---

## Current components

### 1. Heston stochastic volatility model

Implemented under `src/quant_trading_lab/models/heston/`:

**Model**

- Stochastic variance (CIR-type) with mean reversion and leverage effect
- Negative correlation between price and variance (leverage effect)

**Pricing**

- Monte Carlo pricing of European calls under Heston dynamics
- Black–Scholes pricing as a benchmark using a constant “market” volatility

**Trading**

- Mispricing metric  

  `m = (C_Heston - C_BS) / C_BS`
- Simple buy / sell / flat rule based on a relative mispricing threshold
- PnL simulation along simulated Heston paths

The Heston module is documented in detail in  
`src/quant_trading_lab/models/heston/README.md`.

---

### 2. Black–Scholes model

Implemented under `src/quant_trading_lab/models/black_scholes/`:

**Pricing**

- Closed-form pricing of European calls and puts
- Put–call parity helpers
- Implied volatility solvers for calls and puts (bisection-based)

**Greeks**

- Delta, gamma, vega, theta, rho
- Designed for use in hedging simulations and risk analysis

**Strategy / Hedging**

- Discrete-time **delta-hedging** engine for a European call:
  - Short 1 call at $t=0$ at its Black–Scholes price
  - Hedge with the underlying using Black–Scholes delta
  - Track the replicating portfolio value vs theoretical option price
  - Analyse the resulting **hedging error**

The Black–Scholes module is documented in  
`src/quant_trading_lab/models/black_scholes/README.md`.

---

## Real-data notebooks

In `notebooks/` you can find research-style notebooks that connect:

> data → model → pricing → strategy / hedge → PnL → plots

Current notebooks:

- **`heston_model_demo.ipynb`**
  - Simulates and plots Heston price and variance paths
  - Shows how stochastic variance evolves over time
  - Explores the distribution of terminal prices under Heston vs a simple lognormal assumption

- **`black_scholes_delta_hedge.ipynb`**
  - Downloads real equity data (e.g. AAPL, PLTR) with `yfinance`
  - Estimates a constant volatility from historical returns
  - Sets up a 1-year at-the-money European call
  - Shorts 1 call and **delta-hedges** it using Black–Scholes delta
  - Plots:
    - Underlying price
    - Theoretical Black–Scholes option price
    - Replicating portfolio value
    - Hedging error over time

At the bottom of the Black–Scholes notebook you can reuse a simple pattern:

```python
results = run_bs_delta_hedge_for_ticker("AAPL", period="1y")
plot_bs_hedging_results(results)
````

and just change `"AAPL"` to another ticker (e.g. `"PLTR"`).

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

High-level layout:

```text
quant-trading-lab/
├─ README.md                 # this file
├─ LICENSE
├─ src/
│  └─ quant_trading_lab/
│     ├─ __init__.py
│     └─ models/
│        ├─ heston/
│        │  ├─ README.md     # Heston documentation (math + design)
│        │  ├─ params.py     # model & option parameter dataclasses
│        │  ├─ simulation.py # Heston path simulation (full-truncation Euler)
│        │  ├─ pricing.py    # Heston MC pricer + Black–Scholes benchmark
│        │  └─ strategy.py   # mispricing signal + PnL backtest
│        └─ black_scholes/
│           ├─ README.md     # Black–Scholes documentation (assumptions + formulas)
│           ├─ __init__.py
│           ├─ pricing.py    # Closed-form BS pricing + implied vol
│           ├─ greeks.py     # Delta, gamma, vega, theta, rho
│           └─ strategy.py   # Delta-hedging engine and demo
├─ examples/
│  └─ heston_demo.py         # End-to-end demo on synthetic Heston paths
├─ notebooks/
│  ├─ README.md              # Overview of available notebooks
│  ├─ heston_model_demo.ipynb
│  └─ black_scholes_delta_hedge.ipynb
└─ tests/
   ├─ test_heston_pricing.py           # Basic tests for Heston pricing
   ├─ test_black_scholes_pricing.py    # Tests for BS prices & implied vols
   └─ test_black_scholes_greeks.py     # Tests for BS Greeks properties
```

You can extend this structure as the lab grows (e.g. more models, more tests, additional notebooks), but this README now matches what you actually have in place.
