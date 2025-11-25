# Quant Trading Lab

A personal quant research lab containing mathematical finance models, trading strategies, and reproducible experiments in Python.

The goal of this repository is to **bridge the gap between theory and practice**:

- Implement well-known models from quantitative finance (Heston, Black–Scholes, etc.)
- Turn them into systematic trading strategies
- Apply them to both **simulated** and **real market data**
- Keep the code clean and modular, in a style a quant researcher / developer would recognize

---

## Current components

### Heston stochastic volatility model

Implemented under `src/quant_trading_lab/models/heston/`:

- **Model**
  - Stochastic variance (CIR-type) with mean reversion and leverage effect
- **Pricing**
  - Monte Carlo pricing of European calls under Heston dynamics
  - Black–Scholes pricing as a benchmark using a constant “market” volatility
- **Trading**
  - Mispricing metric
    - $$ m = \frac{C^{\text{Heston}} - C^{\text{BS}}}{C^{\text{BS}}} $$
  - Simple buy / sell / flat rule based on a relative mispricing threshold
  - PnL simulation along simulated Heston paths

### Real-data notebooks

In `notebooks/` you can find research-style notebooks that:

- Download **real equity data** (e.g. AAPL, PLTR) with `yfinance`
- Estimate historical volatility from daily returns
- Use the Heston model to price short-dated at-the-money options
- Compare Heston prices to Black–Scholes prices using historical vol as “market vol”
- Run and visualise a mispricing-based options strategy and its PnL

These notebooks are intended to show a full pipeline:

> data → model → pricing → signal → backtest → plots

---

## Planned modules

Planned additions to this lab include:

- **Black–Scholes utilities**
  - Implied volatility and Greeks
  - Simple volatility surface exploration
- **Mean-reversion / pairs trading**
  - OU-type processes and statistical arbitrage toy examples
- **Portfolio & risk models**
  - Markowitz mean–variance optimisation
  - CAPM-style factor overlays and simple risk attribution
- **Volatility trading ideas (simplified)**
  - Variance-swap approximations
  - Gamma / vega-driven strategies in a toy setting

Each model will live in its own subdirectory with:

- A model-specific `README.md` (math + intuition)
- Python implementation (simulation / pricing / strategy logic)
- Optionally a notebook and/or example script

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
│        └─ heston/
│           ├─ README.md     # Heston documentation (math + design)
│           ├─ params.py     # model & option parameter dataclasses
│           ├─ simulation.py # Heston path simulation (full-truncation Euler)
│           ├─ pricing.py    # Heston MC pricer + Black–Scholes benchmark
│           └─ strategy.py   # mispricing signal + PnL backtest
├─ examples/
│  └─ heston_demo.py         # end-to-end demo on synthetic Heston paths
├─ notebooks/
│  └─ heston_...ipynb        # real-data experiments (e.g. AAPL / PLTR mispricing)
└─ tests/
   └─ test_heston_pricing.py # basic unit tests for Heston pricing
