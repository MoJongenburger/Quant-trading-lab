# Notebooks

This folder contains Jupyter notebooks that demonstrate and visualise the models and strategies implemented in this repository.

The notebooks are meant to look and feel like small research notes:
they bridge **theory → model → strategy → results** using real or simulated data.

---

## Current notebooks

### `heston_model_demo.ipynb`

A demonstration of the Heston stochastic volatility model:

- Simulates and plots Heston price and variance paths
- Shows how stochastic variance evolves over time (mean reversion, volatility-of-volatility, leverage effect)
- Illustrates the distribution of terminal prices under Heston vs a simple lognormal assumption

This notebook focuses on **model intuition and dynamics**, not trading.

---

### `black_scholes_delta_hedge.ipynb`

A real-data delta-hedging experiment under the Black–Scholes model:

- Downloads daily close prices for a chosen equity (e.g. AAPL, PLTR) via `yfinance`
- Estimates a constant volatility from historical returns
- Sets up a 1-year at-the-money European call
- Shorts 1 call and **delta-hedges** it using the Black–Scholes delta
- Tracks:
  - The theoretical Black–Scholes option price
  - The value of the replicating (hedged) portfolio
  - The **hedging error** over time

At the bottom of the notebook, you can reuse:

```python
results = run_bs_delta_hedge_for_ticker("AAPL", period="1y")
plot_bs_hedging_results(results)
