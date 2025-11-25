# Notebooks

This folder contains Jupyter notebooks that demonstrate and visualise the models and strategies implemented in this repository.

The notebooks are meant to look and feel like small research notes: they bridge

> theory → model → strategy / hedge → results

using real or simulated data.

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

- Downloads daily close prices for a chosen equity (e.g. `AAPL`, `PLTR`) via `yfinance`  
- Estimates a constant volatility from historical returns  
- Sets up a 1-year at-the-money European call  
- Shorts 1 call and delta-hedges it using the Black–Scholes delta  
- Tracks and plots:
  - The theoretical Black–Scholes option price  
  - The value of the replicating (hedged) portfolio  
  - The hedging error over time  

At the bottom of the notebook, you can reuse:

```python
results = run_bs_delta_hedge_for_ticker("AAPL", period="1y")
plot_bs_hedging_results(results)
````

and simply change `"AAPL"` to another ticker (e.g. `"PLTR"`).

---

### `sabr_smile_trading_demo.ipynb`

A synthetic **SABR smile / relative value** trading experiment:

* Chooses a “true” set of SABR parameters and generates a smooth implied vol smile
* Adds noise to create a pseudo “market” smile (microstructure / quote noise)
* Calibrates SABR back to the noisy market smile via least squares (fixed β)
* Uses the calibrated SABR vols as a **fair value** model
* Builds a simple cross-sectional strategy:

  * Go **long** calls where SABR vol ≫ market vol (underpriced)
  * Go **short** calls where SABR vol ≪ market vol (overpriced)
* Simulates terminal prices under a lognormal model and shows the **PnL distribution**

This notebook links:

> model → calibration → signal → PnL.

---

### `sabr_surface_3d.ipynb`

A visual exploration of the SABR implied volatility **surface**:

* Uses the Hagan lognormal SABR approximation to compute σ(K, T)
* Plots a 3D SABR implied volatility surface over **strike–maturity** space
* Plots 2D smiles (vol vs strike) for several maturities
* Optionally adds noise to the vols to mimic real market data and overlays:

  * a smooth SABR model (wireframe),
  * a noisy “market” surface / scattered points

This notebook is focused on **intuition and visualisation** of how the SABR parameters (α, β, ρ, ν) shape skew, smile curvature and term structure.

---

### `crr_model_trading_demo.ipynb`

A pricing and trading demo for the **Cox–Ross–Rubinstein (CRR) binomial model**:

* Implements European and American option pricing on a recombining CRR tree
* Shows **convergence** of CRR European call prices to the Black–Scholes price as the number of steps `N` increases
* Compares **American vs European** prices:

  * American call vs European call on a non-dividend-paying stock
  * American put vs European put (highlighting early-exercise value)
* Builds a toy **mispricing strategy**:

  * “Market” uses a coarse, low-step CRR tree to price calls
  * The “model” uses a fine, high-step CRR tree as fair value
  * Trades long/short calls across strikes based on relative mispricing
* Simulates terminal prices under the high-step CRR model and plots the **PnL distribution** of the strategy

This notebook illustrates how a simple discrete-time model can be used for both
**theoretical insight** (CRR → Black–Scholes) and **toy trading ideas**.

---

## Planned notebooks

Planned additions include, for example:

* Heston vs Black–Scholes mispricing strategies on real data
* Implied volatility term structures and simple vol surfaces
* Portfolio / risk demos (Markowitz, CAPM-style analysis)

As the repository grows, new notebooks will be added here and linked in this README.
