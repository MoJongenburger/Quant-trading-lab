# Black–Scholes Model

This module implements the **Black–Scholes option pricing model** and basic tools around it (pricing, Greeks, implied volatility), together with a simple trading / hedging example.

It is part of the broader `quant-trading-lab` project and focuses on three layers:

1. **Model** – mathematical definition and assumptions of the Black–Scholes framework  
2. **Pricing** – closed-form formulas for European calls and puts, plus implied volatility  
3. **Trading / Hedging** – a simple delta-hedging style simulation and basic PnL analysis

---

## 1. Model overview

### 1.1 Assumptions

In the classical Black–Scholes setup:

- The underlying asset price $S_t$ follows a **geometric Brownian motion** under the risk-neutral measure:
  
  ```math
  $ dS_t = r S_t\, dt + \sigma S_t\, dW_t, $


where

* $r$ is the constant risk-free rate,

* $\sigma > 0$ is the **constant volatility**,

* $W_t$ is a standard Brownian motion.

* Markets are frictionless (no transaction costs, no bid–ask spreads).

* Trading is continuous; positions can be rebalanced at any time.

* The risk-free rate $r$ and volatility $\sigma$ are known and constant.

* No arbitrage and no dividends (in the basic version here).

Under these assumptions, the log-price is normally distributed and we obtain **closed-form** formulas for European option prices.

---

### 1.2 European call and put pricing

Consider a European call and put with:

* current spot price $S_0$,
* strike $K$,
* maturity $T$ (in years),
* constant risk-free rate $r$,
* constant volatility $\sigma$.

Define

```math
d_1 = \frac{\ln(S_0 / K) + (r + \tfrac{1}{2} \sigma^2) T}{\sigma \sqrt{T}}, 
\qquad
d_2 = d_1 - \sigma \sqrt{T}.
```

Let $N(\cdot)$ be the standard normal CDF. Then:

* **Call price**:

  ```math
  C_0 = S_0 N(d_1) - K e^{-rT} N(d_2).
  ```

* **Put price**:

  ```math
  P_0 = K e^{-rT} N(-d_2) - S_0 N(-d_1).
  ```

These satisfy **put–call parity**:

```math
C_0 - P_0 = S_0 - K e^{-rT}.
```

---

## 2. Greeks (sensitivities)

The **Greeks** measure how the option price reacts to changes in inputs:

* **Delta** ($\Delta$): sensitivity to the underlying price
* **Gamma** ($\Gamma$): sensitivity of delta to the underlying price
* **Vega** ($\nu$): sensitivity to volatility
* **Theta** ($\Theta$): sensitivity to the passage of time
* **Rho** ($\rho$): sensitivity to the interest rate

For example, for a European call in Black–Scholes:

* Delta:

  ```math
  \Delta_{\text{call}} = N(d_1).
  ```

* Gamma (same for calls and puts):

  ```math
  \Gamma = \frac{N'(d_1)}{S_0 \sigma \sqrt{T}},
  ```

  where $N'(x)$ is the standard normal PDF.

* Vega (same for calls and puts):

  ```math
  \nu = S_0 N'(d_1) \sqrt{T}.
  ```

These formulas are implemented in code so they can be used for hedging simulations and risk analysis.

---

## 3. Implied volatility

In practice, markets often quote **option prices**, not volatilities.
The **implied volatility** $\sigma_{\text{impl}}$ is defined as:

> the value of $\sigma$ such that the Black–Scholes price equals the observed market price.

Formally, for a call:

```math
\text{Given } C_{\text{mkt}}, \text{ find } \sigma_{\text{impl}} \text{ such that }
C_{\text{BS}}(S_0, K, T, r, \sigma_{\text{impl}}) = C_{\text{mkt}}.
```

In this module we provide a numerical solver (e.g. Newton or bisection) to invert the Black–Scholes formula and compute implied volatility for calls (and optionally puts).

This is a basic building block for:

* constructing implied volatility **surfaces**,
* comparing **historical vs implied vol**,
* building **volatility trading strategies**.

---

## 4. Trading / hedging example

To keep the module focused yet practical, we include a simple **delta-hedging style** example:

* Simulate an underlying price path using either:

  * geometric Brownian motion (Black–Scholes dynamics), or
  * historical data from a liquid equity (e.g. AAPL, PLTR).
* Start with a short position in a European call.
* At each time step:

  * Compute the Black–Scholes **delta** of the call.
  * Adjust the stock position to match the delta (i.e. maintain a hedged portfolio).
* Track the **PnL of the hedging strategy** versus the theoretical Black–Scholes price.

In a pure Black–Scholes world with continuous trading, perfect replication is possible.
In discrete time with realistic data, a **hedging error** appears – a useful concept for interviews and practical risk discussions.

---

## 5. Implementation layout (in this repo)

The Black–Scholes code in `quant-trading-lab` is organized as:

* `pricing.py`

  * Closed-form pricing functions for European calls and puts
  * Put–call parity helper(s)
  * Implied volatility solver (e.g. for calls)

* `greeks.py`

  * Delta, gamma, vega, etc. for calls and puts
  * Convenience functions suitable for vectorised use in simulations

* `strategy.py`

  * Simple delta-hedging simulation given a price path
  * PnL time series and basic summary statistics

Additional **experiments and visualisations** live in the `notebooks/` directory at the repository root, for example:

* Comparing Black–Scholes prices to historical PnL of hedging on real AAPL / PLTR data
* Simple implied volatility term structure or smile explorations (planned)

---

## 6. Reference

* Fischer Black and Myron Scholes (1973),
  *“The Pricing of Options and Corporate Liabilities”*,
  Journal of Political Economy, Vol. 81, No. 3.


