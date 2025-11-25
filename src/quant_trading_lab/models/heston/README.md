# Heston Stochastic Volatility Model

This module implements the **Heston stochastic volatility model** for option pricing and a simple model-based trading strategy.

It is part of the broader `quant-trading-lab` project and focuses on three layers:

1. **Model** – mathematical definition of the Heston dynamics  
2. **Pricing** – Monte Carlo pricing of European options under Heston, plus a Black–Scholes benchmark  
3. **Trading** – a mispricing-based trading signal and simple PnL simulations (on simulated and real data)

---

## 1. Mathematical overview

### 1.1 Motivation

The standard Black–Scholes model assumes:

- Constant volatility $\sigma$  
- Lognormal stock prices  
- No volatility clustering or skew  

In real markets we observe:

- **Volatility smile / skew** – implied volatility depends on strike and maturity  
- **Volatility clustering** – high-volatility periods tend to follow high-volatility periods  
- **Leverage effect** – negative returns often coincide with rising volatility  

The **Heston model** addresses these limitations by making the **variance of returns stochastic and mean-reverting**, while remaining analytically tractable.

---

### 1.2 Model definition

Under the risk-neutral measure $\mathbb{Q}$, the Heston model is

```math
\begin{aligned}
dS_t &= r S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^{(1)}, \\
dv_t &= \kappa (\theta - v_t)\, dt + \sigma \sqrt{v_t}\, dW_t^{(2)}, \\
dW_t^{(1)} dW_t^{(2)} &= \rho\, dt, \quad \rho \in [-1, 1].
\end{aligned}
````

where:

* $S_t$ – asset price at time $t$
* $v_t$ – instantaneous variance at time $t$
* $r$ – risk-free interest rate
* $\kappa > 0$ – speed of mean reversion of the variance
* $\theta > 0$ – long-run mean of the variance
* $\sigma > 0$ – volatility of variance (“vol-of-vol”)
* $\rho \in [-1, 1]$ – correlation between price and variance Brownian motions
* $v_0 = v_{t=0}$ – initial variance

We collect the parameters as

```math
(\kappa, \theta, \sigma, \rho, v_0, r).
```

The variance process $v_t$ is a **Cox–Ingersoll–Ross (CIR)**-type process, which is designed to remain non-negative.
A standard condition that helps keep $v_t$ strictly positive is the **Feller condition**:

```math
2 \kappa \theta \ge \sigma^2.
```

---

## 2. Intuition

### 2.1 Price process $S_t$

The price dynamics look like Black–Scholes, but with **stochastic volatility** $\sqrt{v_t}$ instead of a constant $\sigma$:

* When $v_t$ is high, the asset is more volatile.
* When $v_t$ is low, the asset is calmer.

### 2.2 Variance process $v_t$

The variance:

* Mean-reverts towards $\theta$ at speed $\kappa$
* Has randomness controlled by $\sigma$ (vol-of-vol)
* Moves more when it is high and less when it is low due to the $\sqrt{v_t}$ term

### 2.3 Correlation $\rho$

The Brownian motions $W^{(1)}$ and $W^{(2)}$ are correlated:

```math
dW_t^{(1)} dW_t^{(2)} = \rho\, dt.
```

For equity markets one typically has $\rho < 0$, capturing the **leverage effect**: price drops tend to coincide with volatility spikes, producing realistic downside skew in implied volatilities.

---

## 3. Option pricing in this module

For a European call with strike $K$ and maturity $T$, the payoff at maturity is

```math
\max(S_T - K, 0).
```

Under the risk-neutral measure, the time-0 price is

```math
C_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}\big[\max(S_T - K, 0)\big].
```

In the full Heston framework, one can derive a **semi-closed-form solution** using the characteristic function of $\log S_T$.

In this module we start with a **Monte Carlo implementation**:

1. Simulate many paths of $(S_t, v_t)$ up to maturity $T$ under the Heston dynamics.
2. Compute the payoff $\max(S_T^{(i)} - K, 0)$ for each path $i$.
3. Discount and average:

```math
C_0^{\text{Heston}}
\approx
e^{-rT} \frac{1}{N} \sum_{i=1}^N \max\big(S_T^{(i)} - K, 0\big).
```

For comparison, we also compute the **Black–Scholes price** $C_0^{\text{BS}}$ for the same option, using a constant volatility $\sigma_{\text{mkt}}$.

---

## 4. Trading strategy design

The aim is not only to price options, but also to illustrate how a stochastic volatility model can drive a **systematic trading signal**.

Conceptually:

1. Assume the “market” prices options using **Black–Scholes** with some volatility $\sigma_{\text{mkt}}$.

2. Use the **Heston model** as our internal view of the world.

3. For a given option (e.g. an at-the-money call), compute:

   * Heston price $C^{\text{Heston}}$ (via Monte Carlo)
   * Black–Scholes price $C^{\text{BS}}$

4. Define a **relative mispricing** metric:

```math
m = \frac{C^{\text{Heston}} - C^{\text{BS}}}{C^{\text{BS}}}.
```

5. Turn this mispricing into a basic trading rule:

* If $m > \tau$:
  Heston says the option is **underpriced** vs Black–Scholes
  → go **long** the option

* If $m < -\tau$:
  Heston says the option is **overpriced**
  → go **short** the option

* If $|m| \le \tau$:
  → **hold / no position**

In the simplest synthetic setting we:

* Simulate an underlying path under the Heston model
* Re-evaluate prices and signals over time
* Track the resulting **PnL** assuming the market uses Black–Scholes prices as observable quotes.

In the **real-data notebooks** (e.g. AAPL, PLTR) we:

* Use daily equity data via `yfinance`
* Estimate historical volatility as a proxy for $\sigma_{\text{mkt}}$
* Apply the same mispricing logic to short-dated at-the-money calls

This explicitly links:

> model → pricing → signal → PnL.

---

## 5. Implementation layout (in this repo)

The Heston code in `quant-trading-lab` is organized as:

* `params.py`

  * Dataclasses for:

    * Heston model parameters $(\kappa, \theta, \sigma, \rho, v_0, r)$
    * Option parameters (strike $K$, maturity $T$)

* `simulation.py`

  * Time-discretised simulation of the Heston SDEs
  * Full-truncation Euler for the variance process
  * Correlated Gaussian shocks for $W^{(1)}$ and $W^{(2)}$

* `pricing.py`

  * Monte Carlo pricing of European options under Heston
  * Closed-form Black–Scholes pricing for comparison

* `strategy.py`

  * Computation of the mispricing metric $m$
  * Buy / sell / hold signal generation
  * Simple PnL backtests on simulated Heston paths

The **notebooks** in the root `notebooks/` folder demonstrate:

* Heston dynamics (simulated paths for $S_t$ and $v_t$)
* Mispricing-based strategies applied to real equity data (e.g. AAPL, PLTR)

---

## 6. Reference

* Steven L. Heston (1993),
  *“A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options”*,
  The Review of Financial Studies, Vol. 6, No. 2.

```

If copying still fights you, you can also:

- Select inside the code block, press `Ctrl+A` (or `Cmd+A` on Mac) to select all inside it, then `Ctrl+C` / `Cmd+C`.
::contentReference[oaicite:0]{index=0}
```
