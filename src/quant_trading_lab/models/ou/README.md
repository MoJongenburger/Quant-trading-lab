# Ornstein–Uhlenbeck (OU) Mean-Reversion Model

This module implements the **Ornstein–Uhlenbeck (OU)** mean-reversion process and basic
calibration tools. It is designed for modelling **stationary**, mean-reverting time series,
such as:

- price spreads in pairs trading,
- short-term interest rates (in simple settings),
- log-spreads or z-scores of co-integrated assets.

It is part of the broader `quant-trading-lab` project and focuses on three layers:

- **Model** – continuous-time OU dynamics and their properties  
- **Simulation** – exact-discretisation paths for Monte Carlo experiments  
- **Calibration** – fitting $(\kappa, \theta, \sigma)$ from discrete observations via an AR(1) view  

Planned notebooks (e.g. `ou_pairs_trading_demo.ipynb`) will show how to:

- estimate OU parameters from a historical spread, and  
- build a simple mean-reversion / pairs trading strategy.

---

## 1. Model overview

### 1.1 Motivation

Many financial time series show **mean-reverting** behaviour, for example:

- spreads between two co-integrated equities,
- deviations from a fundamental value,
- short rates under certain models.

A pure random walk / Brownian motion does not revert; its variance grows without bound.
The Ornstein–Uhlenbeck process introduces a **pull back to a long-run mean**.

The OU process is popular because:

- It is **Gaussian and Markovian**, with closed-form transition densities.
- It has a **stationary distribution** (under mild conditions).
- It is simple to calibrate using linear regression (AR(1) interpretation).

### 1.2 OU dynamics

The continuous-time OU process $(X_t)_{t \ge 0}$ is defined by the SDE:

$$
dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t,
$$

where:

- $X_t$ – state variable at time $t$ (e.g. spread, log-spread, short rate),  
- $\kappa > 0$ – speed of mean reversion,  
- $\theta$ – long-run mean level,  
- $\sigma > 0$ – volatility parameter,  
- $W_t$ – standard Brownian motion.

We typically collect the parameters in the vector $(\kappa, \theta, \sigma, x_0)$,
where $x_0 = X_0$ is the initial state.

---

## 2. Intuition

### 2.1 Mean reversion

The drift term $\kappa(\theta - X_t)$ pulls $X_t$ towards $\theta$:

- If $X_t > \theta$, then $(\theta - X_t) < 0$ and the drift is negative (pulling down).
- If $X_t < \theta$, then $(\theta - X_t) > 0$ and the drift is positive (pulling up).

The parameter $\kappa$ controls **how fast** the process reverts to $\theta$.

A useful quantity is the **half-life** of deviations:

$$
t_{\frac{1}{2}} = \frac{\ln 2}{\kappa},
$$

which is the time it takes for the expected deviation from $\theta$ to be cut in half.

### 2.2 Volatility and stationary distribution

The diffusion term $\sigma\,dW_t$ introduces randomness.

Under mild conditions, the OU process has a **stationary normal distribution**:

$$
X_t \sim \mathcal{N}\left(\theta, \; \frac{\sigma^2}{2\kappa}\right)
\quad \text{(in the long run)}.
$$

So:

- $\theta$ is both the **long-run mean** and the **mode** of the distribution.
- $\sigma^2 / (2\kappa)$ is the long-run variance around $\theta$.

A larger $\kappa$ means **stronger pull** and a tighter distribution around $\theta$,
while a larger $\sigma$ means more randomness and a wider spread.

---

## 3. Exact discretisation and simulation

For a time step $\Delta t$, the OU process has a **closed-form transition**:

$$
X_{t+\Delta t}
= \theta + (X_t - \theta)e^{-\kappa \Delta t}
  + \sigma \sqrt{\frac{1 - e^{-2\kappa \Delta t}}{2\kappa}}\, Z,
$$

where $Z \sim \mathcal{N}(0, 1)$.

This implies:

**Conditional mean**

$$
\mathbb{E}[X_{t+\Delta t} \mid X_t]
= \theta + (X_t - \theta)e^{-\kappa \Delta t},
$$

**Conditional variance**

$$
\mathrm{Var}[X_{t+\Delta t} \mid X_t]
= \frac{\sigma^2}{2\kappa}\left(1 - e^{-2\kappa \Delta t}\right).
$$

In this module we use this **exact discretisation**, not a simple Euler scheme, to
simulate OU paths.

---

## 4. Calibration from discrete-time data

Suppose we observe a time series at equally spaced times:

$$
X_0, X_1, X_2, \dots, X_T
$$

with time step $\Delta t$ (e.g. 1 trading day $\Rightarrow \Delta t = 1/252$ years).

Using the exact discretisation, one can show that:

$$
X_{t+1} = a + b X_t + \varepsilon_t,
$$

where:

- $b = e^{-\kappa \Delta t}$,
- $a = \theta(1 - b)$,
- $\varepsilon_t$ is Gaussian noise with variance
  $$
  \mathrm{Var}(\varepsilon_t)
  = \sigma^2 \cdot \frac{1 - e^{-2\kappa \Delta t}}{2\kappa}.
  $$

This is just an **AR(1) model** in discrete time. We can:

1. **Run a simple OLS regression:**

   $$
   X_{t+1} = a + b X_t + \varepsilon_t
   $$

   to estimate $a$ and $b$.

2. **Recover continuous-time parameters:**

   - Mean-reversion speed:
     $$
     \kappa = -\frac{1}{\Delta t}\ln b,
     $$
   - Long-run mean:
     $$
     \theta = \frac{a}{1 - b}.
     $$

3. **Estimate $\sigma$ from the residual variance:**

   Let

   $$
   \hat{\varepsilon}_t = X_{t+1} - (a + b X_t),
   $$

   and define

   $$
   \hat{\sigma}_\varepsilon^2
   \approx \frac{1}{n-2} \sum_t \hat{\varepsilon}_t^2.
   $$

   Use the relationship

   $$
   \hat{\sigma}_\varepsilon^2
   = \sigma^2 \cdot \frac{1 - e^{-2\kappa \Delta t}}{2\kappa}
   $$

   to solve for $\sigma$:

   $$
   \sigma
   = \sqrt{
       \hat{\sigma}_\varepsilon^2
       \cdot
       \frac{2\kappa}{1 - e^{-2\kappa \Delta t}}
     }.
   $$

In this module, `estimate_ou_from_series` wraps exactly this logic:

- takes a 1D time series and $\Delta t$,
- estimates $(a, b)$ by OLS,
- converts to $(\kappa, \theta, \sigma)$,
- returns an `OUParams` object with these estimates and $x_0 = X_0$.

---

## 5. Implementation layout (in this repo)

Current OU implementation:

```text
src/quant_trading_lab/models/ou/
├─ __init__.py
└─ ou.py
````

### `OUParams`

Dataclass encapsulating the OU parameters:

* `kappa` – mean-reversion speed
* `theta` – long-run mean
* `sigma` – volatility
* `x0` – initial state $X_0$

Basic validation is performed (e.g. `kappa > 0`, `sigma > 0`).

### `ou_exact_step(x_t, dt, params, rng=None)`

Implements the exact transition

$$
X_{t+\Delta t}
= \theta + (X_t - \theta)e^{-\kappa \Delta t}

* \sigma \sqrt{\frac{1 - e^{-2\kappa \Delta t}}{2\kappa}}, Z,
  $$

with $Z \sim \mathcal{N}(0, 1)$.

* Works for scalar or vector `x_t`.
* Accepts a NumPy random generator for reproducibility.

### `simulate_ou_paths(params, T, n_steps, n_paths=1, rng=None)`

* Simulates `n_paths` OU paths from $0$ to $T$ using `n_steps` steps.
* Returns:

  * `t_grid` – time grid of shape `(n_steps + 1,)`,
  * `paths` – simulated paths of shape `(n_paths, n_steps + 1)`.

### `estimate_ou_from_series(x, dt)`

* Estimates OU parameters $(\kappa, \theta, \sigma)$ from a 1D time series `x`
  using the AR(1) regression view.
* Returns an `OUParams` instance with the estimated parameters and `x0 = x[0]`.

Planned notebooks:

* `notebooks/ou_pairs_trading_demo.ipynb` (planned) – estimate OU parameters from
  a real spread (e.g. two equities), construct a z-score mean-reversion strategy,
  and backtest PnL over time.

Unit tests (to be added) will live in:

```text
tests/test_ou_model.py
```

to verify that:

* simulated paths have the correct mean/variance behaviour, and
* calibration approximately recovers known parameters on synthetic data.

---

## 6. References

* Uhlenbeck, G. E., and Ornstein, L. S. (1930),
  *On the Theory of the Brownian Motion*, Physical Review.

* Many quantitative finance texts (e.g. interest-rate modelling, statistical arbitrage)
  discuss the OU process as a canonical mean-reverting model.


