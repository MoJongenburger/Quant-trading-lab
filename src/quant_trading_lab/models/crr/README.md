# Cox–Ross–Rubinstein (CRR) Binomial Model

This module implements the **Cox–Ross–Rubinstein (CRR)** binomial tree for
option pricing. It can price European and American calls and puts on a
non–dividend-paying underlying, using a recombining binomial tree built
from volatility.

It is part of the broader `quant-trading-lab` project and complements the
continuous-time Black–Scholes model with a discrete-time, tree-based view.

---

## 1. Model overview

In the CRR model, the stock price $S_t$ evolves on a binomial tree:

- At each time step of length $\Delta t = T / N$:
  - The price moves **up** by a factor $u$
  - or **down** by a factor $d$

Under the **risk-neutral measure**, the one-step dynamics are:

- $S_{t+\Delta t} = S_t u$ with probability $p$
- $S_{t+\Delta t} = S_t d$ with probability $(1-p)$

The standard CRR parameterisation expresses $(u, d, p)$ in terms of volatility
$\sigma$ and risk-free rate $r$:

$$
\begin{aligned}
\Delta t &= T / N, \\\\
u &= e^{\sigma \sqrt{\Delta t}}, \\\\
d &= \frac{1}{u}, \\\\
p &= \frac{e^{r \Delta t} - d}{u - d}.
\end{aligned}
$$

with $0 \le p \le 1$.

---

## 2. European option pricing

For a European call or put, we:

1. **Build the stock price tree** at maturity $T$:

   $$
   S_{N,j} = S_0 u^j d^{N-j}, \quad j = 0, 1, \dots, N
   $$

2. Compute terminal **payoffs**:

   - Call: $\max(S_{N,j} - K, 0)$  
   - Put:  $\max(K - S_{N,j}, 0)$

3. Perform **backward induction** under the risk-neutral measure:

   $$
   V_{n,j} = e^{-r \Delta t} \Big( p V_{n+1,j+1} + (1-p) V_{n+1,j} \Big),
   $$

   until we reach $V_{0,0}$, the time–0 price.

In this case there is **no early exercise** – the option is held to maturity.

---

## 3. American option pricing

For American options, early exercise is allowed at each node.

During backward induction we replace the continuation value by:

$$
V_{n,j} = \max\Big(
    e^{-r \Delta t} (p V_{n+1,j+1} + (1-p) V_{n+1,j}),
    \text{intrinsic}(S_{n,j})
\Big)
$$

where $\text{intrinsic}(S_{n,j})$ is:

- Call: $\max(S_{n,j} - K, 0)$  
- Put:  $\max(K - S_{n,j}, 0)$

This yields the American option price at the root $V_{0,0}$.

For a **non–dividend-paying** stock, the American call price should be virtually
identical to the European call price (no early exercise premium). For puts,
early exercise can add value, especially for deep in-the-money contracts.

---

## 4. Implementation layout

Current CRR implementation:

```text
src/quant_trading_lab/models/crr/
├─ __init__.py   # public interface
└─ pricing.py    # CRRParams + European & American pricing
````

### `CRRParams`

Dataclass that encapsulates:

* `S0` – spot price
* `K` – strike
* `T` – maturity (years)
* `r` – risk-free rate (continuous compounding)
* `sigma` – volatility
* `N` – number of time steps
* `option_type` – `"call"` or `"put"`

It also performs basic validation (e.g. `N > 0`, `T > 0`, `sigma > 0`).

### `crr_price_european(params: CRRParams) -> float`

* Builds a CRR tree using the standard $(u, d, p)$ parameterisation.
* Performs risk-neutral backward induction **without** early exercise.
* Returns the European option price at time 0.

### `crr_price_american(params: CRRParams) -> float`

* Same tree and induction as above, but at each node:

  * compares the continuation value with the intrinsic value,
  * and takes the maximum (early exercise).

---

## 5. Tests and usage

Unit tests for the CRR model live in:

```text
tests/test_crr_pricing.py
```

They check that:

* European CRR prices converge to Black–Scholes prices as `N` becomes large.
* American call prices on non-dividend stocks are almost identical to
  European call prices.
* American put prices are never lower than European put prices.

Example usage:

```python
from quant_trading_lab.models.crr import CRRParams, crr_price_european

params = CRRParams(
    S0=100.0,
    K=100.0,
    T=1.0,
    r=0.02,
    sigma=0.20,
    N=200,
    option_type="call",
)

price = crr_price_european(params)
print(price)
```

