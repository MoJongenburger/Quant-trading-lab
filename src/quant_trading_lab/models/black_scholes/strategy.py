"""
Delta-hedging strategy under the Black–Scholes model.

This module provides:

- A simple geometric Brownian motion (GBM) simulator for the underlying
- A discrete-time delta-hedging simulation for a European call option

The main idea:

- Start short 1 call at its Black–Scholes price
- Use Black–Scholes delta to hedge with the underlying
- Rebalance delta along a price path (simulated or real)
- Track replicating portfolio value vs theoretical option price
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Optional

import numpy as np

from .pricing import black_scholes_call
from .greeks import delta as bs_delta


@dataclass
class DeltaHedgeResult:
    """
    Container for delta-hedging simulation results.
    """
    times: np.ndarray                 # shape (n_steps+1,)
    S: np.ndarray                     # underlying prices
    option_prices: np.ndarray         # theoretical BS call prices over time
    deltas: np.ndarray                # hedge ratios over time
    cash: np.ndarray                  # cash account over time
    replicating_values: np.ndarray    # cash + delta * S
    hedging_errors: np.ndarray        # replicating_values - option_prices


def simulate_gbm_path(
    S0: float,
    T: float,
    n_steps: int,
    mu: float,
    sigma: float,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single geometric Brownian motion (GBM) path:

        dS_t = mu S_t dt + sigma S_t dW_t

    Parameters
    ----------
    S0 : float
        Initial price.
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    mu : float
        Drift (can be r under risk-neutral measure, or something else).
    sigma : float
        Volatility.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    times : np.ndarray
        Array of times of shape (n_steps + 1,).
    S : np.ndarray
        Simulated price path of shape (n_steps + 1,).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    times = np.linspace(0.0, T, n_steps + 1)
    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S0

    for i in range(n_steps):
        z = rng.standard_normal()
        S[i + 1] = S[i] * np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z)

    return times, S


def delta_hedge_call_path(
    S: np.ndarray,
    times: np.ndarray,
    K: float,
    r: float,
    sigma: float,
) -> DeltaHedgeResult:
    """
    Run a discrete-time delta-hedging simulation for a European call
    along a given price path.

    We assume:
    - We are SHORT 1 call at t=0
    - We receive the Black–Scholes call premium at t=0
    - We hedge by holding delta_t units of the underlying

    The replicating portfolio is:

        V_t = cash_t + delta_t * S_t

    We compare V_t to the theoretical Black–Scholes price C_t and define
    the **hedging error** as:

        error_t = V_t - C_t

    Parameters
    ----------
    S : np.ndarray
        Underlying price path, shape (n_steps + 1,).
    times : np.ndarray
        Corresponding times in years, shape (n_steps + 1,).
        Must be increasing and end at maturity T.
    K : float
        Strike of the European call.
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Volatility (Black–Scholes).

    Returns
    -------
    DeltaHedgeResult
        Object with times, S, option prices, deltas, cash, replicating
        portfolio values, and hedging errors.
    """
    S = np.asarray(S, dtype=float)
    times = np.asarray(times, dtype=float)

    if S.shape != times.shape:
        raise ValueError("S and times must have the same shape.")

    n_points = S.shape[0]
    if n_points < 2:
        raise ValueError("Need at least two time points for a path.")

    T = float(times[-1])

    option_prices = np.zeros(n_points, dtype=float)
    deltas = np.zeros(n_points, dtype=float)
    cash = np.zeros(n_points, dtype=float)
    replicating_values = np.zeros(n_points, dtype=float)
    hedging_errors = np.zeros(n_points, dtype=float)

    # --- t = 0: initialize ---
    S0 = S[0]
    tau0 = max(T - times[0], 0.0)

    C0 = black_scholes_call(S0, K, tau0, r, sigma)
    option_prices[0] = C0

    # Initial delta
    delta0 = 0.0
    if tau0 > 0.0 and sigma > 0.0:
        delta0 = bs_delta(S0, K, tau0, r, sigma, option_type="call")
    deltas[0] = delta0

    # Short 1 call -> receive C0
    # Buy delta0 shares for hedging -> pay delta0 * S0
    # Remaining goes to cash
    cash[0] = C0 - delta0 * S0

    replicating_values[0] = cash[0] + delta0 * S0
    hedging_errors[0] = replicating_values[0] - C0

    # --- t > 0: rebalance delta and accrue interest ---
    for i in range(1, n_points):
        dt = times[i] - times[i - 1]
        if dt < 0.0:
            raise ValueError("times must be increasing.")

        # Accrue interest on cash
        cash[i] = cash[i - 1] * exp(r * dt)

        St = S[i]
        tau = max(T - times[i], 0.0)

        if tau > 0.0 and sigma > 0.0:
            Ct = black_scholes_call(St, K, tau, r, sigma)
            delt = bs_delta(St, K, tau, r, sigma, option_type="call")
        else:
            # At maturity: option price = payoff; delta ~ indicator{S_T > K}
            Ct = max(St - K, 0.0)
            delt = 1.0 if St > K else 0.0

        option_prices[i] = Ct

        # Rebalance: change in delta
        delta_prev = deltas[i - 1]
        delta_change = delt - delta_prev

        # Buy/sell delta_change shares at current price
        # If delta_change > 0 -> buy shares -> cash decreases
        # If delta_change < 0 -> sell shares -> cash increases
        cash[i] -= delta_change * St
        deltas[i] = delt

        replicating_values[i] = cash[i] + delt * St
        hedging_errors[i] = replicating_values[i] - Ct

    return DeltaHedgeResult(
        times=times,
        S=S,
        option_prices=option_prices,
        deltas=deltas,
        cash=cash,
        replicating_values=replicating_values,
        hedging_errors=hedging_errors,
    )


if __name__ == "__main__":
    # Small demonstration when running this module directly.
    # This won't run in your tests, but is handy for quick local checks.
    S0_demo = 100.0
    K_demo = 100.0
    T_demo = 1.0
    r_demo = 0.02
    sigma_demo = 0.2

    times_demo, S_demo = simulate_gbm_path(
        S0=S0_demo,
        T=T_demo,
        n_steps=252,
        mu=r_demo,          # risk-neutral drift
        sigma=sigma_demo,
        seed=42,
    )

    result = delta_hedge_call_path(
        S=S_demo,
        times=times_demo,
        K=K_demo,
        r=r_demo,
        sigma=sigma_demo,
    )

    print("Delta-hedging demo (GBM path):")
    print(f"Final replicating value: {result.replicating_values[-1]:.4f}")
    print(f"Final option price     : {result.option_prices[-1]:.4f}")
    print(f"Final hedging error    : {result.hedging_errors[-1]:.4f}")
