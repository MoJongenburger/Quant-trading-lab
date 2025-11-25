"""
Trading strategy based on Heston vs Black–Scholes mispricing.

We assume:
- The *market* quotes Black–Scholes prices with a constant volatility sigma_mkt.
- We (the model user) have a Heston view of the world.

At each time step along a simulated Heston path:
- We compute the option price under Heston (via Monte Carlo) and under Black–Scholes.
- We measure relative mispricing m = (C_heston - C_bs) / C_bs.
- We generate a buy/sell/hold signal based on a threshold on m.
- We mark PnL using the Black–Scholes price as the "market" price.
"""

from __future__ import annotations

from typing import Literal, Optional, Dict

import numpy as np

from .params import HestonParams, OptionParams
from .pricing import price_european_call_heston_mc, black_scholes_call
from .simulation import simulate_heston_paths

Signal = Literal["buy", "sell", "hold"]


def compute_mispricing(
    S: float,
    v: float,
    option: OptionParams,
    params: HestonParams,
    sigma_mkt: float,
    mc_steps: int = 80,
    mc_paths: int = 5_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """
    Compute Heston vs Black–Scholes mispricing from the current state (S, v).

    Parameters
    ----------
    S : float
        Current spot price S_t.
    v : float
        Current variance v_t.
    option : OptionParams
        Option parameters at this time (strike, remaining maturity).
    params : HestonParams
        Base Heston parameters.
    sigma_mkt : float
        Market (Black–Scholes) volatility.
    mc_steps : int, optional
        Number of time steps for Heston Monte Carlo pricing.
    mc_paths : int, optional
        Number of Monte Carlo paths.
    rng : np.random.Generator, optional
        Random number generator for the Monte Carlo simulation.

    Returns
    -------
    C_heston : float
        Heston Monte Carlo call price from state (S, v).
    C_bs : float
        Black–Scholes call price with volatility sigma_mkt.
    mispricing : float
        Relative mispricing (C_heston - C_bs) / C_bs (0.0 if C_bs is zero).
    """
    # We reuse the same Heston parameters but reset v0 to the current variance v_t.
    local_params = HestonParams(
        kappa=params.kappa,
        theta=params.theta,
        sigma=params.sigma,
        rho=params.rho,
        v0=v,
        r=params.r,
    )

    C_heston = price_european_call_heston_mc(
        S0=S,
        option=option,
        params=local_params,
        n_steps=mc_steps,
        n_paths=mc_paths,
        rng=rng,
    )

    C_bs = black_scholes_call(
        S0=S,
        option=option,
        r=params.r,
        sigma=sigma_mkt,
    )

    if C_bs <= 0.0:
        mispricing = 0.0
    else:
        mispricing = (C_heston - C_bs) / C_bs

    return C_heston, C_bs, mispricing


def generate_signal(mispricing: float, rel_threshold: float) -> Signal:
    """
    Convert a mispricing value into a discrete trading signal.

    Parameters
    ----------
    mispricing : float
        Relative mispricing (C_heston - C_bs) / C_bs.
    rel_threshold : float
        Threshold for taking action (e.g. 0.05 = 5%).

    Returns
    -------
    signal : {"buy", "sell", "hold"}
        Trading decision.
    """
    if mispricing > rel_threshold:
        return "buy"
    if mispricing < -rel_threshold:
        return "sell"
    return "hold"


def run_mispricing_strategy_backtest(
    S0: float,
    option: OptionParams,
    params: HestonParams,
    sigma_mkt: float,
    rel_threshold: float = 0.05,
    n_steps_path: int = 50,
    seed_path: Optional[int] = 123,
    mc_steps_per_node: int = 40,
    mc_paths_per_node: int = 3_000,
) -> Dict[str, np.ndarray]:
    """
    Run a simple mispricing-based strategy along a single simulated Heston path.

    We:
    - Simulate one "realized" Heston path for (S_t, v_t) from t=0 to t=T.
    - At each time step, compute Heston and BS prices for the remaining maturity.
    - Generate buy/sell/hold signals from the mispricing.
    - Trade at the Black–Scholes price and track the portfolio value.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    option : OptionParams
        Option with initial maturity T and strike K. The maturity is assumed to
        decrease linearly as time evolves along the path.
    params : HestonParams
        Heston model parameters.
    sigma_mkt : float
        Market volatility used in Black–Scholes pricing.
    rel_threshold : float, optional
        Relative mispricing threshold for taking trades.
    n_steps_path : int, optional
        Number of time steps in the simulated Heston path.
    seed_path : int, optional
        Random seed for the outer Heston path (the "realized" world).
    mc_steps_per_node : int, optional
        Time steps used in the inner Monte Carlo pricing at each node.
    mc_paths_per_node : int, optional
        Number of Monte Carlo paths used in the inner Heston pricing at each node.

    Returns
    -------
    results : dict of str -> ndarray
        Dictionary containing:
        - "times": time grid
        - "S": spot prices along the path
        - "v": variances along the path
        - "bs_prices": Black–Scholes prices along the path
        - "heston_prices": Heston prices along the path
        - "mispricings": mispricing series
        - "signals": integer-encoded signals (+1 buy, -1 sell, 0 hold)
        - "positions": option positions over time
        - "portfolio_values": portfolio value over time
    """
    T = option.T

    # Outer path: one "true" Heston path
    rng_path = np.random.default_rng(seed_path)
    times, S_paths, v_paths = simulate_heston_paths(
        S0=S0,
        params=params,
        T=T,
        n_steps=n_steps_path,
        n_paths=1,
        rng=rng_path,
    )
    S = S_paths[0]
    v = v_paths[0]

    n_points = n_steps_path + 1

    bs_prices = np.zeros(n_points)
    heston_prices = np.zeros(n_points)
    mispricings = np.zeros(n_points)
    signals = np.zeros(n_points, dtype=int)
    positions = np.zeros(n_points, dtype=int)
    portfolio_values = np.zeros(n_points)

    cash = 0.0
    pos = 0

    # Separate RNG for inner pricing to keep things tidy
    rng_inner = np.random.default_rng(seed_path + 1 if seed_path is not None else None)

    for i, t in enumerate(times):
        # Remaining time to maturity
        tau = max(option.T - t, 0.0)
        option_t = OptionParams(K=option.K, T=tau)

        C_heston, C_bs, m = compute_mispricing(
            S=S[i],
            v=v[i],
            option=option_t,
            params=params,
            sigma_mkt=sigma_mkt,
            mc_steps=mc_steps_per_node,
            mc_paths=mc_paths_per_node,
            rng=rng_inner,
        )

        heston_prices[i] = C_heston
        bs_prices[i] = C_bs
        mispricings[i] = m

        sig = generate_signal(m, rel_threshold)
        if sig == "buy":
            target_pos = 1
            signals[i] = 1
        elif sig == "sell":
            target_pos = -1
            signals[i] = -1
        else:
            target_pos = 0
            signals[i] = 0

        # Trade towards target position at current BS price
        delta = target_pos - pos
        if delta != 0:
            # Buying options costs cash (delta > 0); selling generates cash (delta < 0)
            cash -= delta * C_bs
            pos = target_pos

        positions[i] = pos
        portfolio_values[i] = cash + pos * C_bs

    return {
        "times": times,
        "S": S,
        "v": v,
        "bs_prices": bs_prices,
        "heston_prices": heston_prices,
        "mispricings": mispricings,
        "signals": signals,
        "positions": positions,
        "portfolio_values": portfolio_values,
    }
