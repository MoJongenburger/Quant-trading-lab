"""
Pricing routines for the Heston stochastic volatility model.

Includes:
- Monte Carlo pricing of European call options under Heston
- Black–Scholes call pricing as a benchmark
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .params import HestonParams, OptionParams
from .simulation import simulate_heston_paths

ArrayLike = Union[np.ndarray, float]


def _norm_cdf(x: ArrayLike) -> ArrayLike:
    """
    Standard normal cumulative distribution function using erf.

    N(x) = 0.5 * (1 + erf(x / sqrt(2))).
    """
    from math import erf, sqrt

    if isinstance(x, float):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    else:
        return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def black_scholes_call(
    S0: float,
    option: OptionParams,
    r: float,
    sigma: float,
) -> float:
    """
    Black–Scholes price of a European call option.

    Parameters
    ----------
    S0 : float
        Spot price.
    option : OptionParams
        Option parameters (strike K, maturity T).
    r : float
        Risk-free interest rate.
    sigma : float
        Constant volatility.

    Returns
    -------
    float
        Call option price under Black–Scholes.
    """
    from math import log, sqrt, exp

    K = option.K
    T = option.T

    if T <= 0.0 or sigma <= 0.0:
        return max(S0 - K, 0.0)

    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    return S0 * Nd1 - K * exp(-r * T) * Nd2


def price_european_call_heston_mc(
    S0: float,
    option: OptionParams,
    params: HestonParams,
    n_steps: int = 200,
    n_paths: int = 50_000,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Monte Carlo price of a European call option under the Heston model.

    Parameters
    ----------
    S0 : float
        Initial asset price S(0).
    option : OptionParams
        Option parameters (strike K, maturity T).
    params : HestonParams
        Heston model parameters.
    n_steps : int, optional
        Number of time steps for the Heston path simulation.
    n_paths : int, optional
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed (ignored if rng is provided).
    rng : np.random.Generator, optional
        NumPy random number generator.

    Returns
    -------
    float
        Estimated call option price under Heston.
    """
    _, S_paths, _ = simulate_heston_paths(
        S0=S0,
        params=params,
        T=option.T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        rng=rng,
    )

    ST = S_paths[:, -1]
    payoff = np.maximum(ST - option.K, 0.0)
    discount_factor = np.exp(-params.r * option.T)

    return float(discount_factor * np.mean(payoff))
