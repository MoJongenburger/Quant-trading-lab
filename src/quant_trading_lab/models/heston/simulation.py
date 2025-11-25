"""
Simulation of the Heston stochastic volatility model.

We use a full-truncation Euler scheme to keep the variance non-negative.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from .params import HestonParams


def simulate_heston_paths(
    S0: float,
    params: HestonParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate joint paths of (S_t, v_t) under the Heston model using a
    full-truncation Euler scheme.

    Parameters
    ----------
    S0 : float
        Initial asset price S(0).
    params : HestonParams
        Heston model parameters (kappa, theta, sigma, rho, v0, r).
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps in [0, T].
    n_paths : int
        Number of Monte Carlo paths to simulate.
    seed : int, optional
        Random seed for reproducibility (ignored if rng is provided).
    rng : np.random.Generator, optional
        NumPy random number generator. If None, a new Generator is created
        from the given seed.

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
        Time grid from 0 to T.
    S_paths : ndarray, shape (n_paths, n_steps + 1)
        Simulated asset price paths.
    v_paths : ndarray, shape (n_paths, n_steps + 1)
        Simulated variance paths.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma
    rho = params.rho
    v0 = params.v0
    r = params.r

    times = np.linspace(0.0, T, n_steps + 1)

    S_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    v_paths = np.empty((n_paths, n_steps + 1), dtype=float)

    S_paths[:, 0] = S0
    v_paths[:, 0] = v0

    for t in range(n_steps):
        # Draw independent standard normals
        Z1 = rng.standard_normal(size=n_paths)
        Z_perp = rng.standard_normal(size=n_paths)

        # Correlated shocks
        Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * Z_perp

        v_prev = v_paths[:, t]
        S_prev = S_paths[:, t]

        v_pos = np.maximum(v_prev, 0.0)

        # Variance update (full truncation Euler)
        v_next = (
            v_prev
            + kappa * (theta - v_pos) * dt
            + sigma * np.sqrt(v_pos) * sqrt_dt * Z2
        )
        v_next = np.maximum(v_next, 0.0)

        # Price update
        S_next = S_prev * np.exp(
            (r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1
        )

        v_paths[:, t + 1] = v_next
        S_paths[:, t + 1] = S_next

    return times, S_paths, v_paths
