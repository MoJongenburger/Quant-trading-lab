"""
Ornstein–Uhlenbeck (OU) mean-reversion model.

This module provides:

- A parametric OU process specification (kappa, theta, sigma, x0)
- Simulation of OU paths using the exact discretisation
- Calibration of OU parameters from discrete-time observations via
  an AR(1) regression interpretation

Typical usage
-------------
from quant_trading_lab.models.ou import (
    OUParams,
    simulate_ou_paths,
    estimate_ou_from_series,
)

params = OUParams(kappa=1.5, theta=0.0, sigma=0.3, x0=0.0)
t, paths = simulate_ou_paths(params, T=1.0, n_steps=252, n_paths=10)

# Calibration example (e.g. from a spread time series)
estimated = estimate_ou_from_series(x_obs, dt=1/252)
"""

from .ou import (
    OUParams,
    simulate_ou_paths,
    ou_exact_step,
    estimate_ou_from_series,
)

__all__ = [
    "OUParams",
    "simulate_ou_paths",
    "ou_exact_step",
    "estimate_ou_from_series",
]
