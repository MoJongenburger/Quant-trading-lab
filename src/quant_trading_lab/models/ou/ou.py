from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Optional, Tuple

import numpy as np


@dataclass
class OUParams:
    """
    Parameters for the Ornstein–Uhlenbeck (OU) mean-reversion process.

    Continuous-time SDE:
        dX_t = kappa * (theta - X_t) dt + sigma dW_t

    where
    -------
    kappa > 0 : speed of mean reversion
    theta     : long-run mean level
    sigma > 0 : volatility parameter
    x0        : initial level X_0
    """

    kappa: float
    theta: float
    sigma: float
    x0: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive.")


def ou_exact_step(
    x_t: np.ndarray | float,
    dt: float,
    params: OUParams,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Single exact Euler step for the OU process over interval dt.

    Exact discretisation:
        X_{t+dt} = theta + (X_t - theta) * e^{-kappa dt}
                   + sigma * sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) * Z,

    where Z ~ N(0, 1).

    Parameters
    ----------
    x_t : float or np.ndarray
        Current state(s) X_t.
    dt : float
        Time increment (in years).
    params : OUParams
        OU parameters (kappa, theta, sigma, x0).
    rng : np.random.Generator, optional
        Random number generator. If None, uses default RNG.

    Returns
    -------
    np.ndarray
        Next state(s) X_{t+dt} with the same shape as x_t.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    x_t_arr = np.asarray(x_t, dtype=float)

    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma

    exp_term = exp(-kappa * dt)
    mean = theta + (x_t_arr - theta) * exp_term
    var = (sigma**2) * (1.0 - exp(-2.0 * kappa * dt)) / (2.0 * kappa)

    z = rng.normal(loc=0.0, scale=1.0, size=x_t_arr.shape)
    x_next = mean + np.sqrt(var) * z
    return x_next


def simulate_ou_paths(
    params: OUParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate OU paths using the exact discretisation.

    Parameters
    ----------
    params : OUParams
        OU parameters (kappa, theta, sigma, x0).
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int, default 1
        Number of independent paths to simulate.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    t_grid : np.ndarray, shape (n_steps + 1,)
        Time grid from 0 to T.
    paths : np.ndarray, shape (n_paths, n_steps + 1)
        Simulated OU paths, each row corresponds to one path.
    """
    if T <= 0.0:
        raise ValueError("T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = params.x0

    x_t = paths[:, 0]
    for i in range(1, n_steps + 1):
        x_t = ou_exact_step(x_t, dt=dt, params=params, rng=rng)
        paths[:, i] = x_t

    return t_grid, paths


def estimate_ou_from_series(
    x: np.ndarray,
    dt: float,
) -> OUParams:
    """
    Estimate OU parameters (kappa, theta, sigma) from a time series using
    an AR(1) regression interpretation.

    Discrete-time approximation:
        X_{t+1} = a + b * X_t + epsilon_t,
    where
        b = exp(-kappa * dt),
        a = theta * (1 - b).

    We estimate a, b by OLS on (X_t, X_{t+1}) data and then back out:
        kappa = - (1/dt) * ln(b),
        theta = a / (1 - b),
        sigma from residual variance:
            Var(epsilon) = sigma^2 * (1 - e^{-2 kappa dt}) / (2 kappa).

    Parameters
    ----------
    x : np.ndarray
        1D array of observations X_0, X_1, ..., X_T at fixed spacing dt.
    dt : float
        Time step between observations (in years).

    Returns
    -------
    OUParams
        Estimated OU parameters with x0 set to the first observation.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array of observations.")
    if len(x) < 3:
        raise ValueError("Need at least 3 observations to estimate OU parameters.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    X_t = x[:-1]
    X_next = x[1:]
    n = len(X_t)

    # OLS for AR(1): X_next = a + b * X_t + error
    X_design = np.vstack([np.ones_like(X_t), X_t]).T
    # Beta_hat = (X^T X)^{-1} X^T y
    beta_hat, _, _, _ = np.linalg.lstsq(X_design, X_next, rcond=None)
    a_hat, b_hat = beta_hat

    if not (0.0 < b_hat < 1.0):
        # In some cases (e.g. extremely weak/negative mean reversion) b_hat
        # may fall outside (0, 1). We still try to estimate, but warn user.
        # For documentation: kappa will become small or negative.
        # Here we allow it but could clamp/log if needed.
        pass

    # Back out continuous-time parameters
    kappa_hat = -log(b_hat) / dt
    theta_hat = a_hat / (1.0 - b_hat)

    # Residuals and variance of epsilon
    eps_hat = X_next - (a_hat + b_hat * X_t)
    var_eps_hat = float(np.sum(eps_hat**2) / (n - 2))  # unbiased-ish

    # var_eps = sigma^2 * (1 - e^{-2 kappa dt}) / (2 kappa)
    denom = (1.0 - exp(-2.0 * kappa_hat * dt)) / (2.0 * kappa_hat)
    sigma_hat = np.sqrt(max(var_eps_hat / denom, 1e-16))

    return OUParams(
        kappa=float(kappa_hat),
        theta=float(theta_hat),
        sigma=float(sigma_hat),
        x0=float(x[0]),
    )
