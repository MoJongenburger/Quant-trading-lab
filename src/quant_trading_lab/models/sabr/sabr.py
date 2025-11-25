"""
Core SABR functions: Hagan lognormal implied volatility approximation.

This module implements the classic lognormal SABR model:

    dF_t = alpha_t F_t^beta dW_1
    dalpha_t = nu alpha_t dW_2
    dW_1 dW_2 = rho dt

and the corresponding Hagan et al. approximation for the
Black–Scholes implied volatility sigma_BS(F0, K, T).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


Number = Union[float, np.ndarray]


@dataclass
class SabrParams:
    """
    Parameters of the lognormal SABR model.

    Attributes
    ----------
    alpha : float
        Initial volatility level (alpha_0 > 0).
    beta : float
        Elasticity parameter in [0, 1].
    rho : float
        Correlation between forward and volatility in [-1, 1].
    nu : float
        Volatility of volatility (nu > 0).
    """
    alpha: float
    beta: float
    rho: float
    nu: float


def sabr_implied_vol_hagan(
    F0: float,
    K: Number,
    T: float,
    params: SabrParams,
    eps: float = 1e-07,
) -> Number:
    """
    Hagan et al. lognormal SABR implied volatility approximation.

    Parameters
    ----------
    F0 : float
        Forward price at time 0 (must be > 0).
    K : float or np.ndarray
        Strike(s) (must be > 0). Can be a scalar or vector.
    T : float
        Time to maturity in years (T >= 0).
    params : SabrParams
        SABR parameters (alpha, beta, rho, nu).
    eps : float, optional
        Small threshold to decide when F0 ~ K (ATM limit).

    Returns
    -------
    sigma_BS : float or np.ndarray
        Black–Scholes implied volatility according to the
        Hagan SABR approximation.

    Notes
    -----
    This is the lognormal SABR formula as in Hagan et al. (2002),
    "Managing Smile Risk". For strikes close to the forward (F0 ~ K),
    an ATM limit is used to avoid numerical issues.
    """
    alpha = float(params.alpha)
    beta = float(params.beta)
    rho = float(params.rho)
    nu = float(params.nu)

    if F0 <= 0.0:
        raise ValueError("F0 must be positive in lognormal SABR.")
    if T < 0.0:
        raise ValueError("T must be non-negative.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if nu <= 0.0:
        raise ValueError("nu must be positive.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1].")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")

    K = np.asarray(K, dtype=float)
    if np.any(K <= 0.0):
        raise ValueError("All strikes K must be positive in lognormal SABR.")

    one_minus_beta = 1.0 - beta

    # Handle ATM and non-ATM separately for numerical stability
    log_FK = np.log(F0 / K)
    is_atm = np.abs(log_FK) < eps
    is_not_atm = ~is_atm

    sigma = np.zeros_like(K, dtype=float)

    # --- ATM case (F0 ≈ K) ---
    if np.any(is_atm):
        # F0 ~ K => (F0 K)^((1 - beta)/2) ~ F0^(1 - beta)
        F_beta = F0 ** one_minus_beta

        # Leading term
        sigma_atm = alpha / F_beta

        # Time-dependent correction
        term1 = ((one_minus_beta ** 2) / 24.0) * (alpha ** 2) / (F_beta ** 2)
        term2 = 0.25 * rho * beta * nu * alpha / F_beta
        term3 = ((2.0 - 3.0 * rho * rho) / 24.0) * (nu ** 2)

        correction = 1.0 + T * (term1 + term2 + term3)

        sigma[is_atm] = sigma_atm * correction

    # --- Non-ATM case (F0 != K) ---
    if np.any(is_not_atm):
        K_ = K[is_not_atm]
        log_FK_ = log_FK[is_not_atm]

        FK_beta = (F0 * K_) ** (0.5 * one_minus_beta)
        z = (nu / alpha) * FK_beta * log_FK_

        # x(z) term
        sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
        numerator = sqrt_term + z - rho
        denominator = 1.0 - rho
        x_z = np.log(numerator / denominator)

        # A(F0, K) in Hagan formula
        A = alpha / (FK_beta * (1.0 + ((one_minus_beta ** 2) / 24.0) * (log_FK_ ** 2)
                                + ((one_minus_beta ** 4) / 1920.0) * (log_FK_ ** 4)))

        # z / x(z)
        zx = z / x_z

        # Time-dependent correction B(F0, K)
        term1 = ((one_minus_beta ** 2) / 24.0) * (alpha ** 2) / (FK_beta ** 2)
        term2 = 0.25 * rho * beta * nu * alpha / FK_beta
        term3 = ((2.0 - 3.0 * rho * rho) / 24.0) * (nu ** 2)
        B = 1.0 + T * (term1 + term2 + term3)

        sigma_non_atm = A * zx * B
        sigma[is_not_atm] = sigma_non_atm

    # If K was scalar, return scalar
    if sigma.size == 1:
        return float(sigma[0])
    return sigma
