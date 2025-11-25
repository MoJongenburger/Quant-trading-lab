"""
Greeks for Black–Scholes European options.

Provides:
- Delta (call/put)
- Gamma
- Vega
- Theta (call/put, simplified convention)
- Rho (call/put)
"""

from __future__ import annotations

from math import exp, sqrt
from typing import Literal

import numpy as np

from .pricing import _normal_pdf, _normal_cdf, _d1_d2


OptionType = Literal["call", "put"]


def delta(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """
    Black–Scholes delta for a European call or put.

    Parameters
    ----------
    S0, K, T, r, sigma : float
        Standard Black–Scholes inputs.
    option_type : {"call", "put"}
        Type of the option.

    Returns
    -------
    float
        Delta of the option with respect to the underlying price.
    """
    if T <= 0.0 or sigma <= 0.0:
        # At maturity, delta is 0 or 1 for calls, -1 or 0 for puts; we keep it simple:
        if option_type == "call":
            return 1.0 if S0 > K else 0.0
        else:
            return -1.0 if S0 < K else 0.0

    d1, _ = _d1_d2(S0, K, T, r, sigma)

    if option_type == "call":
        return float(_normal_cdf(d1))
    elif option_type == "put":
        return float(_normal_cdf(d1) - 1.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black–Scholes gamma for a European option (same for calls and puts).
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0

    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return float(_normal_pdf(d1) / (S0 * sigma * sqrt(T)))


def vega(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black–Scholes vega for a European option (same for calls and puts).

    Returns vega as derivative w.r.t. volatility (per unit of sigma, not per %).
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0

    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return float(S0 * _normal_pdf(d1) * sqrt(T))


def theta(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """
    Black–Scholes theta (time decay) for a European call or put.

    Convention: returns dPrice/dT (per year). Sign depends on option type.

    Parameters
    ----------
    S0, K, T, r, sigma : float
        Standard Black–Scholes inputs.
    option_type : {"call", "put"}
        Type of the option.

    Returns
    -------
    float
        Theta of the option.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0

    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    pdf_d1 = _normal_pdf(d1)

    first_term = - (S0 * pdf_d1 * sigma) / (2.0 * sqrt(T))

    if option_type == "call":
        second_term = -r * K * exp(-r * T) * _normal_cdf(d2)
    elif option_type == "put":
        second_term = r * K * exp(-r * T) * _normal_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(first_term + second_term)


def rho(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """
    Black–Scholes rho for a European call or put.

    Rho is the sensitivity of the option price to the risk-free rate.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0

    _, d2 = _d1_d2(S0, K, T, r, sigma)

    if option_type == "call":
        return float(K * T * exp(-r * T) * _normal_cdf(d2))
    elif option_type == "put":
        return float(-K * T * exp(-r * T) * _normal_cdf(-d2))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
