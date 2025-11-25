"""
Black–Scholes closed-form pricing and implied volatility.

This module provides:
- European call and put pricing
- Put–call parity helpers
- A simple implied volatility solver for calls (and puts)
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Literal, Optional

import numpy as np


def _normal_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """
    Standard normal probability density function.
    """
    x = np.asarray(x, dtype=float)
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)


def _normal_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """
    Standard normal cumulative distribution function.
    """
    from math import erf

    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _d1_d2(S0: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """
    Compute d1 and d2 for the Black–Scholes formula.

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility of the underlying.

    Returns
    -------
    d1 : float
    d2 : float
    """
    if T <= 0.0 or sigma <= 0.0:
        # edge cases handled at higher level; here just avoid division by zero
        return np.inf, np.inf

    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def black_scholes_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black–Scholes price of a European call option.

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility of the underlying.

    Returns
    -------
    float
        Theoretical call option price in the Black–Scholes model.
    """
    if T <= 0.0 or sigma <= 0.0:
        return max(S0 - K, 0.0)

    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    Nd1 = _normal_cdf(d1)
    Nd2 = _normal_cdf(d2)
    return float(S0 * Nd1 - K * exp(-r * T) * Nd2)


def black_scholes_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black–Scholes price of a European put option.

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility of the underlying.

    Returns
    -------
    float
        Theoretical put option price in the Black–Scholes model.
    """
    if T <= 0.0 or sigma <= 0.0:
        return max(K - S0, 0.0)

    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    Nd1_neg = _normal_cdf(-d1)
    Nd2_neg = _normal_cdf(-d2)
    return float(K * exp(-r * T) * Nd2_neg - S0 * Nd1_neg)


def put_from_call(
    call_price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
) -> float:
    """
    Recover put price from call price via put–call parity.

    P = C - S0 + K e^{-rT}
    """
    return float(call_price - S0 + K * exp(-r * T))


def call_from_put(
    put_price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
) -> float:
    """
    Recover call price from put price via put–call parity.

    C = P + S0 - K e^{-rT}
    """
    return float(put_price + S0 - K * exp(-r * T))


def implied_vol_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    call_price: float,
    sigma_lower: float = 1e-6,
    sigma_upper: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute Black–Scholes implied volatility for a European call using bisection.

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    call_price : float
        Observed market price of the call.
    sigma_lower : float, optional
        Lower bound of the volatility search interval.
    sigma_upper : float, optional
        Upper bound of the volatility search interval.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of bisection iterations.

    Returns
    -------
    float
        Implied volatility sigma such that
        black_scholes_call(S0, K, T, r, sigma) ≈ call_price.

    Raises
    ------
    ValueError
        If the market price is outside the no-arbitrage bounds
        given the search interval for volatility.
    """
    # Simple intrinsic value bounds
    intrinsic = max(S0 - K * exp(-r * T), 0.0)
    if call_price < intrinsic:
        raise ValueError("Call price below intrinsic value; no implied vol.")

    # Values at bounds
    price_low = black_scholes_call(S0, K, T, r, sigma_lower)
    price_high = black_scholes_call(S0, K, T, r, sigma_upper)

    if not (price_low <= call_price <= price_high):
        raise ValueError(
            "Call price is outside the achievable range "
            "for the given volatility bounds."
        )

    a = sigma_lower
    b = sigma_upper

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        price_mid = black_scholes_call(S0, K, T, r, mid)

        if abs(price_mid - call_price) < tol:
            return float(mid)

        if price_mid < call_price:
            a = mid
        else:
            b = mid

    # If not converged, return midpoint with a warning-style behavior
    return float(0.5 * (a + b))


def implied_vol_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    put_price: float,
    sigma_lower: float = 1e-6,
    sigma_upper: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute Black–Scholes implied volatility for a European put.

    Internally converts the put to an equivalent call via put–call parity
    and then applies `implied_vol_call`.

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    put_price : float
        Observed market price of the put.
    sigma_lower, sigma_upper, tol, max_iter :
        See `implied_vol_call`.

    Returns
    -------
    float
        Implied volatility sigma.
    """
    call_equiv = call_from_put(put_price, S0, K, T, r)
    return implied_vol_call(
        S0=S0,
        K=K,
        T=T,
        r=r,
        call_price=call_equiv,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        tol=tol,
        max_iter=max_iter,
    )
