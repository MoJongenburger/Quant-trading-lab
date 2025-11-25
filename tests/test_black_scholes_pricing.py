"""
Basic tests for Black–Scholes pricing and implied volatility.

These tests are designed to be:
- Simple
- Deterministic (no randomness)
- Focused on key properties a quant would care about
"""

import math

import numpy as np

from quant_trading_lab.models.black_scholes.pricing import (
    black_scholes_call,
    black_scholes_put,
    implied_vol_call,
    implied_vol_put,
    put_from_call,
    call_from_put,
)


# Common test parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.02
sigma = 0.2
TOL = 1e-8


def test_put_call_parity():
    """
    Check that call and put satisfy put–call parity:

        C - P = S0 - K e^{-rT}
    """
    C = black_scholes_call(S0, K, T, r, sigma)
    P = black_scholes_put(S0, K, T, r, sigma)

    lhs = C - P
    rhs = S0 - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10


def test_put_call_parity_helpers():
    """
    Check that put_from_call and call_from_put are consistent with parity.
    """
    C = black_scholes_call(S0, K, T, r, sigma)
    P = black_scholes_put(S0, K, T, r, sigma)

    P_from_C = put_from_call(C, S0, K, T, r)
    C_from_P = call_from_put(P, S0, K, T, r)

    assert abs(P_from_C - P) < 1e-10
    assert abs(C_from_P - C) < 1e-10


def test_call_price_monotone_in_vol():
    """
    For fixed S0, K, T, r, the call price should increase with volatility.
    """
    sigmas = [0.05, 0.1, 0.2, 0.4]
    prices = [black_scholes_call(S0, K, T, r, s) for s in sigmas]

    # Each next price should be >= previous price
    for p_low, p_high in zip(prices[:-1], prices[1:]):
        assert p_high >= p_low - 1e-12  # small numerical tolerance


def test_call_price_bounds():
    """
    Basic no-arbitrage bounds for a call:

        max(S0 - K e^{-rT}, 0) <= C <= S0
    """
    C = black_scholes_call(S0, K, T, r, sigma)
    lower_bound = max(S0 - K * math.exp(-r * T), 0.0)
    upper_bound = S0

    assert lower_bound - 1e-12 <= C <= upper_bound + 1e-12


def test_implied_vol_call_recovers_sigma():
    """
    Implied vol should recover the original sigma used to price the call.
    """
    C = black_scholes_call(S0, K, T, r, sigma)
    sigma_impl = implied_vol_call(
        S0=S0,
        K=K,
        T=T,
        r=r,
        call_price=C,
    )
    assert abs(sigma_impl - sigma) < 1e-4


def test_implied_vol_put_recovers_sigma():
    """
    Implied vol for the put should also recover the original sigma.
    """
    P = black_scholes_put(S0, K, T, r, sigma)
    sigma_impl = implied_vol_put(
        S0=S0,
        K=K,
        T=T,
        r=r,
        put_price=P,
    )
    assert abs(sigma_impl - sigma) < 1e-4
