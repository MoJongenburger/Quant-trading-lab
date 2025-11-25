"""
Tests for Black–Scholes Greeks.

We check:
- Basic sign / range properties for delta, gamma, vega, rho
- Consistency with pricing via finite differences
"""

import math

import numpy as np

from quant_trading_lab.models.black_scholes.pricing import (
    black_scholes_call,
    black_scholes_put,
)
from quant_trading_lab.models.black_scholes.greeks import (
    delta,
    gamma,
    vega,
    theta,
    rho,
)

S0 = 100.0
K = 100.0
T = 1.0
r = 0.02
sigma = 0.2


def test_call_delta_between_0_and_1():
    """
    Call delta should be in (0, 1) for non-degenerate case.
    """
    d = delta(S0, K, T, r, sigma, option_type="call")
    assert 0.0 < d < 1.0


def test_put_delta_between_minus1_and_0():
    """
    Put delta should be in (-1, 0) for non-degenerate case.
    """
    d = delta(S0, K, T, r, sigma, option_type="put")
    assert -1.0 < d < 0.0


def test_delta_put_call_parity_relation():
    """
    From put–call parity C - P = S0 - K e^{-rT},
    differentiating w.r.t S0 implies:

        delta_call - delta_put = 1
    """
    d_call = delta(S0, K, T, r, sigma, option_type="call")
    d_put = delta(S0, K, T, r, sigma, option_type="put")
    assert abs((d_call - d_put) - 1.0) < 1e-10


def test_gamma_positive():
    """
    Gamma should be strictly positive for non-degenerate options.
    """
    g = gamma(S0, K, T, r, sigma)
    assert g > 0.0


def test_vega_positive():
    """
    Vega should be strictly positive (price increases with volatility).
    """
    v = vega(S0, K, T, r, sigma)
    assert v > 0.0


def test_rho_signs():
    """
    For calls, rho > 0 (higher rates -> higher call value).
    For puts, rho < 0 (higher rates -> lower put value, all else equal).
    """
    rho_call = rho(S0, K, T, r, sigma, option_type="call")
    rho_put = rho(S0, K, T, r, sigma, option_type="put")

    assert rho_call > 0.0
    assert rho_put < 0.0


def test_call_theta_negative_atm():
    """
    For an at-the-money call (no dividends), theta should be negative
    in the simple Black–Scholes world: time decay hurts the holder.
    """
    th = theta(S0, K, T, r, sigma, option_type="call")
    assert th < 0.0


def test_delta_matches_finite_difference():
    """
    Delta ≈ dPrice/dS using a central finite difference on the call price.
    """
    h = 0.01 * S0  # 1% bump
    C_up = black_scholes_call(S0 + h, K, T, r, sigma)
    C_down = black_scholes_call(S0 - h, K, T, r, sigma)
    fd_delta = (C_up - C_down) / (2.0 * h)

    d = delta(S0, K, T, r, sigma, option_type="call")
    assert abs(fd_delta - d) < 1e-4


def test_gamma_matches_finite_difference():
    """
    Gamma ≈ second derivative of price w.r.t S using central finite difference:

        Gamma ≈ (C(S+h) - 2C(S) + C(S-h)) / h^2
    """
    h = 0.01 * S0  # 1% bump
    C_up = black_scholes_call(S0 + h, K, T, r, sigma)
    C_mid = black_scholes_call(S0, K, T, r, sigma)
    C_down = black_scholes_call(S0 - h, K, T, r, sigma)
    fd_gamma = (C_up - 2.0 * C_mid + C_down) / (h * h)

    g = gamma(S0, K, T, r, sigma)
    assert abs(fd_gamma - g) < 1e-4
