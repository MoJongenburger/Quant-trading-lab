import math

from quant_trading_lab.models.crr import (
    CRRParams,
    crr_price_european,
    crr_price_american,
)

# We use the Black–Scholes analytic formulas as a benchmark
from quant_trading_lab.models.black_scholes.pricing import (
    bs_call_price,
    bs_put_price,
)


def test_european_call_converges_to_black_scholes():
    """
    For large N the CRR European call price should be very close
    to the Black–Scholes price with the same parameters.
    """
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.20

    bs = bs_call_price(S0, K, T, r, sigma)

    params = CRRParams(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        N=1000,
        option_type="call",
    )
    crr = crr_price_european(params)

    # within 1 cent is more than good enough for a unit test
    assert abs(crr - bs) < 1e-2


def test_european_put_converges_to_black_scholes():
    """
    Same convergence check for a European put.
    """
    S0 = 95.0
    K = 100.0
    T = 1.0
    r = 0.03
    sigma = 0.25

    bs = bs_put_price(S0, K, T, r, sigma)

    params = CRRParams(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        N=1000,
        option_type="put",
    )
    crr = crr_price_european(params)

    assert abs(crr - bs) < 1e-2


def test_american_call_close_to_european_on_non_dividend_stock():
    """
    For a non-dividend-paying stock the American call should be
    worth (almost) the same as the European call.
    """
    params = CRRParams(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        N=500,
        option_type="call",
    )

    eur = crr_price_european(params)
    amer = crr_price_american(params)

    assert amer >= eur  # no-arbitrage
    # economically, they should be extremely close
    assert abs(amer - eur) < 1e-3


def test_american_put_at_least_european_put():
    """
    American put should never be worth less than the European put.
    Early exercise can add value, especially for deep ITM puts.
    """
    params = CRRParams(
        S0=80.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.25,
        N=500,
        option_type="put",
    )

    eur = crr_price_european(params)
    amer = crr_price_american(params)

    assert amer >= eur
