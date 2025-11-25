import os
import sys

# Make sure we can import from src/
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

import numpy as np

from quant_trading_lab.models.heston.params import HestonParams, OptionParams
from quant_trading_lab.models.heston.pricing import (
    price_european_call_heston_mc,
    black_scholes_call,
)


def test_heston_price_is_positive():
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
        v0=0.04,
        r=0.02,
    )
    option = OptionParams(K=100.0, T=1.0)
    S0 = 100.0

    price = price_european_call_heston_mc(
        S0=S0,
        option=option,
        params=params,
        n_steps=100,
        n_paths=5_000,
        seed=123,
    )

    assert price > 0.0


def test_black_scholes_increases_with_sigma():
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
        v0=0.04,
        r=0.02,
    )
    option = OptionParams(K=100.0, T=1.0)
    S0 = 100.0

    price_low = black_scholes_call(S0, option, r=params.r, sigma=0.1)
    price_high = black_scholes_call(S0, option, r=params.r, sigma=0.3)

    assert price_high > price_low
