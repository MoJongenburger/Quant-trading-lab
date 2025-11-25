"""
Cox–Ross–Rubinstein (CRR) binomial option pricing model.

This module provides functions to price European and American calls and puts
on a non–dividend-paying underlying using the CRR binomial tree.

Typical usage
-------------
from quant_trading_lab.models.crr import (
    CRRParams,
    crr_price_european,
    crr_price_american,
)

params = CRRParams(
    S0=100.0,
    K=100.0,
    T=1.0,
    r=0.02,
    sigma=0.20,
    N=200,
    option_type="call",
)

price_eur = crr_price_european(params)
price_am  = crr_price_american(params)
"""

from .pricing import (
    CRRParams,
    crr_price_european,
    crr_price_american,
)

__all__ = [
    "CRRParams",
    "crr_price_european",
    "crr_price_american",
]
