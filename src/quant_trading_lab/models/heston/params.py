"""
Parameter definitions for the Heston stochastic volatility model.
"""

from dataclasses import dataclass


@dataclass
class HestonParams:
    """
    Parameters of the Heston stochastic volatility model.

    Under the risk-neutral measure Q, the Heston model is

        dS_t = r S_t dt + sqrt(v_t) S_t dW_1(t)
        dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_2(t)

    with corr(dW_1, dW_2) = rho.

    Attributes
    ----------
    kappa : float
        Speed of mean reversion of the variance.
    theta : float
        Long-run mean of the variance.
    sigma : float
        Volatility of variance ("vol-of-vol").
    rho : float
        Correlation between the price and variance Brownian motions.
    v0 : float
        Initial variance v(0).
    r : float
        Risk-free interest rate.
    """

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    r: float


@dataclass
class OptionParams:
    """
    Parameters of a European option.

    Attributes
    ----------
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    """

    K: float
    T: float
