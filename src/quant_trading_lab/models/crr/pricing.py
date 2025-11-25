from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal


OptionType = Literal["call", "put"]


@dataclass
class CRRParams:
    """
    Parameters for Cox–Ross–Rubinstein (CRR) binomial option pricing.

    This dataclass assumes the standard CRR parameterisation where the
    up/down factors are derived from volatility `sigma`:

        dt = T / N
        u  = exp(sigma * sqrt(dt))
        d  = 1 / u
        p  = (exp(r * dt) - d) / (u - d)

    Attributes
    ----------
    S0 : float
        Spot price at time 0.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Volatility of the underlying.
    N : int
        Number of time steps in the binomial tree (N >= 1).
    option_type : {"call", "put"}
        Payoff type. Defaults to "call".
    """

    S0: float
    K: float
    T: float
    r: float
    sigma: float
    N: int
    option_type: OptionType = "call"

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("N (number of steps) must be positive.")
        if self.T <= 0.0:
            raise ValueError("T (maturity) must be positive.")
        if self.sigma <= 0.0:
            raise ValueError("sigma (volatility) must be positive.")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")


def _crr_ud_p(r: float, sigma: float, T: float, N: int) -> tuple[float, float, float, float]:
    """
    Compute (dt, u, d, p) for the standard CRR binomial tree.

    Parameters
    ----------
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    N : int
        Number of steps.

    Returns
    -------
    dt : float
    u : float
    d : float
    p : float
        Risk-neutral probability of an up move.
    """
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    disc = exp(r * dt)
    denom = u - d
    if denom == 0.0:
        raise ValueError("u and d coincide; invalid CRR parameters.")
    p = (disc - d) / denom
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Risk-neutral probability out of [0,1]: p={p:.4f}")
    return dt, u, d, p


def _payoff(spot: float, K: float, option_type: OptionType) -> float:
    if option_type == "call":
        return max(spot - K, 0.0)
    else:
        return max(K - spot, 0.0)


def crr_price_european(params: CRRParams) -> float:
    """
    Price a European call or put using the CRR binomial model.

    This uses the standard risk-neutral backward induction on a
    recombining binomial tree.

    Parameters
    ----------
    params : CRRParams
        Model and contract parameters.

    Returns
    -------
    float
        European option price at time 0.
    """
    dt, u, d, p = _crr_ud_p(params.r, params.sigma, params.T, params.N)
    disc = exp(-params.r * dt)

    # Terminal payoffs at maturity
    values = [
        _payoff(
            spot=params.S0 * (u ** j) * (d ** (params.N - j)),
            K=params.K,
            option_type=params.option_type,
        )
        for j in range(params.N + 1)
    ]

    # Backward induction for European option (no early exercise)
    for n in range(params.N - 1, -1, -1):
        values = [
            disc * (p * values[j + 1] + (1.0 - p) * values[j])
            for j in range(n + 1)
        ]

    return float(values[0])


def crr_price_american(params: CRRParams) -> float:
    """
    Price an American call or put using the CRR binomial model.

    Uses backward induction with early-exercise at each node.

    Notes
    -----
    For a non–dividend-paying underlying, the American call price
    should be very close to the European call price (no early exercise
    advantage). For puts, early exercise can matter.

    Parameters
    ----------
    params : CRRParams
        Model and contract parameters.

    Returns
    -------
    float
        American option price at time 0.
    """
    dt, u, d, p = _crr_ud_p(params.r, params.sigma, params.T, params.N)
    disc = exp(-params.r * dt)

    # Stock prices at maturity
    spots = [
        params.S0 * (u ** j) * (d ** (params.N - j))
        for j in range(params.N + 1)
    ]
    # Option values at maturity (intrinsic value)
    values = [
        _payoff(spot=s, K=params.K, option_type=params.option_type)
        for s in spots
    ]

    # Backward induction with early exercise
    for n in range(params.N - 1, -1, -1):
        new_values = []
        new_spots = []
        for j in range(n + 1):
            s_ij = params.S0 * (u ** j) * (d ** (n - j))
            continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            exercise = _payoff(spot=s_ij, K=params.K, option_type=params.option_type)
            new_values.append(max(continuation, exercise))
            new_spots.append(s_ij)
        values = new_values
        spots = new_spots

    return float(values[0])
