import os
import sys

import numpy as np

# Ensure src/ is on the path when running from the repo root
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from quant_trading_lab.models.heston.params import HestonParams, OptionParams
from quant_trading_lab.models.heston.pricing import (
    price_european_call_heston_mc,
    black_scholes_call,
)
from quant_trading_lab.models.heston.strategy import run_mispricing_strategy_backtest


def main():
    # Heston model parameters (illustrative equity-like set)
    params = HestonParams(
        kappa=2.0,
        theta=0.04,   # long-run variance -> ~20% vol
        sigma=0.5,
        rho=-0.7,
        v0=0.04,
        r=0.02,
    )

    S0 = 100.0
    option = OptionParams(K=100.0, T=1.0)

    # Assume the market uses a flat BS vol equal to sqrt(theta)
    sigma_mkt = float(np.sqrt(params.theta))

    # --- Initial pricing comparison at t = 0 ---
    C_heston_0 = price_european_call_heston_mc(
        S0=S0,
        option=option,
        params=params,
        n_steps=200,
        n_paths=30_000,
        seed=123,
    )
    C_bs_0 = black_scholes_call(
        S0=S0,
        option=option,
        r=params.r,
        sigma=sigma_mkt,
    )

    print("=== Initial pricing comparison (t = 0) ===")
    print(f"S0                 : {S0:.2f}")
    print(f"K, T               : {option.K:.2f}, {option.T:.2f} year")
    print(f"r                  : {params.r:.4f}")
    print(f"sigma_mkt (BS vol) : {sigma_mkt:.4f}")
    print(f"Heston MC price    : {C_heston_0:.4f}")
    print(f"Black–Scholes price: {C_bs_0:.4f}")
    if C_bs_0 > 0:
        m0 = (C_heston_0 - C_bs_0) / C_bs_0
        print(f"Relative mispricing: {m0 * 100:.2f}%")
    print()

    # --- Run mispricing-based strategy along a Heston path ---
    results = run_mispricing_strategy_backtest(
        S0=S0,
        option=option,
        params=params,
        sigma_mkt=sigma_mkt,
        rel_threshold=0.05,      # 5% mispricing threshold
        n_steps_path=40,
        seed_path=42,
        mc_steps_per_node=30,
        mc_paths_per_node=2_000,
    )

    times = results["times"]
    S = results["S"]
    portfolio_values = results["portfolio_values"]

    print("=== Backtest summary ===")
    print(f"Horizon        : {times[-1]:.2f} years")
    print(f"Initial spot   : {S[0]:.2f}")
    print(f"Final spot     : {S[-1]:.2f}")
    print(f"Initial value  : {portfolio_values[0]:.4f}")
    print(f"Final value    : {portfolio_values[-1]:.4f}")
    print(f"Total PnL      : {portfolio_values[-1] - portfolio_values[0]:.4f}")


if __name__ == "__main__":
    main()
