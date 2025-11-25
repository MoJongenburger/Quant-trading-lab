"""
Basic tests for the SABR model implementation.

We check:

- Hagan SABR implied vols are positive
- ATM limit behaves sensibly
- Negative rho produces a downward skew
- Least-squares calibration can approximately recover known parameters
  from a synthetic smile (using grid-search fallback)
"""

import numpy as np

from quant_trading_lab.models.sabr.sabr import SabrParams, sabr_implied_vol_hagan
from quant_trading_lab.models.sabr.calibration import calibrate_sabr_ls


F0 = 100.0
T = 1.0


def test_sabr_implied_vol_positive():
    """
    SABR implied vols should be strictly positive for reasonable parameters.
    """
    params = SabrParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.6)
    strikes = np.array([60, 80, 90, 100, 110, 120, 140], dtype=float)

    vols = sabr_implied_vol_hagan(F0=F0, K=strikes, T=T, params=params)
    assert np.all(vols > 0.0)


def test_sabr_implied_vol_atm_limit():
    """
    For very small maturity and F0 ~ K, the SABR implied vol should be close
    to alpha / F0^(1 - beta), i.e. the leading ATM term (time correction small).
    """
    alpha = 0.25
    beta = 0.5
    rho = -0.3
    nu = 0.5

    params = SabrParams(alpha=alpha, beta=beta, rho=rho, nu=nu)

    T_small = 1e-4
    K_atm = F0

    sigma_atm = sabr_implied_vol_hagan(F0=F0, K=K_atm, T=T_small, params=params)
    expected = alpha / (F0 ** (1.0 - beta))

    rel_diff = abs(sigma_atm - expected) / expected
    assert rel_diff < 1e-2  # within 1% is fine given approximations


def test_sabr_negative_rho_downward_skew():
    """
    With negative rho, SABR should produce higher implied vols for low strikes
    than for high strikes (downward skew).
    """
    params = SabrParams(alpha=0.3, beta=0.5, rho=-0.6, nu=0.7)

    K_low = 80.0
    K_atm = 100.0
    K_high = 120.0

    sigma_low = sabr_implied_vol_hagan(F0=F0, K=K_low, T=T, params=params)
    sigma_atm = sabr_implied_vol_hagan(F0=F0, K=K_atm, T=T, params=params)
    sigma_high = sabr_implied_vol_hagan(F0=F0, K=K_high, T=T, params=params)

    # Downward skew: low strike vol highest, high strike vol lowest (typical pattern)
    assert sigma_low > sigma_atm
    assert sigma_atm >= sigma_high


def test_sabr_calibration_recovers_params_approximately():
    """
    Create a synthetic smile from known SABR parameters and check that
    least-squares calibration (with fixed beta) approximately recovers them.

    We use use_scipy=False to force the grid-search fallback, so the test
    does not depend on SciPy being installed.
    """
    beta_true = 0.5
    params_true = SabrParams(alpha=0.25, beta=beta_true, rho=-0.4, nu=0.8)

    strikes = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)
    vols_true = sabr_implied_vol_hagan(F0=F0, K=strikes, T=T, params=params_true)

    # Optionally add a tiny bit of noise to avoid perfection (but keep it small)
    rng = np.random.default_rng(42)
    noise = 0.0 * rng.normal(scale=1e-3, size=vols_true.shape)  # currently zero
    vols_obs = vols_true + noise

    calib = calibrate_sabr_ls(
        F0=F0,
        T=T,
        strikes=strikes,
        market_vols=vols_obs,
        beta=beta_true,
        use_scipy=False,  # force grid-search so test is environment-independent
    )

    alpha_hat = calib.params.alpha
    rho_hat = calib.params.rho
    nu_hat = calib.params.nu

    # Coarse but reasonable tolerances for grid-search calibration
    assert abs(alpha_hat - params_true.alpha) < 0.1
    assert abs(rho_hat - params_true.rho) < 0.3
    assert abs(nu_hat - params_true.nu) < 0.4
