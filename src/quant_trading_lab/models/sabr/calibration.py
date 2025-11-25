"""
Calibration utilities for the lognormal SABR model.

We fit SABR parameters to a set of market implied volatilities
(sigma_mkt(K_i)) for a given forward F0 and maturity T, using
the Hagan lognormal SABR approximation.

Typical use:

    params = calibrate_sabr_ls(
        F0=100.0,
        T=1.0,
        strikes=np.array([...]),
        market_vols=np.array([...]),
        beta=0.5,
    )

The calibration keeps beta fixed and optimises over (alpha, rho, nu).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sabr import SabrParams, sabr_implied_vol_hagan


@dataclass
class SabrCalibrationResult:
    """
    Result of a SABR calibration.

    Attributes
    ----------
    params : SabrParams
        Calibrated SABR parameters.
    beta_fixed : float
        The beta that was fixed during calibration.
    error : float
        Final value of the objective function (e.g. sum of squared errors).
    n_iter : int
        Number of iterations / evaluations used (if available).
    method : str
        Short description of the calibration method used,
        e.g. "scipy-LBFGS" or "grid-search".
    """
    params: SabrParams
    beta_fixed: float
    error: float
    n_iter: int
    method: str


def _sabr_calibration_objective(
    alpha: float,
    rho: float,
    nu: float,
    F0: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Least-squares objective:

        sum_i w_i * (sigma_sabr(K_i) - sigma_mkt(K_i))^2
    """
    params = SabrParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    model_vols = sabr_implied_vol_hagan(F0=F0, K=strikes, T=T, params=params)

    residuals = model_vols - market_vols

    if weights is not None:
        residuals = residuals * weights

    return float(np.sum(residuals**2))


def calibrate_sabr_ls(
    F0: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 0.5,
    weights: Optional[np.ndarray] = None,
    alpha_init: Optional[float] = None,
    rho_init: float = -0.1,
    nu_init: float = 0.5,
    use_scipy: bool = True,
) -> SabrCalibrationResult:
    """
    Calibrate lognormal SABR parameters (alpha, rho, nu) for a fixed beta
    via least squares against market implied volatilities.

    Parameters
    ----------
    F0 : float
        Forward price at time 0 (must be > 0).
    T : float
        Time to maturity in years (T >= 0).
    strikes : array-like
        Array of strikes K_i (> 0).
    market_vols : array-like
        Array of corresponding market implied vols sigma_mkt(K_i).
    beta : float, optional
        Fixed beta in [0, 1]. Common choices: 0.5, 0.7, 1.0.
    weights : array-like, optional
        Optional weights w_i for each strike (same shape as strikes).
        If None, all weights are treated as 1.
    alpha_init : float, optional
        Initial guess for alpha. If None, we use the ATM vol as a proxy.
    rho_init : float, optional
        Initial guess for rho in [-1, 1].
    nu_init : float, optional
        Initial guess for nu (> 0).
    use_scipy : bool, optional
        If True, try to use scipy.optimize.minimize with L-BFGS-B.
        If scipy is not available or use_scipy=False, fall back to a simple
        grid-search over (alpha, rho, nu).

    Returns
    -------
    SabrCalibrationResult
        Calibrated SABR parameters and diagnostics.

    Notes
    -----
    This is a deliberately simple calibrator for pedagogical / research
    purposes. In production one would typically add:
    - stronger parameter bounds,
    - more robust error handling,
    - multi-start or global optimisation for difficult smiles.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)

    if strikes.shape != market_vols.shape:
        raise ValueError("strikes and market_vols must have the same shape.")
    if np.any(strikes <= 0.0):
        raise ValueError("All strikes must be positive in lognormal SABR.")
    if F0 <= 0.0:
        raise ValueError("F0 must be positive in lognormal SABR.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")

    n_points = strikes.size
    if n_points < 3:
        raise ValueError("Need at least 3 strikes to calibrate a smile meaningfully.")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != strikes.shape:
            raise ValueError("weights must have the same shape as strikes.")

    # Rough ATM-based initial guess for alpha if not provided
    if alpha_init is None:
        # Pick strike closest to F0 as ATM
        idx_atm = int(np.argmin(np.abs(strikes - F0)))
        sigma_atm = float(market_vols[idx_atm])
        if sigma_atm <= 0.0:
            sigma_atm = float(np.mean(market_vols[market_vols > 0.0]))
        alpha_init = sigma_atm * (F0 ** (1.0 - beta))

    # Try SciPy optimisation first (if available and allowed)
    if use_scipy:
        try:
            from scipy.optimize import minimize

            def objective_vec(x):
                a, r, n = x
                return _sabr_calibration_objective(
                    alpha=a,
                    rho=r,
                    nu=n,
                    F0=F0,
                    T=T,
                    strikes=strikes,
                    market_vols=market_vols,
                    beta=beta,
                    weights=weights,
                )

            # Initial guess and bounds
            x0 = np.array([alpha_init, rho_init, nu_init], dtype=float)

            bounds = [
                (1e-4, 5.0),    # alpha
                (-0.999, 0.999),  # rho
                (1e-4, 5.0),    # nu
            ]

            res = minimize(
                objective_vec,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
            )

            alpha_opt, rho_opt, nu_opt = res.x
            error_opt = float(res.fun)
            n_iter = int(res.nfev)

            params_opt = SabrParams(
                alpha=float(alpha_opt),
                beta=float(beta),
                rho=float(rho_opt),
                nu=float(nu_opt),
            )

            return SabrCalibrationResult(
                params=params_opt,
                beta_fixed=float(beta),
                error=error_opt,
                n_iter=n_iter,
                method="scipy-LBFGS",
            )

        except Exception:
            # Fall back to grid search if scipy is not available or fails
            pass

    # --- Fallback: simple grid search over (alpha, rho, nu) ---

    # Define coarse grids (for demonstration / research, not production)
    alpha_grid = np.linspace(0.5 * alpha_init, 1.5 * alpha_init, 7)
    rho_grid = np.linspace(-0.9, 0.9, 7)
    nu_grid = np.linspace(0.1, 2.0, 7)

    best_error = float("inf")
    best_params = None
    eval_count = 0

    for a in alpha_grid:
        for r in rho_grid:
            for n in nu_grid:
                err = _sabr_calibration_objective(
                    alpha=a,
                    rho=r,
                    nu=n,
                    F0=F0,
                    T=T,
                    strikes=strikes,
                    market_vols=market_vols,
                    beta=beta,
                    weights=weights,
                )
                eval_count += 1

                if err < best_error:
                    best_error = err
                    best_params = SabrParams(
                        alpha=float(a),
                        beta=float(beta),
                        rho=float(r),
                        nu=float(n),
                    )

    if best_params is None:
        raise RuntimeError("SABR grid search calibration failed to find parameters.")

    return SabrCalibrationResult(
        params=best_params,
        beta_fixed=float(beta),
        error=float(best_error),
        n_iter=int(eval_count),
        method="grid-search",
    )
