"""
SABR stochastic volatility model.

This package provides:
- SabrParams dataclass
- Hagan lognormal SABR implied volatility approximation

Calibration utilities will live in `calibration.py`.
"""

from .sabr import SabrParams, sabr_implied_vol_hagan  # noqa: F401
