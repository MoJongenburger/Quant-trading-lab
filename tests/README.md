# Tests

This folder contains unit tests for the models and utilities implemented in `quant-trading-lab`.

The goal of the tests is to ensure that:

- Core pricing formulas are internally consistent  
- Greeks behave as expected (e.g. small finite-difference checks)  
- Calibration routines recover known parameters on synthetic data  
- The code remains robust as the project grows  

---

## Current test files

### `test_heston_pricing.py`

Basic tests for the Heston Monte Carlo pricer:

- Checks that Heston prices converge for increasing number of paths  
- Compares Heston Monte Carlo prices to Black–Scholes prices in the limit of “almost constant variance”  
- Verifies that prices are positive and behave sensibly when parameters change  

---

### `test_black_scholes_pricing.py`

Tests for the Black–Scholes pricing utilities:

- Verifies call/put prices against known benchmark values  
- Checks **put–call parity**  
- Tests implied volatility solvers by re-pricing options with the recovered vol  

---

### `test_black_scholes_greeks.py`

Tests for Black–Scholes Greeks:

- Finite-difference checks for delta, gamma, vega, theta, rho  
- Basic sign and monotonicity checks (e.g. call delta ∈ (0, 1))  

---

### `test_sabr_model.py`

Tests for the SABR module:

- Ensures Hagan SABR implied vols are **positive**  
- Checks that negative $\rho$ produces a **downward skew**  
- Verifies that the grid-search calibration approximately recovers known SABR parameters on a synthetic smile  

---

### `test_crr_pricing.py`

Tests for the Cox–Ross–Rubinstein (CRR) binomial model:

- Checks that **European** CRR call and put prices converge to the corresponding Black–Scholes prices as the number of steps `N` becomes large  
- Verifies that, on a non-dividend-paying underlying, the **American call** price is very close to the European call price (no early exercise premium)  
- Confirms that the **American put** is never worth less than the European put, and can be strictly more valuable due to early exercise  

---

## How to run the tests (optional)

If you clone the repo locally and have `pytest` installed, you can run:

```bash
pytest tests
````

from the repository root to execute the full test suite.

