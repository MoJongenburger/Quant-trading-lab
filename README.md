# Quant Trading Lab

A personal **quant research lab** containing mathematical finance models, trading strategies, and reproducible experiments in Python.

The goal of this repository is to bridge the gap between **theory and practice**:

- Implement well-known models from **quantitative finance** (Heston, Black–Scholes, etc.)
- Turn them into **systematic trading strategies**
- Write **clean, well-documented code** that a quant researcher or developer would recognize

---

## Contents

Current and planned components:

-  **Heston Stochastic Volatility Model**
  - Option pricing via Monte Carlo under Heston dynamics
  - Comparison to Black–Scholes pricing
  - Simple mispricing-based trading strategy and PnL simulation

-  **Planned modules**
  - Black–Scholes implied volatility surface & Greeks
  - Mean-reversion / pairs trading strategies
  - Basic portfolio and risk models (e.g. Markowitz, CAPM-inspired overlays)
  - Volatility trading ideas (variance swaps, gamma scalping – simplified)

Each model will live in its own subdirectory with:

- A **model-specific README** (math + intuition)
- **Python implementation** (simulation, pricing, strategy)
- Optionally a **notebook** or example script

---

## Repository structure

The lab is organized to look like a small research/codebase rather than a single script:

```text
quant-trading-lab/
├─ README.md
├─ requirements.txt           # or pyproject.toml if using poetry
├─ src/
│  └─ quant_trading_lab/
│     ├─ __init__.py
│     ├─ common/              # shared utilities (rng, plotting, helpers)
│     └─ models/
│        └─ heston/
│           ├─ README.md      # detailed Heston explanation
│           ├─ params.py      # model & option parameter classes
│           ├─ simulation.py  # Heston path simulation
│           ├─ pricing.py     # Heston MC pricing + Black–Scholes pricing
│           ├─ strategy.py    # trading rules & PnL simulation
│           └─ plotting.py    # optional: visualization helpers
├─ examples/
│  └─ heston_demo.py          # end-to-end demo: pricing + trading signal + PnL
└─ tests/
   ├─ test_heston_pricing.py
   └─ test_heston_strategy.py

