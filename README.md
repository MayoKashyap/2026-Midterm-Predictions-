# 2026 Midterm Election Forecaster

A probabilistic forecasting system for the 2026 U.S. House and Senate elections.

## Architecture

- **Layer 1 — Fundamentals:** XGBoost regressor trained on historical results, incumbency, Cook PVI, and economic data
- **Layer 2 — Bayesian Poll Aggregator:** PyMC model with house-effect corrections and Gaussian random walk for opinion drift
- **Layer 3 — Election Simulator:** 10,000-run Monte Carlo with correlated national shock

## Setup

```bash
pip install -r requirements.txt
```

## Data Sources

- MIT Election Lab — historical House + Senate results
- Cook PVI — partisan lean per district/state
- FRED — GDP growth, unemployment
- 538 — polling data and 2022 forecast benchmark
- Redistricting Data Hub — post-2020 Census redistricting flags
