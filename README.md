# quantresearchv1

A first pass at quantitative finance research: Markov regime models, gap-continuation statistics across four sectors, and an intraday backtest with a held-out split.

Written up as a case study: **Four times the instrument answered for me** — every confident result turned out to be a property of the tool that measured it.

## Studies

| Script | What it does |
|---|---|
| `markov_regimes.py` | Labels daily bars into market states, builds transition matrices, derives expected regime duration |
| `markov_dashboard.py` | Same, rendered as an HTML dashboard with dated JSON run history |
| `markov_intraday.py` | Intraday version (2m bars) plus a 60/40 design/held-out backtest with costs |
| `system_map.py` | Lead-lag correlation and Markov-switching regimes across commodity/equity systems |
| `supernova_scanner.py` | Gap-continuation study, small-cap biotech |
| `cannabis_scanner.py` / `mining_scanner.py` | Same study, other sectors |
| `gap_diagnostics.py` | Validation pass: baselines, Wilson intervals, year splits, permutation clustering test |
| `cannabis_clean.py` | Data-quality pass that withdrew the cannabis sector |
| `final_numbers.py` | Publication numbers for biotech and AI tech |

## Findings

Gap-ups fade below their own baseline on ordinary volume and continue above it on heavy volume — replicated across two unrelated small-cap universes, 1,773 gaps. Biotech is stable across 2024–2026; AI's low-volume signal decayed to nothing by 2026.

No exit rule, no P&L. This is a measured base rate and a filter, not a strategy.

## Caveats

- The Markov persistence result is reproduced by a random walk — the regimes are a property of the 20-day smoother, not the market. Needs a surrogate null.
- The backtest loses out-of-sample. That result is published rather than tuned away.
- Two sectors were tested and withdrawn: bitcoin miners for name concentration, cannabis because its price data is unusable.

## Running

Install: pip install yfinance pandas numpy statsmodels

Then: python3 final_numbers.py
