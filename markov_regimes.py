"""
ORIC Markov Regime Study — first build outside thinkorswim
==========================================================
Labels each trading day with a market "state", counts state-to-state
transitions, and builds a Markov transition matrix. Then checks whether
those transitions are actually different from random (the whole point).

Run locally:
    pip install yfinance pandas numpy
    python oric_markov_regimes.py

Or paste into a QuantConnect research notebook and swap the yfinance
download for qb.History (marked below).

Key design choice for a biotech like ORIC: we add explicit GAP states,
because catalyst days (trial readouts, FDA news) violate the smooth
day-to-day behavior a Markov chain assumes. Instead of pretending they
don't exist, we make them their own state and measure what happens
AFTER them. That's the memoryless property working for us.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. GET DATA
# ----------------------------------------------------------------------
import sys
# Ticker comes from the command line: python3 markov_regimes.py NVDA
# If none is given, defaults to ORIC.
TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "ORIC"
BENCHMARK = "SPY"          # a liquid, non-gappy ticker for comparison
LOOKBACK = "2y"

import yfinance as yf
def get_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=LOOKBACK, interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):      # yfinance quirk
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# QuantConnect alternative:
#   qb = QuantBook()
#   sym = qb.AddEquity("ORIC").Symbol
#   df = qb.History(sym, timedelta(days=730), Resolution.Daily)

# ----------------------------------------------------------------------
# 2. LABEL EACH DAY WITH A STATE
# ----------------------------------------------------------------------
GAP_THRESHOLD = 0.05       # 5% overnight gap = catalyst day for a biotech
SMA_LEN = 20

STATES = ["GAP_UP", "GAP_DOWN", "TREND_UP", "TREND_DOWN", "CHOP"]

def label_states(df: pd.DataFrame) -> pd.Series:
    sma = df["Close"].rolling(SMA_LEN).mean()
    sma_slope = sma.diff()
    overnight_gap = df["Open"] / df["Close"].shift(1) - 1

    conditions = [
        overnight_gap >= GAP_THRESHOLD,
        overnight_gap <= -GAP_THRESHOLD,
        (df["Close"] > sma) & (sma_slope > 0),
        (df["Close"] < sma) & (sma_slope < 0),
    ]
    choices = ["GAP_UP", "GAP_DOWN", "TREND_UP", "TREND_DOWN"]
    labels = np.select(conditions, choices, default="CHOP")
    s = pd.Series(labels, index=df.index, name="state")
    return s.iloc[SMA_LEN:]            # drop the SMA warm-up period

# ----------------------------------------------------------------------
# 3. BUILD THE TRANSITION MATRIX
# ----------------------------------------------------------------------
def transition_matrix(states: pd.Series) -> pd.DataFrame:
    counts = pd.DataFrame(0, index=STATES, columns=STATES, dtype=float)
    for prev, curr in zip(states[:-1], states[1:]):
        counts.loc[prev, curr] += 1
    # rows -> probabilities: P(next state | current state)
    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums.replace(0, np.nan), axis=0)
    return probs.round(3), counts

# ----------------------------------------------------------------------
# 4. SANITY CHECK: IS THIS DIFFERENT FROM RANDOM?
# ----------------------------------------------------------------------
# If we shuffle the state sequence, all Markov structure is destroyed;
# transitions collapse toward the base frequencies of each state.
# Compare the real diagonal (persistence) against the shuffled one.
def shuffled_baseline(states: pd.Series, n_shuffles: int = 200) -> pd.Series:
    diag = []
    vals = states.values.copy()
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        rng.shuffle(vals)
        probs, _ = transition_matrix(pd.Series(vals))
        diag.append(np.diag(probs.fillna(0).values))
    return pd.Series(np.mean(diag, axis=0), index=STATES)

# ----------------------------------------------------------------------
# 5. RUN IT
# ----------------------------------------------------------------------
def analyze(ticker: str):
    print("=" * 60)
    print(f"  {ticker}  —  Markov regime analysis ({LOOKBACK} daily bars)")
    print("=" * 60)
    df = get_daily(ticker)
    states = label_states(df)

    freq = states.value_counts(normalize=True).reindex(STATES).fillna(0)
    print("\nState frequencies (how often each regime occurs):")
    print(freq.round(3).to_string())

    probs, counts = transition_matrix(states)
    print("\nTransition matrix  P(column | row):")
    print(probs.to_string())
    print("\nTransition counts (trust rows with more observations):")
    print(counts.astype(int).to_string())

    # Persistence: how long does each regime last, on average?
    # For a Markov chain, expected duration in state i = 1 / (1 - P_ii)
    print("\nExpected regime duration in days (1 / (1 - P_ii)):")
    for s in STATES:
        p = probs.loc[s, s]
        if pd.notna(p) and p < 1:
            print(f"  {s:<11} {1 / (1 - p):5.1f}")

    baseline = shuffled_baseline(states)
    print("\nPersistence vs. shuffled baseline (real should beat random")
    print("if regimes are genuine and not an artifact of labeling):")
    comp = pd.DataFrame({"real_P_ii": np.diag(probs.fillna(0).values),
                         "shuffled_P_ii": baseline.round(3)}, index=STATES)
    print(comp.round(3).to_string())
    return probs

if __name__ == "__main__":
    oric = analyze(TICKER)
    spy = analyze(BENCHMARK)   # compare: how different is a gappy biotech
                               # from a liquid index ETF?

# ----------------------------------------------------------------------
# WHAT TO LOOK FOR (and what to build next)
# ----------------------------------------------------------------------
# 1. GAP rows on ORIC: what actually follows a catalyst gap? If
#    GAP_DOWN -> TREND_DOWN is high, gaps tend to start trends (momentum);
#    if GAP_DOWN -> CHOP/TREND_UP dominates, gaps tend to mean-revert.
#    Row counts will be small — that's honest; don't over-trust them.
# 2. TREND_UP persistence on SPY vs ORIC: liquid indexes usually show
#    stronger, cleaner regime persistence than single biotechs.
# 3. If real P_ii barely beats the shuffled baseline, the states are
#    poorly defined — redefine before building anything on top.
# 4. Next steps: rolling-window matrices (regimes drift), then move to
#    5m bars for the tickers you day trade, then gate your EMA/VWAP
#    entries on the current state and backtest with vs. without.
