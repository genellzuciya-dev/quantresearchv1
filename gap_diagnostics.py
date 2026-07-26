"""
Gap study — diagnostics pass
============================
Runs on YOUR machine (needs yfinance + network). One download per ticker,
reused across all six checks. Prints a plain-text report.

    python3 gap_diagnostics.py            # all four baskets
    python3 gap_diagnostics.py cannabis   # just one
    python3 gap_diagnostics.py > diagnostics.txt

WHAT CHANGED vs the scanners
----------------------------
vol_x now uses .shift(1) before the rolling mean, so a gap day's own volume
is not in its own 20-day average. Every tier boundary moves. Both the old
and new numbers are printed side by side so the shift is visible.
"""

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

LOOKBACK = "2y"
GAP_MIN = 0.03
VOL_TIERS = [(0, 1.5, "<1.5x"), (1.5, 2, "1.5-2x"), (2, 3, "2-3x"),
             (3, 5, "3-5x"), (5, 999, "5x+")]
OUTLIER_FLAG = 1.00        # intraday move above +100% = almost certainly a split artifact
N_SHUFFLES = 300           # for the clustering null

BASKETS = {
    "biotech": ["ORIC", "VSTM", "OLMA", "NUVB", "KYMR", "RXRX", "CRBU", "SANA",
                "BEAM", "NTLA", "CRSP", "FATE", "RLAY", "ARVN", "IMVT", "KROS",
                "RCKT", "DAWN", "TSHA", "VERA", "AVXL", "PGEN", "CGEM", "MREO",
                "CABA"],
    "ai_tech": ["SOUN", "BBAI", "AI", "VERI", "GFAI", "SERV", "RGTI", "QBTS",
                "IONQ", "LAES", "POET", "AEVA", "OUST", "MVIS", "KSCP", "AITX",
                "CXAI", "INOD", "AUR", "ARBE", "NNDM", "PRSO"],
    "btc_miners": ["MARA", "RIOT", "CLSK", "BITF", "HUT", "CIFR", "WULF", "CORZ",
                   "BTDR", "HIVE", "BTBT", "CANG", "SDIG", "GREE", "DGHI"],
    "cannabis": ["TLRY", "CGC", "ACB", "SNDL", "OGI", "CRON", "VFF", "GRWG",
                 "AKAN", "CURLF", "GTBIF", "TCNNF", "CRLBF", "AAWH", "MSOS",
                 "YOLO", "HITI", "SHWZ", "MRMD", "GDNSF", "TSNDF", "CLVR"],
}

# Cannabis basket contains 2 ETFs and 7 OTC names. OTC opening prints in Yahoo
# data are thin and often stale, which manufactures both fake gaps and fake
# fades. ETFs cannot have idiosyncratic catalysts by construction.
EXCLUDE = {
    "cannabis": {"MSOS": "ETF", "YOLO": "ETF",
                 "CURLF": "OTC", "GTBIF": "OTC", "TCNNF": "OTC", "CRLBF": "OTC",
                 "AAWH": "OTC", "GDNSF": "OTC", "TSNDF": "OTC"},
}

_CACHE = {}


# ------------------------------------------------------------------ data
def get_daily(ticker):
    if ticker in _CACHE:
        return _CACHE[ticker]
    import yfinance as yf
    df = yf.download(ticker, period=LOOKBACK, interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    _CACHE[ticker] = df.dropna()
    return _CACHE[ticker]


def prep(df):
    """Adds gap, intraday, and BOTH vol_x definitions."""
    d = df.copy()
    d["prev_close"] = d["Close"].shift(1)
    d["gap"] = d["Open"] / d["prev_close"] - 1
    d["intraday"] = d["Close"] / d["Open"] - 1
    # OLD: rolling window includes the day being measured (the bug)
    d["vol_x_old"] = d["Volume"] / d["Volume"].rolling(20).mean()
    # NEW: prior 20 days only
    d["vol_x"] = d["Volume"] / d["Volume"].shift(1).rolling(20).mean()
    return d.dropna()


def load_basket(name, listed_only=False):
    tickers = BASKETS[name]
    dropped = EXCLUDE.get(name, {}) if listed_only else {}
    frames, missing = [], []
    for t in tickers:
        if t in dropped:
            continue
        try:
            df = get_daily(t)
            if len(df) < 40:
                missing.append((t, "not enough history"))
                continue
            d = prep(df)
            d["ticker"] = t
            frames.append(d)
        except Exception as e:
            missing.append((t, str(e)[:40]))
    return (pd.concat(frames) if frames else None), missing, dropped


# ------------------------------------------------------------------ tests
def wilson(k, n):
    """Wilson 95% interval — behaves better than normal approx at small n."""
    if n == 0:
        return (float("nan"),) * 3
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def tier_table(all_rows, col="vol_x"):
    ups = all_rows[all_rows["gap"] >= GAP_MIN]
    out = []
    for lo, hi, label in VOL_TIERS:
        t = ups[(ups[col] >= lo) & (ups[col] < hi)]
        n = len(t)
        k = int((t["intraday"] > 0).sum())
        p, lo_ci, hi_ci = wilson(k, n)
        out.append((label, n, p, lo_ci, hi_ci,
                    t["intraday"].mean() * 100 if n else float("nan"),
                    t["intraday"].median() * 100 if n else float("nan")))
    return out


def test_1_outliers(all_rows):
    """Reverse-split artifacts: intraday moves that cannot be real."""
    ups = all_rows[all_rows["gap"] >= GAP_MIN]
    top = ups.nlargest(8, "intraday")[["ticker", "gap", "intraday", "vol_x"]]
    flagged = ups[ups["intraday"] >= OUTLIER_FLAG]
    return top, flagged


def test_2_baseline(all_rows):
    """The unconditional rate. What a random day does, with no gap filter."""
    everything = all_rows
    nongap = all_rows[all_rows["gap"].abs() < 0.01]
    smallgap = all_rows[(all_rows["gap"] >= 0.01) & (all_rows["gap"] < GAP_MIN)]
    rows = []
    for label, sub in [("all days", everything), ("flat open (<1%)", nongap),
                       ("small gap 1-3%", smallgap)]:
        n = len(sub)
        k = int((sub["intraday"] > 0).sum())
        rows.append((label, n) + wilson(k, n))
    return rows


def test_3_by_year(all_rows):
    ups = all_rows[all_rows["gap"] >= GAP_MIN].copy()
    ups["year"] = ups.index.year
    out = []
    for yr, sub in ups.groupby("year"):
        low = sub[sub["vol_x"] < 1.5]
        high = sub[sub["vol_x"] >= 3]
        out.append((yr, len(low), (low["intraday"] > 0).mean() if len(low) else np.nan,
                    len(high), (high["intraday"] > 0).mean() if len(high) else np.nan))
    return out


def test_4_concentration(all_rows):
    """Is the high-volume result three names in a trenchcoat?"""
    hi = all_rows[(all_rows["gap"] >= GAP_MIN) & (all_rows["vol_x"] >= 3)]
    by = hi.groupby("ticker")["intraday"].agg(["size", lambda s: (s > 0).mean()])
    by.columns = ["gaps", "cont"]
    by = by.sort_values("gaps", ascending=False)
    total = by["gaps"].sum()
    top3 = by["gaps"].head(3).sum() / total if total else np.nan
    return by, top3, total


def test_5_clustering(all_rows, rng=None):
    """Do gap days cluster on shared calendar dates more than chance?

    Statistic: mean fraction of the universe gapping, on days where >=1 name
    gapped. Null: shuffle each ticker's gap-day flags across its own trading
    dates, preserving each name's gap frequency but destroying shared timing.
    """
    rng = rng or np.random.default_rng(7)
    piv = (all_rows.assign(g=(all_rows["gap"] >= GAP_MIN).astype(int))
                   .pivot_table(index=all_rows.index, columns="ticker",
                                values="g", aggfunc="max"))
    piv = piv.dropna(how="all").fillna(0).astype(int)
    if piv.shape[1] < 3:
        return None

    def stat(mat):
        per_day = mat.sum(axis=1)
        active = per_day[per_day > 0]
        return (active / mat.shape[1]).mean() if len(active) else np.nan

    obs = stat(piv.values)
    null = []
    for _ in range(N_SHUFFLES):
        m = piv.values.copy()
        for j in range(m.shape[1]):
            rng.shuffle(m[:, j])
        null.append(stat(m))
    null = np.array(null)
    z = (obs - null.mean()) / null.std() if null.std() else np.nan
    return {"observed": obs, "null_mean": null.mean(), "null_sd": null.std(),
            "z": z, "n_names": piv.shape[1], "n_days": piv.shape[0]}


def test_6_cluster_vs_isolated(all_rows):
    """The one that links mechanism to outcome.

    Split gap days into 'cluster' (many names gapped that day) and 'isolated'
    (this name gapped nearly alone), then compare continuation. If correlated
    catalysts are what break continuation, isolated gaps should continue better.
    """
    ups = all_rows[all_rows["gap"] >= GAP_MIN].copy()
    breadth = ups.groupby(ups.index).size()
    ups["breadth"] = ups.index.map(breadth)
    n_names = all_rows["ticker"].nunique()
    thresh = max(2, int(np.ceil(0.15 * n_names)))
    out = []
    for label, sub in [(f"isolated (<{thresh} names)", ups[ups["breadth"] < thresh]),
                       (f"cluster (>={thresh} names)", ups[ups["breadth"] >= thresh])]:
        for vlabel, vsub in [("all vol", sub), ("3x+ vol", sub[sub["vol_x"] >= 3])]:
            n = len(vsub)
            k = int((vsub["intraday"] > 0).sum())
            out.append((label, vlabel, n) + wilson(k, n))
    return out, thresh


# ------------------------------------------------------------------ report
def pct(x):
    return "  n/a" if x != x else f"{x*100:5.1f}%"


def run(name, listed_only=False):
    tag = f"{name}{' [listed only]' if listed_only else ''}"
    print("\n" + "=" * 72)
    print(f"  {tag.upper()}")
    print("=" * 72)

    all_rows, missing, dropped = load_basket(name, listed_only)
    if all_rows is None:
        print("  no data")
        return
    print(f"  {all_rows['ticker'].nunique()} names · {len(all_rows)} trading days")
    if dropped:
        print(f"  excluded: " + ", ".join(f"{k}({v})" for k, v in dropped.items()))
    if missing:
        print(f"  failed:   " + ", ".join(f"{t}({e})" for t, e in missing))

    # --- 1. outliers
    print("\n-- 1. OUTLIER SCAN (reverse-split artifacts) " + "-" * 27)
    top, flagged = test_1_outliers(all_rows)
    for ix, r in top.iterrows():
        mark = "  <-- IMPOSSIBLE" if r["intraday"] >= OUTLIER_FLAG else ""
        print(f"   {ix.date()}  {r['ticker']:6} gap {r['gap']*100:+7.1f}%  "
              f"intraday {r['intraday']*100:+8.1f}%  vol {r['vol_x']:5.1f}x{mark}")
    print(f"   {len(flagged)} day(s) above +{OUTLIER_FLAG*100:.0f}% intraday "
          f"-- exclude these before publishing if any.")

    # --- 2. baseline
    print("\n-- 2. UNCONDITIONAL BASELINE (close > open) " + "-" * 28)
    for label, n, p, lo, hi in test_2_baseline(all_rows):
        print(f"   {label:18} n={n:6d}   {pct(p)}   [{pct(lo)},{pct(hi)} ]")

    # --- 3. tiers, old vs fixed
    print("\n-- 3. VOLUME TIERS: buggy vs fixed vol_x " + "-" * 31)
    old = tier_table(all_rows, "vol_x_old")
    new = tier_table(all_rows, "vol_x")
    print(f"   {'tier':8} {'n(old)':>7} {'old':>7}   {'n(fix)':>7} {'fixed':>7} "
          f"{'95% CI':>16} {'mean':>8} {'median':>8}")
    for o, nw in zip(old, new):
        print(f"   {nw[0]:8} {o[1]:7d} {pct(o[2])}   {nw[1]:7d} {pct(nw[2])} "
              f"[{pct(nw[3])},{pct(nw[4])} ] {nw[5]:7.1f}% {nw[6]:7.1f}%")
    print("   mean vs median gap: a large spread means fat tails, not a typical day.")

    # --- 4. by year
    print("\n-- 4. BY YEAR (does the floor track the tape?) " + "-" * 25)
    print(f"   {'year':6} {'n<1.5x':>8} {'cont':>7}   {'n>=3x':>8} {'cont':>7}")
    for yr, nl, cl, nh, ch in test_3_by_year(all_rows):
        print(f"   {yr:6} {nl:8d} {pct(cl)}   {nh:8d} {pct(ch)}")

    # --- 5. concentration
    print("\n-- 5. CONCENTRATION of the 3x+ result " + "-" * 34)
    by, top3, total = test_4_concentration(all_rows)
    print(f"   {total} gaps at 3x+; top 3 names = {top3*100:.0f}% of them")
    for t, r in by.head(6).iterrows():
        print(f"   {t:6} {int(r['gaps']):4d} gaps   {pct(r['cont'])}")

    # --- 6. clustering
    print("\n-- 6. CLUSTERING vs shuffled null " + "-" * 38)
    c = test_5_clustering(all_rows)
    if c:
        print(f"   universe breadth on active days: observed {c['observed']*100:.1f}%")
        print(f"   shuffled null: {c['null_mean']*100:.1f}% (sd {c['null_sd']*100:.2f}) "
              f"-> z = {c['z']:+.1f}")
        print("   z above ~2 means gap days genuinely share calendar dates.")

    # --- 7. the mechanism test
    print("\n-- 7. CLUSTER vs ISOLATED gap days " + "-" * 37)
    rows, thresh = test_6_cluster_vs_isolated(all_rows)
    for label, vlabel, n, p, lo, hi in rows:
        print(f"   {label:22} {vlabel:8} n={n:5d}  {pct(p)}  [{pct(lo)},{pct(hi)} ]")
    print("   If correlated catalysts break continuation, isolated should beat cluster.")


if __name__ == "__main__":
    which = sys.argv[1:] or list(BASKETS)
    for name in which:
        if name not in BASKETS:
            print(f"unknown basket: {name}")
            continue
        run(name)
        if name in EXCLUDE:
            run(name, listed_only=True)
    print("\ndone. paste this whole output back into the chat.\n")
