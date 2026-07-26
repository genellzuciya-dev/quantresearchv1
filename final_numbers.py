"""
Final numbers — biotech + AI tech
=================================
The publication dataset. Three changes from the original scanners:

  1. vol_x uses the PRIOR 20 days (.shift(1)), so a gap day's own volume
     is not inside its own average.
  2. Rows with |gap| or |intraday| beyond 100% are dropped as data errors.
  3. AITX is out. 25% of its days print open == close (median price $0.11);
     that is tick-size quantization, not trading.

Continuation is reported two ways, because the choice matters:
  INCL FLAT  close > open        -- the trader's view: flat = no gain = miss
  EXCL FLAT  close > open | close != open  -- the unbiased directional rate

Flat days cluster in quiet tiers, so INCL FLAT makes the gradient look
steeper. Both are printed so the page can state which one it used.

    python3 final_numbers.py | tee final_numbers.txt
"""

import numpy as np
import pandas as pd

BASKETS = {
    "biotech": ["ORIC", "VSTM", "OLMA", "NUVB", "KYMR", "RXRX", "CRBU", "SANA",
                "BEAM", "NTLA", "CRSP", "FATE", "RLAY", "ARVN", "IMVT", "KROS",
                "RCKT", "TSHA", "VERA", "AVXL", "PGEN", "CGEM", "MREO", "CABA"],
    # AITX dropped: 25.35% flat prints, median price $0.11
    "ai_tech": ["SOUN", "BBAI", "AI", "VERI", "GFAI", "SERV", "RGTI", "QBTS",
                "IONQ", "LAES", "POET", "AEVA", "OUST", "MVIS", "KSCP",
                "CXAI", "INOD", "AUR", "ARBE", "NNDM", "PRSO"],
}

TIERS = [(0, 1.5, "<1.5x"), (1.5, 2, "1.5-2x"), (2, 3, "2-3x"),
         (3, 5, "3-5x"), (5, 999, "5x+")]
SANE = 1.00


def load(tickers):
    import yfinance as yf
    frames = []
    for t in tickers:
        d = yf.download(t, period="2y", interval="1d",
                        auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna()
        if len(d) < 40:
            print(f"  skipped {t}")
            continue
        d = d.copy()
        d["gap"] = d["Open"] / d["Close"].shift(1) - 1
        d["intraday"] = d["Close"] / d["Open"] - 1
        d["vol_x"] = d["Volume"] / d["Volume"].shift(1).rolling(20).mean()
        d = d.dropna()
        d["ticker"] = t
        frames.append(d)
    df = pd.concat(frames)
    bad = df[(df["gap"].abs() > SANE) | (df["intraday"].abs() > SANE)]
    if len(bad):
        print(f"  dropped {len(bad)} corrupt row(s): "
              f"{sorted(bad['ticker'].unique())}")
    return df.drop(bad.index)


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def rates(sub):
    """(incl_flat, excl_flat, n, n_nonflat, flat_pct)"""
    n = len(sub)
    if n == 0:
        return (np.nan,) * 5
    up = int((sub["intraday"] > 0).sum())
    flat = int((sub["intraday"] == 0).sum())
    nf = n - flat
    return up / n, (up / nf if nf else np.nan), n, nf, flat / n * 100


def run(name, tickers):
    print(f"\n{'='*76}\n  {name.upper()}\n{'='*76}")
    d = load(tickers)
    print(f"  {d['ticker'].nunique()} names · {len(d)} trading days")

    b_in, b_ex, bn, bnf, bflat = rates(d)
    print(f"\n  BASELINE (any day)   incl flat {b_in*100:.1f}%   "
          f"excl flat {b_ex*100:.1f}%   ({bflat:.2f}% of days flat)")

    ups = d[d["gap"] >= 0.03]
    print(f"\n  TIERS ({len(ups)} up-gaps >= 3%)")
    print(f"  {'tier':8}{'n':>6}{'incl':>8}{'excl':>8}{'95% CI (excl)':>20}"
          f"{'vs base':>9}{'flat%':>7}{'median':>8}")
    for lo, hi, lab in TIERS:
        t = ups[(ups["vol_x"] >= lo) & (ups["vol_x"] < hi)]
        p_in, p_ex, n, nf, fl = rates(t)
        if n == 0:
            continue
        _, l, h = wilson(int((t["intraday"] > 0).sum()), nf)
        print(f"  {lab:8}{n:6d}{p_in*100:7.1f}%{p_ex*100:7.1f}%"
              f" [{l*100:6.1f},{h*100:6.1f}]{(p_ex-b_ex)*100:+8.1f}"
              f"{fl:6.1f}%{t['intraday'].median()*100:7.1f}%")

    print(f"\n  BY YEAR (excl flat)")
    u = ups.copy()
    u["year"] = u.index.year
    for yr, sub in u.groupby("year"):
        lo_t, hi_t = sub[sub["vol_x"] < 1.5], sub[sub["vol_x"] >= 3]
        _, f_ex, fn, _, _ = rates(lo_t)
        _, g_ex, gn, _, _ = rates(hi_t)
        print(f"    {yr}   <1.5x n={fn:4d} {f_ex*100:5.1f}%"
              f"    >=3x n={gn:4d} {g_ex*100:5.1f}%")

    hi3 = ups[ups["vol_x"] >= 3]
    by = hi3.groupby("ticker")["intraday"].agg(
        ["size", lambda s: (s > 0).sum() / max((s != 0).sum(), 1)])
    by.columns = ["gaps", "cont"]
    by = by.sort_values("gaps", ascending=False)
    print(f"\n  PER NAME at 3x+ ({len(hi3)} gaps, top3 = "
          f"{by['gaps'].head(3).sum()/max(len(hi3),1)*100:.0f}%)")
    for t, r in by.head(8).iterrows():
        print(f"    {t:7}{int(r['gaps']):4d} gaps  {r['cont']*100:5.1f}%")

    breadth = ups.groupby(ups.index).size()
    u2 = ups.copy()
    u2["breadth"] = u2.index.map(breadth)
    thr = max(2, int(np.ceil(0.15 * d["ticker"].nunique())))
    print(f"\n  ISOLATED vs CLUSTER at 3x+ (threshold {thr} names, excl flat)")
    for lab, sub in [("isolated", u2[(u2["breadth"] < thr) & (u2["vol_x"] >= 3)]),
                     ("cluster ", u2[(u2["breadth"] >= thr) & (u2["vol_x"] >= 3)])]:
        _, p_ex, n, nf, _ = rates(sub)
        _, l, h = wilson(int((sub["intraday"] > 0).sum()), nf)
        print(f"    {lab}  n={n:4d}  {p_ex*100:5.1f}% [{l*100:5.1f},{h*100:5.1f}]")


if __name__ == "__main__":
    for k, v in BASKETS.items():
        run(k, v)
    print("\ndone.\n")
