"""
Cannabis clean pass
===================
Two fixes, then rerun the same tests:

  1. DROP CORRUPT ROWS. SHWZ has days Yahoo never split-adjusted:
     +1718%, +1144%, +126% intraday, and a +6567% gap. Those are data
     errors, not trades. Any row with a gap or intraday move beyond
     +/-100% gets thrown out, whatever the ticker.

  2. DROP NON-LISTED NAMES. SHWZ and MRMD are OTCQX, which the earlier
     filter missed, plus the 7 OTC names and 2 ETFs it did catch.
     OTC opening prints are thin and often stale, which manufactures
     both fake gaps and fake fades.

Everything is printed before AND after so you can see exactly what the
cleaning changed. Run from the Trading Research folder:

    python3 cannabis_clean.py | tee cannabis_clean.txt
"""

import numpy as np
import pandas as pd

FULL = ["TLRY", "CGC", "ACB", "SNDL", "OGI", "CRON", "VFF", "GRWG", "AKAN",
        "CURLF", "GTBIF", "TCNNF", "CRLBF", "AAWH", "MSOS", "YOLO", "HITI",
        "SHWZ", "MRMD", "GDNSF", "TSNDF", "CLVR"]

NOT_LISTED = {"MSOS": "ETF", "YOLO": "ETF", "CURLF": "OTC", "GTBIF": "OTC",
              "TCNNF": "OTC", "CRLBF": "OTC", "AAWH": "OTC", "GDNSF": "OTC",
              "TSNDF": "OTC", "SHWZ": "OTC (missed before)",
              "MRMD": "OTC (missed before)"}

LISTED = [t for t in FULL if t not in NOT_LISTED]
SANE = 1.00          # anything beyond +/-100% in a day is a data error
TIERS = [(0, 1.5, "<1.5x"), (1.5, 2, "1.5-2x"), (2, 3, "2-3x"),
         (3, 5, "3-5x"), (5, 999, "5x+")]


def load(tickers):
    import yfinance as yf
    frames = []
    for t in tickers:
        df = yf.download(t, period="2y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 40:
            print(f"  skipped {t}: not enough history")
            continue
        d = df.copy()
        d["gap"] = d["Open"] / d["Close"].shift(1) - 1
        d["intraday"] = d["Close"] / d["Open"] - 1
        d["vol_x"] = d["Volume"] / d["Volume"].shift(1).rolling(20).mean()
        d = d.dropna()
        d["ticker"] = t
        frames.append(d)
    return pd.concat(frames)


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def report(d, label):
    print(f"\n{'='*66}\n  {label}\n{'='*66}")
    print(f"  {d['ticker'].nunique()} names · {len(d)} trading days")

    n = len(d)
    k = int((d["intraday"] > 0).sum())
    base, blo, bhi = wilson(k, n)
    print(f"\n  BASELINE (close>open, all days): {base*100:.1f}% "
          f"[{blo*100:.1f}, {bhi*100:.1f}]")

    ups = d[d["gap"] >= 0.03]
    print(f"\n  TIERS ({len(ups)} up-gaps >=3%)")
    print(f"  {'tier':8}{'n':>6}{'cont':>8}{'95% CI':>18}"
          f"{'vs base':>10}{'mean':>8}{'median':>8}")
    for lo, hi, lab in TIERS:
        t = ups[(ups["vol_x"] >= lo) & (ups["vol_x"] < hi)]
        p, l, h = wilson(int((t["intraday"] > 0).sum()), len(t))
        if len(t) == 0:
            continue
        print(f"  {lab:8}{len(t):6d}{p*100:7.1f}% [{l*100:6.1f},{h*100:6.1f}]"
              f"{(p-base)*100:+9.1f}{t['intraday'].mean()*100:7.1f}%"
              f"{t['intraday'].median()*100:7.1f}%")

    print(f"\n  BY YEAR")
    ups2 = ups.copy()
    ups2["year"] = ups2.index.year
    for yr, sub in ups2.groupby("year"):
        lo_t, hi_t = sub[sub["vol_x"] < 1.5], sub[sub["vol_x"] >= 3]
        f = (lo_t["intraday"] > 0).mean() if len(lo_t) else np.nan
        g = (hi_t["intraday"] > 0).mean() if len(hi_t) else np.nan
        print(f"    {yr}   <1.5x n={len(lo_t):4d} {f*100:5.1f}%"
              f"    >=3x n={len(hi_t):4d} {g*100:5.1f}%")

    hi3 = ups[ups["vol_x"] >= 3]
    by = hi3.groupby("ticker")["intraday"].agg(["size", lambda s: (s > 0).mean()])
    by.columns = ["gaps", "cont"]
    by = by.sort_values("gaps", ascending=False)
    print(f"\n  PER NAME at 3x+ ({len(hi3)} gaps, top3 = "
          f"{by['gaps'].head(3).sum()/max(len(hi3),1)*100:.0f}%)")
    for t, r in by.iterrows():
        print(f"    {t:7}{int(r['gaps']):4d} gaps  {r['cont']*100:5.1f}%")

    breadth = ups.groupby(ups.index).size()
    u = ups.copy()
    u["breadth"] = u.index.map(breadth)
    thr = max(2, int(np.ceil(0.15 * d["ticker"].nunique())))
    print(f"\n  ISOLATED vs CLUSTER at 3x+ (threshold {thr} names)")
    for lab, sub in [("isolated", u[(u["breadth"] < thr) & (u["vol_x"] >= 3)]),
                     ("cluster ", u[(u["breadth"] >= thr) & (u["vol_x"] >= 3)])]:
        p, l, h = wilson(int((sub["intraday"] > 0).sum()), len(sub))
        print(f"    {lab}  n={len(sub):4d}  {p*100:5.1f}% "
              f"[{l*100:5.1f},{h*100:5.1f}]")


if __name__ == "__main__":
    print("Downloading full basket...")
    raw = load(FULL)

    bad = raw[(raw["intraday"].abs() > SANE) | (raw["gap"].abs() > SANE)]
    print(f"\nCORRUPT ROWS DROPPED: {len(bad)}")
    for ix, r in bad.sort_values("intraday", ascending=False).iterrows():
        print(f"   {ix.date()}  {r['ticker']:6} gap {r['gap']*100:+9.1f}%  "
              f"intraday {r['intraday']*100:+9.1f}%")
    clean = raw.drop(bad.index)

    print(f"\nNON-LISTED NAMES DROPPED: " +
          ", ".join(f"{k}({v})" for k, v in NOT_LISTED.items()))

    report(raw,  "A · AS PUBLISHED (all 22 names, corrupt rows included)")
    report(clean, "B · CORRUPT ROWS REMOVED ONLY")
    report(clean[clean["ticker"].isin(LISTED)],
           "C · CLEAN + LISTED ONLY  <- this is the one for the page")
    print("\ndone.\n")
