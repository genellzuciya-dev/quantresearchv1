"""
Fade Study — the SELL side of the supernova trade
==================================================
The scanner found the ENTRY (3x+ volume gap -> ~63% continue). This tool
solves the missing half: once you're in, WHEN does the move top out?

It reads INTRADAY bars and, for each big-gap session, measures the shape:
  - how far it ran from the open (the peak)
  - WHEN the peak landed (which part of the day)
  - how much it HELD by the close vs gave back
  - a verdict: ran & held / spike & fade / grind

Reads from EITHER source:
  1. Your thinkorswim CSV exports  (preferred — deep intraday history)
  2. Yahoo 5m  (fallback — only ~60 days, often empty for old gaps)

--- thinkorswim CSV mode -------------------------------------------------
Export a chart to CSV (per stock, per day or multi-day, 2m/5m). Drop the
files in a folder, then:
    python3 fade_study.py --csv ~/Downloads/tos_exports/
It auto-detects thinkorswim's column format (see load_tos_csv).

--- Yahoo mode -----------------------------------------------------------
    python3 fade_study.py ORIC VSTM OLMA
(Will say "no big gaps in window" if none fell in the last ~60 days.)

Needs only: pandas, numpy  (+ yfinance for Yahoo mode)
"""

import sys, os, glob
from datetime import date
import numpy as np
import pandas as pd

GAP_MIN = 0.05          # only study sessions that gapped at least this much
INTERVAL = "5m"
INTRADAY_PERIOD = "60d"


# ---------------------------------------------------------------- loaders
def load_tos_csv(path):
    """Load a thinkorswim chart-export CSV into OHLCV with a datetime index.
    thinkorswim exports vary; we handle the common layouts:
      - columns like Time/Date, Open, High, Low, Close, Volume
      - a combined 'Time' with date+time, or separate Date & Time
    """
    raw = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}

    # find datetime
    dt = None
    if "time" in cols and "date" in cols:
        dt = pd.to_datetime(raw[cols["date"]].astype(str) + " "
                            + raw[cols["time"]].astype(str), errors="coerce")
    elif "datetime" in cols:
        dt = pd.to_datetime(raw[cols["datetime"]], errors="coerce")
    elif "time" in cols:
        dt = pd.to_datetime(raw[cols["time"]], errors="coerce")
    elif "date" in cols:
        dt = pd.to_datetime(raw[cols["date"]], errors="coerce")
    else:
        dt = pd.to_datetime(raw.iloc[:, 0], errors="coerce")

    def col(*names):
        for n in names:
            if n in cols:
                return raw[cols[n]]
        return None

    out = pd.DataFrame({
        "Open": col("open"), "High": col("high"),
        "Low": col("low"), "Close": col("close", "last"),
        "Volume": col("volume", "vol"),
    })
    out.index = dt
    out = out[out.index.notna()].dropna(subset=["Open", "Close"])
    return out.sort_index()


def load_csv_folder(folder):
    """Load every CSV in a folder, group bars by ticker guessed from filename
    (e.g. ORIC_2026-07-25_5m.csv -> ORIC)."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        print(f"No CSVs found in {folder}")
        return {}
    by_ticker = {}
    for f in files:
        name = os.path.basename(f)
        ticker = name.split("_")[0].split(".")[0].upper()
        try:
            bars = load_tos_csv(f)
            if ticker in by_ticker:
                by_ticker[ticker] = pd.concat([by_ticker[ticker], bars])
            else:
                by_ticker[ticker] = bars
            print(f"  loaded {name} -> {ticker} ({len(bars)} bars)")
        except Exception as e:
            print(f"  skipped {name}: {e}")
    return {t: b.sort_index()[~b.sort_index().index.duplicated()]
            for t, b in by_ticker.items()}


def get_intraday_yahoo(ticker):
    import yfinance as yf
    df = yf.download(ticker, period=INTRADAY_PERIOD, interval=INTERVAL,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# --------------------------------------------------------------- analysis
def session_shapes(bars):
    """For each trading day in the bars, if it gapped >= GAP_MIN vs the prior
    session's close, describe the intraday shape."""
    if bars.empty:
        return []
    bars = bars.sort_index()
    days = sorted(set(bars.index.normalize()))
    shapes = []
    prev_close = None
    for day in days:
        d = bars[bars.index.normalize() == day]
        if len(d) < 5:
            prev_close = d["Close"].iloc[-1] if len(d) else prev_close
            continue
        o = d["Open"].iloc[0]
        gap = (o / prev_close - 1) if prev_close else 0.0
        prev_close = d["Close"].iloc[-1]
        if gap < GAP_MIN:
            continue
        hi_i = int(d["High"].values.argmax())
        hi = d["High"].iloc[hi_i]
        lo_after_i = None
        close = d["Close"].iloc[-1]
        run = (hi / o - 1) * 100
        held = (close / o - 1) * 100
        give_back = run - held
        frac = hi_i / (len(d) - 1)
        when = ("first 30m" if frac < 0.08 else "morning" if frac < 0.33
                else "midday" if frac < 0.66 else "afternoon" if frac < 0.92
                else "close")
        shape = ("ran & held" if held > run * 0.6 else
                 "spike & fade" if held < run * 0.3 else "grind")
        # time of peak in clock terms (approx, assumes 9:30 open, bar spacing)
        peak_time = d.index[hi_i].strftime("%H:%M")
        shapes.append({"date": day.date(), "gap": gap * 100, "run": run,
                       "held": held, "give_back": give_back, "peak_when": when,
                       "peak_time": peak_time, "shape": shape})
    return shapes


def summarize(all_shapes):
    """Aggregate across all gap sessions: where does the peak usually land,
    and how often does the move hold vs fade?"""
    if not all_shapes:
        return None
    df = pd.DataFrame(all_shapes)
    when_counts = df["peak_when"].value_counts()
    shape_counts = df["shape"].value_counts()
    return {
        "n": len(df),
        "avg_run": df["run"].mean(),
        "avg_held": df["held"].mean(),
        "avg_giveback": df["give_back"].mean(),
        "pct_held": (df["shape"] == "ran & held").mean() * 100,
        "pct_faded": (df["shape"] == "spike & fade").mean() * 100,
        "peak_when": when_counts.to_dict(),
        "shape_mix": shape_counts.to_dict(),
    }


# --------------------------------------------------------------- rendering
def heat(p):
    if p is None or pd.isna(p):
        return "background:#151a21;color:#5a6270"
    t = min(max(p, 0), 1) ** 0.7
    r, g, b = int(0x1b + t * 20), int(0x21 + t * 158), int(0x2b + t * 100)
    return f"background:rgb({r},{g},{b});color:{'#0e1116' if t > 0.55 else '#e8e6e0'}"


def shape_color(shape):
    return {"ran & held": "var(--win)", "spike & fade": "var(--loss)",
            "grind": "var(--amber)"}.get(shape, "var(--dim)")


def summary_card(summ, source):
    if not summ:
        return ('<section class="card"><h2>No gap sessions found</h2>'
                '<div class="note">No sessions gapped &ge;5% in the available '
                'intraday data. In Yahoo mode this is expected (only ~60 days). '
                'Export thinkorswim CSVs of your actual gap days and run with '
                '<code>--csv</code> to build this out.</div></section>')
    when_order = ["first 30m", "morning", "midday", "afternoon", "close"]
    when_bars = ""
    total = summ["n"]
    for w in when_order:
        c = summ["peak_when"].get(w, 0)
        pct = c / total * 100 if total else 0
        when_bars += (f'<div class="wrow"><span class="wlabel">{w}</span>'
                      f'<div class="wtrack"><div class="wfill" style="width:{pct:.0f}%"></div></div>'
                      f'<span class="wpct">{pct:.0f}% ({c})</span></div>')
    return (f'<section class="card pooled"><div class="card-head">'
            f'<h2>Fade summary</h2><span class="range">{summ["n"]} gap sessions · '
            f'{source}</span></div>'
            f'<div class="stats">'
            f'<div class="stat"><div class="label">Avg run from open</div>'
            f'<div class="big" style="color:var(--win)">+{summ["avg_run"]:.1f}%</div>'
            f'<div class="sub">peak reached intraday</div></div>'
            f'<div class="stat"><div class="label">Avg held to close</div>'
            f'<div class="big">{summ["avg_held"]:+.1f}%</div>'
            f'<div class="sub">what you keep if you hold all day</div></div>'
            f'<div class="stat"><div class="label">Avg given back</div>'
            f'<div class="big" style="color:var(--loss)">-{summ["avg_giveback"]:.1f}%</div>'
            f'<div class="sub">run minus held = the fade</div></div>'
            f'<div class="stat"><div class="label">Ran &amp; held</div>'
            f'<div class="big">{summ["pct_held"]:.0f}%</div>'
            f'<div class="sub">vs {summ["pct_faded"]:.0f}% spike &amp; fade</div></div>'
            f'</div>'
            f'<h3>When does the peak land? <span class="hint">this is your exit '
            f'timing \u2014 sell into strength before the fade</span></h3>'
            f'<div class="whenchart">{when_bars}</div>'
            f'<div class="note"><em>How to use this.</em> If most peaks land in '
            f'the first 30m / morning, the move is a <em>spike</em> \u2014 you exit '
            f'into the open ramp, you do not hold. If peaks spread to midday/'
            f'afternoon, the move <em>trends</em> and a trailing stop makes sense. '
            f'The "given back" number is what selling at the close costs you vs '
            f'selling at the peak \u2014 the bigger it is, the more exit timing '
            f'matters.</div></section>')


def detail_table(shapes):
    if not shapes:
        return ""
    rows = ""
    for s in sorted(shapes, key=lambda x: x["date"], reverse=True)[:40]:
        rows += (f'<tr><th>{s["date"]}</th>'
                 f'<td class="dim">gap {s["gap"]:+.0f}%</td>'
                 f'<td class="win">ran +{s["run"]:.0f}%</td>'
                 f'<td class="dim">held {s["held"]:+.0f}%</td>'
                 f'<td class="loss">gave back {s["give_back"]:.0f}%</td>'
                 f'<td class="dim">peak {s["peak_time"]} ({s["peak_when"]})</td>'
                 f'<td style="color:{shape_color(s["shape"])}">{s["shape"]}</td></tr>')
    return (f'<section class="card"><div class="card-head"><h2>Every gap session</h2>'
            f'<span class="range">newest first</span></div>'
            f'<table class="matrix"><tbody>{rows}</tbody></table></section>')


CSS = """:root{--ink:#0e1116;--panel:#151a21;--panel2:#1b212b;--line:#28303c;
--paper:#e8e6e0;--dim:#8b93a0;--faint:#5a6270;--win:#2fbf8f;--loss:#e4574f;--amber:#e5a83b}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--ink);color:var(--paper);font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:15px;line-height:1.6}
.wrap{max-width:1060px;margin:0 auto;padding:0 28px 80px}
header{padding:44px 0 8px}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
h1{font-family:Archivo,sans-serif;font-size:30px;margin:6px 0 2px}
.subtitle{color:var(--dim);font-size:14px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px 26px;margin:22px 0}
.card.pooled{border-color:var(--amber)}
.card-head{display:flex;align-items:baseline;gap:14px;margin-bottom:12px}
.card-head h2{font-family:Archivo,sans-serif;font-size:22px}
.range{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px 14px}
.stat .label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.stat .big{font-family:Archivo,sans-serif;font-size:24px;font-weight:600;margin:2px 0}
.stat .sub{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--faint)}
h3{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin:16px 0 10px;font-weight:500}
.hint{text-transform:none;letter-spacing:0;color:var(--faint);font-size:11px;margin-left:8px}
.whenchart{display:flex;flex-direction:column;gap:6px;margin-bottom:6px}
.wrow{display:flex;align-items:center;gap:10px;font-family:'IBM Plex Mono',monospace;font-size:12px}
.wlabel{width:78px;color:var(--dim);text-align:right}
.wtrack{flex:1;height:12px;background:var(--panel2);border-radius:6px;overflow:hidden}
.wfill{height:100%;background:var(--amber)}
.wpct{width:88px;color:var(--dim)}
table.matrix{border-collapse:separate;border-spacing:3px;width:100%;font-family:'IBM Plex Mono',monospace}
.matrix tbody th{text-align:left;color:var(--paper);font-size:12px;padding-right:10px}
.matrix td{text-align:center;padding:7px 6px;border-radius:6px;font-size:12px}
.matrix td.dim{background:var(--panel2);color:var(--dim)}
.matrix td.win{background:var(--panel2);color:var(--win)}
.matrix td.loss{background:var(--panel2);color:var(--loss)}
.note{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px 16px;font-size:13px;color:var(--dim);margin-top:12px}
.note code{font-family:'IBM Plex Mono',monospace;color:var(--paper)}
.note em{color:var(--paper);font-style:normal}
footer{margin-top:36px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);text-align:center}"""


def build(shapes, source):
    summ = summarize(shapes)
    body = summary_card(summ, source) + detail_table(shapes)
    html = (f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Fade Study</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
            f'family=IBM+Plex+Sans:wght@400;500;600&family=Archivo:wght@600;700&display=swap" rel="stylesheet">'
            f'<style>{CSS}</style></head><body><div class="wrap"><header>'
            f'<div class="eyebrow">Fade study · the SELL side · gaps &ge;{GAP_MIN*100:.0f}%</div>'
            f'<h1>Fade Study</h1>'
            f'<div class="subtitle">The scanner found the entry. This finds the exit: '
            f'once a gap runs, when does it top out?</div></header>'
            f'{body}<footer>Generated {date.today()} · not trade advice</footer>'
            f'</div></body></html>')
    with open("fade_study.html", "w") as f:
        f.write(html)
    print(f"\nWrote fade_study.html  ({summ['n'] if summ else 0} gap sessions)")
    print("Open with:  open fade_study.html")


def main(argv):
    if argv and argv[0] == "--csv":
        folder = argv[1] if len(argv) > 1 else "."
        print(f"Loading thinkorswim CSVs from {folder}...")
        data = load_csv_folder(folder)
        all_shapes = []
        for t, bars in data.items():
            s = session_shapes(bars)
            for x in s:
                x["ticker"] = t
            all_shapes += s
            print(f"  {t}: {len(s)} gap sessions")
        build(all_shapes, f"thinkorswim CSV · {len(data)} tickers")
    else:
        tickers = [t.upper() for t in argv] or ["ORIC"]
        print(f"Yahoo mode ({INTERVAL}, ~{INTRADAY_PERIOD})...")
        all_shapes = []
        for t in tickers:
            try:
                bars = get_intraday_yahoo(t)
                s = session_shapes(bars)
                for x in s:
                    x["ticker"] = t
                all_shapes += s
                print(f"  {t}: {len(s)} gap sessions")
            except Exception as e:
                print(f"  {t}: skipped ({e})")
        build(all_shapes, f"Yahoo {INTERVAL}")


if __name__ == "__main__":
    main(sys.argv[1:])
