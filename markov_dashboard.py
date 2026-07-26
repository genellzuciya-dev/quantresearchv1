"""
Markov Regime Dashboard  (v2 — with run history)
================================================
Analyzes tickers, writes markov_regimes.html (current run) AND
markov_history.html (every run you've ever done, dated).

History lives in markov_history.json next to the script — plain text,
so you can back it up, version it, or read it yourself.

Run:
    python3 markov_dashboard.py MRVL CRWV AAOI NVDA SPY
    open markov_regimes.html
"""

import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

LOOKBACK = "2y"
GAP_THRESHOLD = 0.05
SMA_LEN = 20
STATES = ["GAP_UP", "GAP_DOWN", "TREND_UP", "TREND_DOWN", "CHOP"]
LBL = {"GAP_UP": "Gap U", "GAP_DOWN": "Gap D", "TREND_UP": "Trend U",
       "TREND_DOWN": "Trend D", "CHOP": "Chop"}
HISTORY_FILE = "markov_history.json"


# ---------------------------------------------------------------- analysis
def get_daily(ticker):
    import yfinance as yf
    df = yf.download(ticker, period=LOOKBACK, interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def label_states(df):
    sma = df["Close"].rolling(SMA_LEN).mean()
    slope = sma.diff()
    gap = df["Open"] / df["Close"].shift(1) - 1
    cond = [gap >= GAP_THRESHOLD, gap <= -GAP_THRESHOLD,
            (df["Close"] > sma) & (slope > 0),
            (df["Close"] < sma) & (slope < 0)]
    lab = np.select(cond, ["GAP_UP", "GAP_DOWN", "TREND_UP", "TREND_DOWN"],
                    default="CHOP")
    return pd.Series(lab, index=df.index, name="state").iloc[SMA_LEN:]


def transition_matrix(s):
    c = pd.DataFrame(0, index=STATES, columns=STATES, dtype=float)
    for a, b in zip(s[:-1], s[1:]):
        c.loc[a, b] += 1
    return c.div(c.sum(axis=1).replace(0, np.nan), axis=0), c


def shuffled_baseline(s, n=200):
    d, vals = [], np.asarray(s.tolist())
    rng = np.random.default_rng(42)
    for _ in range(n):
        rng.shuffle(vals)
        p, _ = transition_matrix(pd.Series(vals))
        d.append(np.diag(p.fillna(0).values))
    return pd.Series(np.mean(d, axis=0), index=STATES)


def analyze(t):
    df = get_daily(t)
    s = label_states(df)
    p, c = transition_matrix(s)
    b = shuffled_baseline(s)
    dur = {x: (1 / (1 - p.loc[x, x])
               if pd.notna(p.loc[x, x]) and p.loc[x, x] < 1 else None)
           for x in STATES}
    return {"ticker": t, "n_days": len(s), "start": str(s.index[0].date()),
            "end": str(s.index[-1].date()),
            "freq": s.value_counts(normalize=True).reindex(STATES).fillna(0),
            "probs": p, "counts": c, "baseline": b, "durations": dur,
            "gap_days": int((s == "GAP_UP").sum() + (s == "GAP_DOWN").sum())}


# ---------------------------------------------------------------- history
def to_record(res):
    """Flatten one analysis into a JSON-safe row for the history file."""
    p = res["probs"]
    return {
        "ticker": res["ticker"],
        "run_date": str(date.today()),
        "window": f'{res["start"]} to {res["end"]}',
        "n_days": res["n_days"],
        "gap_days": res["gap_days"],
        "freq": {s: round(float(res["freq"][s]), 3) for s in STATES},
        "persist": {s: (None if pd.isna(p.loc[s, s]) else round(float(p.loc[s, s]), 3))
                    for s in STATES},
        "baseline": {s: round(float(res["baseline"][s]), 3) for s in STATES},
        "duration": {s: (None if res["durations"][s] is None
                         else round(float(res["durations"][s]), 1)) for s in STATES},
        # the cross-ticker regularity worth tracking over time:
        "up_to_down": (None if pd.isna(p.loc["TREND_UP", "TREND_DOWN"])
                       else round(float(p.loc["TREND_UP", "TREND_DOWN"]), 3)),
        "chop_resolves_up": (None if pd.isna(p.loc["CHOP", "TREND_UP"])
                             else round(float(p.loc["CHOP", "TREND_UP"]), 3)),
        "chop_resolves_down": (None if pd.isna(p.loc["CHOP", "TREND_DOWN"])
                               else round(float(p.loc["CHOP", "TREND_DOWN"]), 3)),
        # full matrix, so the history page can render the same card as the live run
        "probs": {a: {b: (None if pd.isna(p.loc[a, b]) else round(float(p.loc[a, b]), 3))
                      for b in STATES} for a in STATES},
        "counts": {a: {b: int(res["counts"].loc[a, b]) for b in STATES} for a in STATES},
    }


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception as e:
        print(f"  (couldn't read {HISTORY_FILE}: {e} — starting fresh)")
        return []


def save_history(records):
    """Append today's records; one entry per ticker per day (re-runs replace)."""
    hist = load_history()
    keys = {(r["ticker"], r["run_date"]) for r in records}
    hist = [h for h in hist if (h["ticker"], h["run_date"]) not in keys]
    hist.extend(records)
    hist.sort(key=lambda r: (r["run_date"], r["ticker"]), reverse=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(hist, f, indent=1)
    return hist


# ---------------------------------------------------------------- rendering
def heat(p):
    if p is None or pd.isna(p):
        return "background:#151a21;color:#5a6270"
    t = min(max(p, 0), 1) ** 0.7
    r, g, b = int(0x1b + t * 20), int(0x21 + t * 158), int(0x2b + t * 100)
    return f"background:rgb({r},{g},{b});color:{'#0e1116' if t > 0.55 else '#e8e6e0'}"


COLORS = {"GAP_UP": "#e5a83b", "GAP_DOWN": "#b06a2c", "TREND_UP": "#2fbf8f",
          "TREND_DOWN": "#e4574f", "CHOP": "#5a6270"}


def card(rec):
    """Render one full ticker card from a history record. Used by BOTH pages,
    so the latest run and the archive always show identical detail."""
    segs = legend = ""
    for s in STATES:
        pct = rec["freq"][s] * 100
        if pct > 0:
            segs += f'<div style="width:{pct:.1f}%;background:{COLORS[s]}"></div>'
            legend += (f'<span><i style="background:{COLORS[s]}"></i>'
                       f'{LBL[s]} {pct:.0f}%</span>')
    stats = ""
    for s in ["TREND_UP", "TREND_DOWN", "CHOP"]:
        d, real, base = rec["duration"][s], rec["persist"][s], rec["baseline"][s]
        if d is None or real is None:
            continue
        stats += (f'<div class="stat"><div class="label">{LBL[s]}</div>'
                  f'<div class="big">{d}<span class="unit">days</span></div>'
                  f'<div class="sub">persist {format(real, ".2f")} '
                  f'<span class="edge">vs {format(base, ".2f")} rand</span></div></div>')
    stats += (f'<div class="stat"><div class="label">Gap days</div>'
              f'<div class="big">{rec["gap_days"]}'
              f'<span class="unit">/ {rec["n_days"]}</span></div>'
              f'<div class="sub">catalyst freq</div></div>')

    matrix = ""
    if rec.get("probs"):
        head = "".join(f"<th>{LBL[s]}</th>" for s in STATES)
        rows = ""
        for a in STATES:
            cells = ""
            for b in STATES:
                p = rec["probs"][a][b]
                n = rec["counts"][a][b]
                v = "-" if p is None else format(p, ".2f")
                cells += f'<td style="{heat(p)}">{v}<span class="n">{n}</span></td>'
            rows += f"<tr><th>{LBL[a]}</th>{cells}</tr>"
        matrix = (f'<h3>Transition matrix <span class="hint">P(next | current)</span></h3>'
                  f'<table class="matrix"><thead><tr><th></th>{head}</tr></thead>'
                  f'<tbody>{rows}</tbody></table>')

    return (f'<section class="card"><div class="card-head"><h2>{rec["ticker"]}</h2>'
            f'<span class="range">{rec["window"]} · {rec["n_days"]} bars</span></div>'
            f'<div class="freqbar">{segs}</div><div class="legend">{legend}</div>'
            f'<div class="stats">{stats}</div>{matrix}</section>')


def history_page(hist):
    by_date = {}
    for r in hist:
        by_date.setdefault(r["run_date"], []).append(r)

    blocks = ""
    for run_date in sorted(by_date, reverse=True):
        recs = sorted(by_date[run_date], key=lambda x: x["ticker"])
        rows = ""
        for r in recs:
            up, dn, ch = r["persist"]["TREND_UP"], r["persist"]["TREND_DOWN"], r["persist"]["CHOP"]
            dur_up, u2d = r["duration"]["TREND_UP"], r["up_to_down"]
            base_up = r["baseline"]["TREND_UP"]
            edge = None if up is None or base_up is None else up - base_up
            rows += (
                f'<tr><th>{r["ticker"]}</th>'
                f'<td class="dim">{r["n_days"]}</td>'
                f'<td style="{heat(up)}">{"-" if up is None else format(up, ".2f")}</td>'
                f'<td style="{heat(dn)}">{"-" if dn is None else format(dn, ".2f")}</td>'
                f'<td style="{heat(ch)}">{"-" if ch is None else format(ch, ".2f")}</td>'
                f'<td class="dim">{"-" if dur_up is None else dur_up}</td>'
                f'<td class="dim">{r["gap_days"]}</td>'
                f'<td class="dim">{"-" if u2d is None else format(u2d, ".3f")}</td>'
                f'<td class="edge">{"-" if edge is None else "+" + format(edge, ".2f")}</td>'
                f'<td class="dim">{format(r["freq"]["TREND_UP"], ".0%")}</td></tr>')

        head = ('<th></th><th>bars</th><th>Trend U<br>persist</th><th>Trend D<br>persist</th>'
                '<th>Chop<br>persist</th><th>Up<br>days</th><th>gap<br>days</th>'
                '<th>U&rarr;D</th><th>edge vs<br>random</th><th>% time<br>up</th>')
        blocks += (f'<div class="daymark"><span class="daydate">{run_date}</span>'
                   f'<span class="range">{len(recs)} ticker'
                   f'{"s" if len(recs) != 1 else ""}</span></div>'
                   f'<section class="card"><h3>Side by side</h3>'
                   f'<table class="matrix hist"><thead><tr>{head}</tr></thead>'
                   f'<tbody>{rows}</tbody></table></section>')
        blocks += "".join(card(r) for r in recs)

    tickers = sorted({r["ticker"] for r in hist})
    summary = (f'<div class="note">{len(hist)} runs logged across {len(tickers)} '
               f'ticker{"s" if len(tickers) != 1 else ""}: {", ".join(tickers)}. '
               f'Stored in <code>markov_history.json</code> — back that file up and '
               f'your research log survives anything.</div>')
    return page("Run History",
                "Every analysis you've run, newest first — same detail as the live run. "
                "Compare a ticker across dates to see whether its structure is stable or drifting.",
                summary + blocks,
                nav='<a href="markov_regimes.html">&larr; Latest run</a>')


CSS = """:root{--ink:#0e1116;--panel:#151a21;--panel2:#1b212b;--line:#28303c;
--paper:#e8e6e0;--dim:#8b93a0;--faint:#5a6270;--win:#2fbf8f}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--ink);color:var(--paper);font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:15px;line-height:1.6}
.wrap{max-width:1060px;margin:0 auto;padding:0 28px 80px}
header{padding:44px 0 8px}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
h1{font-family:Archivo,sans-serif;font-size:30px;margin:6px 0 2px}
.subtitle{color:var(--dim);font-size:14px;margin-bottom:10px}
nav{font-family:'IBM Plex Mono',monospace;font-size:12px;margin-bottom:22px}
nav a{color:var(--win);text-decoration:none;border-bottom:1px solid transparent}
nav a:hover{border-color:var(--win)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px 26px;margin:22px 0}
.card-head{display:flex;align-items:baseline;gap:14px;margin-bottom:14px}
.card-head h2{font-family:Archivo,sans-serif;font-size:22px}
.range{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint)}
.freqbar{display:flex;height:14px;border-radius:7px;overflow:hidden;border:1px solid var(--line)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 18px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--dim)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px 14px}
.stat .label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.stat .big{font-family:Archivo,sans-serif;font-size:24px;font-weight:600;margin:2px 0}
.stat .unit{font-size:12px;color:var(--faint);margin-left:4px;font-weight:400}
.stat .sub{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--faint)}
.stat .edge{color:var(--win)}
h3{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin:8px 0 10px;font-weight:500}
.hint{text-transform:none;letter-spacing:0;color:var(--faint);font-size:11px;margin-left:8px}
table.matrix{border-collapse:separate;border-spacing:3px;width:100%;font-family:'IBM Plex Mono',monospace}
.matrix th{font-size:11px;color:var(--dim);font-weight:500;padding:4px 6px;text-align:center;line-height:1.3}
.matrix tbody th{text-align:right;padding-right:10px;color:var(--paper);font-size:13px}
.matrix td{text-align:center;padding:8px 4px 5px;border-radius:6px;font-size:13px}
.matrix td .n{display:block;font-size:9px;opacity:.65;margin-top:1px}
.hist td.dim{background:var(--panel2);color:var(--dim)}
.hist td.edge{background:var(--panel2);color:var(--win)}
.daymark{display:flex;align-items:baseline;gap:12px;margin:38px 0 -6px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.daydate{font-family:Archivo,sans-serif;font-size:20px;font-weight:700}
.note{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px 16px;font-size:13px;color:var(--dim);margin-top:18px}
.note code{font-family:'IBM Plex Mono',monospace;color:var(--paper)}
footer{margin-top:36px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);text-align:center}"""


def page(title, subtitle, body, nav=""):
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
            f'family=IBM+Plex+Sans:wght@400;500;600&family=Archivo:wght@600;700&display=swap" rel="stylesheet">'
            f'<style>{CSS}</style></head><body><div class="wrap"><header>'
            f'<div class="eyebrow">Regime study · {LOOKBACK} daily bars · 20-SMA + 5% gaps</div>'
            f'<h1>{title}</h1><div class="subtitle">{subtitle}</div>'
            f'<nav>{nav}</nav></header>{body}'
            f'<footer>Generated {date.today()}</footer></div></body></html>')


# ---------------------------------------------------------------- main
def build(tickers):
    cards, records = "", []
    for t in tickers:
        print(f"Analyzing {t}...")
        try:
            rec = to_record(analyze(t))
            cards += card(rec)
            records.append(rec)
        except Exception as e:
            print(f"  skipped {t}: {e}")

    if not records:
        print("Nothing analyzed — check the ticker symbols.")
        return

    with open("markov_regimes.html", "w") as f:
        f.write(page("Markov Regime Dashboard",
                     "Rows are today's state, columns tomorrow's. Greener = more likely. "
                     "Small number = how many times it happened.",
                     cards + '<div class="note">In-sample only. Trust cells with high '
                     'counts; gap rows are usually thin. Hypotheses, not trade signals.</div>',
                     nav='<a href="markov_history.html">Run history &rarr;</a>'))

    hist = save_history(records)
    with open("markov_history.html", "w") as f:
        f.write(history_page(hist))

    print(f"\nWrote markov_regimes.html  (this run: {', '.join(r['ticker'] for r in records)})")
    print(f"Wrote markov_history.html  ({len(hist)} runs logged)")
    print("Open with:  open markov_regimes.html")


if __name__ == "__main__":
    build([t.upper() for t in sys.argv[1:]] or ["ORIC", "NVDA", "SPY"])
