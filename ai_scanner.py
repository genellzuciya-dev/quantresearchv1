"""
Supernova Scanner — daily gap-continuation + intraday shape (Markov style)
==========================================================================
Hunts explosive moves instead of smooth trends. Two layers:

  LAYER 1 (daily): for each big overnight gap, does the DAY continue in the
                   gap's direction or fade? Split by gap size and volume.
                   This is the core probability: P(continue | gapped big).

  LAYER 2 (intraday): for the biggest recent gap days, zoom into 5-min bars
                   and show the SHAPE — did it run all day, spike-and-fade,
                   or grind? So you see what a real ignition looks like.

Supernova logic, not trend logic:
  - The GAP is the signal, not noise. We build the whole tool around it.
  - VOLUME confirms: a gap on huge volume is real; on light volume it's a trap.
  - FLOAT is the fuel: low float = violent moves. We fetch & show it.

Run:
    python3 supernova_scanner.py                      # default biotech basket
    python3 supernova_scanner.py ORIC NUVB OLMA VSTM
    open ai_supernova.html

Needs only: yfinance, pandas, numpy  (all already installed)
"""

import sys
from datetime import date
import numpy as np
import pandas as pd

# Small-cap / clinical-stage biotech basket (+ ORIC). Edit freely.
DEFAULT_BASKET = [
    # low-float AI / small-cap tech complex — biotech analog with
    # STOCK-SPECIFIC catalysts (customer wins, chip contracts, AI-pivot news).
    # Deliberately NOT the mega-caps (NVDA, AMD) — those are too big to supernova.
    "SOUN", "BBAI", "AI", "VERI", "GFAI", "SERV", "RGTI", "QBTS",
    "IONQ", "LAES", "POET", "AEVA", "OUST", "MVIS", "KSCP", "AITX",
    "CXAI", "INOD", "AUR", "ARBE", "NNDM", "PRSO",
]

DAILY_LOOKBACK = "2y"
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "60d"
GAP_BUCKETS = [(0.03, 0.07, "3-7%"), (0.07, 0.15, "7-15%"),
               (0.15, 0.30, "15-30%"), (0.30, 10.0, "30%+")]
# Volume tiers (x normal). The study: at what volume multiple does gap
# continuation flip from coin-flip to reliable? That's your real threshold.
VOL_TIERS = [(0, 1.5, "<1.5x"), (1.5, 2, "1.5-2x"), (2, 3, "2-3x"),
             (3, 5, "3-5x"), (5, 999, "5x+")]
TOP_N_INTRADAY = 3          # zoom into this many biggest recent gaps


# ------------------------------------------------------------------ data
def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_daily(ticker):
    import yfinance as yf
    return _flatten(yf.download(ticker, period=DAILY_LOOKBACK, interval="1d",
                                auto_adjust=True, progress=False)).dropna()


def get_intraday(ticker):
    import yfinance as yf
    return _flatten(yf.download(ticker, period=INTRADAY_PERIOD,
                                interval=INTRADAY_INTERVAL,
                                auto_adjust=True, progress=False)).dropna()


def get_float(ticker):
    """Shares float in millions, or None. Low float = supernova fuel."""
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).get_info()
        f = info.get("floatShares") or info.get("sharesOutstanding")
        return (f / 1e6) if f else None
    except Exception:
        return None


# --------------------------------------------------------------- analysis
def gap_analysis(df):
    """For each day, overnight gap = open vs prior close. Continuation =
    did close finish beyond the open in the gap's direction (i.e. the gap
    kept going intraday)? Also flag volume vs its 20-day average."""
    d = df.copy()
    d["prev_close"] = d["Close"].shift(1)
    d["gap"] = d["Open"] / d["prev_close"] - 1
    d["intraday"] = d["Close"] / d["Open"] - 1        # open -> close move
    vol_avg = d["Volume"].rolling(20).mean()
    d["vol_x"] = d["Volume"] / vol_avg                # volume as x of normal
    d = d.dropna()

    rows = []
    for lo, hi, label in GAP_BUCKETS:
        up = d[(d["gap"] >= lo) & (d["gap"] < hi)]
        # continuation for an UP gap = positive intraday (kept running up)
        if len(up):
            cont = (up["intraday"] > 0).mean()
            avg_cont = up["intraday"].mean() * 100
            hi_vol = up[up["vol_x"] >= 2]
            cont_hivol = (hi_vol["intraday"] > 0).mean() if len(hi_vol) else None
            rows.append({"bucket": label, "dir": "UP", "n": len(up),
                         "cont": cont, "avg_move": avg_cont,
                         "n_hivol": len(hi_vol), "cont_hivol": cont_hivol})
    return d, rows


def volume_tier_analysis(d):
    """For UP gaps of any size >=3%, bucket by volume multiple and measure
    continuation. This is THE study: find where the edge turns on."""
    ups = d[d["gap"] >= 0.03]
    rows = []
    for lo, hi, label in VOL_TIERS:
        tier = ups[(ups["vol_x"] >= lo) & (ups["vol_x"] < hi)]
        if len(tier):
            cont = (tier["intraday"] > 0).mean()
            rows.append({"tier": label, "n": len(tier), "cont": cont,
                         "avg_move": tier["intraday"].mean() * 100})
        else:
            rows.append({"tier": label, "n": 0, "cont": None, "avg_move": None})
    return rows


def biggest_gaps(d, n):
    """Return the n largest up-gap days (date, gap, intraday, vol_x)."""
    ups = d[d["gap"] > 0].nlargest(n, "gap")
    return [{"date": ix.date(), "gap": r["gap"] * 100,
             "intraday": r["intraday"] * 100, "vol_x": r["vol_x"]}
            for ix, r in ups.iterrows()]


def intraday_shape(ticker, gap_dates):
    """For each big-gap date we can find in the 5m window, describe the shape:
    where the high came (early/mid/late) and how much of the run held."""
    try:
        bars = get_intraday(ticker)
    except Exception:
        return []
    if bars.empty:
        return []
    out = []
    for gd in gap_dates:
        day = bars[bars.index.normalize() == pd.Timestamp(gd["date"])]
        if len(day) < 5:
            continue
        o = day["Open"].iloc[0]
        hi_i = day["High"].values.argmax()
        hi = day["High"].iloc[hi_i]
        close = day["Close"].iloc[-1]
        run = (hi / o - 1) * 100
        held = (close / o - 1) * 100
        frac = hi_i / (len(day) - 1)           # 0 = open, 1 = close
        when = "early" if frac < 0.33 else ("midday" if frac < 0.66 else "late")
        give_back = run - held
        shape = ("ran & held" if held > run * 0.6 else
                 "spike & fade" if held < run * 0.3 else "grind")
        out.append({"date": gd["date"], "gap": gd["gap"], "run": run,
                    "held": held, "give_back": give_back, "peak_when": when,
                    "shape": shape})
    return out


# --------------------------------------------------------------- rendering
def heat(p):
    if p is None or pd.isna(p):
        return "background:#151a21;color:#5a6270"
    t = min(max(p, 0), 1) ** 0.7
    r, g, b = int(0x1b + t * 20), int(0x21 + t * 158), int(0x2b + t * 100)
    return f"background:rgb({r},{g},{b});color:{'#0e1116' if t > 0.55 else '#e8e6e0'}"


def ticker_card(ticker):
    try:
        df = get_daily(ticker)
    except Exception as e:
        return f'<section class="card"><h2>{ticker}</h2><div class="note">no data ({e})</div></section>'
    if len(df) < 40:
        return f'<section class="card"><h2>{ticker}</h2><div class="note">not enough history</div></section>'

    d, rows = gap_analysis(df)
    flt = get_float(ticker)
    flt_str = f"{flt:.0f}M float" if flt else "float n/a"
    flt_tag = ""
    if flt is not None:
        tier = ("ultra-low <20M" if flt < 20 else "low <75M" if flt < 75
                else "mid <300M" if flt < 300 else "high 300M+")
        flt_tag = f'<span class="flt {"hot" if flt < 75 else ""}">{flt_str} · {tier}</span>'

    # gap-continuation table
    grows = ""
    for r in rows:
        ch = heat(r["cont"])
        chv = heat(r["cont_hivol"])
        cv = "-" if r["cont_hivol"] is None else f"{r['cont_hivol']:.0%}"
        grows += (f'<tr><th>{r["bucket"]}</th>'
                  f'<td class="dim">{r["n"]}</td>'
                  f'<td style="{ch}">{r["cont"]:.0%}</td>'
                  f'<td class="dim">{r["avg_move"]:+.1f}%</td>'
                  f'<td class="dim">{r["n_hivol"]}</td>'
                  f'<td style="{chv}">{cv}</td></tr>')
    if not grows:
        grows = '<tr><td colspan="6" class="dim">no gaps in range</td></tr>'

    gtable = (f'<table class="matrix"><thead><tr><th>gap size</th><th>days</th>'
              f'<th>continued</th><th>avg day</th><th>hi-vol days</th>'
              f'<th>continued<br>(hi-vol)</th></tr></thead><tbody>{grows}</tbody></table>')

    # VOLUME-TIER STUDY: where does the edge turn on?
    vrows = ""
    vtiers = volume_tier_analysis(d)
    prev_cont = None
    for r in vtiers:
        if r["cont"] is None:
            vrows += (f'<tr><th>{r["tier"]}</th><td class="dim">0</td>'
                      f'<td class="dim">-</td><td class="dim">-</td>'
                      f'<td class="dim"></td></tr>')
            continue
        ch = heat(r["cont"])
        # flag the tier where continuation first crosses 60% (the turn-on)
        turn = ""
        if r["cont"] >= 0.6 and (prev_cont is None or prev_cont < 0.6):
            turn = '<span class="turn">edge turns on</span>'
        prev_cont = r["cont"]
        vrows += (f'<tr><th>{r["tier"]}</th>'
                  f'<td class="dim">{r["n"]}</td>'
                  f'<td style="{ch}">{r["cont"]:.0%}</td>'
                  f'<td class="dim">{r["avg_move"]:+.1f}%</td>'
                  f'<td class="verdict">{turn}</td></tr>')
    vtable = (f'<h3>Volume-tier study <span class="hint">at what volume does '
              f'continuation turn on? (all gaps &ge;3%)</span></h3>'
              f'<table class="matrix"><thead><tr><th>volume</th><th>gap days</th>'
              f'<th>continued</th><th>avg day</th><th></th></tr></thead>'
              f'<tbody>{vrows}</tbody></table>')

    # intraday shapes for biggest gaps
    bg = biggest_gaps(d, TOP_N_INTRADAY)
    shapes = intraday_shape(ticker, bg)
    if shapes:
        srows = ""
        for s in shapes:
            srows += (f'<tr><th>{s["date"]}</th>'
                      f'<td class="dim">gap {s["gap"]:+.0f}%</td>'
                      f'<td class="win">ran +{s["run"]:.0f}%</td>'
                      f'<td class="dim">held {s["held"]:+.0f}%</td>'
                      f'<td class="loss">gave back {s["give_back"]:.0f}%</td>'
                      f'<td class="dim">peak {s["peak_when"]}</td>'
                      f'<td class="shape">{s["shape"]}</td></tr>')
        stable = (f'<h3>Intraday shape · biggest recent gaps <span class="hint">'
                  f'{INTRADAY_INTERVAL} bars</span></h3>'
                  f'<table class="matrix"><tbody>{srows}</tbody></table>')
    else:
        stable = ('<h3>Intraday shape</h3><div class="note">No big gaps fell inside '
                  f'the {INTRADAY_PERIOD} intraday window (Yahoo only serves ~60 days '
                  'of 5m bars). The daily table above still covers 2 years.</div>')

    return (f'<section class="card"><div class="card-head"><h2>{ticker}</h2>{flt_tag}</div>'
            f'<h3>Daily gap continuation <span class="hint">does the gap keep running intraday?</span></h3>'
            f'{gtable}{vtable}{stable}</section>')


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
.card-head{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
.card-head h2{font-family:Archivo,sans-serif;font-size:22px}
.flt{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:6px;padding:2px 8px}
.flt.hot{color:var(--amber);border-color:var(--amber)}
h3{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin:16px 0 10px;font-weight:500}
.hint{text-transform:none;letter-spacing:0;color:var(--faint);font-size:11px;margin-left:8px}
table.matrix{border-collapse:separate;border-spacing:3px;width:100%;font-family:'IBM Plex Mono',monospace}
.matrix th{font-size:11px;color:var(--dim);font-weight:500;padding:4px 8px;text-align:center;line-height:1.3}
.matrix tbody th{text-align:left;color:var(--paper);font-size:12px}
.matrix td{text-align:center;padding:7px 6px;border-radius:6px;font-size:13px}
.matrix td.dim{background:var(--panel2);color:var(--dim)}
.matrix td.win{background:var(--panel2);color:var(--win)}
.matrix td.loss{background:var(--panel2);color:var(--loss)}
.matrix td.shape{background:var(--panel2);color:var(--amber);font-size:11px}
.matrix td.verdict{background:var(--panel2);text-align:left;padding-left:10px}
.turn{color:var(--amber);font-size:10px;letter-spacing:.06em;text-transform:uppercase}
.card.pooled{border-color:var(--amber)}
.conf{font-size:9px;padding:1px 5px;border-radius:4px;margin-left:4px}
.conf.solid{background:rgba(47,191,143,.15);color:var(--win)}
.conf.ok{background:rgba(229,168,59,.12);color:var(--amber)}
.conf.thin{background:rgba(228,87,79,.12);color:var(--loss)}
.note{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px 16px;font-size:13px;color:var(--dim);margin-top:12px}
.note em{color:var(--paper);font-style:normal}
footer{margin-top:36px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);text-align:center}"""


def pooled_volume_study(tickers):
    """Pool UP-gaps across ALL tickers, then measure continuation per volume
    tier. This is where thin per-stock samples become trustworthy totals."""
    frames = []
    for t in tickers:
        try:
            d, _ = gap_analysis(get_daily(t))
            ups = d[d["gap"] >= 0.03][["gap", "intraday", "vol_x"]].copy()
            ups["ticker"] = t
            frames.append(ups)
        except Exception:
            continue
    if not frames:
        return None
    allg = pd.concat(frames)
    rows = []
    prev = None
    for lo, hi, label in VOL_TIERS:
        tier = allg[(allg["vol_x"] >= lo) & (allg["vol_x"] < hi)]
        if len(tier):
            cont = (tier["intraday"] > 0).mean()
            rows.append({"tier": label, "n": len(tier), "cont": cont,
                         "avg_move": tier["intraday"].mean() * 100,
                         "turn": cont >= 0.6 and (prev is None or prev < 0.6)})
            prev = cont
        else:
            rows.append({"tier": label, "n": 0, "cont": None,
                         "avg_move": None, "turn": False})
    return {"rows": rows, "total_gaps": len(allg), "n_tickers": len(frames)}


def pooled_card(pooled):
    if not pooled:
        return ""
    rows = ""
    for r in pooled["rows"]:
        if r["cont"] is None:
            rows += (f'<tr><th>{r["tier"]}</th><td class="dim">0</td>'
                     f'<td class="dim">-</td><td class="dim">-</td><td></td></tr>')
            continue
        turn = '<span class="turn">edge turns on</span>' if r["turn"] else ""
        # confidence hint from sample size
        conf = "solid" if r["n"] >= 30 else "thin" if r["n"] < 10 else "ok"
        rows += (f'<tr><th>{r["tier"]}</th>'
                 f'<td class="dim">{r["n"]} <span class="conf {conf}">{conf}</span></td>'
                 f'<td style="{heat(r["cont"])}">{r["cont"]:.0%}</td>'
                 f'<td class="dim">{r["avg_move"]:+.1f}%</td>'
                 f'<td class="verdict">{turn}</td></tr>')
    return (f'<section class="card pooled"><div class="card-head">'
            f'<h2>Whole-basket volume study</h2>'
            f'<span class="range">{pooled["total_gaps"]} up-gaps pooled across '
            f'{pooled["n_tickers"]} names</span></div>'
            f'<h3>Continuation by volume tier <span class="hint">this is the one '
            f'with real sample sizes — trust these over the per-stock cells</span></h3>'
            f'<table class="matrix"><thead><tr><th>volume</th><th>gap days</th>'
            f'<th>continued</th><th>avg day</th><th></th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
            f'<div class="note"><em>Read this first.</em> Pooling every stock\'s '
            f'gaps makes each volume tier a real sample instead of 2-3 days. Where '
            f'"edge turns on" here is the threshold to set in your thinkorswim live '
            f'scanner. The per-stock cards below show which names drive it — but '
            f'a 100% on 2 days there is noise; this pooled table is the signal.</div>'
            f'</section>')


def build(tickers):
    print(f"Scanning {len(tickers)} names for supernova structure...")
    print("Building pooled cross-basket study...")
    pooled = pooled_card(pooled_volume_study(tickers))
    cards = ""
    for t in tickers:
        print(f"  {t}...")
        cards += ticker_card(t)
    cards = pooled + cards

    intro = ('<div class="note"><em>How to read this.</em> The daily table asks: '
             'when this stock gapped up by X%, how often did it keep running that '
             'day (not fade)? The hi-vol column is the one that matters — a gap on '
             '2x+ normal volume is a real ignition; a quiet gap is a trap. Float is '
             'the fuel: under ~75M shares (amber) is where violent moves live. The '
             'intraday table shows the SHAPE of the biggest recent gaps — did they '
             'run and hold, or spike and fade? That tells you whether these are '
             'catchable or whether they trap you at the top.</div>')

    html = (f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>AI Supernova Scanner</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
            f'family=IBM+Plex+Sans:wght@400;500;600&family=Archivo:wght@600;700&display=swap" rel="stylesheet">'
            f'<style>{CSS}</style></head><body><div class="wrap"><header>'
            f'<div class="eyebrow">AI supernova study · daily gaps + {INTRADAY_INTERVAL} shape · AI / small-cap tech basket</div>'
            f'<h1>AI Supernova Scanner</h1>'
            f'<div class="subtitle">Hunting explosive gaps, not smooth trends. '
            f'Does a big gap keep running, and what does the day look like?</div>'
            f'</header>{intro}{cards}<footer>Generated {date.today()} · not trade advice</footer>'
            f'</div></body></html>')

    with open("ai_supernova.html", "w") as f:
        f.write(html)
    print("\nWrote ai_supernova.html")
    print("Open with:  open ai_supernova.html")


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or DEFAULT_BASKET
    build(tickers)
