"""
ORIC Backtest — dressed up as HTML, appended to the intraday study
==================================================================
Runs the exit-on-chop vs scale-and-trail comparison and writes a styled
backtest section, then STITCHES it into markov_intraday.html so the
regime study and its backtest live on one page.

Run AFTER markov_intraday.py (it needs that file to exist):
    python3 markov_intraday.py ORIC 2m
    python3 backtest_html.py ORIC 2m
    open markov_intraday.html
"""

import sys, os
from datetime import date
import numpy as np
import pandas as pd

INTERVAL = "2m"
PERIOD_FOR = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d"}
SMA_LEN = 20
GAP_THRESHOLD = 0.02
COST_PER_SIDE = 0.002
TRAIL_PCT = 0.015
TARGET_HTML = "markov_intraday.html"


def get_bars(ticker, interval):
    import yfinance as yf
    df = yf.download(ticker, period=PERIOD_FOR.get(interval, "60d"),
                     interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def add_indicators(df):
    """EMA9 on close, plus VWAP that RESETS each session (intraday-correct)."""
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()
    if "Volume" in df.columns and "High" in df.columns:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        session = df.index.normalize()               # group by calendar day
        pv = (typical * df["Volume"]).groupby(session).cumsum()
        vol = df["Volume"].groupby(session).cumsum().replace(0, np.nan)
        df["vwap"] = (pv / vol).fillna(df["Close"])
    else:
        df["vwap"] = df["Close"]                       # fallback: no volume
    return df


def label_states(df):
    df = add_indicators(df)
    sma = df["Close"].rolling(SMA_LEN).mean()
    slope = sma.diff()
    gap = df["Open"] / df["Close"].shift(1) - 1
    cond = [gap >= GAP_THRESHOLD, gap <= -GAP_THRESHOLD,
            (df["Close"] > sma) & (slope > 0), (df["Close"] < sma) & (slope < 0)]
    out = df.copy()
    out["state"] = np.select(cond, ["GAP_UP", "GAP_DOWN", "TREND_UP", "TREND_DOWN"],
                             default="CHOP")
    return out.iloc[SMA_LEN:]


# --- ENTRY GATE -------------------------------------------------------
# A trade is allowed to open only when ALL of these hold on the bar:
#   1. current bar is an up-trend/gap-up state
#   2. previous bar was ALSO an up-trend state  (2 consecutive up bars)
#   3. price is above the session VWAP           (right side of the day)
#   4. price is at or above EMA9                 (short-term momentum intact)
# Toggle any of these with the flags below to test their individual effect.
REQUIRE_TWO_BARS = True
REQUIRE_ABOVE_VWAP = True
REQUIRE_ABOVE_EMA9 = True
EMA9_TOLERANCE = 0.001     # "close to EMA9" = within 0.1% below it counts


def _enter_gate(bars, i):
    st = bars["state"].iloc[i]
    if st not in ("TREND_UP", "GAP_UP"):
        return False
    if REQUIRE_TWO_BARS:
        if i == 0 or bars["state"].iloc[i - 1] not in ("TREND_UP", "GAP_UP"):
            return False
    px = bars["Close"].iloc[i]
    if REQUIRE_ABOVE_VWAP and px < bars["vwap"].iloc[i]:
        return False
    if REQUIRE_ABOVE_EMA9 and px < bars["ema9"].iloc[i] * (1 - EMA9_TOLERANCE):
        return False
    return True


def run_exit_on_chop(bars):
    trades, pos, entry = [], 0, 0.0
    for i in range(len(bars)):
        st, px = bars["state"].iloc[i], bars["Close"].iloc[i]
        if pos == 0 and _enter_gate(bars, i): pos, entry = 1, px
        elif pos == 1 and st in ("CHOP", "TREND_DOWN", "GAP_DOWN"):
            trades.append((px / entry - 1) - 2 * COST_PER_SIDE); pos = 0
    return np.array(trades)


def run_scale_and_trail(bars):
    trades, pos, entry, peak = [], 0, 0.0, 0.0
    for i in range(len(bars)):
        st, px = bars["state"].iloc[i], bars["Close"].iloc[i]
        if pos == 0 and _enter_gate(bars, i): pos, entry, peak = 1.0, px, px
        elif pos == 1.0:
            peak = max(peak, px)
            if st in ("CHOP", "TREND_DOWN", "GAP_DOWN"):
                trades.append(0.5 * ((px / entry - 1) - 2 * COST_PER_SIDE)); pos = 0.5
        elif pos == 0.5:
            peak = max(peak, px)
            if px <= peak * (1 - TRAIL_PCT):
                trades.append(0.5 * ((px / entry - 1) - 2 * COST_PER_SIDE)); pos = 0
    return np.array(trades)


def score(trades):
    if len(trades) == 0: return None
    w, l = trades[trades > 0], trades[trades <= 0]
    eq = np.cumprod(1 + trades); peak = np.maximum.accumulate(eq)
    return {"n": len(trades), "total": (eq[-1] - 1) * 100,
            "win_rate": len(w) / len(trades) * 100,
            "avg_win": (w.mean() * 100) if len(w) else 0.0,
            "avg_loss": (l.mean() * 100) if len(l) else 0.0,
            "exp": trades.mean() * 100, "dd": ((eq - peak) / peak).min() * 100,
            "equity": eq}


# ------------------------------------------------------------- rendering
GREEN, RED, DIM = "#2fbf8f", "#e4574f", "#8b93a0"


def sign_color(v, good_high=True):
    if v is None: return DIM
    pos = v > 0
    return GREEN if (pos == good_high) else RED


def metric(label, value, color):
    return (f'<div class="bt-metric"><div class="bt-label">{label}</div>'
            f'<div class="bt-val" style="color:{color}">{value}</div></div>')


def strat_row(name, s, is_holdout):
    if s is None:
        return f'<div class="bt-strat"><div class="bt-name">{name}</div><div class="bt-metrics">no trades</div></div>'
    ec = sign_color(s["exp"])
    tc = sign_color(s["total"])
    verdict = ""
    if is_holdout:
        ok = s["exp"] > 0
        verdict = (f'<span class="bt-verdict {"pass" if ok else "fail"}">'
                   f'{"EDGE (out-of-sample)" if ok else "NO EDGE (out-of-sample)"}</span>')
    m_trades = metric("trades", str(s["n"]), DIM)
    m_win = metric("win rate", format(s["win_rate"], ".0f") + "%", DIM)
    m_exp = metric("expectancy", format(s["exp"], "+.3f") + "%", ec)
    m_aw = metric("avg win", format(s["avg_win"], "+.2f") + "%", GREEN)
    m_al = metric("avg loss", format(s["avg_loss"], "+.2f") + "%", RED)
    m_tot = metric("total", format(s["total"], "+.1f") + "%", tc)
    m_dd = metric("max DD", format(s["dd"], ".1f") + "%", RED)
    return (f'<div class="bt-strat"><div class="bt-name">{name}{verdict}</div>'
            f'<div class="bt-metrics">'
            f'{m_trades}{m_win}{m_exp}{m_aw}{m_al}{m_tot}{m_dd}'
            f'</div></div>')


def sparkline(equity, w=260, h=44):
    if equity is None or len(equity) < 2: return ""
    lo, hi = equity.min(), equity.max()
    rng = (hi - lo) or 1
    pts = " ".join(f"{i/(len(equity)-1)*w:.1f},{h-(v-lo)/rng*h:.1f}"
                   for i, v in enumerate(equity))
    col = GREEN if equity[-1] >= 1 else RED
    return (f'<svg viewBox="0 0 {w} {h}" class="spark" preserveAspectRatio="none">'
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.5"/></svg>')


def backtest_section(ticker, interval):
    bars = label_states(get_bars(ticker, interval))
    split = int(len(bars) * 0.6)
    design, held = bars.iloc[:split], bars.iloc[split:]

    dA, dB = score(run_exit_on_chop(design)), score(run_scale_and_trail(design))
    hA, hB = score(run_exit_on_chop(held)), score(run_scale_and_trail(held))

    def block(title, A, B, holdout):
        sub = ("The only scores that count — the rule never saw this data."
               if holdout else "You may peek. In-sample results are NOT evidence.")
        sparks = ""
        if holdout and A and B:
            sparks = (f'<div class="bt-sparks">'
                      f'<div><span class="bt-label">A equity</span>{sparkline(A["equity"])}</div>'
                      f'<div><span class="bt-label">B equity</span>{sparkline(B["equity"])}</div>'
                      f'</div>')
        cls = "holdout" if holdout else "design"
        return (f'<div class="bt-half {cls}"><div class="bt-half-head">'
                f'<h4>{title}</h4><span class="bt-sub">{sub}</span></div>'
                f'{strat_row("A · exit on chop", A, holdout)}'
                f'{strat_row("B · scale + trail", B, holdout)}{sparks}</div>')

    gates = []
    if REQUIRE_TWO_BARS: gates.append("2 up bars")
    if REQUIRE_ABOVE_VWAP: gates.append(">VWAP")
    if REQUIRE_ABOVE_EMA9: gates.append(">=EMA9")
    gate_desc = " + ".join(gates) if gates else "any trend bar"

    # the takeaway line, computed from the actual held-out numbers
    takeaway = ""
    if hA and hB:
        if hB["exp"] > 0 and hB["exp"] > hA["exp"]:
            takeaway = ("Out-of-sample, B (scale + trail) shows positive expectancy and "
                        "beats A — scaling out is a real improvement, not hindsight.")
        elif hB["exp"] > hA["exp"]:
            hb_exp = format(hB["exp"], "+.3f"); ha_exp = format(hA["exp"], "+.3f")
            hb_dd = format(hB["dd"], ".0f"); ha_dd = format(hA["dd"], ".0f")
            takeaway = ("Out-of-sample, both rules lose — but B loses far less than A "
                        f"(expectancy {hb_exp}% vs {ha_exp}%, drawdown {hb_dd}% vs "
                        f"{ha_dd}%). Scaling helps, but this naive trend-entry has no "
                        "tradeable edge. The weak link is the ENTRY, not the exit — "
                        "try requiring confirmation before entering.")
        else:
            takeaway = ("Out-of-sample, A beats B here — unusual; worth re-checking on a "
                        "different window before trusting either.")

    return (f'<section class="card bt"><div class="card-head">'
            f'<h2>Backtest · exit-on-chop vs scale-and-trail</h2>'
            f'<span class="range">{ticker} {interval} · {len(bars)} bars · '
            f'{COST_PER_SIDE*1e4:.0f}bps/side · {TRAIL_PCT*100:.1f}% trail · entry: {gate_desc}</span></div>'
            f'{block("DESIGN half", dA, dB, False)}'
            f'{block("HELD-OUT half", hA, hB, True)}'
            f'<div class="note"><em>Read the held-out half only.</em> {takeaway} '
            f'The split (design 60% / held-out 40%) is the guardrail against '
            f'curve-fitting: a rule that only wins on the half it was designed on '
            f'is a fantasy. Raise the cost or change the entry and rerun — but never '
            f'tune a parameter to make the design half prettier.</div></section>')


BT_CSS = """
.bt .bt-half{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.bt .bt-half.holdout{border-color:var(--win)}
.bt-half-head{display:flex;align-items:baseline;gap:12px;margin-bottom:10px}
.bt-half-head h4{font-family:Archivo,sans-serif;font-size:15px}
.bt-sub{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--faint)}
.bt-strat{padding:8px 0;border-top:1px solid var(--line)}
.bt-name{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--paper);margin-bottom:6px;display:flex;align-items:center;gap:10px}
.bt-verdict{font-size:9px;letter-spacing:.08em;padding:2px 7px;border-radius:4px}
.bt-verdict.pass{background:rgba(47,191,143,.15);color:var(--win)}
.bt-verdict.fail{background:rgba(228,87,79,.15);color:var(--loss)}
.bt-metrics{display:flex;flex-wrap:wrap;gap:16px}
.bt-metric{min-width:64px}
.bt-label{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.bt-val{font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:500}
.bt-sparks{display:flex;gap:24px;margin-top:10px}
.bt-sparks svg{display:block;width:260px;height:44px;margin-top:3px}
.spark{overflow:visible}
"""


def main(ticker, interval):
    print(f"Backtesting {ticker} @ {interval}...")
    section = backtest_section(ticker, interval)

    if not os.path.exists(TARGET_HTML):
        # standalone fallback page if the intraday study hasn't been run
        html = (f'<!DOCTYPE html><html><head><meta charset="UTF-8">'
                f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
                f'family=IBM+Plex+Sans:wght@400;500;600&family=Archivo:wght@600;700&display=swap" rel="stylesheet">'
                f'<style>:root{{--ink:#0e1116;--panel:#151a21;--panel2:#1b212b;--line:#28303c;'
                f'--paper:#e8e6e0;--dim:#8b93a0;--faint:#5a6270;--win:#2fbf8f;--loss:#e4574f}}'
                f'body{{background:var(--ink);color:var(--paper);font-family:"IBM Plex Sans",sans-serif;'
                f'max-width:1060px;margin:0 auto;padding:40px 28px}}'
                f'.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px 26px}}'
                f'.card-head{{display:flex;align-items:baseline;gap:14px;margin-bottom:14px}}'
                f'.card-head h2{{font-family:Archivo,sans-serif;font-size:22px}}'
                f'.range{{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--faint)}}'
                f'.note{{background:var(--panel2);border:1px solid var(--line);border-radius:9px;'
                f'padding:12px 16px;font-size:13px;color:var(--dim);margin-top:14px}}'
                f'.note em{{color:var(--paper);font-style:normal}}{BT_CSS}</style></head>'
                f'<body>{section}</body></html>')
        with open(TARGET_HTML, "w") as f:
            f.write(html)
        print(f"  {TARGET_HTML} didn't exist — wrote a standalone backtest page.")
        print(f"  (Run markov_intraday.py first to get the full regime study too.)")
        return

    with open(TARGET_HTML) as f:
        page = f.read()

    # remove any prior backtest section (so reruns replace, not stack)
    import re
    page = re.sub(r'<section class="card bt">.*?</section>', '', page, flags=re.DOTALL)
    # inject our CSS once
    if ".bt .bt-half" not in page:
        page = page.replace("</style>", BT_CSS + "</style>", 1)
    # insert the section just before the footer
    page = page.replace('<footer>', section + '<footer>', 1)

    with open(TARGET_HTML, "w") as f:
        f.write(page)
    print(f"  Appended backtest to {TARGET_HTML}")
    print(f"Open with:  open {TARGET_HTML}")


if __name__ == "__main__":
    raw = sys.argv[1:]
    interval = INTERVAL
    tickers = []
    for a in raw:
        if a.lower() in PERIOD_FOR: interval = a.lower()
        else: tickers.append(a.upper())
    main(tickers[0] if tickers else "ORIC", interval)
