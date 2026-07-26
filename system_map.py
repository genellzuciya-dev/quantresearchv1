"""
System Map — network analysis + regime reading, in that order
=============================================================
Step 1 (network): how do the members of a business system move together,
                  and does anyone LEAD anyone else?
Step 2 (system):  build one equal-weighted series representing the system.
Step 3 (regime):  read what state the system is in — via a fitted
                  Markov-switching model if statsmodels is installed,
                  otherwise via the same rule-based labeling as the
                  Markov dashboard.

The first ticker you pass is the ANCHOR (the one you actually want to hold).
Everything else is a partner / peer / supplier.

Systems are tabbed. Edit the SYSTEMS dict below, or pass groups on the
command line separated by a slash:

    python3 system_map.py                      # uses SYSTEMS below
    python3 system_map.py UNP WAB CSX / NVDA AMAT TSM
    open system_map.html

Optional, unlocks the fitted model:
    pip3 install statsmodels
"""

import sys
from datetime import date

import numpy as np
import pandas as pd

LOOKBACK = "2y"
MAX_LAG = 5

# Each entry: "Tab name": [anchor, partner, partner, ...]
# The FIRST ticker is the anchor — the one you'd actually hold.
SYSTEMS = {
    "UNP · rail & freight": ["UNP", "WAB", "GBX", "TRN", "CSX", "NSC"],
    "NVDA · AI compute":    ["NVDA", "AMAT", "TSM", "MRVL", "AVGO", "VRT"],
    # Coffee is an INVERSE system: the bean price is a cost to the roaster,
    # not a shared revenue driver. Expect weak or negative linkage — that is
    # the finding, not a failure. KC=F is the Coffee C futures contract;
    # EWZ is Brazil, where the weather that moves coffee actually happens.
    "SBUX · coffee chain":  ["SBUX", "KC=F", "KDP", "NSRGY", "DNUT", "EWZ"],
    # Cocoa: same inverse structure as coffee, far more violent supply story.
    # CC=F is cocoa futures; MDLZ and HSY are the big chocolate buyers.
    "HSY · cocoa chain":    ["HSY", "CC=F", "MDLZ", "NSRGY", "SBUX"],
    # Grain complex — UNP sits INSIDE this one. If UNP correlates more with
    # grain than with its own rail peers, your rail anchor is a grain proxy.
    "Grain complex":        ["ADM", "BG", "ZC=F", "ZS=F", "ZW=F", "DE", "NTR", "UNP"],
    # Irrigation & water — the binding constraint on global agriculture.
    # VMI and LNN are center-pivot irrigation; CTVA is seed genetics.
    "Water & irrigation":   ["VMI", "LNN", "CTVA", "DE", "NTR", "MOS"],
}


# ------------------------------------------------------------------ data
def get_returns(tickers):
    import yfinance as yf
    px = {}
    for t in tickers:
        df = yf.download(t, period=LOOKBACK, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print(f"  no data for {t} — skipping")
            continue
        px[t] = df["Close"]
    prices = pd.DataFrame(px).dropna()
    return prices.pct_change().dropna()


# -------------------------------------------------------------- analysis
def lead_lag(rets, anchor):
    """Does a partner's move today predict the anchor's move later?

    partner_leads: corr(anchor[t], partner[t-k])  -> partner is early
    anchor_leads : corr(partner[t], anchor[t-k])  -> anchor is early
    """
    out = []
    a = rets[anchor]
    n = len(a)
    # rough 5% significance threshold for a correlation with n observations
    thresh = 1.96 / np.sqrt(n)
    for p in rets.columns:
        if p == anchor:
            continue
        row = {"ticker": p, "same_day": float(a.corr(rets[p]))}
        pl = {k: float(a.corr(rets[p].shift(k))) for k in range(1, MAX_LAG + 1)}
        al = {k: float(rets[p].corr(a.shift(k))) for k in range(1, MAX_LAG + 1)}
        row["partner_leads"] = pl
        row["anchor_leads"] = al
        bk = max(pl, key=lambda k: abs(pl[k]))
        ak = max(al, key=lambda k: abs(al[k]))
        row["best_partner_lead"] = (bk, pl[bk])
        row["best_anchor_lead"] = (ak, al[ak])
        row["signal"] = (abs(pl[bk]) > thresh or abs(al[ak]) > thresh)
        out.append(row)
    return out, thresh


def build_basket(rets):
    """Equal-weighted system return series + its cumulative index."""
    basket = rets.mean(axis=1)
    basket.name = "SYSTEM"
    return basket


def fit_regimes(basket):
    """Fitted 2-regime Markov switching model on system returns.
    Returns None if statsmodels isn't installed."""
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        return None
    try:
        mod = MarkovRegression(basket.values * 100, k_regimes=2,
                               trend="c", switching_variance=True)
        res = mod.fit(disp=False)
        probs = pd.DataFrame(res.smoothed_marginal_probabilities,
                             index=basket.index)
        p = np.asarray(res.params)
        # param layout for k=2, trend='c', switching_variance=True:
        #   [p00, p10, const0, const1, sigma2_0, sigma2_1]
        n_trans = 2 * (2 - 1)          # k * (k - 1) transition params
        means = [float(p[n_trans + i]) for i in range(2)]
        sigmas = [float(p[-2]), float(p[-1])]   # these are VARIANCES
        calm = int(np.argmin(sigmas))
        durs = [float(d) for d in res.expected_durations]
        return {"probs": probs, "means": means, "sigmas": sigmas,
                "calm": calm, "durations": durs,
                "current": [float(probs.iloc[-1, i]) for i in range(2)]}
    except Exception as e:
        print(f"  regime fit failed: {e}")
        return None


def rule_regimes(basket):
    """Fallback: same labeling idea as the Markov dashboard, on the basket."""
    idx = (1 + basket).cumprod()
    sma = idx.rolling(20).mean()
    slope = sma.diff()
    lab = np.select([(idx > sma) & (slope > 0), (idx < sma) & (slope < 0)],
                    ["SYSTEM_UP", "SYSTEM_DOWN"], default="SYSTEM_CHOP")
    s = pd.Series(lab, index=idx.index).iloc[20:]
    freq = s.value_counts(normalize=True)
    # persistence
    pers = {}
    for st in ["SYSTEM_UP", "SYSTEM_DOWN", "SYSTEM_CHOP"]:
        pairs = [(a, b) for a, b in zip(s[:-1], s[1:]) if a == st]
        pers[st] = (sum(1 for a, b in pairs if b == st) / len(pairs)) if pairs else None
    return {"states": s, "freq": freq, "persist": pers, "current": s.iloc[-1]}


# ------------------------------------------------------------- rendering
def heat_corr(v):
    if v is None or pd.isna(v):
        return "background:#151a21;color:#5a6270"
    t = min(abs(v), 1) ** 0.7
    if v >= 0:
        r, g, b = int(0x1b + t * 20), int(0x21 + t * 158), int(0x2b + t * 100)
    else:
        r, g, b = int(0x1b + t * 201), int(0x21 + t * 66), int(0x2b + t * 52)
    return f"background:rgb({r},{g},{b});color:{'#0e1116' if t > 0.55 else '#e8e6e0'}"


def corr_table(rets, anchor):
    cols = list(rets.columns)
    c = rets.corr()
    head = "".join(f"<th>{t}</th>" for t in cols)
    rows = ""
    for a in cols:
        cells = ""
        for b in cols:
            v = c.loc[a, b]
            cells += f'<td style="{heat_corr(v)}">{v:.2f}</td>'
        mark = ' class="anchor"' if a == anchor else ""
        rows += f"<tr><th{mark}>{a}</th>{cells}</tr>"
    return (f'<table class="matrix"><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def leadlag_table(ll, thresh, anchor):
    rows = ""
    for r in sorted(ll, key=lambda x: -abs(x["best_partner_lead"][1])):
        pk, pv = r["best_partner_lead"]
        ak, av = r["best_anchor_lead"]
        flag = ""
        if abs(pv) > thresh and abs(pv) >= abs(av):
            flag = f'<span class="lead">{r["ticker"]} leads by {pk}d</span>'
        elif abs(av) > thresh:
            flag = f'<span class="lead">{anchor} leads by {ak}d</span>'
        else:
            flag = '<span class="none">no lead-lag</span>'
        rows += (f'<tr><th>{r["ticker"]}</th>'
                 f'<td style="{heat_corr(r["same_day"])}">{r["same_day"]:.2f}</td>'
                 f'<td style="{heat_corr(pv)}">{pv:+.3f}<span class="n">lag {pk}</span></td>'
                 f'<td style="{heat_corr(av)}">{av:+.3f}<span class="n">lag {ak}</span></td>'
                 f'<td class="verdict">{flag}</td></tr>')
    return (f'<table class="matrix ll"><thead><tr><th></th><th>same day<br>corr</th>'
            f'<th>partner&rarr;{anchor}<br>best lag</th>'
            f'<th>{anchor}&rarr;partner<br>best lag</th>'
            f'<th>read</th></tr></thead><tbody>{rows}</tbody></table>')


def regime_block(fit, rule, basket):
    if fit:
        calm, hot = fit["calm"], 1 - fit["calm"]
        cur_calm = fit["current"][calm] * 100
        label = "CALM" if cur_calm > 50 else "STRESSED"
        conf = cur_calm if cur_calm > 50 else 100 - cur_calm
        stats = (
            f'<div class="stat"><div class="label">System state today</div>'
            f'<div class="big">{label}</div>'
            f'<div class="sub">{conf:.0f}% probability</div></div>'
            f'<div class="stat"><div class="label">Calm regime</div>'
            f'<div class="big">{fit["durations"][calm]:.0f}<span class="unit">days</span></div>'
            f'<div class="sub">mean {fit["means"][calm]:+.2f}%/day · '
            f'vol {np.sqrt(fit["sigmas"][calm]):.2f}</div></div>'
            f'<div class="stat"><div class="label">Stressed regime</div>'
            f'<div class="big">{fit["durations"][hot]:.0f}<span class="unit">days</span></div>'
            f'<div class="sub">mean {fit["means"][hot]:+.2f}%/day · '
            f'vol {np.sqrt(fit["sigmas"][hot]):.2f}</div></div>')
        note = ('<div class="note">Fitted 2-regime Markov switching model '
                '(statsmodels). The regimes were <em>inferred from the data</em> — '
                'nothing was hand-labeled. Expected durations come from the '
                'estimated transition matrix, same 1/(1-p) idea as your '
                'rule-based dashboard.</div>')
    else:
        f = rule["freq"]
        stats = f'<div class="stat"><div class="label">System state today</div><div class="big">{rule["current"].replace("SYSTEM_","")}</div><div class="sub">rule-based</div></div>'
        for st in ["SYSTEM_UP", "SYSTEM_DOWN", "SYSTEM_CHOP"]:
            p = rule["persist"][st]
            stats += (f'<div class="stat"><div class="label">{st.replace("SYSTEM_","")}</div>'
                      f'<div class="big">{f.get(st,0):.0%}</div>'
                      f'<div class="sub">persist {"-" if p is None else format(p,".2f")}</div></div>')
        note = ('<div class="note">Rule-based labeling (20-SMA on the system index). '
                'Install statsmodels — <code>pip3 install statsmodels</code> — and rerun '
                'to get a fitted Markov-switching model that infers the regimes '
                'from the data instead.</div>')
    cum = (1 + basket).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    vol = basket.std() * np.sqrt(252) * 100
    stats += (f'<div class="stat"><div class="label">System basket</div>'
              f'<div class="big">{total:+.0f}<span class="unit">%</span></div>'
              f'<div class="sub">{len(basket)} days · {vol:.0f}% ann vol</div></div>')
    return f'<div class="stats">{stats}</div>{note}'


CSS = """:root{--ink:#0e1116;--panel:#151a21;--panel2:#1b212b;--line:#28303c;
--paper:#e8e6e0;--dim:#8b93a0;--faint:#5a6270;--win:#2fbf8f;--loss:#e4574f;--amber:#e5a83b}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--ink);color:var(--paper);font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:15px;line-height:1.6}
.wrap{max-width:1060px;margin:0 auto;padding:0 28px 80px}
header{padding:44px 0 8px}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
h1{font-family:Archivo,sans-serif;font-size:30px;margin:6px 0 2px}
.subtitle{color:var(--dim);font-size:14px;margin-bottom:22px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.tabbtn{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.04em;
background:var(--panel);color:var(--dim);border:1px solid var(--line);
border-radius:8px;padding:9px 15px;cursor:pointer}
.tabbtn:hover{color:var(--paper)}
.tabbtn.on{background:var(--panel2);color:var(--win);border-color:var(--win)}
.panesub{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);margin-top:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px 26px;margin:22px 0}
.card-head{display:flex;align-items:baseline;gap:14px;margin-bottom:14px}
.card-head h2{font-family:Archivo,sans-serif;font-size:22px}
.range{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px 14px}
.stat .label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.stat .big{font-family:Archivo,sans-serif;font-size:24px;font-weight:600;margin:2px 0}
.stat .unit{font-size:12px;color:var(--faint);margin-left:4px;font-weight:400}
.stat .sub{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--faint)}
h3{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin:14px 0 10px;font-weight:500}
.hint{text-transform:none;letter-spacing:0;color:var(--faint);font-size:11px;margin-left:8px}
table.matrix{border-collapse:separate;border-spacing:3px;width:100%;font-family:'IBM Plex Mono',monospace}
.matrix th{font-size:11px;color:var(--dim);font-weight:500;padding:4px 6px;text-align:center;line-height:1.3}
.matrix tbody th{text-align:right;padding-right:10px;color:var(--paper);font-size:13px}
.matrix tbody th.anchor{color:var(--amber)}
.matrix td{text-align:center;padding:8px 4px 5px;border-radius:6px;font-size:13px}
.matrix td .n{display:block;font-size:9px;opacity:.65;margin-top:1px}
.ll td.verdict{background:var(--panel2);text-align:left;padding-left:12px;font-size:11px}
.lead{color:var(--win)}
.none{color:var(--faint)}
.note{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px 16px;font-size:13px;color:var(--dim);margin-top:14px}
.note code{font-family:'IBM Plex Mono',monospace;color:var(--paper)}
.note em{color:var(--paper);font-style:normal}
footer{margin-top:36px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);text-align:center}"""


def system_body(tickers):
    """Run all three steps for one system; return (html_body, member_list)."""
    anchor = tickers[0]
    print(f"  anchor {anchor}  ·  partners {', '.join(tickers[1:])}")
    rets = get_returns(tickers)
    if anchor not in rets.columns or rets.shape[1] < 2:
        return (f'<div class="note">Not enough data for {anchor} and at least '
                f'one partner.</div>', [])
    print(f"    {len(rets)} overlapping trading days")

    ll, thresh = lead_lag(rets, anchor)
    basket = build_basket(rets)
    fit = fit_regimes(basket)
    rule = rule_regimes(basket)

    body = (
        f'<section class="card"><div class="card-head"><h2>Step 1 · Who moves together</h2>'
        f'<span class="range">daily return correlation</span></div>'
        f'{corr_table(rets, anchor)}'
        f'<div class="note">High correlation means these names are <em>one bet, not '
        f'several</em>. That is what makes the system thesis true and what makes '
        f'owning all of them concentration rather than diversification.</div></section>'

        f'<section class="card"><div class="card-head"><h2>Step 2 · Who moves first</h2>'
        f'<span class="range">|corr| &gt; {thresh:.3f} is the ~5% significance line</span></div>'
        f'{leadlag_table(ll, thresh, anchor)}'
        f'<div class="note">A partner that leads {anchor} is a <em>signal</em>. Daily '
        f'lead-lag correlations are usually tiny and often noise — treat anything '
        f'near the threshold as a hypothesis to re-test on fresh data, not a finding.'
        f'</div></section>'

        f'<section class="card"><div class="card-head"><h2>Step 3 · What state the system is in</h2>'
        f'<span class="range">equal-weighted basket of all {len(rets.columns)} names</span></div>'
        f'{regime_block(fit, rule, basket)}</section>'
    )
    return body, list(rets.columns)


TABS_JS = """
function showTab(i){
  document.querySelectorAll('.tabpane').forEach(function(p,j){
    p.style.display = (j===i) ? 'block' : 'none';
  });
  document.querySelectorAll('.tabbtn').forEach(function(b,j){
    b.className = 'tabbtn' + (j===i ? ' on' : '');
  });
}
"""


def build(systems):
    btns, panes = "", ""
    for i, (name, tickers) in enumerate(systems.items()):
        print(f"\n{name}")
        body, members = system_body([t.upper() for t in tickers])
        btns += (f'<button class="tabbtn{" on" if i == 0 else ""}" '
                 f'onclick="showTab({i})">{name}</button>')
        sub = (f'<div class="panesub">Members: {", ".join(members)}</div>'
               if members else "")
        panes += (f'<div class="tabpane" style="display:'
                  f'{"block" if i == 0 else "none"}">{sub}{body}</div>')

    html = (f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>System Map</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
            f'family=IBM+Plex+Sans:wght@400;500;600&family=Archivo:wght@600;700&display=swap" rel="stylesheet">'
            f'<style>{CSS}</style></head><body><div class="wrap"><header>'
            f'<div class="eyebrow">System study · {LOOKBACK} daily returns</div>'
            f'<h1>System Map</h1>'
            f'<div class="subtitle">Network analysis identifies the system; '
            f'the regime model reads its state.</div>'
            f'<div class="tabs">{btns}</div></header>{panes}'
            f'<footer>Generated {date.today()}</footer></div>'
            f'<script>{TABS_JS}</script></body></html>')

    with open("system_map.html", "w") as f:
        f.write(html)
    print("\nWrote system_map.html")
    print("Open with:  open system_map.html")


if __name__ == "__main__":
    args = [t.upper() for t in sys.argv[1:]]
    if args:
        groups, cur = [], []
        for a in args:
            if a == "/":
                if cur:
                    groups.append(cur)
                cur = []
            else:
                cur.append(a)
        if cur:
            groups.append(cur)
        build({g[0]: g for g in groups})
    else:
        build(SYSTEMS)
