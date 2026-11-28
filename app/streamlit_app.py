"""
Retail Intelligence: customer retention, demand forecasting, and pricing and promotions.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import os
from pathlib import Path

# Point the package at this repository's data, even when it is installed as a regular
# (non-editable) package, as on Streamlit Community Cloud. Must run before the imports below.
os.environ.setdefault("RETAIL_HOME", str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from retail_intelligence.agent import make_client, model_name, run_agent
from retail_intelligence.engine import ACTIONS, SEGMENTS, Engine

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="Retail Intelligence", page_icon=str(ASSETS / "icon.svg"), layout="wide")

REPO_URL = "https://github.com/marianunez-data/retail-intelligence"
AUTHOR = "María Camila González Núñez"

# ----------------------------------------------------------------- design
INK, INK2, LINE = "#14181f", "#5b6472", "#e3e6ea"
BG, NAVY = "#f4f6f9", "#16345c"
SHADOW = "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06)"
BLUE, RED, MUTED = "#2a78d6", "#d0453f", "#9aa1ab"
SEG_COLORS = {"Loyal": "#2f8f63", "Nurture": "#d39a1c", "At risk": "#c94a44"}
SEG_BADGE = {"Loyal": ("#e8f3ed", "#1f6b47"), "Nurture": ("#fbf2de", "#7a5200"),
             "At risk": ("#f9e9e8", "#9a2f2a")}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, p, li, label, input, textarea, button, td, th,
h1, h2, h3, h4, [data-testid="stMarkdownContainer"] {{ font-family: 'Inter', sans-serif; }}
[data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1320px; }}
h1 {{ font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.02em; color: {INK}; margin-bottom: 0 !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; color: {INK}; letter-spacing: -0.01em; }}
.eyebrow {{ font-size: .72rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: {INK2}; margin-bottom: .25rem; }}
.lede {{ color: {INK2}; font-size: .95rem; margin: .35rem 0 1.4rem 0; max-width: 820px; }}
.stApp, [data-testid="stMain"] {{ background: {BG}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #ffffff; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebar"] a[aria-current="page"] {{ background: #eaf2fc; }}
[data-testid="stSidebar"] a[aria-current="page"] span {{ color: {NAVY}; font-weight: 600; }}
[class*="st-key-surface"] {{ background: #fff; box-shadow: {SHADOW}; border-color: {LINE} !important;
    border-radius: 12px !important; }}
.kpi {{ border: 1px solid {LINE}; border-radius: 12px; padding: 16px 18px; background: #fff; height: 100%;
        box-shadow: {SHADOW}; }}
.kpi .label {{ font-size: .78rem; font-weight: 500; color: {INK2}; }}
.kpi .value {{ font-size: 1.65rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; margin-top: 2px; }}
.kpi .sub {{ font-size: .78rem; color: {INK2}; margin-top: 2px; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; }}
.panel {{ border: 1px solid {LINE}; border-radius: 12px; padding: 18px 20px; background: #fff; box-shadow: {SHADOW}; }}
.panel .k {{ font-size: .78rem; color: {INK2}; }}
.panel .v {{ font-size: .95rem; color: {INK}; font-weight: 600; margin-bottom: 10px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
.brand {{ font-weight: 700; font-size: 1.02rem; color: {INK}; letter-spacing: -0.01em; }}
.brand-sub {{ font-size: .78rem; color: {INK2}; }}
.q {{ background: #f6f7f9; border: 1px solid {LINE}; border-radius: 10px; padding: 12px 16px;
      margin: 18px 0 10px 0; color: {INK}; font-weight: 500; }}
.tip {{ position: relative; display: inline-flex; align-items: center; justify-content: center;
        width: 15px; height: 15px; margin-left: 5px; border-radius: 50%; border: 1px solid #c3c8cf;
        color: {INK2}; font-size: 10px; font-weight: 600; cursor: help; vertical-align: 1px; }}
.tip .tiptext {{ visibility: hidden; opacity: 0; position: absolute; z-index: 1000; top: 22px; left: -12px;
        width: 260px; background: {INK}; color: #fff; font-size: .78rem; font-weight: 400; line-height: 1.45;
        text-align: left; padding: 10px 12px; border-radius: 8px; box-shadow: 0 6px 18px rgba(20,24,31,.18);
        transition: opacity .12s ease; }}
.tip:hover .tiptext {{ visibility: visible; opacity: 1; }}
.card {{ border: 1px solid {LINE}; border-radius: 12px; padding: 20px 22px; background: #fff; height: 100%;
         box-shadow: {SHADOW}; }}
.card.flat {{ border: none; box-shadow: none; background: transparent; padding: 4px 4px 0 4px; min-height: 392px; }}
.accent {{ height: 4px; border-radius: 4px; margin: 0 0 18px 0; width: 44px; }}
.mod-icon {{ width: 40px; height: 40px; border-radius: 10px; display: inline-flex; align-items: center;
         justify-content: center; margin-bottom: 14px; }}
.mod-icon svg {{ width: 21px; height: 21px; }}
.hero {{ position: relative; overflow: hidden; border-radius: 16px; padding: 38px 40px 32px 40px; margin-bottom: 26px;
         background: linear-gradient(125deg, #0f1e36 0%, #16345c 55%, #1f5596 100%); color: #fff;
         box-shadow: 0 10px 30px rgba(15,30,54,.18); }}
.hero:after {{ content: ""; position: absolute; right: -80px; top: -120px; width: 420px; height: 420px; border-radius: 50%;
         background: radial-gradient(circle, rgba(94,160,240,.35) 0%, rgba(94,160,240,0) 70%); }}
.hero .h-eyebrow {{ font-size: .72rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: #9cc3f2; }}
.hero .h-title {{ font-size: 2.3rem; font-weight: 700; letter-spacing: -0.03em; margin: 6px 0 10px 0; color: #fff; }}
.hero .h-lede {{ font-size: 1rem; line-height: 1.6; color: #d3e0f0; max-width: 840px; }}
.hero .h-stats {{ display: flex; flex-wrap: wrap; gap: 34px; margin-top: 26px; padding-top: 22px;
         border-top: 1px solid rgba(255,255,255,.14); position: relative; z-index: 1; }}
.hero .h-stat .n {{ font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em; color: #fff; }}
.hero .h-stat .l {{ font-size: .76rem; color: #9fb4cf; margin-top: 2px; }}
.hero .h-tags {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; position: relative; z-index: 1; }}
.hero .h-tag {{ font-size: .74rem; padding: 4px 11px; border-radius: 999px; color: #dbe7f6;
         background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); }}
.section {{ font-size: .72rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: {INK2};
            margin: 30px 0 12px 0; }}
.pillar {{ border: 1px solid {LINE}; border-radius: 12px; background: #fff; padding: 16px 16px 14px 16px; height: 100%;
           box-shadow: {SHADOW}; }}
.pillar .t {{ font-weight: 600; font-size: .9rem; color: {INK}; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }}
.pillar .t .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
.pillar .d {{ font-size: .8rem; color: {INK2}; line-height: 1.5; }}
[data-testid="stPageLink"] a span, [data-testid="stPageLink"] a p {{ color: {BLUE} !important; font-weight: 600; }}
[data-testid="stPageLink"] a:hover {{ background: #eef4fc; }}
.card .eyebrow {{ margin-bottom: .5rem; }}
.card .title {{ font-size: 1.1rem; font-weight: 700; color: {INK}; letter-spacing: -0.01em; }}
.card .body {{ font-size: .88rem; color: {INK2}; margin: .45rem 0 .9rem 0; line-height: 1.5; }}
.card .metric {{ font-size: 1.5rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; }}
.card .metric-sub {{ font-size: .78rem; color: {INK2}; margin-bottom: .6rem; }}
.card ul {{ margin: 0; padding-left: 1.1rem; font-size: .84rem; color: {INK}; line-height: 1.6; }}
.stack {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: .3rem; }}
.chip {{ border: 1px solid {LINE}; border-radius: 999px; padding: 3px 10px; font-size: .76rem;
         color: {INK2}; background: #f6f7f9; }}
.side-foot {{ font-size: .76rem; color: {INK2}; line-height: 1.55; margin-top: .5rem; }}
.side-foot a {{ color: {INK2}; text-decoration: underline; }}
</style>
""", unsafe_allow_html=True)


def chart_theme():
    return {"config": {
        "font": "Inter", "view": {"stroke": None}, "background": "transparent",
        "axis": {"labelColor": INK2, "titleColor": INK2, "labelFontSize": 12, "titleFontSize": 12,
                 "titleFontWeight": 500, "gridColor": "#eef0f3", "domainColor": LINE, "tickColor": LINE},
        "title": {"fontSize": 14, "fontWeight": 600, "color": INK, "anchor": "start", "offset": 12},
    }}


alt.theme.register("ri", enable=True)(chart_theme)


@st.cache_resource
def engine() -> Engine:
    from retail_intelligence.config import MODEL_PATH
    if not MODEL_PATH.exists():   # fresh cloud deployment: training on the committed snapshots takes seconds
        from retail_intelligence.train import main as train
        with st.spinner("Training the retention model for this deployment (a few seconds, first visit only)"):
            train()
    return Engine.load()


E = engine()
M = E.metrics


def s_date() -> str:
    return M["scoring_snapshot"]["cutoff"]


# ---------------------------------------------------------------- helpers
def header(eyebrow: str, title: str, lede: str, about: str | None = None):
    """Page title block. `about` adds a "How to read this page" popover on the right."""
    left, right = st.columns([5, 1], vertical_alignment="bottom")
    with left:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
        st.title(title)
    if about:
        with right.popover("How to read this page", width="stretch"):
            st.markdown(about)
    st.markdown(f'<div class="lede">{lede}</div>', unsafe_allow_html=True)


def tip(text: str) -> str:
    """Small info icon with an explanation on hover."""
    return f'<span class="tip">i<span class="tiptext">{html.escape(text)}</span></span>'


def kpis(items: list[tuple]):
    """KPI cards: (label, value, sub) or (label, value, sub, explanation shown on hover)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, sub = item[:3]
        info = tip(item[3]) if len(item) > 3 and item[3] else ""
        col.markdown(f'<div class="kpi"><div class="label">{label}{info}</div>'
                     f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                     unsafe_allow_html=True)
    st.write("")


def nice(text) -> str:
    """Upper-case source descriptions in title case, keeping short codes readable."""
    return " ".join(w if any(ch.isdigit() for ch in w) else w.capitalize() for w in str(text).split())


def badge(segment: str) -> str:
    bg, fg = SEG_BADGE[segment]
    return f'<span class="badge" style="background:{bg};color:{fg}">{segment}</span>'


def style_segments(df: pd.DataFrame):
    def css(v):
        if v in SEG_BADGE:
            bg, fg = SEG_BADGE[v]
            return f"background-color:{bg};color:{fg};font-weight:600"
        return ""
    return df.style.map(css, subset=[c for c in ["segment"] if c in df.columns])


TABLE_CFG = {
    "CustomerID": st.column_config.NumberColumn("Customer", format="%d"),
    "probability": st.column_config.ProgressColumn(
        "Repurchase probability", format="%.0f%%", min_value=0, max_value=100,
        help="Model probability that the customer orders again in the next 90 days."),
    "segment": st.column_config.TextColumn(
        "Segment", help="Loyal: 60% or more. Nurture: 30 to 60%. At risk: below 30%."),
    "value_tier": st.column_config.TextColumn(
        "Value", help="High value is the top 20% of customers by historical spend."),
    "action": st.column_config.TextColumn(
        "Recommended action", help="Marketing action from the segment and value tier."),
    "recency_days": st.column_config.NumberColumn("Days since last order"),
    "frequency": st.column_config.NumberColumn("Orders"),
    "monetary": st.column_config.NumberColumn("Total spent (£)", format="localized"),
}
LIST_COLS = ["CustomerID", "probability", "segment", "value_tier", "action",
             "recency_days", "frequency", "monetary"]


def table_view(df: pd.DataFrame) -> pd.DataFrame:
    """Display copy: probability in percent points, spend rounded to whole pounds."""
    d = df[[c for c in LIST_COLS if c in df.columns]].copy()
    if "probability" in d:
        d["probability"] = (d["probability"] * 100).round(1)
    if "monetary" in d:
        d["monetary"] = d["monetary"].round(0).astype("int64")
    return d


def customer_table(df: pd.DataFrame, height: int | None = None):
    st.dataframe(style_segments(table_view(df)), hide_index=True, width="stretch",
                 column_config=TABLE_CFG, height=height or "auto")


def segment_chart(df: pd.DataFrame, title: str):
    counts = (df["segment"].value_counts().reindex(SEGMENTS, fill_value=0)
              .rename_axis("segment").reset_index(name="customers"))
    counts["share"] = counts["customers"] / max(int(counts["customers"].sum()), 1)
    base = alt.Chart(counts, title=title).encode(
        y=alt.Y("segment:N", sort=SEGMENTS, title=None,
                scale=alt.Scale(paddingInner=0.38),
                axis=alt.Axis(ticks=False, domain=False, labelFontSize=13, minExtent=72, labelLimit=120)),
        x=alt.X("customers:Q", title="Customers", axis=alt.Axis(grid=False, tickCount=5),
                scale=alt.Scale(domain=[0, max(int(counts["customers"].max()), 1) * 1.35])))
    tooltip = [alt.Tooltip("segment:N", title="Segment"),
               alt.Tooltip("customers:Q", title="Customers", format=","),
               alt.Tooltip("share:Q", title="Share", format=".0%")]
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color("segment:N", legend=None,
                        scale=alt.Scale(domain=SEGMENTS, range=[SEG_COLORS[s] for s in SEGMENTS])),
        tooltip=tooltip)
    text = base.mark_text(align="left", dx=6, color=INK2, fontSize=12).encode(
        text=alt.Text("label:N"), tooltip=tooltip).transform_calculate(
        label="format(datum.customers, ',') + '  (' + format(datum.share, '.0%') + ')'")
    st.altair_chart((bars + text).properties(height=alt.Step(46)), width="stretch")


def driver_chart(exp: pd.DataFrame):
    d = exp.copy()
    d["direction"] = d["shap"].apply(lambda v: "Raises probability" if v >= 0 else "Lowers probability")
    bars = alt.Chart(d).mark_bar(cornerRadius=3, height=16).encode(
        y=alt.Y("label:N", sort=alt.EncodingSortField("shap", op="sum", order="descending"),
                title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=220)),
        x=alt.X("shap:Q", title="SHAP contribution (log-odds)", axis=alt.Axis(tickCount=6)),
        color=alt.Color("direction:N", scale=alt.Scale(
            domain=["Raises probability", "Lowers probability"], range=[BLUE, RED]),
            legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("label:N", title="Feature"),
                 alt.Tooltip("value:Q", title="This customer", format=",.2f"),
                 alt.Tooltip("typical:Q", title="Typical customer", format=",.2f"),
                 alt.Tooltip("shap:Q", title="SHAP (log-odds)", format="+.2f")])
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
    st.altair_chart((rule + bars).properties(height=alt.Step(30)), width="stretch")


# ================================================================== pages
ICONS = {
    "customers": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
                 '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "demand": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "pricing": '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>'
               '<line x1="7" y1="7" x2="7.01" y2="7"/>',
}


def icon(name: str, color: str, bg: str) -> str:
    return (f'<span class="mod-icon" style="background:{bg}"><svg viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg></span>')


def home():
    tags = ["Python", "dbt", "DuckDB", "XGBoost", "SHAP", "scikit-learn", "pyfixest", "Streamlit", "Docker",
            "GitHub Actions", "LLM tool calling"]
    stats = [("1.07M", "invoice lines audited"), ("524,674", "store-product-weeks"), ("3", "validated models"),
             ("42", "dbt data tests"), ("55", "automated tests")]
    st.markdown(
        '<div class="hero"><div class="h-eyebrow">Data science · Analytics engineering · AI engineering</div>'
        '<div class="h-title">Retail Intelligence</div>'
        '<div class="h-lede">Three questions every retailer asks, answered end to end on real data: who will buy '
        'again, how much of each product will be ordered, and which prices and promotions pay. Each module combines '
        'tested dbt models, validated machine learning and a decision tool, with an AI analyst on top.</div>'
        '<div class="h-stats">' + "".join(f'<div class="h-stat"><div class="n">{n}</div><div class="l">{lbl}</div></div>'
                                          for n, lbl in stats) + '</div>'
        '<div class="h-tags">' + "".join(f'<span class="h-tag">{t}</span>' for t in tags) + '</div></div>',
        unsafe_allow_html=True)

    h = M["holdout"]
    D, P = E.demand, E.pricing
    modules = [
        ("customers", BLUE, "#e8f1fc", "Customers", "Customer retention",
         "Scores every customer on the probability of ordering again in the next 90 days, explains each score "
         "and turns it into a marketing action.",
         f"{h['capture_top20']:.0%}", f"of future buyers in the top 20% by score (AUC {h['auc']:.3f})",
         ["Segments and next-best actions", "SHAP explanation per customer", "Campaign lists and batch scoring"],
         PAGE_OVERVIEW, "Open retention overview"),
        ("demand", "#0f8f84", "#e4f4f2", "Planning", "Demand forecasting",
         "Forecasts the units of 1,041 products for the next four weeks with an 80% range for safety stock, "
         "backtested against planner baselines.",
         f"{D.metrics['methods']['Global gradient boosting']['wape_total']:.1%}" if D else "n/a",
         "catalogue error, against 11.5% for the best moving average",
         ["Product forecast with P10 to P90", "Twelve-origin backtest", "Conformal interval calibration"],
         PAGE_DEMAND, "Open demand forecast"),
        ("pricing", "#c7702a", "#fbefe4", "Planning", "Pricing and promotions",
         "Estimates price sensitivity and promotion lift for 55 grocery products and simulates the profit of any "
         "price or promotion.",
         "-1.05 to -2.91" if P else "n/a", "category price elasticities, from fixed-effects regressions",
         ["Elasticity by product with intervals", "Promotion and discount effects",
          "Profit simulator with price advice"],
         PAGE_PRICING, "Open pricing simulator"),
    ]
    st.markdown('<div class="section" style="margin-top:6px">Three modules</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for col, (ic, color, tint, eyebrow, title, body, metric, metric_sub, bullets, page, cta) in zip(cols, modules):
        with col.container(border=True, key=f"surface_mod_{ic}"):
            items = "".join(f"<li>{b}</li>" for b in bullets)
            st.markdown(
                f'<div class="card flat">{icon(ic, color, tint)}'
                f'<div class="eyebrow" style="color:{color}">{eyebrow}</div><div class="title">{title}</div>'
                f'<div class="body">{body}</div><div class="metric">{metric}</div>'
                f'<div class="metric-sub">{metric_sub}</div><ul>{items}</ul></div>', unsafe_allow_html=True)
            st.page_link(page, label=cta + "  →")

    st.markdown('<div class="section">How it is built</div>', unsafe_allow_html=True)
    pillars = [
        (BLUE, "Data", "UCI Online Retail II and dunnhumby Breakfast at the Frat, audited before any modelling."),
        ("#0f8f84", "Engineering", "dbt on DuckDB: staging, intermediate and mart layers, 42 data tests including "
                                   "label leakage."),
        ("#6a5acd", "Modelling", "XGBoost with SHAP, gradient boosting with conformal intervals, fixed-effects "
                                 "elasticities."),
        ("#c7702a", "Validation", "Holdout and out-of-time tests, rolling-origin backtests against business "
                                  "baselines."),
        (NAVY, "Production", "Installable package, Docker, CI with 55 tests and a tool-calling AI analyst."),
    ]
    cols = st.columns(5, gap="small")
    for col, (color, t, d) in zip(cols, pillars):
        col.markdown(f'<div class="pillar"><div class="t"><span class="dot" style="background:{color}"></span>{t}</div>'
                     f'<div class="d">{d}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">Where to start</div>', unsafe_allow_html=True)
    with st.container(border=True, key="surface_start"):
        a, b, c, d, e = st.columns(5)
        a.page_link(PAGE_PRICING, label="Run a promotion scenario")
        b.page_link(PAGE_CUSTOMER, label="See why a customer is at risk")
        c.page_link(PAGE_DEMAND, label="Check a product forecast")
        d.page_link(PAGE_ANALYST, label="Ask the AI analyst")
        e.page_link(REPO_URL, label="Code on GitHub")


def overview():
    header("Customer base", "Retention overview",
           f"Every customer active before {pd.Timestamp(s_date()):%d %B %Y} is scored on the probability "
           "of buying again in the next 90 days, then assigned a segment and a recommended marketing action.",
           about="""
**What the score is.** An XGBoost model estimates, for each customer, the probability of placing at least one
order in the next 90 days. It learned from what customers did after past cutoff dates.

**Segments.** Loyal (60% or more), Nurture (30 to 60%) and At risk (below 30%).
High value means the top 20% of customers by historical spend.

**Actions.** Segment and value combine into one marketing action. *Priority win-back* is the short list
worth a personal call: high-value customers the model expects to lose.

**Caveat.** Absolute probabilities move with the season; the ranking is what holds (see *Retention model*).
""")
    s = E.summary()
    pw = s["actions"]["Priority win-back"]
    kpis([
        ("Customers scored", f"{s['customers']:,}", "Registered customers with past orders",
         "Every customer with at least one order before the scoring date. Guest checkouts have no ID and are excluded."),
        ("Expected to buy again", f"{s['expected_repurchasers']:,}", "Sum of individual probabilities",
         "Adding up every customer's probability gives the number of customers expected to order in the next 90 days."),
        ("At risk", f"{s['segments']['At risk']:,}",
         f"{s['segments']['At risk'] / s['customers']:.0%} of the customer base",
         "Customers with a repurchase probability below 30%."),
        ("Priority win-back", f"{pw['customers']:,}", f"£{pw['past_spend']:,.0f} historical spend",
         "High-value customers (top 20% by spend) who are at risk. The first list for a retention campaign."),
    ])
    left, right = st.columns([3, 2], gap="large")
    with left:
        segment_chart(E.scored, "Customers by segment")
    with right:
        st.subheader("Recommended actions")
        acts = pd.DataFrame([{"Action": a, "Customers": f"{v['customers']:,}",
                              "Historical spend": f"£{v['past_spend']:,.0f}"}
                             for a, v in s["actions"].items()])
        st.dataframe(acts, hide_index=True, width="stretch")
    st.subheader("Highest priority: high-value customers at risk")
    customer_table(E.find(action="Priority win-back", sort_by="monetary", limit=10))


def campaigns():
    header("Activation", "Campaign builder",
           "Filter the scored customer base and export a contact list for your email or CRM tool.",
           about="""
**How to use it.** Pick the segments or actions to target, narrow by region and spend, and cap the list at the
size your team can contact. The list is sorted by repurchase probability.

**Typical campaigns.**
- *Win-back*: segment At risk, minimum spend £500.
- *Cross-sell*: action Nurture: cross-sell.
- *Loyalty*: segment Loyal, the top 200 contacts.

The export contains the customer ID, probability, segment, value tier and action, ready for an email or CRM tool.
""")
    f = st.columns([2, 2, 1.2, 1.2, 1.2])
    segs = f[0].multiselect("Segment", SEGMENTS, default=["At risk"],
                            help="Leave empty to include every segment.")
    acts = f[1].multiselect("Recommended action", ACTIONS, help="Leave empty to include every action.")
    region = f[2].selectbox("Region", ["All", "United Kingdom", "International"])
    min_spend = f[3].number_input("Min. spend (£)", min_value=0, value=0, step=100,
                                  help="Historical spend of the customer, in pounds.")
    limit = f[4].number_input("Max. contacts", min_value=1, value=500, step=50,
                              help="Size of the list your team can handle.")

    v = E.scored[E.scored["segment"].isin(segs or SEGMENTS) & (E.scored["monetary"] >= min_spend)]
    if acts:
        v = v[v["action"].isin(acts)]
    if region != "All":
        v = v[v["is_uk"] == int(region == "United Kingdom")]
    v = v.head(int(limit))

    kpis([("Customers selected", f"{len(v):,}", "After filters and limit"),
          ("Average probability", f"{v['probability'].mean():.0%}" if len(v) else "n/a", "Of buying again"),
          ("Historical spend", f"£{v['monetary'].sum():,.0f}", "Of the selected customers")])
    customer_table(v, height=440)
    st.download_button("Export contact list (CSV)",
                       v[["CustomerID", "probability", "segment", "value_tier", "action"]].to_csv(index=False),
                       "campaign_list.csv", "text/csv", type="primary", disabled=v.empty)


def customer_page():
    header("Customer", "Customer profile",
           "Look up any customer to see the score, the reasons behind it, and how it would change.",
           about="""
**Why this score.** Each bar is an exact SHAP value: how much one feature moves this customer's score away from
the average customer, in log-odds. Blue bars raise the probability, red bars lower it. Hover a bar to compare the
customer's value with a typical customer.

**Scenario analysis.** Change the profile and the model rescores it instantly. For example, set *Days since last
order* to 10 and *Orders in last 90 days* to 1 to see what a successful win-back purchase would do to the score.
""")
    ids = E.find(action="Priority win-back", sort_by="monetary", limit=1)["CustomerID"].tolist() or [int(E.scored["CustomerID"].iloc[0])]
    all_ids = sorted(E.scored["CustomerID"].astype(int).tolist())
    cid = st.selectbox("Customer ID", all_ids, index=all_ids.index(int(ids[0])),
                       help="Type to search. Opens on the highest-spending priority win-back customer.")
    row = E.customer(cid)

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(
            f'<div class="panel"><div class="k">Repurchase probability</div>'
            f'<div class="v" style="font-size:2rem">{row["probability"]:.0%}</div>'
            f'<div class="k">Segment</div><div class="v">{badge(row["segment"])}</div>'
            f'<div class="k">Value tier</div><div class="v">{row["value_tier"]}</div>'
            f'<div class="k">Recommended action</div><div class="v">{html.escape(row["action"])}</div>'
            f'<div class="k">Profile</div><div class="v" style="font-weight:500">'
            f'{int(row["frequency"])} orders · £{row["monetary"]:,.0f} spent · '
            f'last order {int(row["recency_days"])} days ago</div></div>', unsafe_allow_html=True)
    with right:
        st.subheader("Why this score")
        exp, base = E.explain(row)
        st.caption(f"Exact SHAP values from the model. Starting from the average customer ({base:.0%}), "
                   "each bar shows how much a feature pushes this customer's score up or down. "
                   "Hover for the customer's value and the typical value.")
        driver_chart(exp)

    st.subheader("Scenario analysis")
    st.caption("Adjust the profile to test a scenario, for example a recent purchase after a win-back offer.")
    a, b, c, d, e = st.columns(5)
    gap = row["avg_days_between_orders"]
    sc = {
        "recency_days": a.number_input("Days since last order", 0, 800, int(row["recency_days"]),
                                       help="The strongest driver of the score."),
        "frequency": b.number_input("Orders", 1, 1000, int(row["frequency"]),
                                    help="Distinct invoices before the scoring date."),
        "orders_last_90d": c.number_input("Orders in last 90 days", 0, 500, int(row["orders_last_90d"])),
        "monetary": d.number_input("Total spent (£)", 0.0, value=float(row["monetary"]), step=100.0,
                                   format="%.0f", help="Net of returns."),
        "avg_order_value": e.number_input("Average order value (£)", 0.0, value=float(row["avg_order_value"]),
                                          step=10.0, format="%.0f"),
        "unique_products": a.number_input("Distinct products", 1, 5000, int(row["unique_products"])),
        "tenure_days": b.number_input("Days as customer", 0, 900, int(row["tenure_days"]),
                                      help="Days between the first order and the scoring date."),
        "avg_days_between_orders": c.number_input("Average days between orders", 0.0, 800.0,
                                                  0.0 if pd.isna(gap) else float(gap), 1.0, format="%.1f",
                                                  help="Ignored when the customer has a single order."),
        "return_rate": d.number_input("Return rate", 0.0, 5.0, float(row["return_rate"]), 0.01,
                                      format="%.2f", help="Cancelled invoices divided by orders. 0.17 means about one "
                                      "cancellation for every six orders."),
        "is_uk": int(e.selectbox("Region", ["United Kingdom", "International"],
                                 index=0 if row["is_uk"] else 1) == "United Kingdom"),
    }
    if sc["frequency"] == 1:
        sc["avg_days_between_orders"] = float("nan")
    new = E.score(pd.DataFrame([sc])).iloc[0]
    delta = new["probability"] - row["probability"]
    kpis([("Scenario probability", f"{new['probability']:.0%}", f"{delta * 100:+.1f} pts vs current",
           "Percentage points against the customer's current score."),
          ("Scenario segment", new["segment"], f"Current: {row['segment']}"),
          ("Scenario action", new["action"], f"Current: {row['action']}")])


def batch_scoring():
    header("Scoring", "Batch scoring",
           "Upload a CSV with one row per customer. The file is scored with the production model "
           "and returned with probabilities, segments and recommended actions.")
    template = pd.DataFrame([
        {"CustomerID": 1, "recency_days": 20, "frequency": 8, "monetary": 1450, "avg_order_value": 181,
         "unique_products": 64, "tenure_days": 410, "return_rate": 0.05, "orders_last_90d": 2,
         "avg_days_between_orders": 55, "is_uk": 1},
        {"CustomerID": 2, "recency_days": 240, "frequency": 1, "monetary": 320, "avg_order_value": 320,
         "unique_products": 18, "tenure_days": 240, "return_rate": 0, "orders_last_90d": 0,
         "avg_days_between_orders": None, "is_uk": 0}])
    c1, c2 = st.columns([3, 1])
    c1.markdown("Required columns: " + ", ".join(f"`{c}`" for c in E.features) + ". `CustomerID` is optional; "
                "`avg_days_between_orders` may be empty for customers with a single order.")
    c2.download_button("Download template", template.to_csv(index=False), "template.csv", width="stretch")
    sample = c2.button("Score a sample of 200 customers", width="stretch",
                       help="Scores 200 random customers from the dataset, as if they were an uploaded file.")
    if sample:
        st.session_state.batch_sample = True

    file = st.file_uploader("Customer file", type="csv", label_visibility="collapsed")
    if file:
        st.session_state.batch_sample = False
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"The file could not be read: {e}")
            return
        name = file.name
    elif st.session_state.get("batch_sample"):
        df = E.scored.sample(200, random_state=7)[["CustomerID", *E.features]]
        name = "Sample of 200 customers"
    else:
        return
    missing = [c for c in E.features if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return
    num = df[E.features].apply(pd.to_numeric, errors="coerce")
    bad = num.drop(columns="avg_days_between_orders").isna().any(axis=1)
    if bad.any():
        st.warning(f"{int(bad.sum())} rows with empty or non-numeric values were skipped.")
    df = df[~bad].copy()
    df[E.features] = num[~bad]
    if "CustomerID" not in df.columns:
        df.insert(0, "CustomerID", range(1, len(df) + 1))
    if df.empty:
        st.error("No valid rows to score.")
        return
    out = E.score(df)
    kpis([("Customers scored", f"{len(out):,}", name),
          ("Average probability", f"{out['probability'].mean():.0%}", "Of buying again"),
          ("At risk", f"{(out['segment'] == 'At risk').sum():,}", "Need a win-back action")])
    left, right = st.columns([2, 3], gap="large")
    with left:
        segment_chart(out, "Uploaded customers by segment")
    with right:
        customer_table(out, height=320)
    st.download_button("Download scored file (CSV)", out.to_csv(index=False),
                       "scored_customers.csv", "text/csv", type="primary")


def model_page():
    h, o = M["holdout"], M["out_of_time"]
    header("Retention model", "Retention model performance",
           "Validated three ways: cross-validation to choose the model, a 20% holdout it never saw, "
           "and an out-of-time test that trains on June 2011 and predicts September 2011.",
           about="""
**Cumulative gains.** Sort customers by score and contact them from the top. The blue line shows the share of
all future buyers reached; the dashed line is a random list. The gap is the value of the model.

**Calibration.** Customers grouped into ten bands by score. Points on the dashed line mean that a predicted 40%
really buys again about 40% of the time.

**What drives the score.** Average size of each feature's SHAP contribution across customers.

**Why XGBoost.** It is within noise of logistic regression on accuracy, but allows monotone constraints
(more recent customers can never score lower) and gives exact explanations.
""")
    kpis([("Holdout AUC", f"{h['auc']:.3f}", f"Recency rule: {M['holdout_recency_rule']['auc']:.3f}",
           "Probability that a random buyer is ranked above a random non-buyer. 0.5 is chance, 1 is perfect. "
           "The recency rule ranks customers by days since the last order only."),
          ("Top 20% reach", f"{h['capture_top20']:.0%}", "Of all customers who bought again",
           "Contacting the 20% of customers with the highest scores reaches this share of all future buyers."),
          ("Lift", f"{h['lift_top20']:.1f}x", "Over a random contact list",
           "The top 20% by score holds this many times more buyers than a random 20%."),
          ("Out-of-time AUC", f"{o['auc']:.3f}", "Ranking holds on a later period",
           "Trained on data up to June 2011 and tested on September 2011, a period the model never saw.")])
    left, right = st.columns(2, gap="large")
    with left:
        g = E.gains
        line = alt.Chart(g, title="Cumulative gains (holdout)").mark_line(color=BLUE, strokeWidth=2).encode(
            x=alt.X("pct_contacted:Q", title="Share of customers contacted", axis=alt.Axis(format="%")),
            y=alt.Y("pct_captured:Q", title="Share of buyers reached", axis=alt.Axis(format="%")))
        hover = alt.Chart(g).mark_point(size=90, opacity=0).encode(
            x="pct_contacted:Q", y="pct_captured:Q",
            tooltip=[alt.Tooltip("pct_contacted:Q", title="Contacted", format=".0%"),
                     alt.Tooltip("pct_captured:Q", title="Buyers reached", format=".0%")])
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            color=MUTED, strokeDash=[4, 4], strokeWidth=1.5).encode(x="x:Q", y="y:Q")
        lbl = alt.Chart(pd.DataFrame({"x": [0.03, 0.70], "y": [0.93, 0.56],
                                      "t": ["Model ranking", "Random list"],
                                      "c": [BLUE, MUTED]})).mark_text(align="left", fontWeight=600).encode(
            x="x:Q", y="y:Q", text="t:N", color=alt.Color("c:N", scale=None))
        st.altair_chart((diag + line + hover + lbl).properties(height=320), width="stretch")
    with right:
        cal = E.calibration.assign(decile=lambda d: range(1, len(d) + 1))
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            color=MUTED, strokeDash=[4, 4], strokeWidth=1.5).encode(x="x:Q", y="y:Q")
        pts = alt.Chart(cal, title="Calibration by score decile (holdout)").mark_line(
            color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
            x=alt.X("predicted:Q", title="Mean predicted probability", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("observed:Q", title="Observed repurchase rate", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("decile:Q", title="Decile"),
                     alt.Tooltip("predicted:Q", title="Predicted", format=".1%"),
                     alt.Tooltip("observed:Q", title="Observed", format=".1%"),
                     alt.Tooltip("customers:Q", title="Customers")])
        st.altair_chart((diag + pts).properties(height=320), width="stretch")

    left, right = st.columns([3, 2], gap="large")
    with left:
        bars = alt.Chart(E.importance, title="What drives the score (mean absolute SHAP value)").mark_bar(
            color=BLUE, cornerRadiusEnd=3, height=16).encode(
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=220)),
            x=alt.X("mean_abs_shap:Q", title="Average impact on the score (log-odds)"),
            tooltip=[alt.Tooltip("label:N", title="Feature"),
                     alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".3f")])
        st.altair_chart(bars.properties(height=330), width="stretch")
    with right:
        st.subheader("Model comparison (holdout)")
        lg = M["holdout_logistic"]
        comp = pd.DataFrame([
            {"Model": "XGBoost (production)", "AUC": f"{h['auc']:.3f}", "Brier": f"{h['brier']:.3f}",
             "Top 20% reach": f"{h['capture_top20']:.1%}"},
            {"Model": "Logistic regression", "AUC": f"{lg['auc']:.3f}", "Brier": f"{lg['brier']:.3f}",
             "Top 20% reach": f"{lg['capture_top20']:.1%}"},
            {"Model": "Recency rule", "AUC": f"{M['holdout_recency_rule']['auc']:.3f}", "Brier": "n/a",
             "Top 20% reach": "n/a"},
        ])
        st.dataframe(comp, hide_index=True, width="stretch")
        st.caption("The two learned models are within cross-validation noise of each other. XGBoost is used "
                   "for its monotone business constraints, native handling of one-order customers and exact SHAP explanations.")

    st.subheader("Seasonality caveat")
    st.markdown(
        f"Out of time, the model ranks customers as well as on the holdout (AUC {o['auc']:.3f}), but its average "
        f"prediction was {o['mean_prediction']:.0%} while {o['base_rate']:.0%} of customers actually returned: "
        "customers buy far more in the autumn than in the summer. Use the score first to **prioritise** customers; "
        "read absolute probabilities and segment sizes with the season in mind.")

    st.subheader("Methodology")
    tr = M["training_snapshot"]
    st.markdown(f"""
- **Data:** UCI Online Retail II, 1.07M invoice lines (Dec 2009 to Dec 2011), audited and cleaned in dbt on DuckDB
  (staging, intermediate and mart layers, with data tests including a label-leakage test).
- **Target:** at least one order in the 90 days after a cutoff date. Features use only data before the cutoff.
- **Training:** snapshot of {tr['cutoff']}, {tr['customers']:,} customers, {tr['repurchase_rate']:.1%} repurchased.
  Scoring: snapshot of {M['scoring_snapshot']['cutoff']}.
- **Model:** {M['model']}, selected by repeated 5-fold cross-validation against logistic regression,
  random forest and histogram gradient boosting.
- **Segments:** Loyal from {E.loyal_t:.0%}, Nurture {E.nurture_t:.0%} to {E.loyal_t:.0%}, At risk below {E.nurture_t:.0%}.
  High value is the top 20% by spend (from £{E.high_value_t:,.0f}).
- **Full analysis:** `notebooks/01_retention_analysis.ipynb` in the repository.
""")


def demand_page():
    D = E.demand
    if D is None:
        st.info("The demand forecast has not been built on this deployment. Run `python -m retail_intelligence.demand`.")
        return
    m, f = D.metrics, D.forecast
    gb, best = m["methods"]["Global gradient boosting"], m["methods"][m["best_baseline"]]
    start, end = (pd.Timestamp(x) for x in m["forecast_window"])
    header("Planning", "Demand forecast",
           f"Units each product will be ordered from {start:%d %b %Y} to {end:%d %b %Y}, with an 80% range for "
           "safety stock. One global gradient boosting model covers the catalogue; it is tested every two weeks "
           "over the last half year against the moving averages a planner would otherwise use.",
           about="""
**The forecast.** Units each product will be ordered in the next four weeks (P50, the median), with a range from
P10 to P90. The real number should fall inside that range about 8 times out of 10.

**How to use the range.** Order the P50 when a stock-out is cheap; order up to the P90 for products where running
out costs a sale, which accepts about a one-in-ten chance of a stock-out per cycle.

**WAPE.** Weighted absolute percentage error: total units missed divided by total units sold. Lower is better.
Product-level WAPE scores each product; catalogue WAPE scores the total.

**The chart.** Grey is weekly history; blue is the forecast spread evenly over the next four weeks, with the
shaded 80% range.
""")
    best_total = min(v["wape_total"] for k, v in m["methods"].items() if k != "Global gradient boosting")
    kpis([
        ("Products forecast", f"{len(f):,}", f"{f.p50.sum():,.0f} units expected",
         "Products sold in at least 70 of 104 weeks. Together they are about 63% of units and revenue."),
        ("Product-level WAPE", f"{gb['wape_product']:.1%}", f"{m['best_baseline']}: {best['wape_product']:.1%}",
         "Units missed across products divided by units sold, averaged over 12 backtest origins. "
         "Most products sell in lumpy wholesale orders, so part of this error cannot be forecast."),
        ("Catalogue WAPE", f"{gb['wape_total']:.1%}", f"Best baseline: {best_total:.1%}",
         "Error on the total of all products, the number that drives warehouse and staffing plans."),
        ("80% range coverage", f"{m['interval_coverage_calibrated']:.0%}", "After conformal calibration",
         "Share of backtest outcomes that fell inside the P10 to P90 range. The target is 80%."),
    ])

    c1, c2 = st.columns([1, 2])
    cats = ["All categories"] + sorted(f.category.dropna().unique())
    cat = c1.selectbox("Category", cats)
    text = c2.text_input("Search product", placeholder="Description words or product id, e.g. heart, 85123A")
    view = D.find(text or None, None if cat == "All categories" else cat, limit=200)
    if view.empty:
        st.warning("No product matches the filters.")
        return

    labels = {r.product_id: f"{r.product_id}  {nice(r.description)}" for r in view.itertuples()}
    pid = st.selectbox("Product", list(labels), format_func=labels.get,
                       help="Products are sorted by forecast volume. Type to search.")
    row = view[view.product_id == pid].iloc[0]
    left, right = st.columns([3, 1], gap="large")
    with left:
        h = D.product_history(pid, 60)
        span = pd.date_range(pd.Timestamp(row.forecast_start), periods=4, freq="7D")
        fc = pd.DataFrame({"week_start": span, "p10": row.p10 / 4, "p50": row.p50 / 4, "p90": row.p90 / 4})
        base = alt.Chart(h).mark_line(color=MUTED, strokeWidth=1.6).encode(
            x=alt.X("week_start:T", title=None), y=alt.Y("units:Q", title="Units per week"),
            tooltip=[alt.Tooltip("week_start:T", title="Week"), alt.Tooltip("units:Q", title="Units", format=",")])
        band = alt.Chart(fc).mark_area(color=BLUE, opacity=0.2).encode(x="week_start:T", y="p10:Q", y2="p90:Q")
        line = alt.Chart(fc).mark_line(color=BLUE, strokeWidth=3, point=True).encode(
            x="week_start:T", y="p50:Q",
            tooltip=[alt.Tooltip("week_start:T", title="Week"),
                     alt.Tooltip("p50:Q", title="Forecast per week", format=",.0f"),
                     alt.Tooltip("p10:Q", title="P10 per week", format=",.0f"),
                     alt.Tooltip("p90:Q", title="P90 per week", format=",.0f")])
        start_rule = alt.Chart(pd.DataFrame({"x": [span[0]]})).mark_rule(
            color=MUTED, strokeDash=[3, 3]).encode(x="x:T")
        start_lbl = alt.Chart(pd.DataFrame({"x": [span[0]], "t": ["Forecast"]})).mark_text(
            align="right", dx=-6, dy=4, baseline="top", color=BLUE, fontWeight=600, fontSize=12).encode(
            x="x:T", y=alt.value(0), text="t:N")
        st.altair_chart((base + start_rule + start_lbl + band + line).properties(
            height=320, title=f"{nice(row.description)}: weekly units and the next four weeks"), width="stretch")
    with right:
        st.markdown(
            f'<div class="panel"><div class="k">Category</div><div class="v">{html.escape(str(row.category))}</div>'
            f'<div class="k">Forecast, next 4 weeks</div><div class="v" style="font-size:1.6rem">{row.p50:,.0f} units</div>'
            f'<div class="k">80% range{tip("P10 to P90. Order up to the P90 to keep the stock-out risk near one in ten.")}</div>'
            f'<div class="v">{row.p10:,.0f} to {row.p90:,.0f}</div>'
            f'<div class="k">Last 4 weeks</div><div class="v">{row.last_4_weeks:,.0f} units</div>'
            f'<div class="k">13-week average, per 4 weeks</div><div class="v">{row.last_13_weeks_avg:,.0f} units</div></div>',
            unsafe_allow_html=True)

    st.subheader("Products")
    table = view[["product_id", "description", "category", "last_4_weeks", "last_13_weeks_avg", "p10", "p50", "p90"]].copy()
    table["description"] = table["description"].map(nice)
    num = ["last_4_weeks", "last_13_weeks_avg", "p10", "p50", "p90"]
    table[num] = table[num].round(0).astype("int64")
    st.dataframe(table, hide_index=True, width="stretch", height=380, column_config={
        "product_id": "Product", "description": "Description", "category": "Category",
        "last_4_weeks": st.column_config.NumberColumn("Last 4 weeks", format="localized"),
        "last_13_weeks_avg": st.column_config.NumberColumn(
            "13-week avg per 4 weeks", format="localized",
            help="Average weekly units over the last 13 weeks, times four. The moving average a planner would use."),
        "p10": st.column_config.NumberColumn("P10", format="localized", help="Low end of the 80% range."),
        "p50": st.column_config.NumberColumn("Forecast (P50)", format="localized",
                                             help="Median forecast of units in the next four weeks."),
        "p90": st.column_config.NumberColumn("P90", format="localized", help="High end of the 80% range.")})
    st.download_button("Export forecast (CSV)", view.to_csv(index=False), "demand_forecast.csv", "text/csv")

    st.subheader("Backtest: forecast error by method")
    acc = pd.DataFrame([{"Method": k, "Product WAPE": v["wape_product"], "Bias": v["bias"],
                         "Catalogue WAPE": v["wape_total"]} for k, v in m["methods"].items()])
    st.dataframe(acc.style.format({"Product WAPE": "{:.1%}", "Bias": "{:+.1%}", "Catalogue WAPE": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.caption(f"{m['backtest_origins']} forecast origins from {m['backtest_period'][0]} to {m['backtest_period'][1]}, "
               "retrained on past data only at each origin. Lower is better; bias below zero means under-forecasting. "
               "Only one past December exists to learn the pre-Christmas slowdown, so December forecasts are the "
               "least reliable (see notebook 02, section 13).")


def pricing_page():
    P = E.pricing
    if P is None:
        st.info("The pricing model has not been built on this deployment. Run `python -m retail_intelligence.pricing`.")
        return
    prods = P.products
    header("Planning", "Pricing and promotions",
           "Price elasticities and promotion effects estimated with fixed-effects regressions on 156 weeks of a "
           "US grocery chain (dunnhumby Breakfast at the Frat, 77 stores, 4 categories). Simulate a base price "
           "change or a promotion and see units, revenue and gross profit against a regular week.",
           about="""
**Price elasticity.** The percentage change in units for a 1% change in price. At -2.0, a 5% price rise loses
about 10% of units. Below -1, demand is *elastic*: volume reacts more than price.

**How it is estimated.** A log-log regression per category with store-by-product and category-by-week fixed
effects, so the price effect is not confused with store size or seasonal demand. Product estimates are pulled
towards their category average when the data is thin (partial pooling).

**Promotions.** A temporary discount is shown on the shelf tag. A *circular feature* (weekly flyer) and an
*in-store display* add lift on top of the price cut.

**Profit.** The data has no costs, so profit uses the gross margin you set. Recommendations follow the
Lerner rule and stay within the ±10% range the data covers. Treat them as tests, not final prices.
""")
    ce = P.model["category_elasticities"]
    kpis([(c, f"{v['elasticity_base']:.2f}", "Base price elasticity",
           f"A 1% increase in the regular price of {c.lower()} lowers units by about {abs(v['elasticity_base']):.2f}%.")
          for c, v in ce.items()])

    st.subheader("Scenario simulator")
    a, b, c = st.columns([1.3, 2, 1])
    category = a.selectbox("Category", sorted(prods.category.unique()))
    options = prods[prods.category == category]
    labels = {r.upc: f"{nice(r.description)} ({nice(r.manufacturer)})" for r in options.itertuples()}
    upc = b.selectbox("Product", list(labels), format_func=labels.get)
    margin = c.slider("Gross margin at today's price", 10, 60, int(P.model["settings"]["default_gross_margin"] * 100), 5,
                      format="%d%%", help="Costs are not in the data, so profit depends on this assumption.") / 100
    a, b, c, d = st.columns(4)
    change = a.slider("Base price change", -10, 10, 0, 1, format="%+d%%",
                      help="Permanent change to the regular shelf price, within the ±10% range seen in the data.") / 100
    discount = b.slider("Temporary discount", 0, 50, 0, 5, format="%d%%",
                        help="Promotional price cut for one week. Discounts under 10% showed no measurable lift.") / 100
    feature = c.toggle("Featured in circular", help="The product appears in the weekly advertising flyer.")
    display = d.toggle("In-store display", help="The product is placed on a secondary display, such as an end cap.")
    row = P.product(upc)
    r = P.simulate(upc, base_price_change=change, discount=discount, feature=feature, display=display,
                   gross_margin=margin)
    kpis([
        ("Shelf price", f"${r['shelf_price']:.2f}", f"Today ${row.base_price:.2f}"),
        ("Units per week", f"{r['units']:,.0f}", f"{r['units_change']:+.0%} vs a regular week",
         "Units across the whole chain in one week. The baseline is a regular week without promotion; "
         "the scenario applies the product's elasticities and the promotion lifts."),
        ("Revenue per week", f"${r['revenue']:,.0f}", f"{r['revenue_change']:+.0%} vs a regular week"),
        ("Gross profit per week", f"${r['gross_profit']:,.0f}", f"{r['gross_profit_change']:+.0%} vs a regular week",
         "Revenue minus cost, where cost is fixed by the gross margin at today's price. A discount comes "
         "straight out of the margin."),
    ])
    if 0 < discount < 0.10:
        st.caption("Discounts under 10% showed no measurable lift in the data, so they only cost margin.")

    left, right = st.columns([1, 1], gap="large")
    with left:
        grid = pd.DataFrame([{"change": x, **P.simulate(upc, base_price_change=x, gross_margin=margin)}
                             for x in [i / 100 for i in range(-10, 11)]])
        rec, label = pricing_recommend(row, margin)
        curve = alt.Chart(grid, title="Gross profit vs base price change (no promotion)").mark_line(
            color=BLUE, strokeWidth=2.4).encode(
            x=alt.X("change:Q", title="Base price change", axis=alt.Axis(format="+%")),
            y=alt.Y("gross_profit_change:Q", title="Gross profit vs today", axis=alt.Axis(format="+%")),
            tooltip=[alt.Tooltip("change:Q", title="Price change", format="+.0%"),
                     alt.Tooltip("gross_profit_change:Q", title="Profit change", format="+.1%"),
                     alt.Tooltip("units_change:Q", title="Units change", format="+.1%")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED).encode(y="y:Q")
        best = alt.Chart(pd.DataFrame({"x": [rec]})).mark_rule(color=RED, strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((zero + curve + best).properties(height=300), width="stretch")
        st.caption(f"Elasticity {row.elasticity_base_pooled:.2f}. Recommendation at a {margin:.0%} margin: "
                   f"**{label}** ({rec:+.0%}, red line). Recommendations are tests: confirm with a store-level price test.")
    with right:
        eff = pd.DataFrame([{"Category": cat, "Mechanic": PROMO_NAMES[t], "Lift": v["lift"]}
                            for cat, terms in P.model["promo_effects"].items() for t, v in terms.items()])
        bars = alt.Chart(eff, title="Promotion lift on top of the price cut").mark_bar(
            cornerRadiusEnd=3, height=10).encode(
            y=alt.Y("Category:N", title=None, axis=alt.Axis(labelLimit=180)),
            yOffset="Mechanic:N",
            x=alt.X("Lift:Q", title="Extra units", axis=alt.Axis(format="+%")),
            color=alt.Color("Mechanic:N", scale=alt.Scale(domain=list(PROMO_NAMES.values()),
                                                          range=[BLUE, "#7fb0ea", MUTED]),
                            legend=alt.Legend(orient="bottom", title=None, columns=2)),
            tooltip=["Category", "Mechanic", alt.Tooltip("Lift:Q", format="+.0%")])
        st.altair_chart(bars.properties(height=300), width="stretch")

    st.subheader("Elasticity by product")
    t = prods.copy()
    rec = t.apply(lambda r: pricing_recommend(r, margin), axis=1, result_type="expand")
    t["recommended"] = (rec[0] * 100).round(0)
    t["action"] = rec[1].str.split(":").str[0]
    t["description"] = t["description"].map(nice)
    t["promo_frequency"] = (t["promo_frequency"] * 100).round(0)
    t["ci"] = t.apply(lambda r: f"{r.elasticity_base - 1.96 * r.se_base:.2f} to {r.elasticity_base + 1.96 * r.se_base:.2f}", axis=1)
    show = t[["description", "category", "base_price", "elasticity_base_pooled", "ci", "recommended", "action",
              "elasticity_promo_pooled", "promo_frequency"]].sort_values("elasticity_base_pooled")
    st.dataframe(show, hide_index=True, width="stretch", height=420, column_config={
        "description": "Product", "manufacturer": "Manufacturer", "category": "Category",
        "base_price": st.column_config.NumberColumn("Base price", format="$%.2f"),
        "promo_frequency": st.column_config.NumberColumn(
            "Promoted", format="%.0f%%", help="Share of store-weeks with a discount, feature or display."),
        "elasticity_base_pooled": st.column_config.NumberColumn(
            "Price elasticity", format="%.2f",
            help="Percent change in units for a 1% change in the regular price, after partial pooling."),
        "ci": st.column_config.TextColumn(
            "95% CI (unpooled)", help="Confidence interval of the product's own estimate, before pooling."),
        "elasticity_promo_pooled": st.column_config.NumberColumn(
            "Discount elasticity", format="%.2f",
            help="Response to a temporary discount. Usually stronger than to a regular price change."),
        "recommended": st.column_config.NumberColumn(
            "Suggested change", format="%+.0f%%",
            help="Profit-maximising change in the regular price at the margin set above (Lerner rule), capped at ±10%."),
        "action": st.column_config.TextColumn(
            f"Action at {margin:.0%} margin",
            help="Increase: margin gained outweighs volume lost. Decrease: extra volume outweighs margin given up. "
                 "Hold: already close to the best price.")})
    v = P.model["validation"]["all_weeks"]
    st.caption("Validation: on 26 unseen weeks the price and promotion model has a WAPE of "
               f"{v['Seasonality + price and promotions']['wape']:.1%}, against "
               f"{v['Store-product average, last 13 weeks']['wape']:.1%} for each store-product's recent average. "
               "Full analysis in notebooks/03_pricing_and_promotions.ipynb.")


PROMO_NAMES = {"is_feature": "Circular feature", "is_display": "In-store display", "is_tpr_only": "Shelf tag only"}


def pricing_recommend(row, margin):
    from retail_intelligence.pricing import recommend
    return recommend(row, margin)


SUGGESTIONS = {
    "High-value customers at risk": "Which high-value customers are most at risk, and what should we do first?",
    "Christmas demand outlook": "What demand do we expect for Christmas products in the next four weeks?",
    "Most price-sensitive products": "Which grocery products are most and least sensitive to price?",
    "Pizza promotion scenario": "What happens to profit if we run DiGiorno pepperoni pizza at 20% off with a display?",
}


def analyst():
    header("AI analyst", "Ask the data",
           "A tool-using agent over the same engines as this dashboard: customer retention, demand forecasts, "
           "and pricing and promotions. It can list customers and products, explain scores, simulate price "
           "scenarios and run read-only SQL. Every answer shows the steps it took.")
    client = make_client()
    if client is None:
        st.info("The AI analyst is not connected to a language model on this deployment. Set LLM_API_KEY, "
                "LLM_BASE_URL and LLM_MODEL to enable it (see DEPLOY.md). Every other page works without it.")
        analyst_capabilities()
        return

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if not st.session_state.chat:
        st.markdown('<div class="eyebrow">Suggested questions</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (label, q) in enumerate(SUGGESTIONS.items()):
            if cols[i].button(label, key=f"s{i}", width="stretch"):
                st.session_state.pending = q
                st.rerun()

    for turn in st.session_state.chat:
        render_turn(turn["role"], turn["content"], turn.get("result"))

    q = st.chat_input("Ask about customers, demand, prices or promotions")
    q = q or st.session_state.pop("pending", None)
    if not q:
        return
    history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat]
    st.session_state.chat.append({"role": "user", "content": q})
    render_turn("user", q)
    with st.spinner("Analyzing"):
        try:
            res = run_agent(q, E, client, model_name(), history)
        except Exception as e:
            detail = getattr(getattr(e, "response", None), "text", "") or ""
            st.error(f"The language model request failed: {e}. {detail[:400]}".strip())
            st.session_state.chat.pop()
            return
    render_turn("assistant", res.answer, res)
    st.session_state.chat.append({"role": "assistant", "content": res.answer, "result": res})


def analyst_capabilities():
    st.subheader("What the analyst can do")
    cols = st.columns(3, gap="medium")
    groups = [
        ("Customers", ["get_summary: segment sizes and actions", "find_customers: filtered customer lists",
                       "explain_customer: SHAP reasons for one score", "get_model_performance: validation results"]),
        ("Demand", ["get_demand_overview: forecast by category", "find_product_forecasts: product forecasts and ranges"]),
        ("Pricing and data", ["get_pricing_overview: elasticities and promotion lifts",
                              "simulate_price_scenario: price and promotion what-ifs",
                              "run_sql: read-only SQL on three tables"]),
    ]
    for col, (name, tools) in zip(cols, groups):
        items = "".join(f"<li><code>{t.split(':')[0]}</code>{t[len(t.split(':')[0]):]}</li>" for t in tools)
        col.markdown(f'<div class="card" style="min-height:190px"><div class="eyebrow">{name}</div><ul>{items}</ul></div>',
                     unsafe_allow_html=True)
    st.write("")
    st.subheader("Example questions")
    st.markdown("\n".join(f"- {q}" for q in SUGGESTIONS.values()))
    st.caption("The language model never touches the data directly: it can only call these tools. SQL must be a "
               "single SELECT on registered tables, runs on a sandboxed in-memory DuckDB and returns at most 200 rows. "
               "Every answer lists the tools and queries it used.")


def render_turn(role: str, content: str, result=None):
    if role == "user":
        st.markdown(f'<div class="q"><div class="eyebrow">Question</div>{html.escape(content)}</div>',
                    unsafe_allow_html=True)
        return
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Analyst</div>', unsafe_allow_html=True)
        st.markdown(content.replace("$", r"\$"))   # keep prices as text, not LaTeX math
        if result:
            render_result(result)


def readable(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Display formats for agent tables: changes as signed percentages, large values as whole
    numbers with thousands separators, small values with two decimals."""
    t, cfg = t.copy(), {}
    for c in t.columns:
        if not pd.api.types.is_float_dtype(t[c]):
            continue
        if "change" in str(c).lower() or "lift" in str(c).lower():
            t[c] = t[c] * 100
            cfg[c] = st.column_config.NumberColumn(c, format="%+.0f%%")
        elif t[c].abs().max() >= 100:
            t[c] = t[c].round(0)
            cfg[c] = st.column_config.NumberColumn(c, format="localized")
        else:
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f")
    return t, cfg


def render_result(res):
    tables = res.tables
    if tables:
        t = tables[-1]
        if "segment" in t.columns:
            t = t.copy()
            if "probability" in t:
                t["probability"] = (t["probability"] * 100).round(1)
            if "monetary" in t:
                t["monetary"] = t["monetary"].round(0)
            st.dataframe(style_segments(t), hide_index=True, width="stretch",
                         column_config=TABLE_CFG, height=min(36 * (len(t) + 1) + 3, 360))
        else:
            t, cfg = readable(t)
            st.dataframe(t, hide_index=True, width="stretch", column_config=cfg,
                         height=min(36 * (len(t) + 1) + 3, 360))
    if res.steps:
        with st.expander(f"Analysis steps ({len(res.steps)})"):
            for i, s in enumerate(res.steps, 1):
                status = "" if s.ok else f" (failed: {s.note})"
                st.markdown(f"**{i}. {s.tool}**{status}")
                if s.tool == "run_sql":
                    st.code(s.args.get("query", ""), language="sql", wrap_lines=True)
                elif s.args:
                    st.code(", ".join(f"{k}={v!r}" for k, v in s.args.items()), language="text",
                            wrap_lines=True)


# -------------------------------------------------------------- navigation
PAGE_HOME = st.Page(home, title="Overview", default=True)
PAGE_OVERVIEW = st.Page(overview, title="Retention overview", url_path="retention")
PAGE_CUSTOMER = st.Page(customer_page, title="Customer profile", url_path="customer")
PAGE_DEMAND = st.Page(demand_page, title="Demand forecast", url_path="demand")
PAGE_PRICING = st.Page(pricing_page, title="Pricing and promotions", url_path="pricing")
PAGE_ANALYST = st.Page(analyst, title="AI analyst", url_path="analyst")

pg = st.navigation({
    "": [PAGE_HOME],
    "Customers": [PAGE_OVERVIEW, PAGE_CUSTOMER,
                  st.Page(campaigns, title="Campaign builder", url_path="campaigns"),
                  st.Page(batch_scoring, title="Batch scoring", url_path="scoring")],
    "Planning": [PAGE_DEMAND, PAGE_PRICING],
    "Intelligence": [PAGE_ANALYST, st.Page(model_page, title="Retention model", url_path="model")],
})
st.logo(str(ASSETS / "logo.svg"), size="large", link=REPO_URL, icon_image=str(ASSETS / "icon.svg"))
with st.sidebar:
    st.markdown(f'<div class="brand-sub">Customers, demand and pricing on real retail data</div>'
                f'<div class="side-foot">Built by {AUTHOR}<br>'
                f'<a href="{REPO_URL}" target="_blank">Source code on GitHub</a><br>'
                f'Data: UCI Online Retail II (CC BY 4.0), dunnhumby Breakfast at the Frat</div>',
                unsafe_allow_html=True)
pg.run()
