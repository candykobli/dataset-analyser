"""Dataset Analyser — upload any dataset, profile it, filter it, chart it and ask questions about it.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from analyser import ai, charts, engine, loaders
from analyser.engine import AGG_LABEL, fmt_date, measure_label
from analyser.profiling import build_dataset, fmt, pct

st.set_page_config(page_title="Dataset Analyser", page_icon=":material/analytics:", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
  [data-testid="stMetricValue"] {font-size: 1.55rem; font-variant-numeric: tabular-nums;}
  [data-testid="stMetricLabel"] p {font-size: .78rem; text-transform: uppercase; letter-spacing: .05em;}
  .da-sub {color: var(--text-color, inherit); opacity: .7; font-size: .9rem; margin-top: -.6rem;}
  .da-card-title {font-weight: 600; font-size: 1.02rem; line-height: 1.35; margin: .2rem 0 .3rem;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
SAMPLE = ROOT / "sample_data" / "global_sales.csv"
VIEWS = ["Overview", "Explore", "Data", "Insights", "Assistant"]
ss = st.session_state


def secret(name: str, default=None):
    """Read a setting from Streamlit secrets, then from environment variables."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file
        value = None
    return value or os.environ.get(name) or default


# ================================================================= state
for k, v in {"datasets": {}, "active": None, "counter": 0, "chats": {}, "seen_uploads": set(), "view": "Overview",
             "last_view": "Overview", "flash": None}.items():
    ss.setdefault(k, v)


def key(ds, *parts) -> str:
    return "|".join([ds.id, *parts])


def init_builder(ds) -> None:
    first_dim = next((c for c in ds.filter_dims if ds.cols[c]["type"] != "number"), None) or (ds.dims or list(ds.cols))[0]
    ss[key(ds, "b", "type")] = "auto"
    ss[key(ds, "b", "x")] = first_dim
    ss[key(ds, "b", "y")] = ds.primary or ""
    ss[key(ds, "b", "agg")] = "sum" if ds.primary else "count"
    ss[key(ds, "b", "series")] = ""
    ss[key(ds, "b", "grain")] = "auto"
    ss[key(ds, "b", "top")] = 10
    ss[key(ds, "km")] = ds.primary or ""
    ss[key(ds, "kagg")] = "sum"


def add_dataset(name, source, frame, detail="", size=None, example=False):
    ss.counter += 1
    ds = build_dataset(f"d{ss.counter}", name, source, frame, detail, size, example)
    ss.datasets[ds.id] = ds
    ss.active = ds.id
    init_builder(ds)
    return ds


@st.cache_data(show_spinner=False)
def example_frame() -> pd.DataFrame:
    if SAMPLE.exists():
        return pd.read_csv(SAMPLE, dtype=str, keep_default_na=False)
    return engine.make_example()


def load_example():
    add_dataset("Global sales 2024–2026", "Example", example_frame(),
                "Example dataset: about 4,200 orders across 17 countries and 13 products", example=True)


if not ss.datasets:
    load_example()


# ================================================================= callbacks
def set_active():
    ss.active = ss["ds_select"]


def remove_active():
    if ss.active in ss.datasets:
        del ss.datasets[ss.active]
    ss.active = next(iter(ss.datasets), None)
    if not ss.active:
        load_example()


def clear_filters(ds):
    for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith(key(ds, "f", ""))]:
        del ss[k]


def apply_spec(ds, spec: dict, go_to_explore: bool = True):
    """Push a chart spec into the Explore builder."""
    xs = explore_x_options(ds)
    ys = [""] + list(ds.cols)
    if spec.get("type") in charts.CHART_TYPES:
        ss[key(ds, "b", "type")] = spec["type"]
    if spec.get("x") in xs:
        ss[key(ds, "b", "x")] = spec["x"]
    y = spec.get("y") or ""
    ss[key(ds, "b", "y")] = y if y in ys else ""
    agg = spec.get("agg") or ("sum" if y else "count")
    ss[key(ds, "b", "agg")] = agg if agg in AGG_LABEL else "sum"
    ss[key(ds, "b", "series")] = spec.get("series") if spec.get("series") in series_options(ds) else ""
    ss[key(ds, "b", "grain")] = spec.get("grain") if spec.get("grain") in ("auto", "day", "week", "month", "quarter", "year") else "auto"
    top = spec.get("top", 10)
    ss[key(ds, "b", "top")] = top if top in (5, 10, 15, 25, 0) else 10
    ss[key(ds, "b", "stacked")] = bool(spec.get("stacked"))
    if go_to_explore:
        ss.view = "Explore"


def run_nl(ds):
    text = ss.get(key(ds, "nl"), "").strip()
    if not text:
        return
    spec = engine.parse_query(text, ds)
    if not spec:
        ss.flash = ("error", "I could not match that to your columns. Try “<measure> by <column>”, e.g. "
                    + " or ".join(f"“{s}”" for s in engine.suggestions(ds, 2)) + ", or ask the Assistant.")
        return
    apply_spec(ds, spec, go_to_explore=False)
    ss.flash = ("info", f"No measure was named, so {ds.primary} is used.") if spec.get("guessed_measure") else None


def use_suggestion(ds, text):
    ss[key(ds, "nl")] = text
    run_nl(ds)


def queue_question(text):
    ss["pending_q"] = text
    ss.view = "Assistant"


def explore_x_options(ds):
    return list(dict.fromkeys(ds.date_cols + ds.dims + ds.measures))


def series_options(ds):
    return [""] + [c for c in ds.filter_dims if ds.cols[c]["unique"] <= 50]


# ================================================================= sidebar: data sources
with st.sidebar:
    st.markdown("## Dataset Analyser")
    with st.expander("Add data", expanded=all(d.example for d in ss.datasets.values()), icon=":material/upload:"):
        t_up, t_url, t_paste = st.tabs(["Upload", "URL / API", "Paste"])
        with t_up:
            files = st.file_uploader("Upload files", type=loaders.SUPPORTED_EXTENSIONS, accept_multiple_files=True,
                                     help="CSV, TSV, TXT, Excel (every sheet), ODS, JSON, JSON Lines, XML, Parquet. "
                                          "Several files at once is fine.")
            for f in files or []:
                fid = getattr(f, "file_id", None) or f"{f.name}-{f.size}"
                if fid in ss.seen_uploads:
                    continue
                ss.seen_uploads.add(fid)
                try:
                    with st.spinner(f"Reading {f.name}…"):
                        for name, detail, frame in loaders.read_upload(f.name, f.getvalue()):
                            add_dataset(name, detail.split(" ")[0], frame, detail, f.size)
                    ss.flash = ("success", f"Loaded {f.name}")
                except Exception as exc:
                    st.error(f"{f.name}: {exc}")
        with t_url:
            with st.form("url_form", border=False):
                url = st.text_input("URL", placeholder="https://api.example.com/v1/sales",
                                    help="A REST/JSON API, a CSV/Excel/JSON file link, a public Google Sheet, "
                                         "a GitHub file or a Dropbox link.")
                header = st.text_input("Auth header (optional)", type="password", placeholder="Authorization: Bearer …")
                params = st.text_input("Query parameters (optional)", placeholder="limit=1000&country=DK")
                path = st.text_input("Records path (optional)", placeholder="data.items",
                                     help="Where the records sit in a JSON response. Leave empty to detect it.")
                if st.form_submit_button("Fetch", type="primary", width="stretch") and url.strip():
                    try:
                        with st.spinner("Fetching…"):
                            name, detail, frame = loaders.fetch_url(url, header, path, params)
                            add_dataset(name, "URL / API", frame, detail)
                        ss.flash = ("success", f"Loaded {name}")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
        with t_paste:
            with st.form("paste_form", border=False):
                pname = st.text_input("Name", placeholder="Pasted data")
                ptext = st.text_area("Data", height=150, placeholder="country,product,sales\nDenmark,Chair,1200",
                                     help="CSV, tab-separated text copied from Excel, JSON or XML. The format is detected automatically.")
                if st.form_submit_button("Load", type="primary", width="stretch"):
                    try:
                        add_dataset(pname.strip() or "Pasted data", "Pasted", loaders.parse_text(ptext), "Pasted text", len(ptext))
                        ss.flash = ("success", "Loaded pasted data")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
        st.button("Load example dataset", on_click=load_example, width="stretch")

    ds = ss.datasets[ss.active]
    if len(ss.datasets) > 1:
        ids = list(ss.datasets)
        ss["ds_select"] = ss.active
        st.selectbox("Active dataset", ids, key="ds_select", on_change=set_active,
                     format_func=lambda i: f"{ss.datasets[i].name} ({len(ss.datasets[i].df):,} rows)")
        st.button("Remove this dataset", on_click=remove_active, width="stretch", type="tertiary")

    # ------------------------------------------------------------- filters
    st.divider()
    fc1, fc2 = st.columns([1, 1])
    fc1.markdown("#### Filters")
    fc2.button("Clear all", on_click=clear_filters, args=(ds,), type="tertiary", width="stretch")
    filters: dict = {}
    for c in ds.filter_dims[:14]:
        info = ds.cols[c]
        if info["type"] == "bool":
            options = [True, False]
        elif info["dim_like"]:
            options = sorted(int(v) for v in ds.df[c].dropna().unique())
        else:
            options = [v for v, _ in info["top"]]
        if info["count"] > info["filled"]:
            options = options + ["(blank)"]
        sel = st.multiselect(c, options, key=key(ds, "f", c), placeholder=f"All {info['unique']:,}",
                             format_func=lambda v: {True: "Yes", False: "No"}.get(v, str(v)) if isinstance(v, bool) else str(v))
        if sel:
            filters[c] = {"kind": "cat", "values": sel}
    for c in ds.date_cols[:2]:
        lo, hi = ds.cols[c]["min"].date(), ds.cols[c]["max"].date()
        rng = st.date_input(c, value=(lo, hi), min_value=lo, max_value=hi, key=key(ds, "f", c), format="DD/MM/YYYY")
        if isinstance(rng, (list, tuple)) and len(rng) == 2 and (rng[0] != lo or rng[1] != hi):
            filters[c] = {"kind": "range", "min": pd.Timestamp(rng[0]), "max": pd.Timestamp(rng[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)}
    if ds.measures:
        with st.expander("Number ranges"):
            for c in ds.measures[:8]:
                info = ds.cols[c]
                lo, hi = float(info["min"]), float(info["max"])
                if lo == hi:
                    continue
                r = st.slider(c, lo, hi, (lo, hi), key=key(ds, "f", c))
                if r[0] > lo or r[1] < hi:
                    filters[c] = {"kind": "range", "min": r[0] if r[0] > lo else None, "max": r[1] if r[1] < hi else None}
    if not (ds.filter_dims or ds.date_cols or ds.measures):
        st.caption("No columns suitable for filtering were found.")

    # ------------------------------------------------------------- AI settings
    st.divider()
    owner_key = secret("ANTHROPIC_API_KEY")
    model = secret("ANTHROPIC_MODEL", ai.DEFAULT_MODEL)
    with st.expander("AI assistant", icon=":material/smart_toy:"):
        if not ai.available():
            st.caption("Install the `anthropic` package to enable AI answers.")
        elif owner_key:
            st.success("AI answers are switched on for everyone using this app.", icon=":material/check_circle:")
        else:
            st.text_input("Your Anthropic API key (optional)", type="password", key="user_api_key",
                          help="Kept only for this browser session. Get one at console.anthropic.com.")
            st.caption("Without a key, the built-in assistant answers “measure by column” questions, totals, counts, "
                       "trends, data quality and insights. With a key it can also answer open questions.")
    api_key = owner_key or ss.get("user_api_key") or None
    ai_on = bool(api_key and ai.available())

# ================================================================= data in view
df = engine.apply_filters(ds.df, filters)
fsig = repr(sorted((k, repr(v)) for k, v in filters.items()))

# ================================================================= header
head_l, head_r = st.columns([5, 2], vertical_alignment="bottom")
with head_l:
    st.markdown(f"## {ds.name}")
    meta = [ds.detail, f"{len(ds.df):,} rows × {len(ds.cols)} columns", f"loaded {ds.loaded_at:%d %b %Y %H:%M}"]
    st.markdown(f"<div class='da-sub'>{' · '.join(meta)}</div>", unsafe_allow_html=True)
with head_r:
    if ds.example:
        st.caption(":orange-badge[Example data] Add your own from the sidebar.")
desc = engine.describe_filters(filters, ds)
if desc:
    st.caption(f"**{len(df):,} of {len(ds.df):,} rows** · " + " · ".join(desc))

if ss.flash:
    kind, msg = ss.flash
    getattr(st, kind if kind in ("success", "info", "warning", "error") else "info")(msg)
    ss.flash = None

view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed") or ss.last_view
ss.last_view = view


def show_spec(spec, height=380, k=None, frame=None):
    fig, res = charts.figure_from_spec(spec, df if frame is None else frame, ds, height)
    if fig is None:
        st.caption("Nothing to chart for this selection.")
    else:
        st.plotly_chart(fig, width="stretch", key=k, config={"displaylogo": False})
    return res


# ================================================================= overview
def auto_specs(m, agg):
    a = agg if m else "count"
    lab = measure_label(m, a)
    dims = [c for c in ds.filter_dims if ds.cols[c]["unique"] > 1 and ds.cols[c]["type"] != "number"]
    dc = ds.date_cols[0] if ds.date_cols else None
    specs = []
    if dc:
        specs.append((f"{lab} over time", {"type": "area", "x": dc, "y": m, "agg": a, "grain": "auto"}))
    if dims:
        specs.append((f"{lab} by {dims[0]}", {"type": "auto", "x": dims[0], "y": m, "agg": a, "top": 10}))
    if len(dims) > 1:
        specs.append((f"Share of {m or 'rows'} by {dims[1]}",
                      {"type": "doughnut" if ds.cols[dims[1]]["unique"] <= 8 else "hbar", "x": dims[1], "y": m,
                       "agg": a if a in ("sum", "count") else "sum", "top": 8}))
    if dc and len(dims) > 1 and ds.cols[dims[1]]["unique"] <= 8:
        specs.append((f"{lab} over time by {dims[1]}", {"type": "line", "x": dc, "y": m, "agg": a, "series": dims[1], "grain": "auto"}))
    elif m:
        specs.append((f"Distribution of {m}", {"type": "histogram", "y": m}))
    elif len(dims) > 2:
        specs.append((f"Rows by {dims[2]}", {"type": "auto", "x": dims[2], "y": None, "agg": "count", "top": 10}))
    return specs[:4]


def render_overview():
    c1, c2, _ = st.columns([2, 1.4, 4])
    c1.selectbox("Key measure", [""] + ds.measures, key=key(ds, "km"), format_func=lambda v: v or "(count of rows)")
    c2.selectbox("Aggregation", ["sum", "avg", "median", "max", "min"], key=key(ds, "kagg"), format_func=AGG_LABEL.get,
                 disabled=not ss[key(ds, "km")])
    m = ss[key(ds, "km")] or None
    agg = ss[key(ds, "kagg")] if m else "count"

    tiles = []
    tiles.append(("Rows in view", f"{len(df):,}", None,
                  "No filters applied" if len(df) == len(ds.df) else f"{pct(len(df) / len(ds.df))} of {len(ds.df):,} rows"))
    if m:
        pc = engine.period_change(df, ds, m, agg)
        delta = f"{pc['change'] * 100:+.1f}%" if pc and pc["change"] is not None else None
        tiles.append((f"{AGG_LABEL[agg]} {m}", fmt(engine.overall(df, m, agg), True), delta,
                      "Latest 3 full months compared with the 3 before" if delta else f"Across {len(df):,} rows"))
        if agg == "sum":
            tiles.append((f"Average {m} per row", fmt(df[m].mean(), True), None, f"Median {fmt(df[m].median(), True)}"))
        second = next((c for c in ds.measures if c != m and any(w in c.lower() for w in ("profit", "margin", "cost", "units", "quantity", "qty", "volume"))), None)
        if second:
            sv = df[second].sum()
            extra = f" · {pct(sv / df[m].sum())} of {m}" if "profit" in second.lower() and df[m].sum() else ""
            tiles.append((f"Total {second}", fmt(sv, True), None, f"Sum over rows{extra}"))
    if ds.date_cols:
        dc = ds.date_cols[0]
        lo, hi = df[dc].min(), df[dc].max()
        tiles.append(("Date coverage", f"{(hi - lo).days + 1:,} days" if pd.notna(lo) else "–", None,
                      f"{fmt_date(lo)} – {fmt_date(hi)}"))
    q = ds.quality["score"]
    tiles.append(("Data quality", f"{q}/100", None, f"{len(ds.cols)} columns · {len(ds.measures)} measures"))
    cols = st.columns(len(tiles))
    for col, (label, value, delta, helptext) in zip(cols, tiles):
        col.metric(label, value, delta, help=helptext, border=True)

    specs = auto_specs(m, agg)
    for i in range(0, len(specs), 2):
        row = st.columns(2)
        for col, (title, spec) in zip(row, specs[i:i + 2]):
            with col.container(border=True):
                t1, t2 = st.columns([5, 1], vertical_alignment="center")
                t1.markdown(f"**{title}**")
                t2.button("Explore", key=f"ov-ex-{ds.id}-{title}", type="tertiary", on_click=apply_spec, args=(ds, spec))
                show_spec(spec, 300, k=f"ov-{ds.id}-{title}")

    a, b = st.columns([1.1, 1])
    with a.container(border=True):
        st.markdown("**About this data**")
        facts = {"Name": ds.name, "Source": ds.detail}
        if ds.size:
            facts["Size"] = f"{ds.size / 1048576:.1f} MB" if ds.size > 1048576 else f"{ds.size / 1024:.0f} KB"
        facts["Shape"] = f"{len(ds.df):,} rows × {len(ds.cols)} columns"
        facts["Measures"] = ", ".join(ds.measures) or "–"
        facts["Dimensions"] = ", ".join(f"{c} ({ds.cols[c]['unique']})" for c in ds.filter_dims) or "–"
        if ds.date_cols:
            facts["Dates"] = "; ".join(f"{c}: {fmt_date(ds.cols[c]['min'])} – {fmt_date(ds.cols[c]['max'])}" for c in ds.date_cols)
        ids = [c for c, i in ds.cols.items() if i["type"] == "id" or i["is_id"]]
        if ids:
            facts["Identifiers"] = ", ".join(ids)
        st.markdown("\n".join(f"- **{k}:** {v}" for k, v in facts.items()))
    with b.container(border=True):
        color = "green" if q >= 85 else "orange" if q >= 65 else "red"
        st.markdown(f"**Data quality** &nbsp; :{color}-badge[{q}/100]")
        badge = {"ok": ":green-badge[OK]", "high": ":red-badge[High]", "check": ":orange-badge[Check]", "note": ":blue-badge[Note]"}
        st.markdown("\n".join(f"- {badge[sev]} {txt}" for sev, txt in ds.quality["issues"][:8]))

    with st.container(border=True):
        st.markdown("**Column profile** · detected type, completeness and distribution for every column")
        rows = []
        for c, i in ds.cols.items():
            if i["type"] == "number":
                summary = f"{fmt(i.get('min'), True)} – {fmt(i.get('max'), True)} · mean {fmt(i.get('mean'), True)} · σ {fmt(i.get('std'), True)}"
            elif i["type"] == "date":
                summary = f"{fmt_date(i['min'])} – {fmt_date(i['max'])}"
            elif i["type"] == "text":
                summary = f"avg {round(i.get('avg_len', 0))} characters"
            elif i["top"]:
                summary = f"most common: {i['top'][0][0]} ({pct(i['top'][0][1] / max(i['filled'], 1), 0)})"
            else:
                summary = ""
            rows.append({"Column": c, "Type": "period" if i["dim_like"] else "id" if i["is_id"] else i["type"],
                         "Role": i["role"] or "–", "Missing": (i["count"] - i["filled"]) / max(i["count"], 1) * 100,
                         "Unique": i["unique"], "Summary": summary, "Distribution": charts.sparkline_values(i)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Missing": st.column_config.ProgressColumn("Missing", format="%.1f%%", min_value=0, max_value=100),
            "Unique": st.column_config.NumberColumn(format="%d"),
            "Distribution": st.column_config.BarChartColumn("Distribution"),
        })


# ================================================================= explore
def render_explore():
    with st.container(border=True):
        st.markdown("**Ask for a view** · plain language works: “volume per category”, “sales by country top 5”, “monthly profit by region”")
        with st.form(f"nl-{ds.id}", border=False):
            c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
            c1.text_input("Request", key=key(ds, "nl"), placeholder="e.g. sales by country, top 5", label_visibility="collapsed")
            c2.form_submit_button("Show", type="primary", on_click=run_nl, args=(ds,), width="stretch")
        sug = engine.suggestions(ds, 6)
        if sug:
            cols = st.columns(len(sug))
            for col, s in zip(cols, sug):
                col.button(s, key=f"sg-{ds.id}-{s}", on_click=use_suggestion, args=(ds, s), width="stretch")

    with st.container(border=True):
        c = st.columns([1.2, 1.6, 1.6, 1.2, 1.4, 1.1, 0.9])
        c[0].selectbox("Chart", list(charts.CHART_TYPES), key=key(ds, "b", "type"), format_func=charts.CHART_TYPES.get)
        xs = explore_x_options(ds)
        if ss.get(key(ds, "b", "x")) not in xs:
            ss[key(ds, "b", "x")] = xs[0]
        c[1].selectbox("Group by (X)", xs, key=key(ds, "b", "x"),
                       format_func=lambda v: f"{v} · {'period' if ds.cols[v]['dim_like'] else ds.cols[v]['type']}")
        c[2].selectbox("Measure (Y)", [""] + ds.measures + [x for x in ds.cols if x not in ds.measures], key=key(ds, "b", "y"),
                       format_func=lambda v: "(count of rows)" if not v else v if v in ds.measures else f"{v} (distinct)")
        c[3].selectbox("Aggregation", list(AGG_LABEL), key=key(ds, "b", "agg"), format_func=AGG_LABEL.get)
        c[4].selectbox("Split by", series_options(ds), key=key(ds, "b", "series"), format_func=lambda v: v or "None")
        c[5].selectbox("Time grain", ["auto", "day", "week", "month", "quarter", "year"], key=key(ds, "b", "grain"), format_func=str.title)
        c[6].selectbox("Top", [10, 5, 15, 25, 0], key=key(ds, "b", "top"), format_func=lambda v: "All" if v == 0 else str(v))

    spec = {"type": ss[key(ds, "b", "type")], "x": ss[key(ds, "b", "x")], "y": ss[key(ds, "b", "y")] or None,
            "agg": ss[key(ds, "b", "agg")], "series": ss[key(ds, "b", "series")] or None,
            "grain": ss[key(ds, "b", "grain")], "top": ss[key(ds, "b", "top")], "stacked": ss.get(key(ds, "b", "stacked"), False)}
    if not spec["y"]:
        spec["agg"] = "count"
    elif ds.cols[spec["y"]]["type"] != "number" and spec["agg"] != "count":
        spec["agg"] = "distinct"
    if spec["type"] == "histogram" and not spec["y"]:
        spec["y"] = ds.measures[0] if ds.measures else None
    if spec["type"] == "scatter" and ds.cols[spec["x"]]["type"] != "number":
        spec["x"] = next((mm for mm in ds.measures if mm != spec["y"]), None)

    with st.container(border=True):
        if spec["type"] == "histogram":
            title = f"Distribution of {spec['y']}"
        elif spec["type"] == "scatter":
            title = f"{spec['y']} vs {spec['x']}"
        else:
            title = f"{measure_label(spec['y'], spec['agg'])} by {spec['x']}" + (f" and {spec['series']}" if spec["series"] else "")
        st.markdown(f"**{title}** · {len(df):,} rows")
        res = show_spec(spec, 440, k=f"ex-{ds.id}")
    if res is not None and not res.table.empty:
        with st.container(border=True):
            t1, t2 = st.columns([5, 1], vertical_alignment="center")
            t1.markdown("**Result table**")
            table = res.table.copy()
            if res.additive and not res.series and not res.is_time:
                total = table.iloc[:, 0].sum()
                table["Share"] = table.iloc[:, 0] / total * 100 if total else None
            t2.download_button("Download", table.to_csv().encode("utf-8-sig"), f"{title}.csv", "text/csv", width="stretch")
            st.dataframe(table, width="stretch", column_config={
                **{c: st.column_config.NumberColumn(format="localized") for c in res.table.columns},
                "Share": st.column_config.ProgressColumn("Share", format="%.1f%%", min_value=0, max_value=100)})


# ================================================================= data
@st.cache_data(show_spinner=False, max_entries=4)
def to_excel(ds_id: str, sig: str, q: str, _frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        _frame.to_excel(xw, index=False, sheet_name="Data")
    return buf.getvalue()


def render_data():
    c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="bottom")
    q = c1.text_input("Search all columns", key=key(ds, "search"), placeholder="Search…")
    view_df = df
    if q.strip():
        needle = q.strip().lower()
        mask = pd.Series(False, index=df.index)
        for c in df.columns:
            mask |= df[c].astype(str).str.lower().str.contains(needle, regex=False, na=False)
        view_df = df[mask]
    base = ds.name.rsplit(".", 1)[0].replace("/", "-")
    c2.download_button("CSV", view_df.to_csv(index=False).encode("utf-8-sig"), f"{base}.csv", "text/csv", width="stretch",
                       icon=":material/download:")
    if len(view_df) <= 200_000:
        c3.download_button("Excel", to_excel(ds.id, fsig, q, view_df), f"{base}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
                           icon=":material/download:")
    st.caption(f"{len(view_df):,} matching rows. Click a column header to sort.")
    st.dataframe(view_df, width="stretch", height=560, hide_index=True)


# ================================================================= insights
@st.cache_data(show_spinner=False, max_entries=16)
def cached_insights(ds_id: str, sig: str, measure: str | None, _ds, _df):
    return engine.compute_insights(_df, _ds, measure)


def render_insights():
    measure = ss.get(key(ds, "km")) or ds.primary
    ins = cached_insights(ds.id, fsig, measure, ds, df)
    with st.container(border=True):
        h1, h2 = st.columns([5, 1.3], vertical_alignment="center")
        h1.markdown("**Written review**" + ("" if ai_on else " · add an API key under *AI assistant* in the sidebar to enable it"))
        if h2.button("Generate review", type="primary", disabled=not ai_on, width="stretch"):
            with st.spinner("Writing the review… this usually takes 15–40 seconds."):
                try:
                    ss[key(ds, "review", fsig)] = ai.review(api_key, model, ds, df, filters, ins)
                except Exception as exc:
                    st.error(ai.friendly_error(exc))
        if ss.get(key(ds, "review", fsig)):
            st.markdown(ss[key(ds, "review", fsig)])
        elif ai_on:
            st.caption("Writes an executive summary, prioritised recommendations, risks and next questions from the figures below.")

    st.markdown(f"**Automatic findings** · {len(ins)} on {len(df):,} rows in view")
    tone = {"up": ":green-badge[Opportunity]", "down": ":red-badge[Risk]", "warn": ":orange-badge[Attention]", "info": ":blue-badge[Insight]"}
    for i in range(0, len(ins), 2):
        row = st.columns(2)
        for col, item in zip(row, ins[i:i + 2]):
            with col.container(border=True):
                st.markdown(f"{tone[item['tone']]} &nbsp; <span style='opacity:.7;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em'>{item['area']}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='da-card-title'>{item['title']}</div>", unsafe_allow_html=True)
                st.markdown(item["body"])
                st.markdown(f"**Recommendation:** {item['action']}")
                if item.get("spec"):
                    with st.expander("Show chart"):
                        show_spec(item["spec"], 280, k=f"in-{ds.id}-{i}-{item['title']}")


# ================================================================= assistant
def render_message(msg, idx):
    with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/analytics:"):
        for j, spec in enumerate(msg.get("charts", [])):
            frame = ai._row_filter(df, ds, spec.get("row_filters")) if spec.get("row_filters") else None
            if spec.get("filters"):
                frame = engine.apply_filters(df if frame is None else frame, spec["filters"])
            if spec.get("title"):
                st.markdown(f"**{spec['title']}**")
            show_spec({k: v for k, v in spec.items() if k not in ("filters", "row_filters")}, 300, k=f"ch-{ds.id}-{idx}-{j}", frame=frame)
            st.button("Open in Explore", key=f"chx-{ds.id}-{idx}-{j}", type="tertiary", on_click=apply_spec,
                      args=(ds, {k: v for k, v in spec.items() if k not in ("filters", "row_filters")}))
        st.markdown(msg["content"])


def answer(question: str) -> dict:
    spec = engine.parse_query(question, ds)
    simple = spec and len(question.split()) <= 12 and not any(w in question.lower() for w in
                                                               ("why", "explain", "recommend", "should", "forecast", "predict", "what if", "reason"))
    if ai_on and not simple:
        history = [{"role": m["role"], "content": m["content"]} for m in ss.chats.get(ds.id, [])]
        try:
            out = ai.ask(api_key, model, ds, df, filters, history, question)
            return {"role": "assistant", "content": out["text"], "charts": out["charts"]}
        except Exception as exc:
            local = engine.answer_locally(question, ds, df)
            return {"role": "assistant", "content": f"*{ai.friendly_error(exc)}*\n\n{local.text}",
                    "charts": [dict(local.spec, filters=local.filters)] if local.spec else []}
    local = engine.answer_locally(question, ds, df)
    return {"role": "assistant", "content": local.text, "charts": [dict(local.spec, filters=local.filters)] if local.spec else []}


def render_assistant():
    chat = ss.chats.setdefault(ds.id, [])
    top_l, top_r = st.columns([5, 1], vertical_alignment="center")
    mode = "AI answers on" if ai_on else "Built-in assistant (no AI key)"
    top_l.caption(f"Answers use the {len(df):,} rows currently in view. Mode: **{mode}**.")
    if chat and top_r.button("New chat", type="tertiary", width="stretch"):
        ss.chats[ds.id] = []
        st.rerun()
    if not chat:
        intro = (f"I have **{ds.name}** loaded: {len(ds.df):,} rows, measures "
                 f"{', '.join(ds.measures[:5]) or 'none'} and dimensions {', '.join(ds.filter_dims[:5]) or 'none'}. "
                 f"Ask for any measure by any column and I will chart it.")
        if ai_on:
            intro += " I can also explain trends, compare segments and recommend actions."
        with st.chat_message("assistant", avatar=":material/analytics:"):
            st.markdown(intro)
    for idx, msg in enumerate(chat):
        render_message(msg, idx)
    sug = engine.suggestions(ds, 4) + (["What are the main trends and what should we do?"] if ai_on else ["Key insights"])
    cols = st.columns(len(sug))
    for col, s in zip(cols, sug):
        col.button(s, key=f"cs-{ds.id}-{s}", on_click=queue_question, args=(s,), width="stretch")
    prompt = st.chat_input("Ask about your data, e.g. volume per category")
    q = prompt or ss.pop("pending_q", None)
    if q:
        chat.append({"role": "user", "content": q})
        with st.spinner("Working on it…"):
            chat.append(answer(q))
        st.rerun()


{"Overview": render_overview, "Explore": render_explore, "Data": render_data,
 "Insights": render_insights, "Assistant": render_assistant}[view]()
