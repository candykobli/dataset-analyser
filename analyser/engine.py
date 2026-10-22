"""Filtering, aggregation, plain-language queries, the built-in assistant and automatic insights."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profiling import Dataset, fmt, pct

AGG_LABEL = {"sum": "Total", "avg": "Average", "count": "Count of rows", "distinct": "Distinct",
             "median": "Median", "min": "Min", "max": "Max"}
PANDAS_AGG = {"sum": "sum", "avg": "mean", "median": "median", "min": "min", "max": "max"}


def measure_label(y: str | None, agg: str) -> str:
    return "Rows" if agg == "count" or not y else f"{AGG_LABEL[agg]} {y}"


def fmt_date(ts) -> str:
    return "–" if ts is None or pd.isna(ts) else f"{ts.day} {ts.strftime('%b %Y')}"


# ================================================================= filters
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """filters: {col: {"kind": "cat", "values": [...]} | {"kind": "range", "min": x, "max": y}}"""
    mask = pd.Series(True, index=df.index)
    for c, f in filters.items():
        if c not in df.columns:
            continue
        s = df[c]
        if f["kind"] == "cat" and f.get("values"):
            vals = list(f["values"])
            m = s.isin([v for v in vals if v != "(blank)"])
            if "(blank)" in vals:
                m |= s.isna()
            mask &= m.fillna(False).astype(bool)
        elif f["kind"] == "range":
            if f.get("min") is not None:
                mask &= (s >= f["min"]).fillna(False).astype(bool)
            if f.get("max") is not None:
                mask &= (s <= f["max"]).fillna(False).astype(bool)
    return df[mask] if not mask.all() else df


def describe_filters(filters: dict, ds: Dataset) -> list[str]:
    out = []
    for c, f in filters.items():
        if f["kind"] == "cat" and f.get("values"):
            v = [str(x) for x in f["values"]]
            out.append(f"{c}: {', '.join(v[:3])}{f' +{len(v) - 3}' if len(v) > 3 else ''}")
        elif f["kind"] == "range" and (f.get("min") is not None or f.get("max") is not None):
            show = fmt_date if ds.cols[c]["type"] == "date" else fmt
            if f.get("min") is not None and f.get("max") is not None:
                out.append(f"{c}: {show(f['min'])} – {show(f['max'])}")
            elif f.get("min") is not None:
                out.append(f"{c} ≥ {show(f['min'])}")
            else:
                out.append(f"{c} ≤ {show(f['max'])}")
    return out


# ================================================================= aggregation
def auto_grain(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "month"
    days = (s.max() - s.min()).days
    return "year" if days > 1460 else "quarter" if days > 540 else "month" if days > 75 else "week" if days > 20 else "day"


def bucket(s: pd.Series, grain: str) -> pd.Series:
    freq = {"year": "Y", "quarter": "Q", "month": "M", "week": "W-SUN", "day": "D"}[grain]
    return s.dt.to_period(freq).dt.start_time


def bucket_label(ts: pd.Timestamp, grain: str) -> str:
    if grain == "year":
        return str(ts.year)
    if grain == "quarter":
        return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"
    if grain == "month":
        return ts.strftime("%b %Y")
    return fmt_date(ts)


@dataclass
class AggResult:
    table: pd.DataFrame          # index = labels, one column per series
    x: str
    y: str | None
    agg: str
    series: str | None
    is_time: bool
    grain: str | None
    totals: pd.Series
    n_rows: int
    title: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def additive(self) -> bool:
        return self.agg in ("sum", "count")


def _keys(s: pd.Series, info: dict) -> pd.Series:
    if info["type"] == "bool":
        return s.map({True: "Yes", False: "No"}).astype(object).where(s.notna(), "(blank)")
    out = s.astype(object).where(s.notna(), "(blank)")
    return out


def _agg(groups, y, agg):
    if agg == "count":
        return groups.size()
    if agg == "distinct":
        return groups[y].nunique()
    return groups[y].agg(PANDAS_AGG[agg])


def aggregate(df: pd.DataFrame, ds: Dataset, x: str, y: str | None = None, agg: str = "sum",
              series: str | None = None, grain: str = "auto", top: int = 10, sort: str = "desc") -> AggResult:
    if x not in ds.cols:
        raise ValueError(f'Unknown column "{x}"')
    if not y:
        agg = "count"
    elif ds.cols[y]["type"] != "number" and agg not in ("count", "distinct"):
        agg = "distinct"
    xinfo = ds.cols[x]
    is_time = xinfo["type"] == "date"
    d = df[df[x].notna()] if is_time else df
    g = None
    if is_time:
        g = auto_grain(d[x]) if grain in (None, "", "auto") else grain
        xkey = bucket(d[x], g).rename("__x")
    else:
        xkey = _keys(d[x], xinfo).rename("__x")
    if series and series != x:
        skey = _keys(d[series], ds.cols[series]).rename("__s")
        r = _agg(d.groupby([xkey, skey], sort=False), y, agg)
        table = r.unstack("__s")
    else:
        series = None
        r = _agg(d.groupby(xkey, sort=False), y, agg)
        table = r.to_frame(measure_label(y, agg))
    totals = _agg(d.groupby(xkey, sort=False), y, agg)
    additive = agg in ("sum", "count")
    if additive:
        table = table.fillna(0)

    if is_time or xinfo["type"] == "number":
        order = sorted(table.index)
        table, totals = table.loc[order], totals.loc[order]
        if xinfo["type"] == "number" and len(order) > 200:
            table, totals = table.iloc[:200], totals.iloc[:200]
    else:
        totals = totals.sort_values(ascending=(sort == "asc"))
        table = table.loc[totals.index]
        if top and len(table) > top:
            rest = table.index[top:]
            if additive:
                other = table.loc[rest].sum()
                ot = totals.loc[rest].sum()
                table = pd.concat([table.iloc[:top], other.to_frame(f"Other ({len(rest)})").T])
                totals = pd.concat([totals.iloc[:top], pd.Series({f"Other ({len(rest)})": ot})])
            else:
                table, totals = table.iloc[:top], totals.iloc[:top]
    # cap series to 6 + Other
    if series and table.shape[1] > 7:
        sums = table.sum().sort_values(ascending=False)
        keep = list(sums.index[:6])
        if additive:
            other = table[sums.index[6:]].sum(axis=1)
            table = table[keep].assign(Other=other)
        else:
            table = table[keep]
    elif series:
        table = table[table.sum().sort_values(ascending=False).index]
    if is_time:
        labels = [bucket_label(t, g) for t in table.index]
        table.index, totals.index = labels, labels
    else:
        table.index = [str(i) for i in table.index]
        totals.index = [str(i) for i in totals.index]
    table.columns = [str(c) for c in table.columns]
    table.index.name = x
    title = f"{measure_label(y, agg)} by {x}" + (f" and {series}" if series else "")
    return AggResult(table=table, x=x, y=y, agg=agg, series=series, is_time=is_time, grain=g,
                     totals=totals, n_rows=len(d), title=title)


def overall(df: pd.DataFrame, y: str | None, agg: str):
    if not y or agg == "count":
        return len(df)
    s = df[y]
    if agg == "distinct":
        return int(s.nunique())
    s = s.dropna()
    if s.empty:
        return None
    return float(getattr(s, PANDAS_AGG[agg])())


def period_change(df: pd.DataFrame, ds: Dataset, y: str | None, agg: str = "sum"):
    """Last 3 complete months vs the 3 months before (the latest month may be partial)."""
    if not ds.date_cols:
        return None
    res = aggregate(df, ds, ds.date_cols[0], y, agg if agg in ("sum", "avg", "median") else "sum", grain="month")
    v = res.table.iloc[:, 0].tolist()
    if len(v) < 7:
        return None
    last = len(v) - 1
    cur, prev = sum(v[last - 3:last]), sum(v[last - 6:last - 3])
    if agg in ("avg", "median"):
        cur, prev = cur / 3, prev / 3
    return {"change": (cur - prev) / abs(prev) if prev else None, "cur": cur, "prev": prev,
            "labels": (res.table.index[last - 3], res.table.index[last - 1])}


def correlation(df: pd.DataFrame, a: str, b: str) -> float | None:
    d = df[[a, b]].dropna()
    if len(d) < 10 or d[a].std() == 0 or d[b].std() == 0:
        return None
    return float(d[a].corr(d[b]))


# ================================================================= plain-language parsing
MEASURE_SYN = [
    (r"\b(sales|turnover|revenue|income|takings|gmv|net sales|gross sales)\b", r"revenue|sales|turnover|amount|income|gmv|value|total"),
    (r"\b(volume|volumes|units|unit|quantity|quantities|qty|pieces|items sold|amount sold|sold)\b", r"units|quantity|qty|volume|pieces|count|sold"),
    (r"\b(profit|profits|margin|earnings|gain)\b", r"profit|margin|earning|gain"),
    (r"\b(cost|costs|spend|spending|expense|expenses|cogs)\b", r"cost|spend|expense|cogs"),
    (r"\b(price|prices|pricing|unit price|asp)\b", r"price|asp"),
    (r"\b(discount|discounts|rebate)\b", r"discount|rebate"),
]
DIM_SYN = [
    (r"\b(country|countries|nation|nations)\b", r"country|nation|land"),
    (r"\b(region|regions|area|areas|zone|zones|continent|continents)\b", r"region|area|zone|continent|territor"),
    (r"\b(product|products|item|items|sku|skus|article)\b", r"product|item|sku|article|model"),
    (r"\b(category|categories|product line|line|type|types|class)\b", r"categ|line|type|class|family|group"),
    (r"\b(channel|channels|source|sources)\b", r"channel|source|medium"),
    (r"\b(segment|segments|customer type)\b", r"segment"),
    (r"\b(customer|customers|client|clients|account|accounts)\b", r"customer|client|account"),
    (r"\b(city|cities)\b", r"city|town"),
    (r"\b(state|states|province|provinces)\b", r"state|province"),
    (r"\b(brand|brands)\b", r"brand"),
]
TIME_SYN = [
    (r"\b(daily|per day|by day|each day)\b", "day"),
    (r"\b(weekly|per week|by week|each week)\b", "week"),
    (r"\b(monthly|per month|by month|each month|month over month|mom)\b", "month"),
    (r"\b(quarterly|per quarter|by quarter|each quarter)\b", "quarter"),
    (r"\b(yearly|annual|annually|per year|by year|each year|year over year|yoy)\b", "year"),
    (r"\b(over time|trend|trends|evolution|timeline|history|development)\b", "auto"),
]
SPLIT_WORDS = r"\b(?:by|per|across|for each|each|split by|broken down by|grouped by|in each|within)\b"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-.]", " ", s.lower())).strip()


def _stem(w: str) -> str:
    return re.sub(r"(es|s)$", "", re.sub(r"ies$", "y", w))


def find_col_in_text(text: str, candidates: list[str]):
    t = " " + _norm(text) + " "
    best, best_score, best_pos = None, 0, 10 ** 9
    for c in candidates:
        n = _norm(c)
        for v in {n, _stem(n), n.replace(" ", "")}:
            if len(v) < 2:
                continue
            i = t.find(" " + v)
            if i >= 0:
                score = len(v) + 10
                if score > best_score or (score == best_score and i < best_pos):
                    best, best_score, best_pos = c, score, i
        for w in n.split():
            if len(w) > 3:
                i = t.find(" " + _stem(w))
                if i >= 0 and len(w) > best_score:
                    best, best_score, best_pos = c, len(w), i
    return (best, best_pos) if best else None


def syn_match(text: str, table, candidates: list[str]):
    for q, col_re in table:
        m = re.search(q, text, re.I)
        if m:
            c = next((c for c in candidates if re.search(col_re, c, re.I)), None)
            if c:
                return c, m.start()
    return None


def detect_value_filters(text: str, ds: Dataset, exclude: set[str]) -> dict:
    """Spot category values ("Germany", "Electronics") and years ("in 2025") named in a question."""
    t = " " + text.lower() + " "
    found: dict = {}
    for c in ds.filter_dims:
        if c in exclude or ds.cols[c]["type"] == "bool":
            continue
        for v, _ in ds.cols[c]["top"]:
            sv = str(v)
            if len(sv) < 3 or _norm(sv) in {_norm(x) for x in ds.cols}:
                continue
            if re.search(r"(?<![\w])" + re.escape(sv.lower()) + r"(?![\w])", t):
                raw = next((x for x in ds.df[c].dropna().unique()[:2000] if str(x) == sv), sv)
                found.setdefault(c, {"kind": "cat", "values": []})["values"].append(raw)
    if ds.date_cols:
        years = re.findall(r"\b(?:in|for|during|of|year)\s+((?:19|20)\d{2})\b", t) or re.findall(r"\b((?:19|20)\d{2})\b(?!\s*(?:rows|units|orders))", t)
        if years:
            y0, y1 = min(map(int, years)), max(map(int, years))
            found[ds.date_cols[0]] = {"kind": "range", "min": pd.Timestamp(y0, 1, 1), "max": pd.Timestamp(y1, 12, 31, 23, 59)}
    return found


def parse_query(text: str, ds: Dataset) -> dict | None:
    """'volume per category', 'sales by country top 5', 'monthly profit by region' → chart spec."""
    t = " " + text.lower() + " "
    spec = {"type": "auto", "agg": "sum", "top": 10, "grain": "auto", "series": None, "sort": "desc", "stacked": False}
    if re.search(r"\b(average|avg|mean|typical)\b", t):
        spec["agg"] = "avg"
    elif re.search(r"\bmedian\b", t):
        spec["agg"] = "median"
    elif re.search(r"\b(max|maximum|highest single|largest single)\b", t):
        spec["agg"] = "max"
    elif re.search(r"\b(min|minimum|lowest single|smallest single)\b", t):
        spec["agg"] = "min"
    m = re.search(r"\b(top|bottom|best|worst|largest|smallest|biggest)\s+(\d{1,3})\b", t)
    if m:
        spec["top"] = int(m.group(2))
        if re.search(r"bottom|worst|smallest", m.group(1)):
            spec["sort"] = "asc"
    elif re.search(r"\ball\b", t):
        spec["top"] = 0
    if re.search(r"\bpie\b", t):
        spec["type"] = "pie"
    elif re.search(r"\b(donut|doughnut|share|split|breakdown|proportion|mix|percentage)\b|% of", t):
        spec["type"] = "doughnut"
    elif re.search(r"\bline\b", t):
        spec["type"] = "line"
    elif re.search(r"\barea\b", t):
        spec["type"] = "area"
    elif re.search(r"\b(scatter|vs\.?|versus|against|correlat\w*)\b", t):
        spec["type"] = "scatter"
    elif re.search(r"\b(histogram|distribution|spread)\b", t):
        spec["type"] = "histogram"
    elif re.search(r"\b(bar|bars|column chart)\b", t):
        spec["type"] = "bar"
    if re.search(r"\bstack(ed)?\b", t):
        spec["stacked"] = True

    is_count = re.search(r"\b(number of|count|how many|# of|frequency|orders|transactions|records|rows)\b", t)
    mm = find_col_in_text(text, ds.measures) or syn_match(t, MEASURE_SYN, ds.measures)
    if not mm and is_count:
        spec["y"], spec["agg"] = None, "count"
    elif mm:
        spec["y"] = mm[0]
    elif re.search(SPLIT_WORDS, t):
        spec["y"] = ds.primary
        spec["guessed_measure"] = bool(ds.primary)
        if not ds.primary:
            spec["agg"] = "count"
    else:
        return None

    grain = next((g for pat, g in TIME_SYN if re.search(pat, t)), None)
    cands = [c for c in ds.dims + ds.date_cols if c != spec.get("y")]
    parts = re.split(SPLIT_WORDS, t)
    after = " ".join(parts[1:]) if len(parts) > 1 else ""
    scan = after or t
    d1 = find_col_in_text(scan, cands) or syn_match(" " + scan + " ", DIM_SYN, cands)
    if not d1 and after:
        d1 = find_col_in_text(text, cands) or syn_match(t, DIM_SYN, cands)
    found = []
    if d1:
        found.append(d1[0])
        rest = [c for c in cands if c != d1[0]]
        d2 = find_col_in_text(scan, rest) or syn_match(" " + scan + " ", DIM_SYN, rest)
        if d2 and re.search(r"\b(and|split|stack\w*)\b|(by|per) .* (by|and)", scan):
            found.append(d2[0])

    if not after and not grain and spec["type"] not in ("histogram", "scatter"):
        return None
    if spec["type"] == "histogram":
        spec["y"] = spec.get("y") or (ds.measures[0] if ds.measures else None)
        return spec if spec["y"] else None
    if spec["type"] == "scatter":
        others = [c for c in ds.measures if c != spec.get("y")]
        stripped = t.replace((spec.get("y") or "").lower(), " ")
        x2 = find_col_in_text(stripped, others) or syn_match(stripped, MEASURE_SYN, others)
        if not x2 or not spec.get("y"):
            return None
        spec["x"] = x2[0]
        return spec
    if grain:
        dc = next((c for c in found if ds.cols[c]["type"] == "date"), ds.date_cols[0] if ds.date_cols else None)
        if not dc:
            return None
        spec["x"], spec["grain"] = dc, grain
        other = next((c for c in found if ds.cols[c]["type"] != "date"), None)
        if other:
            spec["series"] = other
        if spec["type"] in ("auto", "pie", "doughnut"):
            spec["type"] = "line" if spec["series"] else "area"
        return spec
    if not found:
        return None
    spec["x"] = found[0]
    if len(found) > 1:
        spec["series"] = found[1]
    if spec["type"] in ("pie", "doughnut") and spec["agg"] not in ("sum", "count"):
        spec["type"] = "bar"
    return spec


# ================================================================= built-in assistant
@dataclass
class Answer:
    text: str
    spec: dict | None = None
    table: pd.DataFrame | None = None
    filters: dict | None = None


def suggestions(ds: Dataset, n: int = 4) -> list[str]:
    m = ds.measures
    d = [c for c in ds.filter_dims if ds.cols[c]["type"] != "number"]
    low = str.lower
    vol = next((c for c in m if re.search(r"units|quantity|qty|volume", c, re.I)), None)
    sales = next((c for c in m if re.search(r"revenue|sales|turnover|amount", c, re.I)), m[0] if m else None)
    geo = next((c for c in d if ds.cols[c]["role"] == "geo"), None)
    cat = next((c for c in d if ds.cols[c]["role"] in ("category", "product")), d[0] if d else None)
    prod = next((c for c in d if ds.cols[c]["role"] == "product"), None)
    prof = next((c for c in m if re.search(r"profit|margin", c, re.I)), None)
    out = []
    if vol and cat:
        out.append(f"{'volume' if low(vol) == 'units' else low(vol)} per {low(cat)}")
    if sales and geo:
        out.append(f"{'sales' if re.search('revenue|sales', sales, re.I) else low(sales)} by {low(geo)}")
    elif sales and d:
        out.append(f"{low(sales)} by {low(d[0])}")
    if ds.date_cols and sales:
        out.append(f"monthly {low(sales)}" + (f" by {low(d[1])}" if len(d) > 1 and ds.cols[d[1]]["unique"] <= 8 else ""))
    if prof and (cat or d):
        out.append(f"{low(prof)} by {low(cat or d[0])}")
    if sales and prod:
        out.append(f"top 5 {low(prod)}s by {low(sales)}")
    if sales and d:
        out.append(f"share of {low(sales)} by {low(d[1] if len(d) > 1 else d[0])}")
    if not out and d:
        out.append(f"number of rows by {low(d[0])}")
    return list(dict.fromkeys(out))[:n]


def _md_table(table: pd.DataFrame, share: bool, limit: int = 8) -> str:
    cols = [table.index.name or ""] + list(table.columns) + (["Share"] if share else [])
    total = table.iloc[:, 0].sum() if share else None
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for lab, row in table.head(limit).iterrows():
        cells = [str(lab)] + [fmt(v) for v in row.tolist()]
        if share:
            cells.append(pct(row.iloc[0] / total) if total else "–")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def narrate(res: AggResult, guessed: str | None = None) -> str:
    n = len(res.table)
    if n == 0:
        return "No rows match this request with the current filters."
    if res.is_time:
        tv = res.totals.dropna()
        peak = tv.idxmax()
        msg = (f"**{res.title}** ({res.grain}, {res.n_rows:,} rows). From {tv.index[0]} to {tv.index[-1]} it moved from "
               f"**{fmt(tv.iloc[0], True)}** to **{fmt(tv.iloc[-1], True)}**; the peak was **{peak}** at {fmt(tv[peak], True)}.")
        if n > 1:
            msg += " The latest period may be incomplete."
    else:
        first = res.table.iloc[:, 0]
        total = first.sum()
        msg = f"**{res.title}** ({res.n_rows:,} rows)"
        if not res.series:
            msg += f": **{first.index[0]}** leads with **{fmt(first.iloc[0], True)}**"
            if res.additive and total:
                msg += f" ({pct(first.iloc[0] / total, 0)} of the total)"
            if n > 1:
                msg += f", followed by {first.index[1]} ({fmt(first.iloc[1], True)})"
            msg += "."
        msg += "\n\n" + _md_table(res.table, share=res.additive and not res.series)
        if n > 8:
            msg += f"\n\nShowing 8 of {n}. Open the chart's table for the full list."
    if guessed:
        msg += f"\n\nNo measure was named, so I used **{guessed}**."
    return msg


def dataset_summary(ds: Dataset, df: pd.DataFrame) -> str:
    lines = [f"**{ds.name}** has **{len(df):,} rows** in view (of {len(ds.df):,}) and **{len(ds.cols)} columns**."]
    if ds.measures:
        lines.append("Measures: " + ", ".join(ds.measures) + ".")
    if ds.filter_dims:
        lines.append("Dimensions: " + ", ".join(f"{c} ({ds.cols[c]['unique']})" for c in ds.filter_dims) + ".")
    for c in ds.date_cols:
        lines.append(f"{c} runs from {fmt_date(df[c].min())} to {fmt_date(df[c].max())}.")
    if ds.primary:
        lines.append(f"Total {ds.primary}: **{fmt(df[ds.primary].sum(), True)}**.")
    lines.append(f"Data quality score: **{ds.quality['score']}/100**.")
    return "\n\n".join(lines)


HELP = ("I can answer questions in the form **measure by column**, for example {ex}. I also understand "
        "totals and averages (\"total revenue in 2025\"), counts (\"how many orders in Germany\"), trends "
        "(\"monthly sales by region\"), rankings (\"bottom 5 products by profit\"), shares (\"share of sales by channel\"), "
        "and the words *summary*, *columns*, *data quality*, *correlations* and *insights*.")


def answer_locally(question: str, ds: Dataset, df: pd.DataFrame) -> Answer:
    q = question.strip()
    t = " " + q.lower() + " "
    if re.search(r"\b(help|what can you do|how do i)\b", t):
        return Answer(HELP.format(ex=" or ".join(f"*{s}*" for s in suggestions(ds, 3))))
    if re.search(r"\b(columns|fields|variables|schema)\b", t) and not re.search(SPLIT_WORDS, t):
        rows = [f"| {c} | {i['type']}{' (period)' if i['dim_like'] else ''} | {i['role'] or '–'} | {pct((i['count'] - i['filled']) / max(i['count'], 1), 0)} | {i['unique']:,} |"
                for c, i in ds.cols.items()]
        return Answer("| Column | Type | Role | Blank | Unique |\n|---|---|---|---|---|\n" + "\n".join(rows))
    if re.search(r"\b(quality|missing|blank|duplicate|clean|errors?)\b", t):
        return Answer(f"Data quality score: **{ds.quality['score']}/100**.\n\n" +
                      "\n".join(f"- **{sev.title()}**: {txt}" for sev, txt in ds.quality["issues"]))
    if re.search(r"\b(correlat\w*|relationship|related)\b", t) and not find_col_in_text(q, ds.measures):
        pairs = []
        ms = ds.measures[:10]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                r = correlation(df, ms[i], ms[j])
                if r is not None:
                    pairs.append((abs(r), r, ms[i], ms[j]))
        if not pairs:
            return Answer("There are not enough numeric columns to compare.")
        pairs.sort(reverse=True)
        body = "\n".join(f"| {a} | {b} | {r:+.2f} |" for _, r, a, b in pairs[:8])
        return Answer("Strongest relationships between measures (Pearson r; correlation does not prove cause):\n\n"
                      "| Measure | Measure | r |\n|---|---|---|\n" + body,
                      spec={"type": "scatter", "x": pairs[0][2], "y": pairs[0][3]})
    if re.search(r"\b(insights?|recommend\w*|what should|advice|key findings|takeaways?|main trends?|what stands out|highlights?)\b", t):
        ins = compute_insights(df, ds, ds.primary)
        return Answer("Here is what stands out:\n\n" + "\n".join(
            f"- **{i['title']}.** {i['body']} *Recommendation:* {i['action']}" for i in ins[:6]))
    if re.search(r"\b(summary|summari[sz]e|overview|describe|about (this|the) data)\b", t) and not re.search(SPLIT_WORDS, t):
        return Answer(dataset_summary(ds, df))

    spec = parse_query(q, ds)
    exclude = {spec.get("x"), spec.get("series")} if spec else set()
    vf = detect_value_filters(q, ds, exclude)
    scoped = apply_filters(df, vf) if vf else df
    note = (" Filtered to " + "; ".join(describe_filters(vf, ds)) + ".") if vf else ""
    if spec:
        if spec["type"] in ("histogram", "scatter"):
            return Answer(f"Here is the {'distribution of **' + spec['y'] + '**' if spec['type'] == 'histogram' else '**' + spec['y'] + '** against **' + spec['x'] + '**'} for {len(scoped):,} rows.{note}",
                          spec=spec, filters=vf)
        res = aggregate(scoped, ds, spec["x"], spec.get("y"), spec["agg"], spec.get("series"), spec["grain"], spec["top"], spec["sort"])
        text = narrate(res, ds.primary if spec.get("guessed_measure") else None)
        return Answer(text + (f"\n\n{note.strip()}" if note else ""), spec=spec, table=res.table, filters=vf)

    # single figures: "total revenue", "average price in Germany", "how many orders in 2025"
    mm = find_col_in_text(q, ds.measures) or syn_match(t, MEASURE_SYN, ds.measures)
    agg = "avg" if re.search(r"\b(average|avg|mean)\b", t) else "median" if "median" in t else \
        "max" if re.search(r"\b(max|maximum|highest|largest)\b", t) else "min" if re.search(r"\b(min|minimum|lowest|smallest)\b", t) else "sum"
    if mm:
        v = overall(scoped, mm[0], agg)
        return Answer(f"{AGG_LABEL[agg]} **{mm[0]}**: **{fmt(v)}** across {len(scoped):,} rows.{note}", filters=vf)
    if re.search(r"\b(how many|number of|count)\b", t):
        return Answer(f"**{len(scoped):,} rows** match.{note}", filters=vf)
    if vf:
        lines = [f"**{len(scoped):,} rows** match {'; '.join(describe_filters(vf, ds))}."]
        for m in ds.measures[:4]:
            lines.append(f"- Total {m}: **{fmt(scoped[m].sum(), True)}** (average {fmt(scoped[m].mean(), True)})")
        return Answer("\n".join(lines), filters=vf)
    return Answer("I could not match that to your columns. " + HELP.format(ex=" or ".join(f"*{s}*" for s in suggestions(ds, 3))))


# ================================================================= insights
def compute_insights(df: pd.DataFrame, ds: Dataset, measure: str | None) -> list[dict]:
    m = measure or ds.primary
    agg = "sum" if m else "count"
    label = m or "row count"
    out: list[dict] = []
    if df.empty:
        return [{"tone": "warn", "area": "Filters", "title": "No rows match the current filters",
                 "body": "Every finding needs data in view.", "action": "Loosen or clear the filters.", "spec": None}]
    dc = ds.date_cols[0] if ds.date_cols else None
    dims = [c for c in ds.filter_dims if ds.cols[c]["type"] != "number" and 2 <= ds.cols[c]["unique"] <= 60][:4]

    pc = period_change(df, ds, m, agg) if dc else None
    if pc and pc["change"] is not None:
        up = pc["change"] >= 0
        out.append({"tone": "up" if up else "down", "area": "Trend",
                    "title": f"{'Total ' + m if m else 'Volume'} {'up' if up else 'down'} {pct(abs(pc['change']))} in the latest 3 full months",
                    "body": f"{pc['labels'][0]} – {pc['labels'][1]}: {fmt(pc['cur'], True)} vs {fmt(pc['prev'], True)} in the 3 months before.",
                    "action": "Find which segments drive the growth and protect their supply and budget." if up
                    else "Break the decline down by segment to find where it starts.",
                    "spec": {"type": "area", "x": dc, "y": m, "agg": agg, "grain": "month"}})

    for d in dims:
        res = aggregate(df, ds, d, m, agg, top=0)
        vals = res.table.iloc[:, 0]
        total = vals.sum()
        if not total:
            continue
        share = vals.iloc[0] / total
        if share > 0.35 and len(vals) > 2:
            out.append({"tone": "warn", "area": d, "title": f"{vals.index[0]} accounts for {pct(share, 0)} of {label}",
                        "body": f"A single {d.lower()} carries a large part of the result, out of {len(vals)} in total.",
                        "action": f"Reduce dependency: set growth targets for the next {min(3, len(vals) - 1)} {d.lower()} values.",
                        "spec": {"type": "doughnut", "x": d, "y": m, "agg": agg, "top": 8}})
        elif len(vals) >= 5:
            n20 = max(1, round(len(vals) * 0.2))
            s20 = vals.iloc[:n20].sum() / total
            if s20 > 0.5:
                out.append({"tone": "info", "area": d, "title": f"Top {n20} of {len(vals)} {d.lower()} values make {pct(s20, 0)} of {label}",
                            "body": f"Leaders: {', '.join(vals.index[:min(3, n20 + 1)])}.",
                            "action": "Focus account management and forecasting effort on these first.",
                            "spec": {"type": "hbar", "x": d, "y": m, "agg": agg, "top": 10}})

    if dc and dims:
        mx = df[dc].max()
        cut, cut0 = mx - pd.Timedelta(days=182), mx - pd.Timedelta(days=365)
        val = df[m] if m else pd.Series(1, index=df.index)
        for d in dims[:2]:
            key = df[d].astype(object).where(df[d].notna(), "(blank)").astype(str)
            cur = val[df[dc] > cut].groupby(key[df[dc] > cut]).sum()
            prev = val[(df[dc] > cut0) & (df[dc] <= cut)].groupby(key[(df[dc] > cut0) & (df[dc] <= cut)]).sum()
            tc = cur.sum() or 1
            ch = [(k, (cur.get(k, 0) - prev[k]) / prev[k]) for k in prev.index if prev[k] > 0 and cur.get(k, 0) / tc > 0.02]
            ch.sort(key=lambda x: x[1], reverse=True)
            if len(ch) >= 3:
                (bk, bg), (wk, wg) = ch[0], ch[-1]
                spec = {"type": "line", "x": dc, "y": m, "agg": agg, "series": d, "grain": "month"}
                if bg > 0.05:
                    out.append({"tone": "up", "area": d, "title": f"{bk} is the fastest-growing {d.lower()}: +{pct(bg, 0)}",
                                "body": f"Last 6 months vs the 6 months before, measured on {label}.",
                                "action": f"Back the momentum: increase stock, marketing or coverage for {bk}.", "spec": spec})
                if wg < -0.05:
                    out.append({"tone": "down", "area": d, "title": f"{wk} is declining: {pct(wg, 0)}",
                                "body": f"Last 6 months vs the 6 months before, measured on {label}.",
                                "action": f"Investigate {wk}: pricing, availability and competitor activity are the usual causes.", "spec": spec})

    prof = next((c for c in ds.measures if re.search(r"profit", c, re.I)), None)
    rev = next((c for c in ds.measures if re.search(r"revenue|sales|turnover", c, re.I)), None)
    if prof and rev and dims:
        d = next((x for x in dims if ds.cols[x]["role"] in ("category", "product")), dims[0])
        g = df.groupby(df[d].astype(str))[[prof, rev]].sum()
        g = g[g[rev] > 0]
        if len(g) >= 2:
            mg = (g[prof] / g[rev]).sort_values()
            tot = g[prof].sum() / g[rev].sum()
            out.append({"tone": "warn" if mg.iloc[0] < tot * 0.7 else "info", "area": "Margin",
                        "title": f"{mg.index[0]} has the lowest margin at {pct(mg.iloc[0])}",
                        "body": f"Overall margin is {pct(tot)}; {mg.index[-1]} leads at {pct(mg.iloc[-1])}.",
                        "action": f"Review pricing, discounts and cost of goods for {mg.index[0]}, and shift promotion toward {mg.index[-1]}.",
                        "spec": {"type": "bar", "x": d, "y": prof, "agg": "sum", "top": 10}})
        disc = next((c for c in ds.measures if re.search(r"discount|rebate", c, re.I)), None)
        if disc:
            r = correlation(df, disc, prof)
            if r is not None and r < -0.15:
                out.append({"tone": "warn", "area": "Pricing", "title": f"Higher {disc.lower()} goes with lower {prof.lower()}",
                            "body": f"Correlation of {r:.2f} between {disc} and {prof} per row.",
                            "action": "Set discount guardrails and require approval above the typical level.",
                            "spec": {"type": "bar", "x": disc, "y": prof, "agg": "avg", "top": 0}})

    ms = ds.measures[:8]
    best = None
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            r = correlation(df, ms[i], ms[j])
            if r is not None and 0.5 < abs(r) < 0.97 and (best is None or abs(r) > abs(best[2])):
                best = (ms[i], ms[j], r)
    if best:
        a, b, r = best
        out.append({"tone": "info", "area": "Relationships", "title": f"{a} and {b} move {'together' if r > 0 else 'in opposite directions'}",
                    "body": f"Correlation r = {r:.2f} across {len(df):,} rows. Correlation does not prove cause.",
                    "action": f"Use {a} as a leading indicator when forecasting {b}.", "spec": {"type": "scatter", "x": a, "y": b}})
    if m and ds.cols[m].get("outliers"):
        i = ds.cols[m]
        out.append({"tone": "info", "area": "Outliers", "title": f"{i['outliers']:,} unusual {m} values",
                    "body": f"Values outside {fmt(i['lo'], True)} – {fmt(i['hi'], True)} can skew averages.",
                    "action": "Sort the column in the Data tab and confirm they are genuine.", "spec": {"type": "histogram", "y": m}})
    for sev, txt in [x for x in ds.quality["issues"] if x[0] in ("high", "check")][:2]:
        out.append({"tone": "warn", "area": "Data quality", "title": "Fix before relying on the numbers", "body": txt,
                    "action": "Clean the source or filter affected rows before reporting.", "spec": None})
    return out


# ================================================================= example data
def make_example(n: int = 4200, seed: int = 20261022) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    geo = {"Europe": [("Denmark", 1.1), ("Germany", 1.6), ("France", 1.3), ("United Kingdom", 1.4), ("Sweden", .8), ("Spain", .9)],
           "North America": [("United States", 2.2), ("Canada", .9), ("Mexico", .6)],
           "Asia Pacific": [("Japan", 1.1), ("Australia", .8), ("India", .9), ("Singapore", .5)],
           "Africa": [("Côte d'Ivoire", .45), ("Nigeria", .6), ("South Africa", .55), ("Kenya", .4)]}
    countries = [(c, r, w) for r, lst in geo.items() for c, w in lst]
    products = [("Laptop Pro 14", "Electronics", 1299, 1), ("Wireless Earbuds", "Electronics", 129, 1.6), ("4K Monitor", "Electronics", 399, 1),
                ("Smart Watch", "Electronics", 249, 1.2), ("Air Purifier", "Home", 219, .8), ("Coffee Maker", "Home", 89, 1.3),
                ("Robot Vacuum", "Home", 349, .9), ("Ergo Chair", "Office", 459, 1), ("Standing Desk", "Office", 599, .8),
                ("Desk Lamp", "Office", 49, 1.2), ("Trail Backpack", "Outdoor", 119, 1), ("Camping Tent", "Outdoor", 289, .7),
                ("Water Bottle", "Outdoor", 25, 1.4)]
    cost_ratio = {"Electronics": .7, "Home": .55, "Office": .62, "Outdoor": .48}
    cw = np.array([c[2] for c in countries]); cw /= cw.sum()
    pw = np.array([p[3] for p in products]); pw /= pw.sum()
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-09-30")
    span = (end - start).total_seconds()
    rows = []
    while len(rows) < n:
        t = rng.random()
        ts = start + pd.Timedelta(seconds=t * span)
        mo = ts.month
        c = countries[rng.choice(len(countries), p=cw)]
        p = products[rng.choice(len(products), p=pw)]
        w = 1.35 if mo >= 10 else .8 if mo <= 2 else 1
        if p[1] == "Electronics":
            w *= .75 + .6 * t
        if p[1] == "Office":
            w *= .55 if t > .75 else 1.05 - .2 * t
        if c[1] == "Asia Pacific":
            w *= .6 + .9 * t
        if c[1] == "Africa":
            w *= .7 + .5 * t
        if rng.random() * 2.1 > w:
            continue
        channel = rng.choice(["Online", "Retail", "Partner"], p=[1.5 / 3.1, 1 / 3.1, .6 / 3.1])
        segment = rng.choice(["Consumer", "SMB", "Enterprise"], p=[1.6 / 3.05, 1 / 3.05, .45 / 3.05])
        units = 1 + int(rng.random() ** 2 * (30 if p[2] < 100 else 10 if p[2] < 400 else 4))
        if segment == "Enterprise":
            units *= 3
        if rng.random() < 0.0015:
            units *= 25
        price = round(p[2] * (0.95 + rng.random() * 0.1), 2)
        disc = float(rng.choice([0.1, 0.15, 0.2])) if channel == "Partner" else float(rng.choice([0, 0, 0.05, 0.1]))
        revenue = units * price * (1 - disc)
        cost = units * p[2] * cost_ratio[p[1]] * (0.95 + rng.random() * 0.1)
        rows.append({"Order Date": ts.normalize(), "Region": c[1], "Country": c[0], "Category": p[1], "Product": p[0],
                     "Channel": channel, "Customer Segment": segment, "Units": units, "Unit Price": price,
                     "Discount": None if rng.random() < 0.03 else disc, "Revenue": round(revenue, 2),
                     "Cost": round(cost, 2), "Profit": round(revenue - cost, 2)})
    df = pd.DataFrame(rows).sort_values("Order Date").reset_index(drop=True)
    df.insert(0, "Order ID", [f"SO-{100000 + i}" for i in range(len(df))])
    df["Order Date"] = df["Order Date"].dt.strftime("%Y-%m-%d")
    return df
