"""Optional AI analyst (Anthropic Claude API). The app works fully without it.

The model never receives the raw file: it gets the column profile, a few example rows,
and the results of the calculations it asks the app to run through tools.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from .engine import AGG_LABEL, aggregate, apply_filters, describe_filters, measure_label, overall
from .profiling import Dataset, fmt, pct, profile_column, to_number

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_ROUNDS = 8

_FILTERS = {
    "type": "array",
    "description": "Optional row filters for this call only, combined with AND. Dates as YYYY-MM-DD.",
    "items": {"type": "object", "properties": {
        "column": {"type": "string"},
        "op": {"type": "string", "enum": ["=", "!=", "in", "not in", ">", ">=", "<", "<=", "between", "contains"]},
        "value": {"description": "A value, or a list for in / not in / between."}},
        "required": ["column", "value"]},
}
_AGG = {"type": "string", "enum": ["sum", "avg", "count", "distinct", "median", "min", "max"]}
_GRAIN = {"type": "string", "enum": ["day", "week", "month", "quarter", "year"]}

TOOLS = [
    {"name": "run_query",
     "description": "Aggregate the data the user is viewing (dashboard filters already applied). Example: total Revenue "
                    "grouped by Country. Omit group_by for one overall figure. Returns at most 60 groups.",
     "input_schema": {"type": "object", "properties": {
         "group_by": {"type": "array", "items": {"type": "string"}, "description": "0-2 column names; a date column is bucketed by time_grain."},
         "measure": {"type": "string", "description": "Numeric column to aggregate. Omit to count rows."},
         "agg": _AGG, "time_grain": _GRAIN, "filters": _FILTERS,
         "sort": {"type": "string", "enum": ["desc", "asc"]},
         "limit": {"type": "integer", "description": "Max groups, default 25; 0 = all."}}}},
    {"name": "show_chart",
     "description": "Show a chart to the user and return the plotted numbers. Use it for every breakdown, ranking, share, comparison or trend.",
     "input_schema": {"type": "object", "properties": {
         "chart_type": {"type": "string", "enum": ["bar", "hbar", "line", "area", "pie", "doughnut", "scatter", "histogram"]},
         "x": {"type": "string", "description": "Group-by column (the date column for trends, a numeric column for scatter)."},
         "y": {"type": "string", "description": "Measure column. Omit to count rows."},
         "agg": _AGG, "series": {"type": "string", "description": "Optional second column split into coloured series."},
         "time_grain": _GRAIN, "top_n": {"type": "integer"}, "stacked": {"type": "boolean"},
         "title": {"type": "string"}, "filters": _FILTERS},
         "required": ["chart_type", "title"]}},
    {"name": "get_rows",
     "description": "Return up to 25 individual rows matching filters, optionally sorted. Use to inspect records or outliers.",
     "input_schema": {"type": "object", "properties": {
         "filters": _FILTERS, "columns": {"type": "array", "items": {"type": "string"}},
         "sort_by": {"type": "string"}, "descending": {"type": "boolean"}, "limit": {"type": "integer"}}}},
    {"name": "column_stats",
     "description": "Statistics for one column on the data in view: blanks, distinct values, min/max/mean/median/quartiles, or top values.",
     "input_schema": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
]


def available() -> bool:
    return anthropic is not None


def _col(ds: Dataset, name) -> str:
    if name in (None, ""):
        return None
    n = str(name).strip().lower()
    for c in ds.cols:
        if c.lower() == n:
            return c
    for c in ds.cols:
        if n in c.lower() or c.lower() in n:
            return c
    raise ValueError(f'Unknown column "{name}". Columns: {", ".join(ds.cols)}')


def _row_filter(df: pd.DataFrame, ds: Dataset, filters) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for f in filters:
        c = _col(ds, f.get("column"))
        info = ds.cols[c]
        op = (f.get("op") or ("in" if isinstance(f.get("value"), list) else "=")).lower()
        vals = f.get("value") if isinstance(f.get("value"), list) else [f.get("value")]
        s = df[c]
        if info["type"] == "number":
            vals = to_number(pd.Series(vals, dtype=object)).tolist()
        elif info["type"] == "date":
            vals = [pd.Timestamp(v) for v in vals]
        elif info["type"] == "bool":
            vals = [str(v).lower() in ("true", "yes", "1") for v in vals]
        else:
            s = s.astype(str).str.lower()
            vals = [str(v).lower() for v in vals]
        if op in ("=", "==", "in"):
            m = s.isin(vals)
        elif op in ("!=", "not in"):
            m = ~s.isin(vals)
        elif op == ">":
            m = s > vals[0]
        elif op == ">=":
            m = s >= vals[0]
        elif op == "<":
            m = s < vals[0]
        elif op == "<=":
            m = s <= vals[0]
        elif op == "between":
            m = (s >= vals[0]) & (s <= vals[1])
        elif op == "contains":
            m = s.astype(str).str.contains(str(vals[0]), case=False, regex=False)
        else:
            raise ValueError(f"Unsupported op {op}")
        mask &= m.fillna(False).astype(bool)
    return df[mask]


def _spec(ds: Dataset, a: dict) -> dict:
    gb = a.get("group_by") or []
    x = _col(ds, a.get("x")) if a.get("x") else (_col(ds, gb[0]) if gb else None)
    series = _col(ds, a.get("series")) if a.get("series") else (_col(ds, gb[1]) if len(gb) > 1 else None)
    y = _col(ds, a.get("y") or a.get("measure"))
    agg = (a.get("agg") or ("sum" if y else "count")).lower()
    agg = {"mean": "avg", "average": "avg"}.get(agg, agg)
    return {"x": x, "y": y, "agg": agg, "series": series, "grain": a.get("time_grain") or "auto",
            "top": a.get("top_n", a.get("limit", 15)), "sort": a.get("sort") or "desc",
            "type": a.get("chart_type") or "auto", "stacked": bool(a.get("stacked"))}


def _clean(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float):
        return round(v, 4)
    return v


def execute_tool(name: str, args: dict, ds: Dataset, df: pd.DataFrame, charts: list) -> dict:
    rows = _row_filter(df, ds, args.get("filters"))
    if name == "run_query":
        s = _spec(ds, {**args, "limit": args.get("limit", 25)})
        if not s["x"]:
            return {"rows_matched": len(rows), "measure": measure_label(s["y"], s["agg"]), "value": overall(rows, s["y"], s["agg"])}
        res = aggregate(rows, ds, s["x"], s["y"], s["agg"], s["series"], s["grain"], int(s["top"] or 0), s["sort"])
        groups = [{s["x"]: lab, **{k: _clean(v) for k, v in r.items()}} for lab, r in res.table.head(60).iterrows()]
        return {"rows_matched": len(rows), "measure": measure_label(s["y"], res.agg), "split_by": s["series"],
                "time_grain": res.grain, "groups": groups, "truncated": len(res.table) > 60}
    if name == "show_chart":
        s = _spec(ds, args)
        if args.get("filters"):
            s["row_filters"] = args["filters"]
        s["title"] = args.get("title") or "Chart"
        charts.append(s)
        if s["type"] in ("histogram", "scatter"):
            return {"rendered": True, "rows": len(rows)}
        if not s["x"]:
            raise ValueError("x is required for this chart type")
        res = aggregate(rows, ds, s["x"], s["y"], s["agg"], s["series"], s["grain"], int(s["top"] or 15), s["sort"])
        return {"rendered": True, "points": [{"label": lab, **{k: _clean(v) for k, v in r.items()}} for lab, r in res.table.head(40).iterrows()]}
    if name == "get_rows":
        if args.get("sort_by"):
            rows = rows.sort_values(_col(ds, args["sort_by"]), ascending=not args.get("descending", False))
        cols = [_col(ds, c) for c in args.get("columns") or []] or list(ds.cols)[:15]
        out = rows[cols].head(min(25, int(args.get("limit") or 10)))
        return {"rows_matched": len(rows), "rows": [{k: _clean(v) for k, v in r.items()} for r in out.to_dict("records")]}
    if name == "column_stats":
        c = _col(ds, args["column"])
        p = profile_column(ds.cols[c], df[c])
        o = {"column": c, "type": p["type"], "rows": p["count"], "blanks": p["count"] - p["filled"], "distinct": p["unique"]}
        for k in ("min", "q1", "median", "mean", "q3", "max", "std", "sum", "outliers"):
            if k in p:
                o[k] = _clean(p[k])
        if p["type"] not in ("number", "date"):
            o["top_values"] = [{"value": v, "rows": n} for v, n in p["top"][:15]]
        return o
    raise ValueError(f"Unknown tool {name}")


def schema_text(ds: Dataset) -> str:
    out = []
    for c, i in ds.cols.items():
        s = f'- "{c}" ({"period number" if i["dim_like"] else "id" if i["is_id"] else i["type"]}{", " + i["role"] if i["role"] else ""}'
        blank = i["count"] - i["filled"]
        s += f", {pct(blank / i['count'], 0)} blank)" if blank else ")"
        if i["type"] == "number":
            s += f": min {fmt(i.get('min'))}, max {fmt(i.get('max'))}, mean {fmt(i.get('mean'))}, sum {fmt(i.get('sum'))}"
        elif i["type"] == "date":
            s += f": {i['min']:%Y-%m-%d} to {i['max']:%Y-%m-%d}"
        elif i["type"] in ("category", "bool"):
            s += f": {i['unique']} values: " + " | ".join(str(v)[:40] for v, _ in i["top"][:30]) + (" | …" if i["unique"] > 30 else "")
        else:
            s += f": {i['unique']:,} unique, e.g. " + " | ".join(str(v)[:30] for v, _ in i["top"][:3])
        out.append(s)
    return "\n".join(out)


def system_prompt(ds: Dataset, df: pd.DataFrame, filters: dict) -> str:
    sample = df.head(4).to_dict("records")
    sample = "\n".join(json.dumps({k: _clean(v) for k, v in r.items()}, default=str) for r in sample)
    f = describe_filters(filters, ds)
    return f"""You are the analyst inside a data analysis app. The user is looking at the dataset "{ds.name}" ({ds.detail}).
Total rows: {len(ds.df):,}. Rows after the dashboard filters: {len(df):,}.
Active dashboard filters: {'; '.join(f) if f else 'none'}.
Today is {date.today():%Y-%m-%d}.

Columns:
{schema_text(ds)}

Example rows:
{sample}

How to work:
- Compute every number with the tools. Never estimate or invent figures. Tools already apply the dashboard filters.
- Map everyday words to columns: "sales"/"turnover" means the revenue or sales column; "volume"/"units"/"quantity" means the units or quantity column; "per"/"by"/"for each" means group by that column; "monthly"/"trend"/"over time" means group by the date column with a time_grain.
- For any breakdown, ranking, comparison, share or trend, call show_chart so the user sees it: bar or hbar for rankings, line or area for time, doughnut for shares of a whole (8 slices or fewer), scatter for two measures.
- Use exact category values from the column list. If something the user names is not in the data, say so and offer the closest match.
- Answer in the user's language. Lead with the direct answer and key numbers in **bold**, then 2-4 short bullets of interpretation, and a concrete recommendation when the question is about performance. Stay under 180 words unless asked for detail. Small markdown tables are fine. Do not describe your tool calls."""


def ask(api_key: str, model: str, ds: Dataset, df: pd.DataFrame, filters: dict, history: list, question: str) -> dict:
    """Run one assistant turn. Returns {"text": str, "charts": [spec, ...]}."""
    client = anthropic.Anthropic(api_key=api_key)
    messages = [m for m in history[-12:] if m.get("content")] + [{"role": "user", "content": question}]
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    charts: list = []
    system = system_prompt(ds, df, filters)
    for _ in range(MAX_ROUNDS):
        resp = client.messages.create(model=model, max_tokens=2000, system=system, tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return {"text": text or "I could not produce an answer. Try rephrasing the question.", "charts": charts}
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                out = execute_tool(block.name, block.input or {}, ds, df, charts)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out, default=str)[:30000]})
            except Exception as exc:  # the model sees the error and can correct itself
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {exc}", "is_error": True})
        messages.append({"role": "user", "content": results})
    return {"text": "I stopped after too many steps. Try a more specific question.", "charts": charts}


def review(api_key: str, model: str, ds: Dataset, df: pd.DataFrame, filters: dict, insights: list) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    m = ds.primary
    agg = "sum" if m else "count"
    facts = []
    for d in [c for c in ds.filter_dims if ds.cols[c]["type"] != "number"][:3]:
        r = aggregate(df, ds, d, m, agg, top=12)
        facts.append(f"{measure_label(m, agg)} by {d}: " + ", ".join(f"{k}={fmt(v)}" for k, v in r.table.iloc[:, 0].items()))
    if ds.date_cols:
        r = aggregate(df, ds, ds.date_cols[0], m, agg, grain="quarter")
        facts.append("By quarter: " + ", ".join(f"{k}={fmt(v)}" for k, v in r.table.iloc[:, 0].items()))
    prompt = f"""You are a senior business analyst. Write a review of this dataset for a manager.

Dataset "{ds.name}": {len(df):,} rows in view of {len(ds.df):,}. Filters: {'; '.join(describe_filters(filters, ds)) or 'none'}.
Columns:
{schema_text(ds)}

Key figures:
{chr(10).join(facts)}

Automatic findings:
{chr(10).join(f"- [{i['area']}] {i['title']}. {i['body']}" for i in insights)}

Data quality score {ds.quality['score']}/100. Issues: {' '.join(t for _, t in ds.quality['issues'])}

Write in English with markdown: "### Summary" (2-3 sentences with the most important numbers in bold), "### Recommendations" (4-6 numbered, prioritised actions, each with its evidence and expected impact), "### Risks to watch" (2-3 bullets), "### Questions to investigate next" (3 bullets the user could ask the assistant), "### Data fixes" (only if quality issues matter). Use only the figures given. Under 400 words."""
    resp = client.messages.create(model=model, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    if "Authentication" in name:
        return "The API key was rejected. Check the key in the sidebar or in the app's secrets."
    if "RateLimit" in name:
        return "The AI service is busy or the usage limit was reached. Wait a minute and try again."
    if "NotFound" in name:
        return "The AI model name was not recognised. Set ANTHROPIC_MODEL in the app's secrets to a current model."
    if "Connection" in name or "Timeout" in name:
        return "The AI service could not be reached. Check the internet connection and try again."
    return f"The AI request failed ({name}). The built-in assistant still works for 'measure by column' questions."
