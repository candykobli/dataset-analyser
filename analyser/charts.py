"""Plotly charts built from aggregation results and chart specs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .engine import AggResult, aggregate, apply_filters
from .profiling import Dataset

PALETTE = ["#2c5bd0", "#e0861a", "#17907c", "#c2457a", "#7457c4", "#5c8a25", "#d0533a", "#3f8bb3"]
CHART_TYPES = {"auto": "Auto", "bar": "Bar", "hbar": "Horizontal bar", "line": "Line", "area": "Area",
               "pie": "Pie", "doughnut": "Doughnut", "scatter": "Scatter", "histogram": "Histogram"}


def _layout(fig: go.Figure, height: int, legend: bool) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=30, b=8), showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None),
        hovermode="x unified" if fig.data and fig.data[0].type in ("scatter",) and fig.data[0].mode != "markers" else "closest",
        font=dict(size=12), separators=".,",
    )
    fig.update_xaxes(title=None, showgrid=False)
    fig.update_yaxes(title=None)
    return fig


def resolve_type(chart_type: str, res: AggResult) -> str:
    if chart_type and chart_type != "auto":
        return chart_type
    if res.is_time:
        return "line" if res.series else "area"
    labels = list(res.table.index)
    if len(labels) > 8 or any(len(str(l)) > 14 for l in labels):
        return "hbar"
    return "bar"


def figure_from_result(res: AggResult, chart_type: str = "auto", height: int = 380, stacked: bool = False) -> go.Figure:
    kind = resolve_type(chart_type, res)
    long = res.table.reset_index().melt(id_vars=res.table.index.name or "index", var_name="Series", value_name="Value")
    xname = res.table.index.name or "index"
    multi = res.table.shape[1] > 1
    order = {xname: list(res.table.index)}
    if kind in ("pie", "doughnut"):
        s = res.table.iloc[:, 0].clip(lower=0)
        fig = go.Figure(go.Pie(labels=list(s.index), values=s.values, hole=0.55 if kind == "doughnut" else 0,
                               marker=dict(colors=PALETTE * 4), sort=False, textinfo="percent",
                               hovertemplate="%{label}: %{value:,.2f} (%{percent})<extra></extra>"))
        return _layout(fig, height, True).update_layout(legend=dict(orientation="v", y=0.5, x=1.02, yanchor="middle"))
    if kind in ("line", "area"):
        fn = px.area if kind == "area" and not multi else px.line
        fig = fn(long, x=xname, y="Value", color="Series" if multi else None, category_orders=order,
                 color_discrete_sequence=PALETTE, markers=len(res.table) <= 40)
        fig.update_traces(line=dict(width=2.2), hovertemplate="%{y:,.2f}")
        return _layout(fig, height, multi)
    horizontal = kind == "hbar"
    if len(res.table) > 1 and str(res.table.index[-1]).startswith("Other ("):
        # keep the long tail in tables and pies, but not as one giant bar
        long = long[long[xname] != res.table.index[-1]]
        order = {xname: list(res.table.index[:-1])}
    fig = px.bar(long, x="Value" if horizontal else xname, y=xname if horizontal else "Value",
                 color="Series" if multi else None, orientation="h" if horizontal else "v",
                 barmode="stack" if stacked else "group", category_orders=order, color_discrete_sequence=PALETTE)
    if horizontal:
        height = max(height, min(900, 28 * len(res.table) + 80))
    fig.update_traces(hovertemplate="%{y}: %{x:,.2f}" if horizontal else "%{x}: %{y:,.2f}", marker_line_width=0)
    return _layout(fig, height, multi)


def histogram(df: pd.DataFrame, col: str, height: int = 380) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=PALETTE)
    fig.update_traces(marker_line_width=0, hovertemplate=f"{col} %{{x}}<br>Rows %{{y:,}}<extra></extra>")
    fig = _layout(fig, height, False)
    fig.update_layout(bargap=0.04)
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, height: int = 380) -> go.Figure:
    d = df[[c for c in {x, y, color} if c]].dropna()
    if len(d) > 5000:
        d = d.sample(5000, random_state=0)
    fig = px.scatter(d, x=x, y=y, color=color, opacity=0.55, color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=6))
    fig = _layout(fig, height, bool(color))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return fig


def figure_from_spec(spec: dict, df: pd.DataFrame, ds: Dataset, height: int = 380) -> tuple[go.Figure | None, AggResult | None]:
    """Build a figure for a chart spec (from the parser, an insight, the builder or the AI)."""
    if spec.get("filters"):
        df = apply_filters(df, spec["filters"])
    kind = spec.get("type", "auto")
    if kind == "histogram":
        col = spec.get("y") or (ds.measures[0] if ds.measures else None)
        return (histogram(df, col, height) if col else None), None
    if kind == "scatter":
        x, y = spec.get("x"), spec.get("y")
        if not (x and y):
            return None, None
        return scatter(df, x, y, spec.get("color"), height), None
    if not spec.get("x"):
        return None, None
    res = aggregate(df, ds, spec["x"], spec.get("y"), spec.get("agg") or "sum", spec.get("series"),
                    spec.get("grain") or "auto", int(spec.get("top", 10) if spec.get("top") is not None else 10),
                    spec.get("sort") or "desc")
    if res.table.empty:
        return None, res
    return figure_from_result(res, kind, height, bool(spec.get("stacked"))), res


def sparkline_values(info: dict) -> list[int]:
    if info.get("hist"):
        return info["hist"]
    if info["type"] in ("category", "bool") and info.get("top"):
        return [n for _, n in info["top"][:12]]
    return []
