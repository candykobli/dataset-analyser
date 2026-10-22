"""Clean raw frames, detect column types and roles, profile columns and score data quality."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

MISSING = {"", "na", "n/a", "null", "none", "nan", "nat", "-", "—", "?", "undefined", "#n/a",
           "nil", "#value!", "#div/0!", "#ref!", "<na>"}
BOOLS = {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False,
         "t": True, "f": False, "1": True, "0": False}

ROLE_RULES = [
    ("geo", r"country|nation|region|continent|state|province|city|territor|market|location|area|zone|county|geo|iso|land|site|plant"),
    ("time", r"date|time|day|month|year|quarter|week|period|\bfy"),
    ("product", r"product|item|sku|brand|model|article|material|service|drug|molecule"),
    ("category", r"categ|type|class|segment|group|family|department|division|line|genre|kind|tier|level|status|stage|grade"),
    ("channel", r"channel|source|medium|platform|store|shop|outlet|campaign|device"),
    ("party", r"customer|client|account|company|vendor|supplier|partner|employee|\brep\b|agent|owner|manager|team|user"),
]
MEASURE_PRIORITY = [r"revenue|turnover", r"sales", r"amount|total|value", r"profit|margin",
                    r"quantity|qty|units|volume", r"cost|spend|expense", r"price", r"score|rating"]
ROLE_ORDER = {"geo": 0, "category": 1, "product": 2, "channel": 3, "party": 4, "time": 5}

_CURRENCY = r"USD|EUR|GBP|DKK|SEK|NOK|CHF|JPY|CAD|AUD|XOF|NGN|ZAR|INR|CNY"
_MONTH = r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
DATE_RE = re.compile(
    r"^(\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?([ T]\d{1,2}:\d{2}.*)?"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}([ T]\d{1,2}:\d{2}.*)?"
    r"|\d{4}[- ]?Q[1-4]|Q[1-4][- ]?\d{4}"
    rf"|(?=.*\b({_MONTH})[a-z]*\.?\b)(?=.*\d{{4}}).{{4,40}})$",
    re.I,
)


def detect_role(name: str) -> str | None:
    for role, pattern in ROLE_RULES:
        if re.search(pattern, name, re.I):
            return role
    return None


def missing_mask(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s.isna()
    low = s.astype(str).str.strip().str.lower()
    return s.isna() | low.isin(MISSING)


def to_number(s: pd.Series) -> pd.Series:
    """Parse numbers written as text: currencies, %, (negatives), 1.234,56 and 1,234.56."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    x = s.astype(object).where(s.notna(), None).map(lambda v: "" if v is None else str(v)).str.strip()
    neg = x.str.match(r"^\(.*\)$")
    x = x.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    x = x.str.replace(rf"(?i)^({_CURRENCY})\s*|\s*({_CURRENCY})$", "", regex=True)
    x = x.str.replace(r"(?i)[$€£¥₹₦%]|kr\.?|,-$", "", regex=True)
    x = x.str.replace(r"[\s  ']", "", regex=True)
    eu_thousands = x.str.match(r"^[-+]?\d{1,3}(\.\d{3})+(,\d+)?$")
    eu_decimal = x.str.match(r"^[-+]?\d+,\d{1,2}$") & ~eu_thousands
    other = ~(eu_thousands | eu_decimal)
    x = x.where(~eu_thousands, x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    x = x.where(~eu_decimal, x.str.replace(",", ".", regex=False))
    x = x.where(~other, x.str.replace(",", "", regex=False))
    out = pd.to_numeric(x, errors="coerce").astype(float)
    out[neg.fillna(False).astype(bool)] *= -1
    return out


def parse_dates(s: pd.Series, dayfirst: bool) -> pd.Series:
    x = s.astype(object).where(s.notna(), None).map(lambda v: None if v is None else str(v).strip())
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    valid = x.notna()
    iso = valid & x.fillna("").str.match(r"^\d{4}[-/.]\d{1,2}")
    q = x.fillna("").str.extract(r"^(?:(\d{4})[- ]?Q([1-4])|Q([1-4])[- ]?(\d{4}))$", flags=re.I)
    qmask = q.notna().any(axis=1)
    rest = valid & ~iso & ~qmask

    def conv(part: pd.Series, first: bool) -> pd.Series:
        r = pd.to_datetime(part, errors="coerce", format="mixed", dayfirst=first, utc=True)
        return r.dt.tz_localize(None).astype("datetime64[ns]")

    if iso.any():
        out[iso] = conv(x[iso], False)
    if rest.any():
        out[rest] = conv(x[rest], dayfirst)
    if qmask.any():
        year = q[0].fillna(q[3])[qmask]
        month = ((q[1].fillna(q[2])[qmask].astype(int) - 1) * 3 + 1).astype(str)
        out[qmask] = pd.to_datetime(year + "-" + month + "-01")
    return out


def _detect_dayfirst(strs: pd.Series) -> bool:
    m = strs.str.extract(r"^(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}").dropna()
    if m.empty:
        return True
    a, b = m[0].astype(int), m[1].astype(int)
    if (b > 12).any() and not (a > 12).any():
        return False  # month/day/year
    return True


def infer_column(name: str, s: pd.Series) -> tuple[pd.Series, dict]:
    info = {"name": name, "role": detect_role(name), "type": "empty", "warn": None,
            "is_id": False, "dim_like": False, "date_order": None}
    miss = missing_mask(s)
    vals = s[~miss]
    n = len(vals)
    if n == 0:
        return pd.Series([None] * len(s), index=s.index, dtype=object), info
    sample = vals.sample(5000, random_state=0) if n > 5000 else vals
    obj = sample.infer_objects()
    low = sample.astype(str).str.strip().str.lower()
    uniq_low = set(low.unique())

    # booleans
    if pd.api.types.is_bool_dtype(obj) or (len(uniq_low) <= 2 and uniq_low <= set(BOOLS) and not uniq_low & {"1", "0"}):
        conv = s.astype(str).str.strip().str.lower().map(BOOLS).where(~miss)
        return conv.astype("boolean"), {**info, "type": "bool"}

    # real datetimes (Excel, Parquet)
    if pd.api.types.is_datetime64_any_dtype(obj) or all(isinstance(v, (datetime, pd.Timestamp)) for v in itertools.islice(sample, 200)):
        conv = pd.to_datetime(s.where(~miss), errors="coerce", utc=True).dt.tz_localize(None)
        return conv.astype("datetime64[ns]"), {**info, "type": "date", "role": "time"}

    # numbers
    if pd.api.types.is_numeric_dtype(obj):
        num_ok, leading_zero = 1.0, False
    else:
        num_ok = float(to_number(sample).notna().mean())
        leading_zero = bool(sample.astype(str).str.strip().str.match(r"^0\d+$").any())
    if num_ok >= 0.9 and not leading_zero:
        conv = to_number(s.where(~miss))
        nums = conv.dropna()
        is_int = bool((nums == nums.round()).all())
        if re.search(r"date|day", name, re.I) and is_int and nums.min() > 20000 and nums.max() < 60000:
            dates = pd.to_datetime(conv, unit="D", origin="1899-12-30", errors="coerce")
            return dates.astype("datetime64[ns]"), {**info, "type": "date", "role": "time"}
        out = {**info, "type": "number"}
        bad = n - len(nums)
        if bad:
            out["warn"] = f"{bad:,} value{'s' if bad > 1 else ''} could not be read as numbers and were treated as blank"
        if re.search(r"(^|[^a-z])(id|key|code|zip|postal|phone)([^a-z]|$)", name, re.I) and is_int:
            out["is_id"] = True
        if re.search(r"year|month|quarter|week|^fy", name, re.I) and is_int and nums.nunique() <= 60:
            out["dim_like"] = True
            out["role"] = "time"
            conv = conv.astype("Int64")
        return conv, out

    # dates written as text
    strs = sample.astype(str).str.strip()
    if strs.str.match(DATE_RE).mean() >= 0.9:
        dayfirst = _detect_dayfirst(strs)
        conv = parse_dates(s.where(~miss), dayfirst)
        ok = conv.notna().sum() / n
        if ok >= 0.9:
            bad = int(n - conv.notna().sum())
            return conv, {**info, "type": "date", "role": "time", "date_order": "dmy" if dayfirst else "mdy",
                          "warn": f"{bad:,} values did not match the date format" if bad else None}

    # text
    conv = s.astype(object).where(~miss, None).map(lambda v: None if v is None else str(v).strip())
    nn = conv.dropna()
    u = nn.nunique()
    kind = "text"
    id_name = re.search(r"(^|[^a-z])(id|key|code|number|no|ref|uuid|guid|order|invoice|ticket)([^a-z]|$)", name, re.I)
    if u == len(nn) and len(nn) > 20 and (id_name or nn.head(50).str.contains(r"\d").all()):
        kind = "id"
    elif u <= max(30, min(1000, len(nn) * 0.35)):
        kind = "category"
    out = {**info, "type": kind}
    if 0.5 < num_ok < 0.9:
        out["warn"] = f"Mostly numbers, but {round((1 - num_ok) * 100)}% of values are text, so it is treated as text"
    return conv, out


def profile_column(info: dict, s: pd.Series) -> dict:
    p = dict(info)
    nn = s.dropna()
    p["count"], p["filled"] = len(s), len(nn)
    p["unique"] = int(nn.nunique())
    if info["type"] in ("number", "date") and len(nn):
        v = nn.astype("int64") if info["type"] == "date" else nn.astype(float)
        p.update(min=nn.min(), max=nn.max())
        if info["type"] == "number":
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            p.update(sum=float(v.sum()), mean=float(v.mean()), median=float(med), std=float(v.std() or 0),
                     q1=float(q1), q3=float(q3), lo=float(q1 - 1.5 * iqr), hi=float(q3 + 1.5 * iqr),
                     negatives=int((v < 0).sum()), zeros=int((v == 0).sum()))
            p["outliers"] = int(((v < p["lo"]) | (v > p["hi"])).sum()) if iqr > 0 else 0
        counts, _ = np.histogram(v, bins=16)
        p["hist"] = counts.tolist()
    if info["type"] == "bool":
        top = nn.map({True: "Yes", False: "No"}).value_counts()
    else:
        top = nn.astype(str).value_counts() if info["type"] != "date" else pd.Series(dtype=int)
    p["top"] = list(top.head(400).items())
    if info["type"] == "text" and len(nn):
        p["avg_len"] = float(nn.str.len().mean())
    return p


@dataclass
class Dataset:
    id: str
    name: str
    source: str
    detail: str
    df: pd.DataFrame
    cols: dict
    dups: int
    loaded_at: datetime
    size: int | None = None
    example: bool = False
    measures: list = field(default_factory=list)
    dims: list = field(default_factory=list)
    date_cols: list = field(default_factory=list)
    filter_dims: list = field(default_factory=list)
    primary: str | None = None
    quality: dict = field(default_factory=dict)


def _clean_names(cols) -> list[str]:
    out, seen = [], {}
    for i, c in enumerate(cols):
        name = str(c).strip() if c is not None and str(c).strip() and not str(c).startswith("Unnamed:") else f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 0
        out.append(name)
    return out


def build_dataset(ds_id: str, name: str, source: str, raw: pd.DataFrame, detail: str = "",
                  size: int | None = None, example: bool = False) -> Dataset:
    if raw is None or raw.empty:
        raise ValueError(f"No rows were found in {name}.")
    raw = raw.copy()
    raw.columns = _clean_names(raw.columns)
    raw = raw.reset_index(drop=True)
    cols, data = {}, {}
    for c in raw.columns:
        conv, info = infer_column(c, raw[c])
        if info["type"] == "empty" and re.match(r"^Column \d+$", c):
            continue
        data[c] = conv
        cols[c] = info
    df = pd.DataFrame(data)
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows were found in {name}.")
    for c in cols:
        cols[c] = profile_column(cols[c], df[c])
    dups = int(df.duplicated().sum())
    ds = Dataset(id=ds_id, name=name, source=source, detail=detail or source, df=df, cols=cols, dups=dups,
                 loaded_at=datetime.now(), size=size, example=example)
    ds.measures = [c for c, i in cols.items() if i["type"] == "number" and not i["is_id"] and not i["dim_like"]]
    ds.date_cols = [c for c, i in cols.items() if i["type"] == "date"]
    ds.dims = [c for c, i in cols.items() if i["type"] in ("category", "bool") or i["dim_like"]
               or (i["type"] == "text" and i["unique"] <= 2000)]
    fd = [c for c, i in cols.items() if (i["type"] in ("category", "bool") or i["dim_like"]) and 1 <= i["unique"] <= 400]
    ds.filter_dims = sorted(fd, key=lambda c: (ROLE_ORDER.get(cols[c]["role"], 6), cols[c]["unique"]))
    ds.primary = next((m for pat in MEASURE_PRIORITY for m in ds.measures if re.search(pat, m, re.I)),
                      ds.measures[0] if ds.measures else None)
    ds.quality = quality_report(ds)
    return ds


def fmt(v, compact: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (pd.Timestamp, datetime)):
        return f"{v.day} {v.strftime('%b %Y')}"
    if not isinstance(v, (int, float, np.integer, np.floating)):
        return str(v)
    a = abs(float(v))
    if compact and a >= 10_000:
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= div:
                return f"{v / div:,.1f}{suf}".replace(".0" + suf, suf)
    if a >= 100:
        return f"{v:,.0f}"
    return f"{v:,.2f}".rstrip("0").rstrip(".") if not float(v).is_integer() else f"{int(v):,}"


def pct(v, d: int = 1) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v * 100:.{d}f}%"


def quality_report(ds: Dataset) -> dict:
    rows, ncols = ds.df.shape
    cells = rows * ncols or 1
    missing = sum(i["count"] - i["filled"] for i in ds.cols.values())
    issues, score = [], 100.0
    rate = missing / cells
    score -= min(35, rate * 120)
    if missing:
        worst = sorted([i for i in ds.cols.values() if i["count"] > i["filled"]], key=lambda i: i["filled"] - i["count"])[:3]
        issues.append(("high" if rate > 0.1 else "check",
                       f"{pct(rate)} of all cells are blank. Most gaps: " +
                       ", ".join(f"{i['name']} ({pct((i['count'] - i['filled']) / i['count'], 0)})" for i in worst) + "."))
    if ds.dups:
        score -= min(20, ds.dups / rows * 100)
        issues.append(("check", f"{ds.dups:,} duplicate row{'s' if ds.dups > 1 else ''} ({pct(ds.dups / rows)}). "
                                "Check whether they are real repeats or double-loaded records."))
    constant = [i["name"] for i in ds.cols.values() if i["unique"] == 1 and i["filled"] > 1]
    if constant:
        score -= 2 * len(constant)
        issues.append(("note", f"{', '.join(constant)} {'have' if len(constant) > 1 else 'has'} a single value everywhere and add no information."))
    for i in ds.cols.values():
        if i.get("warn"):
            score -= 3
            issues.append(("check", f"{i['name']}: {i['warn']}."))
    for m in ds.measures:
        i = ds.cols[m]
        if i.get("outliers") and i["outliers"] / max(i["filled"], 1) > 0.002:
            issues.append(("note", f"{m} has {i['outliers']:,} outlier{'s' if i['outliers'] > 1 else ''} outside "
                                   f"{fmt(i['lo'], True)} – {fmt(i['hi'], True)} (1.5 × IQR rule)."))
        if i.get("negatives") and re.search(r"qty|quantity|units|volume|price", m, re.I):
            issues.append(("check", f"{m} contains {i['negatives']:,} negative values; these may be returns or entry errors."))
    empty = [i["name"] for i in ds.cols.values() if i["type"] == "empty"]
    if empty:
        score -= 2 * len(empty)
        issues.append(("check", f"{', '.join(empty)} {'are' if len(empty) > 1 else 'is'} completely empty."))
    if not issues:
        issues.append(("ok", "No blanks, duplicates or format problems found."))
    return {"score": max(0, round(score)), "issues": issues}
