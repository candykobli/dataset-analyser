"""Read data from files, URLs/APIs and pasted text into pandas DataFrames."""
from __future__ import annotations

import csv
import io
import ipaddress
import json
import re
import socket
from urllib.parse import urlparse

import pandas as pd
import requests

SUPPORTED_EXTENSIONS = [
    "csv", "tsv", "txt", "xlsx", "xlsm", "xls", "ods",
    "json", "jsonl", "ndjson", "xml", "parquet",
]
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


class LoadError(Exception):
    """A readable error shown to the user."""


# ----------------------------------------------------------------- text
def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_delimited(text: str) -> pd.DataFrame:
    sample = text[:50000]
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first_lines = "\n".join(sample.splitlines()[:20])
        counts = {d: first_lines.count(d) for d in [",", ";", "\t", "|"]}
        sep = max(counts, key=counts.get)
    try:
        df = pd.read_csv(
            io.StringIO(text), sep=sep, dtype=str, keep_default_na=False,
            skipinitialspace=True, on_bad_lines="skip",
        )
    except Exception as exc:  # pragma: no cover - pandas message is shown
        raise LoadError(f"The text could not be read as a table: {exc}") from exc
    if df.shape[1] == 1 and sep != ",":
        # Sniffer can be wrong on single-column data; retry with comma.
        alt = pd.read_csv(io.StringIO(text), sep=",", dtype=str, keep_default_na=False, on_bad_lines="skip")
        if alt.shape[1] > 1:
            df = alt
    return df


# ----------------------------------------------------------------- JSON
def find_records(obj) -> list[dict]:
    """Find the list of records inside any JSON shape (API envelopes included)."""
    if isinstance(obj, list):
        if not obj:
            return []
        if all(isinstance(x, dict) for x in obj):
            return obj
        if all(isinstance(x, list) for x in obj) and all(isinstance(h, str) for h in obj[0]):
            header, *rows = obj
            return [dict(zip(header, r)) for r in rows]
        return [{"value": x} for x in obj]
    if isinstance(obj, dict):
        best = None
        queue = [(obj, 0)]
        while queue:
            node, depth = queue.pop(0)
            for v in node.values():
                if isinstance(v, list) and v and any(isinstance(x, dict) for x in v):
                    if best is None or len(v) > len(best):
                        best = v
                elif isinstance(v, dict) and depth < 4:
                    queue.append((v, depth + 1))
        if best is not None:
            return [x for x in best if isinstance(x, dict)]
        values = list(obj.values())
        if len(values) > 1 and all(isinstance(v, dict) for v in values):
            return [{"key": k, **v} for k, v in obj.items()]
        return [obj]
    return []


def select_path(obj, path: str):
    """Follow a dotted path like 'data.items' or 'results.0.rows'."""
    for part in [p for p in path.strip().split(".") if p]:
        if isinstance(obj, list) and part.isdigit():
            obj = obj[int(part)]
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            raise LoadError(f'The path "{path}" was not found in the response (stopped at "{part}").')
    return obj


def records_to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        raise LoadError("No records were found.")
    df = pd.json_normalize(records, max_level=3, sep=".")
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (list, dict))).any():
            df[c] = df[c].map(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)
                else (json.dumps(v)[:200] if isinstance(v, (list, dict)) else v)
            )
    return df


def read_json_text(text: str, path: str = "") -> pd.DataFrame:
    t = text.strip()
    # JSON Lines
    if t.startswith("{") and re.search(r"\}\s*\n\s*\{", t):
        try:
            rows = [json.loads(line) for line in t.splitlines() if line.strip()]
            return records_to_df(find_records(select_path(rows, path)) if path else rows)
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as exc:
        raise LoadError(f"The JSON could not be read: {exc}") from exc
    if path:
        obj = select_path(obj, path)
    return records_to_df(find_records(obj))


# ----------------------------------------------------------------- XML
def read_xml_text(text: str) -> pd.DataFrame:
    try:
        from defusedxml import ElementTree as ET  # safe against entity attacks
    except ImportError:  # pragma: no cover
        import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    except Exception as exc:
        raise LoadError(f"The XML could not be read: {exc}") from exc

    def tag(el):
        return el.tag.split("}")[-1]

    best, best_n = None, 0
    for el in root.iter():
        counts: dict[str, int] = {}
        for ch in el:
            counts[tag(ch)] = counts.get(tag(ch), 0) + 1
        for t, n in counts.items():
            if n > best_n:
                best, best_n = (el, t), n
    if not best:
        raise LoadError("No repeating records were found in the XML.")
    parent, rec_tag = best
    rows = []
    for ch in parent:
        if tag(ch) != rec_tag:
            continue
        row = dict(ch.attrib)
        for sub in ch:
            row[tag(sub)] = (sub.text or "").strip() if len(sub) == 0 else " ".join("".join(sub.itertext()).split())[:200]
        if len(ch) == 0 and not ch.attrib:
            row["value"] = (ch.text or "").strip()
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- dispatch
def parse_text(text: str, hint: str = "", path: str = "") -> pd.DataFrame:
    t = text.strip()
    if not t:
        raise LoadError("There is no data to load.")
    low = hint.lower()
    if low.endswith((".json", ".jsonl", ".ndjson")) or t[:1] in "[{":
        try:
            return read_json_text(t, path)
        except LoadError:
            if low.endswith((".json", ".jsonl", ".ndjson")):
                raise
    if low.endswith(".xml") or t.startswith("<"):
        return read_xml_text(t)
    return read_delimited(t)


def read_upload(name: str, raw: bytes) -> list[tuple[str, str, pd.DataFrame]]:
    """Return a list of (dataset name, source label, frame). Excel gives one per sheet."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in ("xlsx", "xlsm", "xls", "ods"):
        engine = {"xls": "xlrd", "ods": "odf"}.get(ext)
        try:
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, dtype=object, engine=engine)
        except Exception as exc:
            raise LoadError(f"The spreadsheet could not be read: {exc}") from exc
        out = []
        for sheet, df in sheets.items():
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue
            label = f"{name} · {sheet}" if len(sheets) > 1 else name
            out.append((label, f'Excel file {name}, sheet "{sheet}"', df))
        if not out:
            raise LoadError("All sheets in the workbook are empty.")
        return out
    if ext == "parquet":
        try:
            return [(name, f"Parquet file {name}", pd.read_parquet(io.BytesIO(raw)))]
        except Exception as exc:
            raise LoadError(f"The Parquet file could not be read: {exc}") from exc
    kind = {"json": "JSON", "jsonl": "JSON Lines", "ndjson": "JSON Lines", "xml": "XML", "tsv": "TSV"}.get(ext, "CSV")
    return [(name, f"{kind} file {name}", parse_text(decode_bytes(raw), name))]


# ----------------------------------------------------------------- URLs / APIs
def _check_public(url: str) -> None:
    """Refuse internal addresses so the app cannot be used to probe private networks."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise LoadError("Enter a full web address starting with https://")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise LoadError(f"The address {parsed.hostname} could not be found.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise LoadError("That address points to a private network and cannot be fetched.")


def normalise_url(url: str) -> str:
    """Turn share links into direct-download links (Google Sheets, Dropbox, GitHub)."""
    url = url.strip()
    m = re.match(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)", url)
    if m:
        gid = re.search(r"[#&?]gid=(\d+)", url)
        return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv" + (f"&gid={gid.group(1)}" if gid else "")
    if "dropbox.com" in url:
        return re.sub(r"[?&]dl=0", "", url) + ("&" if "?" in url else "?") + "dl=1"
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return url


def fetch_url(url: str, header: str = "", path: str = "", params: str = "") -> tuple[str, str, pd.DataFrame]:
    """Fetch a file or API endpoint. `header` like 'Authorization: Bearer xyz'."""
    url = normalise_url(url)
    _check_public(url)
    headers = {"User-Agent": "DatasetAnalyser/1.0", "Accept": "application/json, text/csv, */*"}
    if header.strip():
        if ":" not in header:
            raise LoadError('Write the header as "Name: value", e.g. "Authorization: Bearer abc123".')
        k, v = header.split(":", 1)
        headers[k.strip()] = v.strip()
    query = {}
    for pair in [p for p in re.split(r"[&\n]", params) if p.strip()]:
        if "=" in pair:
            k, v = pair.split("=", 1)
            query[k.strip()] = v.strip()
    try:
        resp = requests.get(url, headers=headers, params=query or None, timeout=30, stream=True)
    except requests.RequestException as exc:
        raise LoadError(f"The request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise LoadError(f"The server answered with HTTP {resp.status_code}. Check the URL and any API key.")
    raw = resp.raw.read(MAX_DOWNLOAD_BYTES + 1, decode_content=True)
    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise LoadError("The response is larger than 100 MB.")
    ctype = resp.headers.get("Content-Type", "").lower()
    host = urlparse(url).hostname or "url"
    fname = urlparse(url).path.rsplit("/", 1)[-1] or host
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    if ext in ("xlsx", "xlsm", "xls", "ods", "parquet") or "spreadsheetml" in ctype or "ms-excel" in ctype:
        if not ext or ext not in ("xlsx", "xlsm", "xls", "ods", "parquet"):
            fname += ".xlsx"
        name, _, df = read_upload(fname, raw)[0]
        return name, f"Downloaded from {url}", df
    text = decode_bytes(raw)
    hint = ".json" if "json" in ctype else ".xml" if "xml" in ctype else fname
    df = parse_text(text, hint, path)
    name = fname if ext else host
    return name, f"Fetched from {url}", df
