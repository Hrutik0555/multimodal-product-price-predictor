"""Regex-based extraction of structured fields from product text."""
import re

import numpy as np
import pandas as pd

_VALUE_RE = re.compile(r"Value:\s*([\d.,]+)", re.IGNORECASE)
_UNIT_RE = re.compile(r"Unit:\s*([A-Za-z ]+)", re.IGNORECASE)
_PACK_RES = [
    re.compile(r"pack\s*of\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*[- ]?(?:pack|count|ct|pcs|pieces)\b", re.IGNORECASE),
    re.compile(r"\bcase\s*of\s*(\d+)", re.IGNORECASE),
]
_BULLET_RE = re.compile(r"bullet\s*point", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# unit -> (kind, factor to base unit: grams / millilitres / count)
_UNIT_TABLE = {
    "gram": ("mass", 1.0), "g": ("mass", 1.0),
    "kilogram": ("mass", 1000.0), "kg": ("mass", 1000.0),
    "ounce": ("mass", 28.3495), "oz": ("mass", 28.3495),
    "pound": ("mass", 453.592), "lb": ("mass", 453.592),
    "milliliter": ("volume", 1.0), "ml": ("volume", 1.0),
    "liter": ("volume", 1000.0), "litre": ("volume", 1000.0), "l": ("volume", 1000.0),
    "fluid ounce": ("volume", 29.5735), "fl oz": ("volume", 29.5735),
    "count": ("count", 1.0), "each": ("count", 1.0), "piece": ("count", 1.0),
}

TEXT_FEATURE_NAMES = [
    "value", "value_base", "is_mass", "is_volume", "is_count", "pack_qty",
    "text_len", "word_count", "digit_count", "upper_ratio", "bullet_count",
    "number_count", "has_value",
]


def _parse_value(text):
    m = _VALUE_RE.search(text)
    if not m:
        return np.nan
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return np.nan


def _parse_pack(text):
    for rx in _PACK_RES:
        m = rx.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return 1.0


def extract_text_features(texts):
    """Return a (n, len(TEXT_FEATURE_NAMES)) float array."""
    rows = []
    for raw in pd.Series(texts).fillna("").astype(str):
        value = _parse_value(raw)
        unit_match = _UNIT_RE.search(raw)
        unit = unit_match.group(1).strip().lower() if unit_match else ""
        kind, factor = _UNIT_TABLE.get(unit, ("", 1.0))
        has_value = float(not np.isnan(value))
        value_clean = 0.0 if np.isnan(value) else value
        letters = sum(c.isalpha() for c in raw)
        rows.append([
            value_clean,
            value_clean * factor,
            float(kind == "mass"),
            float(kind == "volume"),
            float(kind == "count"),
            _parse_pack(raw),
            float(len(raw)),
            float(len(raw.split())),
            float(sum(c.isdigit() for c in raw)),
            sum(c.isupper() for c in raw) / letters if letters else 0.0,
            float(len(_BULLET_RE.findall(raw))),
            float(len(_NUM_RE.findall(raw))),
            has_value,
        ])
    return np.asarray(rows, dtype=np.float64)
