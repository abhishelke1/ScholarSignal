"""Strict input contracts and fixed, interpretable categorical encoding."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Needs support", "On track", "Strong"]
BASE_FEATURES = ["studytime", "failures", "absences", "traveltime", "schoolsup", "famsup", "activities", "higher", "internet"]
NUMERIC = {"studytime": (1, 4), "failures": (0, 3), "absences": (0, 93), "traveltime": (1, 4), "G1": (0, 20), "G2": (0, 20), "G3": (0, 20)}
BINARY = ["schoolsup", "famsup", "activities", "higher", "internet"]
DISPLAY = {"G1": "First-period grade", "G2": "Second-period grade", "studytime": "Weekly study time", "failures": "Past class failures", "absences": "Absences", "traveltime": "Travel time", "schoolsup": "School support", "famsup": "Family support", "activities": "Extracurricular activities", "higher": "Plans higher education", "internet": "Internet access"}


class DataError(ValueError):
    """An actionable user-facing validation error."""


def features(mode: str) -> list[str]:
    if mode not in ("With prior grades", "Early warning"):
        raise DataError("Choose a supported prediction mode.")
    return BASE_FEATURES + (["G1", "G2"] if mode == "With prior grades" else [])


def read_csv(payload: bytes) -> pd.DataFrame:
    if len(payload) > 5 * 1024 * 1024:
        raise DataError("CSV must be smaller than 5 MB.")
    try:
        text = payload.decode("utf-8-sig")
        first = text.splitlines()[0]
        delimiter = ";" if first.count(";") > first.count(",") else ","
        headers = [h.strip() for h in next(csv.reader(io.StringIO(text), delimiter=delimiter))]
        if len(headers) != len(set(headers)):
            raise DataError("CSV contains duplicate column names.")
        frame = pd.read_csv(io.StringIO(text), sep=delimiter)
        frame.columns = frame.columns.str.strip()
        return frame
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, IndexError, StopIteration) as exc:
        raise DataError("Upload a non-empty UTF-8 CSV with comma or semicolon separators.") from exc


def load_demo() -> pd.DataFrame:
    return read_csv((ROOT / "data" / "student-mat.csv").read_bytes())


def validate(frame: pd.DataFrame, mode: str, training: bool = False) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty:
        raise DataError("The dataset has no student rows.")
    if len(frame) > 10000:
        raise DataError("Use at most 10,000 rows per run.")
    required = features(mode) + (["G3"] if training else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataError("Missing required columns: " + ", ".join(missing))
    result = frame[required].copy().reset_index(drop=True)
    notes = []
    for col in required:
        raw = result[col].map(lambda value: np.nan if isinstance(value, str) and not value.strip() else value)
        if col in NUMERIC:
            converted = pd.to_numeric(raw, errors="coerce")
            invalid = raw.notna() & converted.isna()
            lo, hi = NUMERIC[col]
            invalid |= converted.notna() & (~np.isfinite(converted) | (converted < lo) | (converted > hi) | (converted % 1 != 0))
            if invalid.any():
                rows = ", ".join(str(int(i) + 2) for i in invalid[invalid].index[:5])
                raise DataError(f"{col} must contain whole numbers from {lo} to {hi}. Check CSV lines {rows}.")
            result[col] = converted
        else:
            normalized = raw.astype("string").str.strip().str.lower()
            if (~normalized.isin(["yes", "no"]) & normalized.notna()).any():
                raise DataError(f"{col} must be yes, no, or blank.")
            result[col] = normalized
        count = int(result[col].isna().sum())
        if count:
            if col == "G3":
                raise DataError("G3 is the training target and cannot be blank.")
            if training and count == len(result):
                raise DataError(f"{col} is entirely blank; provide observations for training.")
            notes.append(f"{col}: {count} missing value(s), encoded as an explicit unknown category.")
    if result[features(mode)].isna().all(axis=1).any():
        raise DataError("A student row has no observed features. Provide at least one feature value.")
    if training:
        if len(result) < 60:
            raise DataError("Training needs at least 60 rows and 10 examples in each grade band.")
        counts = targets(result["G3"]).value_counts().reindex(LABELS, fill_value=0)
        if (counts < 10).any():
            raise DataError("Training needs at least 10 rows in every band: G3 0–9, 10–13, and 14–20.")
    return result, notes


def targets(grades: pd.Series) -> pd.Series:
    return pd.Series(np.select([grades < 10, grades < 14], LABELS[:2], default=LABELS[2]), index=grades.index, name="Performance")


def encode(frame: pd.DataFrame, mode: str) -> tuple[np.ndarray, list[int]]:
    """Fixed bins use no fitted statistics, so they cannot leak test observations.

    The final category of every feature represents a missing observation.
    """
    columns, sizes = [], []
    for col in features(mode):
        values = frame[col]
        if col in BINARY:
            encoded, n = values.map({"no": 0, "yes": 1}).fillna(2), 3
        elif col in ("G1", "G2"):
            encoded, n = pd.Series(np.select([values < 10, values < 14], [0, 1], default=2), index=values.index), 4
        elif col == "absences":
            encoded, n = pd.Series(np.select([values == 0, values <= 5, values <= 15], [0, 1, 2], default=3), index=values.index), 5
        else:
            encoded, n = values - NUMERIC[col][0], 5
        encoded = encoded.where(values.notna(), n - 1)
        columns.append(encoded.to_numpy(dtype=int))
        sizes.append(n)
    return np.column_stack(columns), sizes


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Neutralize spreadsheet formulas in user-provided text when exporting CSV."""
    safe = frame.copy()
    for col in safe.select_dtypes(include=["object", "string"]).columns:
        safe[col] = safe[col].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
    return safe.to_csv(index=False).encode("utf-8-sig")
