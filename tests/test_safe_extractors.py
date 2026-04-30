from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.statsbomb_safe import safe_event_type, safe_location


def test_safe_location_variants() -> None:
    row_none = pd.Series({"location": None})
    x, y = safe_location(row_none)
    assert np.isnan(x) and np.isnan(y)

    row_empty = pd.Series({"location": []})
    x, y = safe_location(row_empty)
    assert np.isnan(x) and np.isnan(y)

    row_valid = pd.Series({"location": [10.0, 20.0]})
    x, y = safe_location(row_valid)
    assert x == 10.0 and y == 20.0

    row_malformed = pd.Series({"location": "bad"})
    x, y = safe_location(row_malformed)
    assert np.isnan(x) and np.isnan(y)


def test_safe_event_type_variants() -> None:
    row_dict = pd.Series({"type": {"name": "Pass"}})
    assert safe_event_type(row_dict) == "Pass"

    row_str = pd.Series({"type_name": "Carry"})
    assert safe_event_type(row_str) == "Carry"

    row_missing = pd.Series({})
    assert safe_event_type(row_missing) == "unknown"

