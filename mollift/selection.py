"""Reference-free selection using the validation-frozen component-RMSE window."""

import numpy as np

FF_PRIORITY = {"MMFF94s": 0, "MMFF94": 1, "UFF": 2}


def select_candidate(pipeline: str, coords, objectives=None, refiner_rows=None, tolerance=0.10):
    finite = np.isfinite(coords).all(axis=(1, 2))
    if pipeline == "G":
        return 0 if finite[0] else None
    if pipeline == "GS":
        eligible = [(float(v), i) for i, v in enumerate(objectives)
                    if finite[i] and np.isfinite(v)]
        return min(eligible)[1] if eligible else None
    if pipeline not in {"GR", "GSR"}:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    rows = refiner_rows or []
    eligible = [(i, row) for i, row in enumerate(rows)
                if finite[i] and row["accepted"]
                and row.get("energy") is not None and np.isfinite(row["energy"])
                and row.get("projection_rmse") is not None and np.isfinite(row["projection_rmse"])]
    if not eligible:
        return None
    best_projection = min(row["projection_rmse"] for _, row in eligible)
    feasible = [(i, row) for i, row in eligible
                if row["projection_rmse"] <= best_projection + tolerance]
    priority = min(FF_PRIORITY.get(row["force_field"], 99) for _, row in feasible)
    comparable = [(i, row) for i, row in feasible if FF_PRIORITY.get(row["force_field"], 99) == priority]
    return min(comparable, key=lambda pair: (pair[1]["energy"], pair[0]))[0]
