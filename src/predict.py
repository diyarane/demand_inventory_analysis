import os

import joblib
import numpy as np
import pandas as pd


def load_models(models_dir="models/") -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_full_dir = models_dir if os.path.isabs(models_dir) else os.path.join(project_root, models_dir)

    demand = joblib.load(os.path.join(models_full_dir, "demand_model.pkl"))
    sales = joblib.load(os.path.join(models_full_dir, "sales_model.pkl"))
    profit = joblib.load(os.path.join(models_full_dir, "profit_model.pkl"))
    scaler = joblib.load(os.path.join(models_full_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(models_full_dir, "feature_names.pkl"))

    return {
        "demand": demand,
        "sales": sales,
        "profit": profit,
        "scaler": scaler,
        "feature_names": feature_names,
    }


def _prepare(input_dict: dict, models: dict) -> np.ndarray:
    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=models["feature_names"], fill_value=0)
    return models["scaler"].transform(df)


def predict_demand(input_dict, models) -> float:
    x = _prepare(input_dict, models)
    return float(models["demand"].predict(x)[0])


def predict_sales(input_dict, models) -> float:
    x = _prepare(input_dict, models)
    return float(models["sales"].predict(x)[0])


def predict_profit(input_dict, models) -> float:
    x = _prepare(input_dict, models)
    return float(models["profit"].predict(x)[0])


def get_inventory_recommendation(predicted_qty: float) -> dict:
    predicted_qty = float(predicted_qty)
    return {
        "reorder_point": round(predicted_qty * 0.3, 2),
        "order_quantity": round(predicted_qty * 1.5, 2),
        "safety_stock": round(predicted_qty * 0.2, 2),
    }


def get_employee_performance(df: pd.DataFrame) -> list:
    """
    Detect employee column, build display name, compute contribution %.
    Returns list of dicts sorted by total_sales descending.
    """
    def _norm_name(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "Unknown"
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return "Unknown"
        s = " ".join(s.split())
        return s.lower().title()

    id_col = None
    for c in df.columns:
        if any(kw in c.lower() for kw in ["employee_id", "emp_id", "staff_id"]):
            id_col = c
            break

    name_col = None
    for c in df.columns:
        if any(kw in c.lower() for kw in ["employee_name", "emp_name", "staff_name", "name"]):
            name_col = c
            break

    if id_col is None:
        return []

    rev_col = "revenue" if "revenue" in df.columns else None
    if rev_col is None:
        return []

    work = df.copy()
    work[id_col] = work[id_col].astype(str).str.strip()
    if name_col:
        work[name_col] = work[name_col].map(_norm_name)
        work = work[work[name_col] != "Unknown"]

        name_mode = (
            work.groupby(id_col)[name_col]
            .agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else "Unknown")
            .rename("_canonical_name")
        )
        work = work.join(name_mode, on=id_col)
        work["_display"] = work[id_col] + " - " + work["_canonical_name"].astype(str)
    else:
        work["_display"] = work[id_col]

    agg = (
        work.groupby(id_col)
        .agg(
            total_sales=(rev_col, "sum"),
            avg_transaction=(rev_col, "mean"),
            total_orders=(rev_col, "count"),
            employee=("_display", lambda s: s.value_counts().index[0] if len(s) else ""),
        )
        .reset_index()
    )

    agg = agg.drop(columns=[id_col], errors="ignore")

    total = agg["total_sales"].sum()
    agg["contribution_pct"] = (agg["total_sales"] / total * 100).round(2) if total > 0 else 0.0

    agg["total_sales"] = agg["total_sales"].round(2)
    agg["avg_transaction"] = agg["avg_transaction"].round(2)

    return agg.sort_values("total_sales", ascending=False).to_dict(orient="records")