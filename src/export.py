import datetime
import io
from io import BytesIO

import pandas as pd
from flask import send_file
import openpyxl

from src.insights import get_demand_trend, _with_date
from src.predict import get_employee_performance, predict_demand


def df_to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    writer = pd.ExcelWriter(buf, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name="Data")
    writer.close()
    buf.seek(0)
    return buf


def build_forecast_export(df, models) -> BytesIO:
    trend = get_demand_trend(df, models, forecast_days=30)
    actual_df = pd.DataFrame(trend["historical"])
    actual_df["type"] = "actual"
    forecast_df = pd.DataFrame(trend["forecast"])
    forecast_df["type"] = "forecast"
    out = pd.concat([actual_df, forecast_df], ignore_index=True)
    out = out[["date", "type", "quantity"]]
    return df_to_excel_bytes(out)


def build_combined_export(df, models) -> BytesIO:
    trend = get_demand_trend(df, models, forecast_days=30)
    historical_dates = [h["date"] for h in trend["historical"]]
    txns_per_day = float(trend.get("transactions_per_day") or 1.0)
    w = _with_date(df)
    w["__date_str__"] = w["__date__"].dt.date.astype(str)
    hist_df = w[w["__date_str__"].isin(historical_dates)].copy()
    hist_df = hist_df.sort_values("__date__")

    feature_df = hist_df.drop(columns=["quantity", "revenue", "profit", "__date__", "__date_str__"], errors="ignore")
    feature_df = feature_df.reindex(columns=models["feature_names"], fill_value=0)
    scaled = models["scaler"].transform(feature_df)
    pred_vals = models["demand"].predict(scaled).astype(float).tolist()

    grouped_actual = (
        hist_df.groupby("__date__", as_index=False)
        .agg(
            actual_quantity=("quantity", "sum"),
            actual_revenue=("revenue", "sum") if "revenue" in hist_df.columns else ("quantity", "size"),
        )
        .sort_values("__date__")
    )
    if "revenue" not in hist_df.columns:
        grouped_actual["actual_revenue"] = 0.0

    pred_series = (
        pd.DataFrame({"__date__": hist_df["__date__"], "predicted_per_transaction": pred_vals})
        .groupby("__date__", as_index=False)
        .mean()
    )
    pred_series["predicted_quantity"] = (pred_series["predicted_per_transaction"] * txns_per_day).round(2)
    pred_series = pred_series.drop(columns=["predicted_per_transaction"])
    out = grouped_actual.merge(pred_series, on="__date__", how="left")
    out["date"] = out["__date__"].dt.date.astype(str)
    out = out[["date", "actual_quantity", "predicted_quantity", "actual_revenue"]]
    return df_to_excel_bytes(out)


def build_employee_export(df) -> BytesIO:
    rows = get_employee_performance(df)
    out = pd.DataFrame(rows)
    return df_to_excel_bytes(out)


def build_inventory_export(df, models) -> BytesIO:
    from src.insights import get_inventory_intelligence

    rows = get_inventory_intelligence(df, models)
    out = pd.DataFrame(rows)
    return df_to_excel_bytes(out)
