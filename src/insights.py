import numpy as np
import pandas as pd

from src.predict import predict_demand


DATE_CANDIDATES = ["date", "Date", "order_date", "invoice_date", "transaction_date"]

# FESTIVAL DATES
FESTIVAL_DATES = pd.to_datetime([
    "2025-03-12",
    "2025-03-13",
    "2025-03-14",
    "2025-03-15"
])


# ─────────────────────────────────────────────────────────────
# GAUSSIAN FESTIVAL SPIKE
# ─────────────────────────────────────────────────────────────
def festival_boost(date):
    boost = 0
    for f in FESTIVAL_DATES:
        dist = abs((date - f).days)
        boost += np.exp(-(dist**2) / (2 * 1.5**2))
    return 1 + 0.8 * boost


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _get_date_column(df):
    for col in DATE_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _with_date(df):
    out = df.copy()
    col = _get_date_column(out)

    if col:
        out[col] = pd.to_datetime(out[col], errors="coerce")
        out = out.dropna(subset=[col]).sort_values(col)
        out = out.rename(columns={col: "__date__"})
    else:
        out["__date__"] = pd.date_range(end=pd.Timestamp.today(), periods=len(out))

    return out


def _get_txn_per_day(df):
    return max(1.0, len(df) / 60)


# ─────────────────────────────────────────────────────────────
# 🔥 FIXED FORECAST ENGINE
# ─────────────────────────────────────────────────────────────
def _run_forecast(daily_qty, daily_rev, daily_profit,
                  feat_matrix, models, txns_per_day, future_dates):

    rng_d = np.random.RandomState(42)
    rng_price = np.random.RandomState(7)
    rng_cost = np.random.RandomState(99)

    safe_qty = np.where(daily_qty > 0, daily_qty, 1)
    base_price = np.median(daily_rev / safe_qty)
    base_cost = np.median((daily_rev - daily_profit) / safe_qty)

    qty_preds, rev_preds, prof_preds = [], [], []

    for i, date in enumerate(future_dates):

        row = feat_matrix.iloc[i % len(feat_matrix)].copy()
        scaled = models["scaler"].transform(
            pd.DataFrame([row]).reindex(columns=models["feature_names"], fill_value=0)
        )

        base_qty = float(models["demand"].predict(scaled)[0])

        # 🔥 Increased variance
        seasonal = 1 + 0.25 * np.sin(i / 2)
        noise = 1 + rng_d.uniform(-0.15, 0.15)

        demand = base_qty * seasonal * noise
        demand *= festival_boost(date)
        demand = max(0, demand * txns_per_day)

        # SALES
        price = base_price * (1 + rng_price.uniform(-0.1, 0.1))
        price *= (1 + 0.1 * np.sin(i))
        sales = demand * price

        # PROFIT (decoupled)
        cost = base_cost * (1 + rng_cost.uniform(-0.08, 0.08))
        cost *= (1 + 0.05 * np.cos(i))
        profit = sales - (cost * demand)
        profit *= festival_boost(date)

        qty_preds.append(round(demand, 1))
        rev_preds.append(round(sales, 2))
        prof_preds.append(round(max(0, profit), 2))

    return qty_preds, rev_preds, prof_preds


# ─────────────────────────────────────────────────────────────
# MAIN FORECAST FUNCTION
# ─────────────────────────────────────────────────────────────
def get_demand_trend(df, models, forecast_days=30):

    work = _with_date(df)

    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce").fillna(0)
    work["revenue"] = pd.to_numeric(work.get("revenue", 0), errors="coerce").fillna(0)
    work["profit"] = pd.to_numeric(work.get("profit", 0), errors="coerce").fillna(0)

    daily = work.groupby("__date__").sum().sort_index()

    daily_qty = daily["quantity"]
    daily_rev = daily["revenue"]
    daily_profit = daily["profit"]

    txns_per_day = _get_txn_per_day(df)

    last_date = daily.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    base = work.drop(columns=["quantity", "revenue", "profit"], errors="ignore").tail(90)
    feat_matrix = base.reindex(columns=models["feature_names"], fill_value=0)

    qty, rev, prof = _run_forecast(
        daily_qty, daily_rev, daily_profit,
        feat_matrix, models, txns_per_day, future_dates
    )

    hist = daily.tail(60)

    return {
        "historical": [
            {"date": str(d.date()), "quantity": float(q)}
            for d, q in zip(hist.index, hist["quantity"])
        ],
        "forecast": [
            {"date": str(d.date()), "quantity": float(q)}
            for d, q in zip(future_dates, qty)
        ],
        "sales_forecast": [
            {"date": str(d.date()), "revenue": float(r)}
            for d, r in zip(future_dates, rev)
        ],
        "profit_forecast": [
            {"date": str(d.date()), "profit": float(p)}
            for d, p in zip(future_dates, prof)
        ],
    }


# ─────────────────────────────────────────────────────────────
# KPI SUMMARY (RESTORED)
# ─────────────────────────────────────────────────────────────
def get_kpi_summary(df, models):

    revenue = pd.to_numeric(df.get("revenue", 0), errors="coerce").fillna(0)
    quantity = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
    profit = pd.to_numeric(df.get("profit", 0), errors="coerce").fillna(0)

    return {
        "total_revenue": round(float(revenue.sum()), 2),
        "total_orders": int(len(df)),
        "avg_order_value": round(float(revenue.mean()), 2),
        "total_profit": round(float(profit.sum()), 2),
        "profit_margin_pct": round((profit.sum() / revenue.sum() * 100) if revenue.sum() else 0, 2),
    }


# ─────────────────────────────────────────────────────────────
# INVENTORY (RESTORED)
# ─────────────────────────────────────────────────────────────
def get_inventory_intelligence(df, models):

    if "quantity" not in df.columns:
        return []

    avg_demand = df["quantity"].mean()

    return [{
        "daily_predicted_demand": round(avg_demand, 2),
        "status": "stable",
        "action": "Monitor inventory levels"
    }]