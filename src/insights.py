import numpy as np
import pandas as pd

from src.predict import predict_demand


DATE_CANDIDATES = ["date", "Date", "order_date", "invoice_date", "transaction_date"]


def _get_transactions_per_day(df: pd.DataFrame) -> float:
    """
    Calculate average number of transactions per calendar day from dataset.
    Try date columns in order: 'date','Date','order_date','invoice_date','transaction_date'.
    If no date column found, return len(df) / 365 as fallback.
    Returns float, minimum value 1.0 to avoid division by zero.
    """
    date_col = None
    for c in ["date", "Date", "order_date", "invoice_date", "transaction_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        return max(1.0, len(df) / 365)
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if len(dates) == 0:
        return max(1.0, len(df) / 365)
    end = dates.max()
    start = max(dates.min(), end - pd.Timedelta(days=59))
    recent = dates[(dates >= start) & (dates <= end)]
    if len(recent) == 0:
        recent = dates
    n_days = max(1, (recent.max() - recent.min()).days + 1)
    return max(1.0, len(recent) / n_days)


def _get_date_column(df: pd.DataFrame):
    for col in DATE_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _with_date(df: pd.DataFrame):
    out = df.copy()
    date_col = _get_date_column(out)
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).sort_values(date_col)
        out = out.rename(columns={date_col: "__date__"})
    else:
        out["__date__"] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(out), freq="D")
        out = out.sort_values("__date__")
    return out


def _round_num(x, digits=2):
    return round(float(x), digits)


def get_demand_trend(df, models, forecast_days=30) -> dict:
    work = _with_date(df)
    if "quantity" not in work.columns:
        raise ValueError("quantity column not found")
    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce").fillna(0)
    daily_qty = work.groupby("__date__")["quantity"].sum().sort_index()
    daily_rev = None
    if "revenue" in work.columns:
        work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce").fillna(0)
        daily_rev = work.groupby("__date__")["revenue"].sum().sort_index()
    daily_profit = None
    if "profit" in work.columns:
        work["profit"] = pd.to_numeric(work["profit"], errors="coerce").fillna(0)
        daily_profit = work.groupby("__date__")["profit"].sum().sort_index()
    historical = daily_qty.tail(60)
    hist_dates = historical.index.tolist()
    hist_vals = historical.values.tolist()

    if len(daily_qty) == 0:
        raise ValueError("no demand history available")

    last_date = daily_qty.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")

    base_features = work.drop(columns=["quantity", "revenue", "profit"], errors="ignore")
    date_features = [c for c in base_features.columns if str(c).lower() in [d.lower() for d in DATE_CANDIDATES] or c == "__date__"]
    base_features = base_features.drop(columns=date_features, errors="ignore")
    if base_features.empty:
        template = pd.Series(dtype=float)
    else:
        template = base_features.iloc[-1].copy()

    txns_per_day = _get_transactions_per_day(df)
    qty_series = daily_qty.astype(float).tolist()
    rev_series = daily_rev.astype(float).tolist() if daily_rev is not None else None
    profit_series = daily_profit.astype(float).tolist() if daily_profit is not None else None

    price_per_unit = 0.0
    if daily_rev is not None and len(daily_rev) and len(daily_qty):
        aligned = pd.DataFrame({"rev": daily_rev, "qty": daily_qty}).dropna()
        aligned = aligned[aligned["qty"] > 0]
        if len(aligned):
            price_per_unit = float((aligned["rev"] / aligned["qty"]).tail(60).median())

    profit_per_unit = 0.0
    if daily_profit is not None and len(daily_profit) and len(daily_qty):
        alignedp = pd.DataFrame({"profit": daily_profit, "qty": daily_qty}).dropna()
        alignedp = alignedp[alignedp["qty"] > 0]
        if len(alignedp):
            profit_per_unit = float((alignedp["profit"] / alignedp["qty"]).tail(60).median())

    rng = np.random.RandomState(42)
    rows = []
    daily_preds = []
    sales_preds = []
    profit_preds = []
    for _ in future_dates:
        row = template.copy()
        last_1 = float(qty_series[-1]) if len(qty_series) >= 1 else 0.0
        last_7 = qty_series[-7:] if len(qty_series) >= 7 else qty_series[:]
        last_7_mean = float(np.mean(last_7)) if len(last_7) else 0.0
        rev_last_1 = float(rev_series[-1]) if rev_series is not None and len(rev_series) >= 1 else None
        rev_last_7_mean = float(np.mean(rev_series[-7:])) if rev_series is not None and len(rev_series) >= 1 else None
        for c in row.index:
            cl = str(c).lower()
            if "quantity" in cl:
                if "lag_1" in cl:
                    row[c] = last_1
                elif "lag_7" in cl:
                    row[c] = last_7_mean
                elif "rolling_mean_7" in cl:
                    row[c] = last_7_mean
            elif ("revenue" in cl or "sales" in cl) and rev_last_1 is not None:
                if "lag_1" in cl:
                    row[c] = rev_last_1
                elif "lag_7" in cl:
                    row[c] = rev_last_7_mean
                elif "rolling_mean_7" in cl or "rolling_7d_avg" in cl:
                    row[c] = rev_last_7_mean
        row_df = pd.DataFrame([row]).reindex(columns=models["feature_names"], fill_value=0)
        scaled = models["scaler"].transform(row_df)
        per_txn_pred = float(models["demand"].predict(scaled)[0])
        base_daily = per_txn_pred * txns_per_day
        noise = float(rng.uniform(-0.05, 0.05))
        daily_pred = max(0.0, base_daily * (1.0 + noise))
        qty_series.append(daily_pred)
        daily_preds.append(round(float(daily_pred), 1))
        if price_per_unit > 0:
            sales_preds.append(round(float(daily_pred * price_per_unit), 2))
        if profit_per_unit != 0.0:
            profit_preds.append(round(float(daily_pred * profit_per_unit), 2))
        rows.append(row_df.iloc[0].to_dict())

    first_pred = float(daily_preds[0]) if len(daily_preds) else 0.0
    last_pred = float(daily_preds[-1]) if len(daily_preds) else 0.0
    if last_pred > first_pred * 1.02:
        trend_direction = "up"
    elif last_pred < first_pred * 0.98:
        trend_direction = "down"
    else:
        trend_direction = "stable"

    peak_idx = int(np.argmax(daily_preds)) if len(daily_preds) else 0
    peak_date = str(future_dates[peak_idx].date()) if len(future_dates) else ""
    peak_qty = float(daily_preds[peak_idx]) if len(daily_preds) else 0.0

    out = {
        "historical": [{"date": str(d.date()), "quantity": float(q)} for d, q in zip(hist_dates, hist_vals)],
        "forecast": [{"date": str(d.date()), "quantity": float(q)} for d, q in zip(future_dates, daily_preds)],
        "trend_direction": trend_direction,
        "forecast_peak_date": peak_date,
        "forecast_peak_qty": peak_qty,
        "transactions_per_day": round(txns_per_day, 1),
        "scale_note": "Forecast values represent estimated daily demand (units/day)",
        "method_note": "Forecast is based on historical trends using lag features and rolling averages",
    }
    if sales_preds:
        out["sales_forecast"] = [{"date": str(d.date()), "revenue": float(v)} for d, v in zip(future_dates, sales_preds)]
        out["effective_price"] = round(float(price_per_unit), 2)
    if profit_preds:
        out["profit_forecast"] = [{"date": str(d.date()), "profit": float(v)} for d, v in zip(future_dates, profit_preds)]
        out["profit_per_unit_est"] = round(float(profit_per_unit), 2)
    return out


def get_kpi_summary(df, models) -> dict:
    work = _with_date(df)
    revenue = pd.to_numeric(work["revenue"], errors="coerce").fillna(0) if "revenue" in work.columns else pd.Series([0] * len(work))
    quantity = pd.to_numeric(work["quantity"], errors="coerce").fillna(0) if "quantity" in work.columns else pd.Series([0] * len(work))
    profit = pd.to_numeric(work["profit"], errors="coerce").fillna(0) if "profit" in work.columns else pd.Series([0] * len(work))

    total_revenue = float(revenue.sum()) if "revenue" in work.columns else 0.0
    total_orders = int(len(work))
    avg_order_value = float(revenue.mean()) if "revenue" in work.columns and len(work) > 0 else 0.0
    total_profit = float(profit.sum()) if "profit" in work.columns else 0.0
    profit_margin_pct = (total_profit / total_revenue * 100.0) if total_revenue > 0 else 0.0

    daily = pd.DataFrame({"__date__": work["__date__"], "revenue": revenue, "quantity": quantity}).groupby("__date__", as_index=False).sum()
    daily = daily.sort_values("__date__")
    last_30 = daily.tail(30)
    prev_30 = daily.iloc[max(0, len(daily) - 60):max(0, len(daily) - 30)]

    last_rev = float(last_30["revenue"].sum()) if len(last_30) else 0.0
    prev_rev = float(prev_30["revenue"].sum()) if len(prev_30) else 0.0
    revenue_growth_pct = ((last_rev - prev_rev) / prev_rev * 100.0) if prev_rev > 0 else (100.0 if last_rev > 0 else 0.0)

    last_qty = float(last_30["quantity"].sum()) if len(last_30) else 0.0
    prev_qty = float(prev_30["quantity"].sum()) if len(prev_30) else 0.0
    demand_growth_pct = ((last_qty - prev_qty) / prev_qty * 100.0) if prev_qty > 0 else (100.0 if last_qty > 0 else 0.0)

    fi = np.array(models["demand"].feature_importances_, dtype=float)
    names = list(models["feature_names"])
    order = np.argsort(fi)[::-1][:3]
    top_factors = [{"name": str(names[i]), "score": round(float(fi[i]), 4)} for i in order]

    return {
        "total_revenue": _round_num(total_revenue),
        "total_orders": total_orders,
        "avg_order_value": _round_num(avg_order_value),
        "total_profit": _round_num(total_profit),
        "profit_margin_pct": _round_num(profit_margin_pct),
        "revenue_growth_pct": _round_num(revenue_growth_pct),
        "demand_growth_pct": _round_num(demand_growth_pct),
        "top_factors": top_factors,
    }


def get_inventory_intelligence(df, models) -> list:
    txns_per_day = _get_transactions_per_day(df)
    product_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "product" in cl or "item" in cl or "sku" in cl or "category" in cl:
            product_col = c
            break
    if product_col is None:
        return [{"message": "No product column detected"}]
    if "quantity" not in df.columns:
        return [{"message": "quantity column not found"}]

    out = []
    unique_products = df[product_col].dropna().astype(str).unique().tolist()[:20]
    for p in unique_products:
        pdf = df[df[product_col].astype(str) == p].copy()
        if len(pdf) == 0:
            continue
        pdf["quantity"] = pd.to_numeric(pdf["quantity"], errors="coerce").fillna(0)
        current_stock = float(pdf["quantity"].sum())
        feature_row = pdf.drop(columns=["quantity", "revenue", "profit"], errors="ignore").iloc[-1].to_dict()
        avg_units_per_txn = float(predict_demand(feature_row, models))
        daily_predicted_demand = round(avg_units_per_txn * txns_per_day, 1)
        weekly_predicted_demand = round(daily_predicted_demand * 7, 1)

        if daily_predicted_demand > 0:
            days_of_inventory = round(current_stock / daily_predicted_demand, 1)
        else:
            days_of_inventory = 999

        if days_of_inventory > 30:
            status = "overstock"
        elif days_of_inventory < 10:
            status = "understock"
        else:
            status = "optimal"

        reason = f"Stock covers only {days_of_inventory} days based on predicted demand" if status == "understock" else f"Stock covers {days_of_inventory} days based on predicted demand"
        action = reason
        out.append(
            {
                "product": str(p),
                "current_stock": round(current_stock, 2),
                "avg_units_per_transaction": round(avg_units_per_txn, 2),
                "daily_predicted_demand": daily_predicted_demand,
                "weekly_predicted_demand": weekly_predicted_demand,
                "days_of_inventory": days_of_inventory,
                "action": action,
                "status": status,
                "reason": reason,
            }
        )
    return out
