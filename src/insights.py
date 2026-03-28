import numpy as np
import pandas as pd

from src.predict import predict_demand

DATE_CANDIDATES = ["date", "Date", "order_date", "invoice_date", "transaction_date"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_date_column(df: pd.DataFrame):
    for col in DATE_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_col = _get_date_column(out)
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).sort_values(date_col)
        out = out.rename(columns={date_col: "__date__"})
    else:
        out["__date__"] = pd.date_range(
            end=pd.Timestamp.today().normalize(), periods=len(out), freq="D"
        )
        out = out.sort_values("__date__")
    return out


def _get_transactions_per_day(df: pd.DataFrame) -> float:
    date_col = _get_date_column(df)
    if date_col is None:
        return max(1.0, len(df) / 365)
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if len(dates) == 0:
        return max(1.0, len(df) / 365)
    end   = dates.max()
    start = max(dates.min(), end - pd.Timedelta(days=59))
    recent = dates[(dates >= start) & (dates <= end)]
    if len(recent) == 0:
        recent = dates
    n_days = max(1, (recent.max() - recent.min()).days + 1)
    return max(1.0, len(recent) / n_days)


def _round_num(x, digits=2):
    return round(float(x), digits)


def _get_feature_matrix(work: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Return the processed feature matrix from the last 90 rows of actual data,
    aligned to model feature names. Used for sampling during forecast.
    """
    date_drop = [c for c in work.columns if c in DATE_CANDIDATES or c == "__date__"]
    base = work.drop(
        columns=["quantity", "revenue", "profit"] + date_drop, errors="ignore"
    ).tail(90)
    if base.empty:
        return pd.DataFrame(
            np.zeros((1, len(models["feature_names"]))),
            columns=models["feature_names"]
        )
    return base.reindex(columns=models["feature_names"], fill_value=0).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def get_demand_trend(df: pd.DataFrame, models: dict, forecast_days: int = 30) -> dict:
    work = _with_date(df)

    if "quantity" not in work.columns:
        raise ValueError("quantity column not found")
    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce").fillna(0)

    # ── Historical daily totals ───────────────────────────────────────────────
    daily_qty = work.groupby("__date__")["quantity"].sum().sort_index()
    hist_display = daily_qty.tail(60)
    hist_dates = hist_display.index.tolist()
    hist_vals  = hist_display.values.tolist()

    if len(daily_qty) == 0:
        raise ValueError("no demand history available")

    txns_per_day = _get_transactions_per_day(df)
    last_date    = daily_qty.index.max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D",
    )

    # ── Historical revenue & profit (daily) for actuals display ──────────────
    daily_rev    = None
    daily_profit = None

    if "revenue" in work.columns:
        work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce").fillna(0)
        daily_rev = work.groupby("__date__")["revenue"].sum().sort_index()

    if "profit" in work.columns:
        work["profit"] = pd.to_numeric(work["profit"], errors="coerce").fillna(0)
        daily_profit = work.groupby("__date__")["profit"].sum().sort_index()

    # ── Stable ratio: profit / revenue (per day, median of last 60 days) ─────
    # Using ratio ensures profit always moves WITH sales, never against it.
    profit_margin_ratio = 0.0
    if daily_rev is not None and daily_profit is not None:
        aligned = pd.DataFrame({
            "rev":    daily_rev,
            "profit": daily_profit
        }).dropna()
        aligned = aligned[aligned["rev"] > 0]
        if len(aligned) > 0:
            profit_margin_ratio = float(
                (aligned["profit"] / aligned["rev"]).tail(60).median()
            )

    # ── Feature matrix from actual historical rows ────────────────────────────
    # KEY FIX: Instead of repeating one template row, we sample actual rows
    # from the last 90 days of data. Each row has the full natural feature
    # variation the model was trained on, so predictions vary realistically.
    feat_matrix = _get_feature_matrix(work, models)
    n_rows = len(feat_matrix)

    # Exact lag column names from preprocessing.py
    LAG_QTY_1  = "lag_1_quantity"
    LAG_QTY_7  = "lag_7_quantity"
    ROLL_QTY_7 = "rolling_mean_7_quantity"
    LAG_REV_1  = "lag_1_revenue"
    LAG_REV_7  = "lag_7_revenue"
    ROLL_REV_7 = "rolling_mean_7_revenue"

    # Seed rolling buffers in per-transaction scale
    seed_qty = daily_qty.tail(30).values.tolist()
    qty_buf  = [v / txns_per_day for v in seed_qty]

    if daily_rev is not None:
        seed_rev = daily_rev.tail(30).values.tolist()
        rev_buf  = [v / txns_per_day for v in seed_rev]
    else:
        rev_buf = [0.0] * len(qty_buf)

    def _buf_val(buf, n):
        return float(buf[-n]) if len(buf) >= n else (float(buf[-1]) if buf else 0.0)

    def _buf_mean(buf, n=7):
        w = buf[-n:] if len(buf) >= n else buf[:]
        return float(np.mean(w)) if w else 0.0

    rng = np.random.RandomState(42)
    daily_preds  = []
    sales_preds  = []
    profit_preds = []

    # ── Forecast loop ─────────────────────────────────────────────────────────
    for i in range(forecast_days):
        # Sample a real historical feature row (cycles through last 90 rows)
        # This preserves all categorical/encoded feature variation naturally.
        row = feat_matrix.iloc[i % n_rows].copy()

        # Override only the lag features with our rolling buffer values
        # (per-transaction scale — matches exactly how training data was built)
        if LAG_QTY_1  in row.index: row[LAG_QTY_1]  = _buf_val(qty_buf, 1)
        if LAG_QTY_7  in row.index: row[LAG_QTY_7]  = _buf_val(qty_buf, 7)
        if ROLL_QTY_7 in row.index: row[ROLL_QTY_7] = _buf_mean(qty_buf, 7)
        if LAG_REV_1  in row.index: row[LAG_REV_1]  = _buf_val(rev_buf, 1)
        if LAG_REV_7  in row.index: row[LAG_REV_7]  = _buf_val(rev_buf, 7)
        if ROLL_REV_7 in row.index: row[ROLL_REV_7] = _buf_mean(rev_buf, 7)

        row_df = pd.DataFrame([row]).reindex(columns=models["feature_names"], fill_value=0)
        scaled = models["scaler"].transform(row_df)

        # Demand model → per-transaction quantity
        per_txn_qty = max(0.0, float(models["demand"].predict(scaled)[0]))

        # Sales model → per-transaction revenue
        per_txn_rev = max(0.0, float(models["sales"].predict(scaled)[0]))

        # Scale to daily
        noise_qty = rng.uniform(-0.04, 0.04)
        noise_rev = rng.uniform(-0.03, 0.03)
        daily_qty_pred = max(0.0, per_txn_qty * txns_per_day * (1.0 + noise_qty))
        daily_rev_pred = max(0.0, per_txn_rev * txns_per_day * (1.0 + noise_rev))

        daily_preds.append(round(daily_qty_pred, 1))
        sales_preds.append(round(daily_rev_pred, 2))

        # Profit = sales × stable margin ratio
        # This guarantees profit moves WITH sales (never inversely).
        if profit_margin_ratio != 0.0:
            profit_preds.append(round(daily_rev_pred * profit_margin_ratio, 2))

        # Push per-transaction values (NOT daily) into buffers
        qty_buf.append(per_txn_qty)
        rev_buf.append(per_txn_rev)

    # ── Trend direction ───────────────────────────────────────────────────────
    if len(hist_vals) >= 14:
        recent_mean = float(np.mean(hist_vals[-7:]))
        prior_mean  = float(np.mean(hist_vals[-14:-7]))
        if recent_mean > prior_mean * 1.03:
            trend_direction = "up"
        elif recent_mean < prior_mean * 0.97:
            trend_direction = "down"
        else:
            trend_direction = "stable"
    else:
        trend_direction = "stable"

    peak_idx  = int(np.argmax(daily_preds)) if daily_preds else 0
    peak_date = str(future_dates[peak_idx].date()) if len(future_dates) else ""
    peak_qty  = float(daily_preds[peak_idx]) if daily_preds else 0.0

    # ── Historical arrays for sales & profit charts ───────────────────────────
    hist_sales_list = []
    if daily_rev is not None:
        h = daily_rev.tail(60)
        hist_sales_list = [
            {"date": str(d.date()), "revenue": float(v)}
            for d, v in zip(h.index, h.values)
        ]

    hist_profit_list = []
    if daily_profit is not None:
        h = daily_profit.tail(60)
        hist_profit_list = [
            {"date": str(d.date()), "profit": float(v)}
            for d, v in zip(h.index, h.values)
        ]

    out = {
        "historical": [
            {"date": str(d.date()), "quantity": float(q)}
            for d, q in zip(hist_dates, hist_vals)
        ],
        "forecast": [
            {"date": str(d.date()), "quantity": float(q)}
            for d, q in zip(future_dates, daily_preds)
        ],
        "trend_direction":      trend_direction,
        "forecast_peak_date":   peak_date,
        "forecast_peak_qty":    peak_qty,
        "transactions_per_day": round(txns_per_day, 1),
        "scale_note":   "Forecast values represent estimated daily demand (units/day)",
        "method_note":  "Forecast samples real historical feature rows to preserve natural variance",
    }

    if sales_preds:
        out["sales_forecast"]   = [
            {"date": str(d.date()), "revenue": float(v)}
            for d, v in zip(future_dates, sales_preds)
        ]
        out["sales_historical"] = hist_sales_list

    if profit_preds:
        out["profit_forecast"]   = [
            {"date": str(d.date()), "profit": float(v)}
            for d, v in zip(future_dates, profit_preds)
        ]
        out["profit_historical"] = hist_profit_list
        out["profit_margin_ratio"] = round(profit_margin_ratio, 4)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# KPI SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_kpi_summary(df: pd.DataFrame, models: dict) -> dict:
    work = _with_date(df)

    def _col(name):
        return (
            pd.to_numeric(work[name], errors="coerce").fillna(0)
            if name in work.columns
            else pd.Series([0] * len(work))
        )

    revenue  = _col("revenue")
    quantity = _col("quantity")
    profit   = _col("profit")

    total_revenue     = float(revenue.sum())
    total_orders      = int(len(work))
    avg_order_value   = float(revenue.mean()) if len(work) > 0 else 0.0
    total_profit      = float(profit.sum())
    profit_margin_pct = (total_profit / total_revenue * 100.0) if total_revenue > 0 else 0.0

    daily = (
        pd.DataFrame({
            "__date__": work["__date__"],
            "revenue":  revenue,
            "quantity": quantity,
        })
        .groupby("__date__", as_index=False)
        .sum()
        .sort_values("__date__")
    )
    last_30 = daily.tail(30)
    prev_30 = daily.iloc[max(0, len(daily) - 60): max(0, len(daily) - 30)]

    last_rev = float(last_30["revenue"].sum()) if len(last_30) else 0.0
    prev_rev = float(prev_30["revenue"].sum()) if len(prev_30) else 0.0
    revenue_growth_pct = (
        ((last_rev - prev_rev) / prev_rev * 100.0) if prev_rev > 0
        else (100.0 if last_rev > 0 else 0.0)
    )

    last_qty = float(last_30["quantity"].sum()) if len(last_30) else 0.0
    prev_qty = float(prev_30["quantity"].sum()) if len(prev_30) else 0.0
    demand_growth_pct = (
        ((last_qty - prev_qty) / prev_qty * 100.0) if prev_qty > 0
        else (100.0 if last_qty > 0 else 0.0)
    )

    fi    = np.array(models["demand"].feature_importances_, dtype=float)
    names = list(models["feature_names"])
    order = np.argsort(fi)[::-1][:5]
    top_factors = [
        {"name": str(names[i]), "score": round(float(fi[i]), 4)} for i in order
    ]

    return {
        "total_revenue":      _round_num(total_revenue),
        "total_orders":       total_orders,
        "avg_order_value":    _round_num(avg_order_value),
        "total_profit":       _round_num(total_profit),
        "profit_margin_pct":  _round_num(profit_margin_pct),
        "revenue_growth_pct": _round_num(revenue_growth_pct),
        "demand_growth_pct":  _round_num(demand_growth_pct),
        "top_factors":        top_factors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INVENTORY INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def get_inventory_intelligence(df: pd.DataFrame, models: dict) -> list:
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
    for p in df[product_col].dropna().astype(str).unique().tolist()[:20]:
        pdf = df[df[product_col].astype(str) == p].copy()
        if len(pdf) == 0:
            continue
        pdf["quantity"] = pd.to_numeric(pdf["quantity"], errors="coerce").fillna(0)
        current_stock = float(pdf["quantity"].sum())
        feature_row   = (
            pdf.drop(columns=["quantity", "revenue", "profit"], errors="ignore")
            .iloc[-1]
            .to_dict()
        )
        avg_units_per_txn       = float(predict_demand(feature_row, models))
        daily_predicted_demand  = round(avg_units_per_txn * txns_per_day, 1)
        weekly_predicted_demand = round(daily_predicted_demand * 7, 1)
        days_of_inventory = (
            round(current_stock / daily_predicted_demand, 1)
            if daily_predicted_demand > 0 else 999.0
        )

        if days_of_inventory > 60:
            status = "overstock"
            action = f"Overstock — {days_of_inventory} days of supply. Reduce reorder quantity."
        elif days_of_inventory < 15:
            status = "understock"
            action = f"Understock — only {days_of_inventory} days remaining. Reorder now."
        else:
            status = "optimal"
            action = f"Stock optimal — {days_of_inventory} days of supply."

        out.append({
            "product":                   str(p),
            "current_stock":             round(current_stock, 2),
            "avg_units_per_transaction": round(avg_units_per_txn, 2),
            "daily_predicted_demand":    daily_predicted_demand,
            "weekly_predicted_demand":   weekly_predicted_demand,
            "days_of_inventory":         days_of_inventory,
            "action":                    action,
            "status":                    status,
        })

    return sorted(out, key=lambda x: x["days_of_inventory"])