"""
forecast_engine.py
──────────────────
Drop this file into your src/ folder.
Call generate_forecast(df, days=30) from insights.py or app.py.

Festival dates can be passed dynamically:
    generate_forecast(df, days=30, festival_days=["2025-03-12", "2025-03-13"])
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT FESTIVAL CALENDAR
# Add or remove dates here, or pass festival_days= at call time.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FESTIVAL_DAYS = [
    "2025-03-12",
    "2025-03-13",
    "2025-03-14",
    "2025-03-15",
]


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN BUMP — smooth festival impact
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_event_mask(dates: pd.DatetimeIndex,
                         festival_dates: list,
                         peak_multiplier: float = 1.0,
                         sigma_days: float = 1.5) -> np.ndarray:
    """
    For each forecast date, compute a smooth additive multiplier.

    A Gaussian bump centred on each festival date ensures the impact
    builds up and fades gradually (no hard edges).

    Returns an array of shape (len(dates),) where values are in [0, peak_multiplier].
    """
    mask = np.zeros(len(dates))
    date_arr = np.array([d.date() for d in dates])

    for fday_str in festival_dates:
        centre = pd.Timestamp(fday_str).date()
        for i, d in enumerate(date_arr):
            dist = (d - centre).days          # signed distance in days
            # Gaussian kernel: peak at dist=0, falls off with sigma
            mask[i] += peak_multiplier * np.exp(-0.5 * (dist / sigma_days) ** 2)

    # Cap so overlapping festivals don't compound beyond 2× peak
    return np.clip(mask, 0.0, peak_multiplier * 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# TREND + SEASONALITY DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_trend_and_seasonality(series: np.ndarray,
                                   forecast_days: int) -> tuple:
    """
    Fit a linear trend to the historical series, then extract a
    7-day seasonal pattern from the residuals.

    Returns:
        trend_slope    — float, daily change in units
        trend_base     — float, value at the end of the history
        seasonal_cycle — np.ndarray of shape (7,), additive daily offsets
    """
    n = len(series)
    if n < 7:
        return 0.0, float(np.mean(series)), np.zeros(7)

    x = np.arange(n)
    coeffs = np.polyfit(x, series, 1)          # [slope, intercept]
    trend_slope = float(coeffs[0])
    trend_base  = float(np.polyval(coeffs, n - 1))   # last historical value

    # Detrend
    detrended = series - np.polyval(coeffs, x)

    # 7-day seasonal pattern (mean of residuals per weekday)
    seasonal_cycle = np.zeros(7)
    counts         = np.zeros(7)
    for i, val in enumerate(detrended):
        dow = i % 7
        seasonal_cycle[dow] += val
        counts[dow]         += 1
    with np.errstate(invalid="ignore"):
        seasonal_cycle = np.where(counts > 0, seasonal_cycle / counts, 0.0)

    return trend_slope, trend_base, seasonal_cycle


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_forecast(
    df: pd.DataFrame,
    days: int = 30,
    festival_days: list = None,
    # Tunable parameters — override to calibrate to your data
    demand_noise_pct:  float = 0.08,   # ±8 % daily noise on demand
    price_noise_pct:   float = 0.04,   # ±4 % daily noise on price
    cost_noise_pct:    float = 0.05,   # ±5 % daily noise on cost
    festival_demand_boost:  float = 0.70,  # up to +70 % demand at peak
    festival_price_surge:   float = 0.12,  # up to +12 % price at peak
    festival_cost_pressure: float = 0.08,  # up to +8 % cost at peak
    festival_sigma:    float = 1.5,    # Gaussian spread in days
    random_seed:       int   = 42,
) -> dict:
    """
    Generate realistic demand, sales, and profit forecasts.

    Parameters
    ──────────
    df               : DataFrame containing historical data with columns
                       [date_col, quantity, revenue, profit]
    days             : Number of forecast days
    festival_days    : List of ISO date strings  e.g. ["2025-03-12", "2025-03-13"]
                       Defaults to DEFAULT_FESTIVAL_DAYS if None.
    *_noise_pct      : Fractional daily noise for each series (independent RNGs)
    festival_*       : Peak boost fractions applied via Gaussian bump
    festival_sigma   : How wide (in days) the festival effect spreads
    random_seed      : Base seed; each series gets seed+offset for independence

    Returns
    ───────
    dict with keys:
        future_dates       : list of ISO date strings
        demand_forecast    : list of floats  (units/day)
        sales_forecast     : list of floats  (currency/day)
        profit_forecast    : list of floats  (currency/day)
        festival_impact    : list of floats  (0–1 festival intensity per day)
    """

    if festival_days is None:
        festival_days = DEFAULT_FESTIVAL_DAYS

    # ── Locate date and target columns ───────────────────────────────────────
    date_candidates = ["date", "Date", "order_date", "invoice_date", "transaction_date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    work = df.copy()
    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).sort_values(date_col)
        work = work.rename(columns={date_col: "__date__"})
    else:
        work["__date__"] = pd.date_range(
            end=pd.Timestamp.today().normalize(), periods=len(work), freq="D"
        )

    for col in ["quantity", "revenue", "profit"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    # ── Build daily aggregates ────────────────────────────────────────────────
    agg_cols = {c: "sum" for c in ["quantity", "revenue", "profit"] if c in work.columns}
    daily = work.groupby("__date__").agg(agg_cols).sort_index()

    qty_hist  = daily["quantity"].values.astype(float) if "quantity" in daily.columns else np.array([1.0])
    rev_hist  = daily["revenue"].values.astype(float)  if "revenue"  in daily.columns else qty_hist * 100.0
    prof_hist = daily["profit"].values.astype(float)   if "profit"   in daily.columns else qty_hist * 20.0

    last_date    = daily.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")

    # ── Trend + seasonality ───────────────────────────────────────────────────
    qty_slope,  qty_base,  qty_season  = _extract_trend_and_seasonality(qty_hist,  days)
    rev_slope,  rev_base,  rev_season  = _extract_trend_and_seasonality(rev_hist,  days)
    prof_slope, prof_base, prof_season = _extract_trend_and_seasonality(prof_hist, days)

    # ── Derive base price and cost from historical medians ────────────────────
    safe_qty = np.where(qty_hist > 0, qty_hist, 1.0)
    base_price = float(np.median(rev_hist  / safe_qty))
    base_cost  = float(np.median((rev_hist - prof_hist) / safe_qty))

    # Guard: cost must be less than price, both positive
    base_price = max(base_price, 1.0)
    base_cost  = max(min(base_cost, base_price * 0.95), 0.01)

    # ── Gaussian festival impact masks ────────────────────────────────────────
    demand_impact = _gaussian_event_mask(future_dates, festival_days,
                                         peak_multiplier=festival_demand_boost,
                                         sigma_days=festival_sigma)
    price_impact  = _gaussian_event_mask(future_dates, festival_days,
                                         peak_multiplier=festival_price_surge,
                                         sigma_days=festival_sigma)
    cost_impact   = _gaussian_event_mask(future_dates, festival_days,
                                         peak_multiplier=festival_cost_pressure,
                                         sigma_days=festival_sigma)

    # ── Independent RNGs for each series ─────────────────────────────────────
    rng_d = np.random.RandomState(random_seed)
    rng_p = np.random.RandomState(random_seed + 7)
    rng_c = np.random.RandomState(random_seed + 13)

    demand_preds = []
    sales_preds  = []
    profit_preds = []

    for i in range(days):
        dow = future_dates[i].dayofweek     # 0=Mon … 6=Sun
        t   = i + 1                         # steps ahead

        # ── Trend + seasonality base ──────────────────────────────────────────
        qty_trend_val  = qty_base  + qty_slope  * t + qty_season[dow]
        rev_trend_val  = rev_base  + rev_slope  * t + rev_season[dow]
        prof_trend_val = prof_base + prof_slope * t + prof_season[dow]

        # Clamp to sensible floor (10 % of base)
        qty_trend_val  = max(qty_trend_val,  qty_base  * 0.10)
        rev_trend_val  = max(rev_trend_val,  rev_base  * 0.10)
        prof_trend_val = max(prof_trend_val, 1.0)

        # ── Independent noise ─────────────────────────────────────────────────
        noise_d = 1.0 + rng_d.uniform(-demand_noise_pct, demand_noise_pct)
        noise_p = 1.0 + rng_p.uniform(-price_noise_pct,  price_noise_pct)
        noise_c = 1.0 + rng_c.uniform(-cost_noise_pct,   cost_noise_pct)

        # ── Festival multipliers (smooth, Gaussian) ───────────────────────────
        d_mult = 1.0 + demand_impact[i]   # e.g. 1.0 → 1.70 at peak
        p_mult = 1.0 + price_impact[i]    # e.g. 1.0 → 1.12 at peak
        c_mult = 1.0 + cost_impact[i]     # e.g. 1.0 → 1.08 at peak

        # ── Compute demand ────────────────────────────────────────────────────
        demand = max(0.0, qty_trend_val * d_mult * noise_d)

        # ── Compute price and cost independently ──────────────────────────────
        price = base_price * p_mult * noise_p
        cost  = base_cost  * c_mult * noise_c

        # ── Sales = demand × price ────────────────────────────────────────────
        sales = demand * price

        # ── Profit = sales - (cost × demand) ─────────────────────────────────
        profit = sales - (cost * demand)
        profit = max(0.0, profit)    # floor at zero

        demand_preds.append(round(demand, 2))
        sales_preds.append(round(sales,  2))
        profit_preds.append(round(profit, 2))

    # ── Light smoothing pass (removes single-step spikes from noise) ──────────
    # Uses gaussian_filter1d with sigma=0.6 — smooths noise but preserves
    # the festival bump shape which is already smooth from the Gaussian mask.
    demand_smooth = gaussian_filter1d(demand_preds, sigma=0.6).tolist()
    sales_smooth  = gaussian_filter1d(sales_preds,  sigma=0.6).tolist()
    profit_smooth = gaussian_filter1d(profit_preds, sigma=0.6).tolist()

    # Normalise festival impact to 0–1 for frontend display
    impact_norm = (demand_impact / (festival_demand_boost * 2.0 + 1e-9)).tolist()

    return {
        "future_dates":     [str(d.date()) for d in future_dates],
        "demand_forecast":  [round(v, 1) for v in demand_smooth],
        "sales_forecast":   [round(v, 2) for v in sales_smooth],
        "profit_forecast":  [round(v, 2) for v in profit_smooth],
        "festival_impact":  [round(v, 4) for v in impact_norm],
    }