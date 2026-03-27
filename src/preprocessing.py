import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(csv_path: str) -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_full_path = csv_path if os.path.isabs(csv_path) else os.path.join(project_root, csv_path)

    df = pd.read_csv(csv_full_path)

    date_cols = [c for c in df.columns if c.lower() in {"date", "order_date"}]
    object_cols = [c for c in df.columns if df[c].dtype == object]
    high_card_cols = [
        c
        for c in object_cols
        if c not in date_cols and df[c].nunique(dropna=False) > 50
    ]
    if high_card_cols:
        df = df.drop(columns=high_card_cols)

    for col in ["quantity", "revenue", "profit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)

        engineered_cols = []
        for base_col in ["quantity", "revenue"]:
            if base_col in df.columns:
                col_lag_1 = f"lag_1_{base_col}"
                col_lag_7 = f"lag_7_{base_col}"
                col_roll_7 = f"rolling_mean_7_{base_col}"
                df[col_lag_1] = df[base_col].shift(1)
                df[col_lag_7] = df[base_col].shift(7)
                df[col_roll_7] = df[base_col].rolling(window=7, min_periods=7).mean()
                engineered_cols.extend([col_lag_1, col_lag_7, col_roll_7])

        if engineered_cols:
            df[engineered_cols] = df[engineered_cols].fillna(0)

        df = df.drop(columns=date_cols, errors="ignore")

    df = df.dropna()

    remaining_object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if remaining_object_cols:
        df = pd.get_dummies(df, columns=remaining_object_cols, drop_first=True)

    if "quantity" not in df.columns:
        raise ValueError("quantity column not found")
    if "revenue" not in df.columns:
        raise ValueError("revenue column not found")
    if "profit" not in df.columns:
        raise ValueError("profit column not found")

    y_qty = df["quantity"]
    y_rev = df["revenue"]
    y_profit = df["profit"]

    X = df.drop(columns=["quantity", "revenue", "profit"], errors="ignore")
    id_like_cols = [c for c in X.columns if c.lower() == "index" or c.lower().endswith("_id")]
    if id_like_cols:
        X = X.drop(columns=id_like_cols, errors="ignore")

    feature_names = X.columns.tolist()

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train_df = X.iloc[train_idx]
    X_test_df = X.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    y_train_qty = y_qty.iloc[train_idx]
    y_test_qty = y_qty.iloc[test_idx]
    y_train_rev = y_rev.iloc[train_idx]
    y_test_rev = y_rev.iloc[test_idx]
    y_train_profit = y_profit.iloc[train_idx]
    y_test_profit = y_profit.iloc[test_idx]

    processed = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_qty": y_train_qty,
        "y_test_qty": y_test_qty,
        "y_train_rev": y_train_rev,
        "y_test_rev": y_test_rev,
        "y_train_profit": y_train_profit,
        "y_test_profit": y_test_profit,
        "scaler": scaler,
        "feature_names": feature_names,
    }

    processed_path = os.path.join(project_root, "data", "processed", "processed_data.pkl")
    joblib.dump(processed, processed_path)
    return processed