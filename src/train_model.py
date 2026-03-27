import os

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.preprocessing import preprocess


def train_all(csv_path="data/processed/cleaned_data.csv"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    data = preprocess(csv_path)
    X_train = data["X_train"]
    X_test = data["X_test"]

    scaler = data["scaler"]
    feature_names = data["feature_names"]

    params = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}

    demand_model = RandomForestRegressor(**params)
    demand_model.fit(X_train, data["y_train_qty"])
    demand_pred = demand_model.predict(X_test)
    demand_mae = mean_absolute_error(data["y_test_qty"], demand_pred)
    demand_r2 = r2_score(data["y_test_qty"], demand_pred)
    print(f"demand_model MAE: {demand_mae:.4f} R2: {demand_r2:.4f}")

    sales_model = RandomForestRegressor(**params)
    sales_model.fit(X_train, data["y_train_rev"])
    sales_pred = sales_model.predict(X_test)
    sales_mae = mean_absolute_error(data["y_test_rev"], sales_pred)
    sales_r2 = r2_score(data["y_test_rev"], sales_pred)
    print(f"sales_model MAE: {sales_mae:.4f} R2: {sales_r2:.4f}")

    profit_model = RandomForestRegressor(**params)
    profit_model.fit(X_train, data["y_train_profit"])
    profit_pred = profit_model.predict(X_test)
    profit_mae = mean_absolute_error(data["y_test_profit"], profit_pred)
    profit_r2 = r2_score(data["y_test_profit"], profit_pred)
    print(f"profit_model MAE: {profit_mae:.4f} R2: {profit_r2:.4f}")

    joblib.dump(demand_model, os.path.join(models_dir, "demand_model.pkl"))
    joblib.dump(sales_model, os.path.join(models_dir, "sales_model.pkl"))
    joblib.dump(profit_model, os.path.join(models_dir, "profit_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))


if __name__ == "__main__":
    train_all()