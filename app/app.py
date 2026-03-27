import sys
import os
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask import send_file
import numpy as np
from src.predict import load_models, predict_demand, predict_sales, predict_profit
from src.predict import get_inventory_recommendation, get_employee_performance
from src.insights import get_demand_trend, get_kpi_summary, get_inventory_intelligence
from src.export import build_forecast_export, build_combined_export, build_employee_export, build_inventory_export
import pandas as pd

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
MODELS = load_models()
DF = pd.read_csv("data/processed/cleaned_data.csv")


@app.route("/")
def index():
    return redirect(url_for("manager_dashboard"))


@app.route("/manager_dashboard")
def manager_dashboard():
    demand_model = MODELS["demand"]
    feature_names = MODELS["feature_names"]
    scores = demand_model.feature_importances_
    idx = np.argsort(scores)[::-1][:10]
    feature_importances = [{"name": feature_names[i], "score": float(scores[i])} for i in idx]
    return render_template("manager_dashboard.html", feature_importances=feature_importances)


@app.route("/employee_dashboard")
def employee_dashboard():
    employee_data = get_employee_performance(DF)
    return render_template("employee_dashboard.html", employee_data=employee_data)


@app.route("/api/predict/demand", methods=["POST"])
def api_predict_demand():
    try:
        payload = request.get_json(force=True) or {}
        prediction = predict_demand(payload, MODELS)
        inventory = get_inventory_recommendation(prediction)
        return jsonify({"prediction": prediction, "inventory": inventory})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/sales", methods=["POST"])
def api_predict_sales():
    try:
        payload = request.get_json(force=True) or {}
        prediction = predict_sales(payload, MODELS)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/profit", methods=["POST"])
def api_predict_profit():
    try:
        payload = request.get_json(force=True) or {}
        prediction = predict_profit(payload, MODELS)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/employee_stats", methods=["GET"])
def api_employee_stats():
    try:
        return jsonify(get_employee_performance(DF))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/demand_trend')
def demand_trend():
    try:
        days = int(request.args.get('forecast_days', 30))
        return jsonify(get_demand_trend(DF, MODELS, forecast_days=days))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/kpi_summary')
def kpi_summary():
    try:
        return jsonify(get_kpi_summary(DF, MODELS))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/inventory_intelligence')
def inventory_intelligence():
    try:
        return jsonify(get_inventory_intelligence(DF, MODELS))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/export/forecast')
def export_forecast():
    buf = build_forecast_export(DF, MODELS)
    return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name=f'demand_forecast_{datetime.date.today()}.xlsx')


@app.route('/export/combined')
def export_combined():
    buf = build_combined_export(DF, MODELS)
    return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name=f'historical_predicted_{datetime.date.today()}.xlsx')


@app.route('/export/employees')
def export_employees():
    buf = build_employee_export(DF)
    return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name=f'employee_performance_{datetime.date.today()}.xlsx')


@app.route('/export/inventory')
def export_inventory():
    buf = build_inventory_export(DF, MODELS)
    return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name=f'inventory_recommendations_{datetime.date.today()}.xlsx')


if __name__ == "__main__":
    app.run(debug=True)