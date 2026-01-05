from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Resolve project root so model paths work regardless of where the app is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"
TARGET_NAMES_PATH = PROJECT_ROOT / "models" / "target_names.joblib"

EXPECTED_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

app = Flask(__name__)

# Load model artifacts once at startup
model = joblib.load(MODEL_PATH)
target_names = joblib.load(TARGET_NAMES_PATH)


@app.get("/health")
def health() -> tuple:
    return jsonify(status="ok"), 200


@app.post("/predict")
def predict() -> tuple:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    missing = [feature for feature in EXPECTED_FEATURES if feature not in payload]
    if missing:
        return jsonify(error="Missing required features.", missing=missing), 400

    try:
        sample = {feature: float(payload[feature]) for feature in EXPECTED_FEATURES}
    except (TypeError, ValueError):
        return jsonify(error="All feature values must be numeric."), 400

    X = pd.DataFrame([sample])

    try:
        pred_idx = int(model.predict(X)[0])
    except Exception as exc:  # noqa: BLE001 - surface model errors to the client
        return jsonify(error="Model prediction failed.", detail=str(exc)), 500

    response = {
        "prediction": target_names[pred_idx],
        "class_index": pred_idx,
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        response["probabilities"] = {
            target_names[i]: float(prob) for i, prob in enumerate(proba)
        }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
