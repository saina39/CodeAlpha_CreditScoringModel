"""
predict.py
----------
Loads the trained model + preprocessing artifacts and exposes a single
function, predict_credit_risk(), that app.py calls with raw user input.

Keeping this logic separate from app.py keeps Flask routes thin and makes
the prediction logic testable/reusable on its own.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "model"

_model = None
_scaler = None
_encoders = None
_metadata = None


def _load_artifacts():
    """Lazy-load model artifacts once and cache them in module-level vars."""
    global _model, _scaler, _encoders, _metadata

    if _model is None:
        model_path = f"{MODEL_DIR}/credit_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Run 'python train.py' first to train and save the model."
            )
        _model = joblib.load(model_path)
        _scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        _encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
        with open(f"{MODEL_DIR}/metadata.json") as f:
            _metadata = json.load(f)

    return _model, _scaler, _encoders, _metadata


def get_metadata():
    """Expose metadata (best model name, metrics, feature list) to app.py."""
    _, _, _, metadata = _load_artifacts()
    return metadata


def predict_credit_risk(input_dict):
    """
    Takes a dict of raw form inputs (matching the training feature names)
    and returns a dict with the prediction, label, and confidence.

    Expected keys in input_dict:
        age, income, employment_length, debt, credit_history_length,
        num_credit_cards, num_loans, monthly_emi, payment_history,
        housing, purpose, job_type, existing_defaults
    """
    model, scaler, encoders, metadata = _load_artifacts()
    feature_cols = metadata["feature_cols"]
    cat_cols = metadata["cat_cols"]

    row = {}
    for col in feature_cols:
        if col in cat_cols:
            le = encoders[col]
            value = input_dict[col]
            # Handle unseen categories gracefully by falling back to the
            # most common class already known to the encoder.
            if value not in le.classes_:
                value = le.classes_[0]
            row[col] = le.transform([value])[0]
        else:
            row[col] = float(input_dict[col])

    X = pd.DataFrame([row])[feature_cols]
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    high_risk_prob = float(proba[1])
    low_risk_prob = float(proba[0])
    confidence = high_risk_prob if prediction == 1 else low_risk_prob

    return {
        "prediction": int(prediction),
        "label": "High Risk" if prediction == 1 else "Low Risk",
        "confidence": round(confidence * 100, 2),
        "high_risk_probability": round(high_risk_prob * 100, 2),
        "low_risk_probability": round(low_risk_prob * 100, 2),
        "eligible_for_loan": bool(prediction == 0),
    }


if __name__ == "__main__":
    # Quick manual test
    sample = {
        "age": 35,
        "income": 55000,
        "employment_length": 8,
        "debt": 12000,
        "credit_history_length": 10,
        "num_credit_cards": 2,
        "num_loans": 1,
        "monthly_emi": 800,
        "payment_history": "good",
        "housing": "own",
        "purpose": "car",
        "job_type": "salaried",
        "existing_defaults": 0,
    }
    result = predict_credit_risk(sample)
    print(json.dumps(result, indent=2))
