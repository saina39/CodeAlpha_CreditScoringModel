"""
app.py
------
Flask web application for the Credit Scoring Model.

Routes:
    GET  /            -> input form (index.html)
    POST /predict      -> validates input, runs prediction, renders result.html
    GET  /api/metrics   -> JSON of model comparison metrics (used by charts in JS)

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, jsonify, render_template, request

from predict import get_metadata, predict_credit_risk

app = Flask(__name__)

PAYMENT_HISTORY_OPTIONS = ["excellent", "good", "average", "poor"]
HOUSING_OPTIONS = ["own", "rent", "with_parents"]
PURPOSE_OPTIONS = ["car", "education", "business", "home_renovation", "personal", "medical"]
JOB_TYPE_OPTIONS = ["salaried", "self_employed", "unemployed", "student"]


def validate_input(form):
    """Server-side validation. Returns (cleaned_dict, list_of_errors)."""
    errors = []
    cleaned = {}

    def get_float(name, min_val, max_val, label):
        raw = form.get(name, "").strip()
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return None
        if value < min_val or value > max_val:
            errors.append(f"{label} must be between {min_val} and {max_val}.")
            return None
        return value

    cleaned["age"] = get_float("age", 18, 100, "Age")
    cleaned["income"] = get_float("income", 0, 10_000_000, "Income")
    cleaned["employment_length"] = get_float("employment_length", 0, 60, "Employment length")
    cleaned["debt"] = get_float("debt", 0, 10_000_000, "Debt")
    cleaned["credit_history_length"] = get_float("credit_history_length", 0, 60, "Credit history length")
    cleaned["num_credit_cards"] = get_float("num_credit_cards", 0, 50, "Number of credit cards")
    cleaned["num_loans"] = get_float("num_loans", 0, 50, "Number of loans")
    cleaned["monthly_emi"] = get_float("monthly_emi", 0, 1_000_000, "Monthly EMI")
    cleaned["existing_defaults"] = get_float("existing_defaults", 0, 20, "Existing defaults")

    payment_history = form.get("payment_history", "")
    if payment_history not in PAYMENT_HISTORY_OPTIONS:
        errors.append("Please select a valid payment history option.")
    cleaned["payment_history"] = payment_history

    housing = form.get("housing", "")
    if housing not in HOUSING_OPTIONS:
        errors.append("Please select a valid housing option.")
    cleaned["housing"] = housing

    purpose = form.get("purpose", "")
    if purpose not in PURPOSE_OPTIONS:
        errors.append("Please select a valid loan purpose.")
    cleaned["purpose"] = purpose

    job_type = form.get("job_type", "")
    if job_type not in JOB_TYPE_OPTIONS:
        errors.append("Please select a valid job type.")
    cleaned["job_type"] = job_type

    return cleaned, errors


@app.route("/")
def index():
    return render_template(
        "index.html",
        payment_history_options=PAYMENT_HISTORY_OPTIONS,
        housing_options=HOUSING_OPTIONS,
        purpose_options=PURPOSE_OPTIONS,
        job_type_options=JOB_TYPE_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    cleaned, errors = validate_input(request.form)

    if errors:
        return render_template(
            "index.html",
            payment_history_options=PAYMENT_HISTORY_OPTIONS,
            housing_options=HOUSING_OPTIONS,
            purpose_options=PURPOSE_OPTIONS,
            job_type_options=JOB_TYPE_OPTIONS,
            errors=errors,
            form_data=request.form,
        )

    try:
        result = predict_credit_risk(cleaned)
    except FileNotFoundError as e:
        return render_template(
            "index.html",
            payment_history_options=PAYMENT_HISTORY_OPTIONS,
            housing_options=HOUSING_OPTIONS,
            purpose_options=PURPOSE_OPTIONS,
            job_type_options=JOB_TYPE_OPTIONS,
            errors=[str(e)],
            form_data=request.form,
        )

    return render_template("result.html", result=result, input_data=cleaned)


@app.route("/api/metrics")
def api_metrics():
    """Returns model comparison metrics as JSON for the charts on the dashboard."""
    try:
        metadata = get_metadata()
        return jsonify(metadata["results"])
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run train.py first."}), 404


if __name__ == "__main__":
    app.run(debug=True)
