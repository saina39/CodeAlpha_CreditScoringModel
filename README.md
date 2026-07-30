# Credit Scoring Model

An end-to-end machine learning web application that predicts whether a loan
applicant is **Low Risk** or **High Risk**, built with Python, Flask,
Scikit-learn, and XGBoost, with a responsive glassmorphism UI (dark/light mode).

---

## Features

- **Data preprocessing**: missing value handling, deduplication, encoding, scaling
- **EDA**: correlation heatmap, distribution plots, class balance, feature importance
- **Model comparison**: Logistic Regression, Decision Tree, Random Forest, XGBoost
  — evaluated on Accuracy, Precision, Recall, F1, and ROC-AUC, with the best
  model auto-selected and saved
- **Flask web app**: form-based input, server-side validation, prediction with
  confidence score
- **Modern UI**: glassmorphism cards, dark/light mode toggle, animated buttons,
  loading spinner, live model-metrics dashboard (Chart.js), mobile-responsive

---

## Project Structure

```
Credit-Scoring-Model/
│
├── dataset/
│   ├── generate_dataset.py   # generates a realistic synthetic dataset
│   └── credit_data.csv       # created by generate_dataset.py (or your own data)
│
├── model/                    # created by train.py
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── metadata.json
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/               # EDA plots saved here by train.py
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Getting Started (Local Setup)

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

This project ships with a script that generates a realistic, German-Credit-style
synthetic dataset (1,500 applicants) so you can run everything immediately
without needing an external download:

```bash
python dataset/generate_dataset.py
```

> **Want to use the real UCI German Credit dataset instead?** Download it from
> https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data, save it as
> `dataset/credit_data.csv` with matching column names (or adjust the column
> names at the top of `train.py`), and skip this step.

### 4. Train the models

```bash
python train.py
```

This will:
- Clean the data and print a missing-value / duplicate report
- Save EDA charts to `static/images/`
- Train all 4 models and print a comparison table
- Save the best model (by ROC-AUC) to `model/credit_model.pkl`

### 5. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## How It Works

1. **`train.py`** loads the CSV, cleans it, encodes categorical fields
   (`payment_history`, `housing`, `purpose`, `job_type`) with `LabelEncoder`,
   scales numeric fields with `StandardScaler`, then trains and compares four
   classifiers. The best model (by ROC-AUC) plus the scaler, encoders, and a
   `metadata.json` (feature list + metrics) are saved to `model/`.
2. **`predict.py`** loads those artifacts once and exposes
   `predict_credit_risk(input_dict)`, which encodes/scales a single applicant's
   data the same way and returns the predicted label with confidence.
3. **`app.py`** serves the form (`index.html`), validates submitted data
   server-side, calls `predict.py`, and renders the result (`result.html`).
   It also exposes `/api/metrics`, which the dashboard on the home page
   fetches to draw the model-comparison chart.

---

## Deployment on Render

1. Push this project to a GitHub repository (the trained `model/` files can be
   committed, or regenerated on first deploy — see note below).
2. Go to [render.com](https://render.com) → **New** → **Web Service** →
   connect your GitHub repo.
3. Configure the service:
   - **Build Command:**
     ```
     pip install -r requirements.txt && python dataset/generate_dataset.py && python train.py
     ```
   - **Start Command:**
     ```
     gunicorn app:app
     ```
     (add `gunicorn` to `requirements.txt` first: `pip install gunicorn` then
     `pip freeze >> requirements.txt`, or simply add the line manually)
4. Set the **Python version** in an environment variable if needed, e.g.
   `PYTHON_VERSION=3.11.9`.
5. Deploy. Render will build the app, train the model as part of the build
   step, and start the Flask app with Gunicorn.

> **Note:** Running `train.py` during the build step keeps the repo lightweight
> and guarantees a freshly trained model on every deploy. Alternatively, train
> locally and commit the `model/` folder, then just use
> `pip install -r requirements.txt` as the build command.

---

## Future Improvements

- **SHAP explanations**: add SHAP values so each prediction shows *why* the
  model flagged an applicant as high/low risk (great for interview talking
  points — explainability is a common ask in credit scoring).
- **Feature importance visualization on the result page**: show which of the
  applicant's specific inputs pushed the score up or down.
- **REST API endpoint**: add a `/api/predict` JSON endpoint (accepting/returning
  JSON instead of form data) so the model can be consumed by other apps or
  tested with Postman.
- **Model monitoring**: log predictions over time to detect data drift.
- **Hyperparameter tuning**: use `GridSearchCV` / `Optuna` to squeeze more
  performance out of Random Forest / XGBoost.
- **Authentication**: add login so the tool can be used by multiple loan
  officers with saved prediction history.

---

## Tech Stack

Python · Flask · Scikit-learn · XGBoost · Pandas · NumPy · Matplotlib · Seaborn
· Joblib · HTML5 · CSS3 (Glassmorphism) · JavaScript · Chart.js
