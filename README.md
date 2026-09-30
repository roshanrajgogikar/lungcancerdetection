# Lung Cancer Prediction

A responsive web app that recreates the supplied glassmorphism sign-in and screening dashboard. It includes account registration, sign-in and logout, a 15-field questionnaire, a live model comparison panel, top feature signals, and a prediction result card.

This rebuild follows the attached report and screenshots, which specify **Logistic Regression and Random Forest**. The earlier project summary said “linear regression”; that is a different algorithm. Logistic Regression is the classifier shown in the supplied references and is paired with Random Forest here.

## Important use limitation

This is an educational survey-classification prototype, not a medical device. The source survey has 309 records and self-reported inputs. The displayed model score is the classifier's output for that survey data; it is not a calibrated clinical probability, diagnosis, or a way to rule out cancer. Do not use it to make medical decisions.

If the app cannot download or read the survey, it trains on generated demo records so the website still opens. In that case a warning banner identifies the synthetic fallback, and the metrics and predictions are for interface demonstration only.

## Project Structure
```text
Lung-Cancer-Prediction/
├── app.py
├── model.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── data/
│   └── lung-cancer-survey.csv
├── artifacts/
│   ├── model.pkl
│   └── metadata.json
├── instance/
│   └── app.db
└── static/
    ├── index.html
    ├── styles.css
    └── app.js
```

## Run locally on Windows

1. Install Python 3.11.
2. Open PowerShell in this project folder and run:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```

3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000), create an account, then sign in.

On its first start, the backend attempts to download the public survey CSV from the source linked below and saves it under `data/`. If you already have the CSV, save it as `data/lung-cancer-survey.csv` before starting the server. Training runs when the app starts; the selected model and training metadata are written to `artifacts/`.

## Run with Docker

From this folder. Copy `.env.example` to `.env` first if you want to set deployment options:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Then open [http://localhost:8000](http://localhost:8000). Docker Compose keeps the SQLite account database, trained artifacts, and any downloaded dataset in local folders. For a public HTTPS deployment, set `COOKIE_SECURE=true` and serve the container behind a TLS-enabled reverse proxy. Use persistent storage for `instance/`; otherwise accounts are lost when the deployment's filesystem is replaced.

## Accounts and data

- Email and bcrypt-hashed passwords are stored in a local SQLite database at `instance/app.db`.
- Session tokens are random, stored as hashes in SQLite, and sent in an HttpOnly, SameSite=Lax cookie.
- The questionnaire is sent to the backend for inference and is not saved as prediction history.
- Model training compares a StandardScaler + Logistic Regression pipeline with a class-balanced Random Forest. Each is evaluated on a stratified 20% holdout and five-fold cross-validation. The winner is selected by cross-validation accuracy, with ROC-AUC as a tie-breaker.
- The dashboard reports the measured scores from the current training run; it does not hard-code the figures from the report.

## Project files

```text
app.py                 FastAPI endpoints, SQLite accounts, sessions
model.py               Dataset loading, classifier training, metrics, artifacts
static/index.html      Sign-in and dashboard structure
static/styles.css      Responsive dark glassmorphism interface
static/app.js          Forms, authentication, model cards, predictions
requirements.txt       Python dependencies
Dockerfile             Single-container deployment
docker-compose.yml     Local deployment with persistent data folders
data/                  Downloaded survey CSV (if available)
artifacts/              Model pickle and computed metadata
instance/               SQLite user accounts and sessions
```

## Dataset reference

The training pipeline uses the `Lung Cancer Survey.csv` published in [ShinjiniShome/lung_cancer_survey_dataviz](https://github.com/ShinjiniShome/lung_cancer_survey_dataviz). It expects the reported 15 survey features and `LUNG_CANCER` target. The original report and screenshots are included by the project owner as implementation references.


