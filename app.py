"""Lung Cancer Prediction: Streamlit screening prototype."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy.engine import Engine

from auth import authenticate_user, build_engine, register_user, validate_registration
from model import FEATURES, LABELS, train_and_save

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="Training the survey classifiers…")
def load_models() -> tuple[dict[str, Any], dict[str, Any]]:
    return train_and_save()


@st.cache_resource(show_spinner=False)
def load_database(database_url: str | None) -> Engine:
    return build_engine(database_url)


def database_url_from_secrets() -> str | None:
    try:
        value = st.secrets.get("DATABASE_URL")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv("DATABASE_URL") or None


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(ellipse at 7% 2%, rgba(30,199,174,.2), transparent 28%),
            radial-gradient(ellipse at 99% 38%, rgba(190,64,100,.16), transparent 27%),
            radial-gradient(ellipse at 46% 99%, rgba(242,160,78,.13), transparent 25%),
            linear-gradient(120deg,#071522 0%,#08101d 52%,#07111f 100%);
          color: #f3f6fb;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1rem; }
        .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 2rem; }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
          border-color: rgba(191,210,233,.16) !important; border-radius: 22px !important;
          background: linear-gradient(140deg,rgba(29,40,56,.82),rgba(8,17,30,.82));
          box-shadow: 0 22px 65px rgba(0,0,0,.2); backdrop-filter: blur(16px);
        }
        [data-testid="stForm"] { border: 0; padding: 0; }
        h1, h2, h3, p, label, [data-testid="stMetricLabel"] { color: #f3f6fb; }
        h1 { letter-spacing: -.045em; }
        h2, h3 { letter-spacing: -.025em; }
        .eyebrow { color:#69ecd1; font-size:10px; font-weight:800; letter-spacing:.22em; text-transform:uppercase; margin:0 0 8px; }
        .muted { color:#a6b1c2; }
        .hero-title { color:#f3f6fb; font-size:clamp(46px,5.6vw,68px); font-weight:780; letter-spacing:-.06em; line-height:.99; margin:8px 0 18px; }
        .hero-copy { color:#b2bdcb; font-size:14px; line-height:1.7; margin:0 0 20px; }
        .fact-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-top:18px; }
        .fact-card { min-height:94px; padding:13px 14px; border:1px solid rgba(196,211,230,.13); border-radius:15px; background:rgba(233,242,255,.07); }
        .fact-card b { display:block; color:#eef2f7; font-size:12px; line-height:1.45; }
        .fact-card .eyebrow { margin-bottom:7px; }
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] > div > div {
          color:#f2f5fa !important; background:rgba(3,11,22,.78) !important; border-radius:11px !important;
          border-color:rgba(173,194,219,.16) !important;
        }
        div[data-testid="stRadio"] [role="radiogroup"] { gap:6px; padding:5px; border:1px solid rgba(195,209,229,.09); border-radius:14px; background:rgba(222,229,241,.07); }
        div[data-testid="stRadio"] label { padding:7px 12px; border-radius:10px; }
        .stButton > button, [data-testid="stFormSubmitButton"] > button {
          min-height:43px; border:0; border-radius:12px; color:#08111b !important;
          background:linear-gradient(105deg,#65ecd0 0%,#f8bd67 100%) !important; font-weight:800;
          box-shadow:0 8px 22px rgba(69,224,194,.13);
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { filter:brightness(1.04); box-shadow:0 12px 28px rgba(69,224,194,.2); }
        .quiet-button .stButton > button { color:#dbe2ec !important; background:rgba(231,239,252,.05) !important; border:1px solid rgba(189,207,228,.15) !important; }
        [data-testid="stMetric"] { border:1px solid rgba(193,211,231,.12); border-radius:14px; padding:12px 14px; background:rgba(221,234,250,.05); }
        [data-testid="stMetricValue"] { font-size:1.25rem; }
        .signal-chip { display:inline-flex; gap:7px; align-items:center; padding:6px 10px; margin:3px 4px 3px 0; border:1px solid rgba(182,202,225,.15); border-radius:999px; color:#d7e0ea; background:rgba(217,230,248,.08); font-size:10px; text-transform:uppercase; }
        .signal-chip b { color:#69ecd1; }
        .notice { padding:12px 15px; margin:0 0 16px; border:1px solid rgba(255,196,102,.25); border-radius:13px; color:#f1d19d; background:rgba(125,83,32,.16); font-size:12px; }
        .small-note { color:#95a2b4; font-size:11px; line-height:1.65; }
        .result-score { font-size:36px; font-weight:780; letter-spacing:-.05em; line-height:1.1; }
        .result-high { color:#ff8795; }
        .result-low { color:#72e8ca; }
        @media (max-width: 760px) {
          .block-container { padding:1rem 1rem 2rem; }
          .fact-grid { grid-template-columns:1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sign_in(engine: Engine, persistent_database: bool) -> None:
    left, right = st.columns([1.74, 0.91], gap="medium", vertical_alignment="center")
    with left:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Secure access</div>', unsafe_allow_html=True)
            st.markdown('<div class="hero-title">Lung Cancer<br>Prediction</div>', unsafe_allow_html=True)
            st.markdown('<div class="hero-copy">Log in to compare Logistic Regression against Random Forest and explore survey-based screening scores.</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="fact-grid">'
                '<div class="fact-card"><div class="eyebrow">Models</div><b>Logistic Regression +<br>Random Forest</b></div>'
                '<div class="fact-card"><div class="eyebrow">Output</div><b>Cancer / No Cancer<br>with model score</b></div>'
                '<div class="fact-card"><div class="eyebrow">Experience</div><b>Responsive glass<br>dashboard</b></div>'
                '</div>',
                unsafe_allow_html=True,
            )
    with right:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Email access</div>', unsafe_allow_html=True)
            st.markdown("## Enter dashboard")
            st.markdown('<p class="muted">Create an account with your email, then sign in to unlock predictions and metrics.</p>', unsafe_allow_html=True)
            mode = st.radio("Account access", ["Sign In", "Create Account"], horizontal=True, label_visibility="collapsed", key="auth_mode")
            if st.session_state.get("_previous_auth_mode") != mode:
                st.session_state.pop("auth_message", None)
                st.session_state["_previous_auth_mode"] = mode
            if mode == "Sign In":
                with st.form("sign_in_form", clear_on_submit=False):
                    email = st.text_input("Email", placeholder="you@example.com", autocomplete="email", key="login_email")
                    password = st.text_input("Password", type="password", placeholder="Enter your password", autocomplete="current-password", key="login_password")
                    submitted = st.form_submit_button("Log In", use_container_width=True)
                if submitted:
                    clean_email = email.strip().lower()
                    if not clean_email or not password:
                        st.session_state["auth_message"] = ("error", "Enter your email and password.")
                    elif authenticate_user(engine, clean_email, password):
                        st.session_state["auth_user"] = clean_email
                        st.rerun()
                    else:
                        st.session_state["auth_message"] = ("error", "Email or password is incorrect.")
                st.markdown('<p class="small-note">Sign in with the email and password from your account.</p>', unsafe_allow_html=True)
            else:
                with st.form("create_account_form", clear_on_submit=False):
                    email = st.text_input("Email", placeholder="you@example.com", autocomplete="email", key="register_email")
                    password = st.text_input("Password", type="password", placeholder="Minimum 8 characters", autocomplete="new-password", key="register_password")
                    confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password", autocomplete="new-password", key="register_confirm")
                    submitted = st.form_submit_button("Create Account", use_container_width=True)
                if submitted:
                    clean_email, error = validate_registration(email, password, confirm)
                    if error:
                        st.session_state["auth_message"] = ("error", error)
                    else:
                        ok, message = register_user(engine, clean_email, password)
                        st.session_state["auth_message"] = ("success" if ok else "error", message)
                st.markdown('<p class="small-note">Passwords are stored as bcrypt hashes. Sign in after creating your account.</p>', unsafe_allow_html=True)
            message = st.session_state.get("auth_message")
            if message:
                kind, text = message
                (st.success if kind == "success" else st.error)(text)
            if not persistent_database:
                st.caption("Local SQLite account storage is active. For hosted use, configure DATABASE_URL with a persistent PostgreSQL database.")
    st.markdown('<p class="small-note" style="text-align:center;margin-top:12px">Educational screening prototype only. Not a medical diagnosis.</p>', unsafe_allow_html=True)


def format_percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def render_model_comparison(metadata: dict[str, Any]) -> None:
    model_names = list(metadata.get("metrics", {}))
    columns = st.columns(max(1, len(model_names)), gap="small")
    for column, name in zip(columns, model_names):
        metrics = metadata["metrics"][name]
        selected = name == metadata.get("best_model")
        with column:
            with st.container(border=True):
                st.markdown(f"### {name}")
                if selected:
                    st.markdown('<div class="eyebrow">Selected model</div>', unsafe_allow_html=True)
                st.metric("Test accuracy", format_percent(metrics["accuracy"]))
                for label, key in (("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1"), ("5-fold CV accuracy", "cv_accuracy")):
                    st.markdown(f'<div class="small-note">{label}<span style="float:right;color:#eef2f7">{format_percent(metrics[key])}</span></div>', unsafe_allow_html=True)


def make_prediction(model_bundle: dict[str, Any], metadata: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    row = {"GENDER": 1 if answers["gender"] == "Male" else 0, "AGE": int(answers["age"])}
    for name in FEATURES:
        if name not in {"GENDER", "AGE"}:
            key = name.lower()
            row[name] = 1 if answers[key] == "Yes" else 0
    frame = pd.DataFrame([{name: row[name] for name in FEATURES}], columns=FEATURES)
    model = model_bundle["model"]
    label = int(model.predict(frame)[0])
    positive_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(frame)[0][positive_index])
    return {
        "label": label,
        "probability": probability,
        "probability_percent": round(probability * 100, 1),
        "model": model_bundle["best_model"],
        "synthetic_fallback": bool(metadata["synthetic_fallback"]),
    }


def render_dashboard(engine: Engine, persistent_database: bool, model_bundle: dict[str, Any], metadata: dict[str, Any]) -> None:
    with st.container(border=True):
        heading, account = st.columns([5, 1], vertical_alignment="center")
        with heading:
            st.markdown('<div class="eyebrow">Lung cancer survey screening</div>', unsafe_allow_html=True)
            st.title("Clinical Risk Estimator")
            st.caption("Compare two classifiers trained on a public survey dataset, then explore a survey-based prediction from patient features.")
        with account:
            st.caption(st.session_state.get("auth_user", ""))
            if st.button("Log out", use_container_width=True):
                for key in ("auth_user", "prediction_result", "prediction_message"):
                    st.session_state.pop(key, None)
                st.rerun()
        best_name = metadata.get("best_model", "—")
        best_accuracy = metadata.get("metrics", {}).get(best_name, {}).get("accuracy", 0)
        stat_columns = st.columns(3)
        stat_columns[0].metric("Best model", best_name)
        stat_columns[1].metric("Survey rows", f"{int(metadata.get('dataset_rows', 0)):,}")
        stat_columns[2].metric("Selected test accuracy", format_percent(best_accuracy))

    if metadata.get("synthetic_fallback"):
        st.markdown('<div class="notice">Training with a synthetic fallback dataset. Scores and predictions are for interface demonstration only.</div>', unsafe_allow_html=True)
    if not persistent_database:
        st.info("This deployment is using local SQLite. Hosted platforms may discard local files; configure a persistent PostgreSQL DATABASE_URL before accepting customer accounts.")

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Model performance</div>', unsafe_allow_html=True)
        st.subheader("Accuracy test")
        st.caption("Held-out scores and five-fold cross-validation results from this training run.")
        render_model_comparison(metadata)

    form_column, result_column = st.columns([1.62, 0.96], gap="medium", vertical_alignment="top")
    with form_column:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Prediction form</div>', unsafe_allow_html=True)
            st.subheader("Patient features")
            with st.form("prediction_form", clear_on_submit=False):
                answers: dict[str, Any] = {}
                binary_names = [name for name in FEATURES if name not in {"GENDER", "AGE"}]
                field_rows: list[tuple[str | None, str | None]] = [("AGE", "GENDER")] + [(binary_names[i], binary_names[i + 1] if i + 1 < len(binary_names) else None) for i in range(0, len(binary_names), 2)]
                for left_name, right_name in field_rows:
                    left_col, right_col = st.columns(2, gap="small")
                    for column, name in ((left_col, left_name), (right_col, right_name)):
                        if name is None:
                            continue
                        with column:
                            if name == "AGE":
                                answers["age"] = st.text_input("Age", placeholder="e.g. 58", key="feature_age")
                            elif name == "GENDER":
                                answers["gender"] = st.selectbox("Gender", ["Select", "Male", "Female"], key="feature_gender")
                            else:
                                key = name.lower()
                                answers[key] = st.selectbox(LABELS[name].replace("_", " ").title(), ["Select", "No", "Yes"], key=f"feature_{key}")
                submitted = st.form_submit_button("Run screening estimate", use_container_width=True)
            if submitted:
                missing = [key for key, value in answers.items() if not value or value == "Select"]
                if missing:
                    st.session_state["prediction_message"] = "Please complete all fields before submitting."
                    st.session_state.pop("prediction_result", None)
                else:
                    try:
                        age = int(answers["age"])
                        if not 18 <= age <= 120:
                            raise ValueError
                        answers["age"] = age
                        st.session_state["prediction_result"] = make_prediction(model_bundle, metadata, answers)
                        st.session_state.pop("prediction_message", None)
                    except ValueError:
                        st.session_state["prediction_message"] = "Enter an age between 18 and 120."
                        st.session_state.pop("prediction_result", None)
            if st.session_state.get("prediction_message"):
                st.error(st.session_state["prediction_message"])

    with result_column:
        with st.container(border=True):
            result = st.session_state.get("prediction_result")
            if result:
                high = result["label"] == 1
                badge = "HIGHER SURVEY MATCH" if high else "LOWER SURVEY MATCH"
                css_class = "result-high" if high else "result-low"
                st.markdown(f'<div class="eyebrow">{badge}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-score {css_class}">{result["probability_percent"]:.1f}%</div>', unsafe_allow_html=True)
                st.subheader("Cancer-associated pattern detected" if high else "No-cancer pattern detected")
                st.progress(min(100, max(0, int(result["probability_percent"]))))
                if high:
                    st.caption("The model matched this record to the cancer-associated class in its training survey. This is not a diagnosis.")
                else:
                    st.caption("The model matched this record to the no-cancer class in its training survey. A lower score cannot rule out cancer.")
                st.caption(f"Selected classifier: {result['model']}")
            else:
                st.markdown('<div class="eyebrow">Model score</div>', unsafe_allow_html=True)
                st.subheader("Your result will appear here")
                st.progress(0)
                st.caption("Complete the questionnaire to see the selected model’s survey classification.")
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Performance text</div>', unsafe_allow_html=True)
            st.subheader(metadata.get("best_model", "Selected model"))
            if metadata.get("synthetic_fallback"):
                st.caption("This demo run uses generated fallback records because the public survey could not be loaded. Its metrics do not describe patient data.")
            else:
                st.caption(f"{metadata.get('best_model')} was selected by five-fold cross-validation on the {int(metadata.get('dataset_rows', 0)):,}-row survey dataset.")
            st.markdown('<div class="eyebrow">Top feature signals</div>', unsafe_allow_html=True)
            chips = "".join(f'<span class="signal-chip">{item["label"]} <b>{item["score"]:.2f}</b></span>' for item in metadata.get("feature_importances", []))
            st.markdown(chips or "Signals are not available yet.", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Important note</div>', unsafe_allow_html=True)
            st.markdown("This educational prototype classifies patterns in a small survey dataset. Its score is not a clinical probability or diagnosis, and it cannot rule cancer in or out. For symptoms or health concerns, contact a licensed clinician.")


def main() -> None:
    inject_styles()
    database_url = database_url_from_secrets()
    engine = load_database(database_url)
    model_bundle, model_metadata = load_models()
    persistent_database = engine.dialect.name != "sqlite"
    if st.session_state.get("auth_user"):
        render_dashboard(engine, persistent_database, model_bundle, model_metadata)
    else:
        render_sign_in(engine, persistent_database)


if __name__ == "__main__":
    main()
