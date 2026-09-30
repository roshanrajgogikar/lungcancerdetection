"""Train and serve the screening classifiers used by the Lung Cancer Prediction app.

The survey is a small, self-reported dataset and is suitable for an educational
prototype only. The probability returned by these models is not a clinical risk.
"""
from __future__ import annotations

import json
import os
import pickle
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
DATA_PATH = Path(os.getenv("DATASET_PATH", str(DATA_DIR / "lung-cancer-survey.csv")))
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DATASET_URL = "https://raw.githubusercontent.com/ShinjiniShome/lung_cancer_survey_dataviz/main/Lung%20Cancer%20Survey.csv"

FEATURES = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING",
    "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN",
]
LABELS = {
    "GENDER": "Gender", "AGE": "Age", "SMOKING": "Smoking",
    "YELLOW_FINGERS": "Yellow fingers", "ANXIETY": "Anxiety",
    "PEER_PRESSURE": "Peer pressure", "CHRONIC_DISEASE": "Chronic disease",
    "FATIGUE": "Fatigue", "ALLERGY": "Allergy", "WHEEZING": "Wheezing",
    "ALCOHOL_CONSUMING": "Alcohol consuming", "COUGHING": "Coughing",
    "SHORTNESS_OF_BREATH": "Shortness of breath",
    "SWALLOWING_DIFFICULTY": "Swallowing difficulty", "CHEST_PAIN": "Chest pain",
}


def canonical_name(name: Any) -> str:
    return "_".join("".join(ch if ch.isalnum() else " " for ch in str(name).strip().upper()).split())


def _ensure_dataset() -> tuple[pd.DataFrame, bool, str]:
    """Load the referenced survey, fetching it once when possible."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if not DATA_PATH.exists():
        try:
            request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "LungCancerPrediction/1.0"})
            with urllib.request.urlopen(request, timeout=6) as response:
                content = response.read(2_000_000)
            if not content.startswith(b"GENDER,") and b"LUNG_CANCER" not in content[:1000].upper():
                raise ValueError("The dataset response did not look like the expected survey CSV.")
            DATA_PATH.write_bytes(content)
            downloaded = True
        except Exception:
            return _synthetic_dataset(), True, "Synthetic demo fallback"

    try:
        frame = pd.read_csv(DATA_PATH)
        frame.columns = [canonical_name(column) for column in frame.columns]
        target = next((name for name in ("LUNG_CANCER", "TARGET", "CANCER", "LABEL", "DIAGNOSIS") if name in frame.columns), None)
        if target is None or not set(FEATURES).issubset(frame.columns):
            raise ValueError("The survey is missing required feature or target columns.")
        frame = frame[FEATURES + [target]].copy()
        for name in FEATURES:
            if name == "AGE":
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
            else:
                frame[name] = _coerce_binary(frame[name], gender=(name == "GENDER"))
        frame["_TARGET"] = _coerce_binary(frame[target], gender=False)
        frame = frame.drop(columns=[target]) if target != "_TARGET" else frame
        frame["AGE"] = frame["AGE"].fillna(frame["AGE"].median())
        for name in FEATURES:
            if name != "AGE":
                mode = frame[name].mode(dropna=True)
                frame[name] = frame[name].fillna(int(mode.iloc[0]) if not mode.empty else 0)
        frame = frame.dropna(subset=["_TARGET"])
        frame["_TARGET"] = frame["_TARGET"].astype(int)
        if len(frame) < 30 or frame["_TARGET"].nunique() != 2:
            raise ValueError("The survey does not contain enough records from both classes.")
        source = "Public GitHub survey CSV" if downloaded else "Local survey CSV"
        return frame[FEATURES + ["_TARGET"]], False, source
    except Exception:
        return _synthetic_dataset(), True, "Synthetic demo fallback"


def _coerce_binary(series: pd.Series, gender: bool = False) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    if gender:
        mapped = values.map({"male": 1, "m": 1, "man": 1, "female": 0, "f": 0, "woman": 0, "1": 1, "0": 0})
    else:
        present = values.isin({"yes", "y", "true", "present", "positive"})
        absent = values.isin({"no", "n", "false", "absent", "negative"})
        mapped = pd.Series(np.nan, index=series.index, dtype=float)
        mapped.loc[present] = 1
        mapped.loc[absent] = 0
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_values = set(numeric.dropna().unique())
        if 2 in numeric_values:
            mapped.loc[numeric == 1] = 0
            mapped.loc[numeric == 2] = 1
        else:
            mapped.loc[numeric == 0] = 0
            mapped.loc[numeric == 1] = 1
    return mapped


def _synthetic_dataset(rows: int = 900) -> pd.DataFrame:
    """Deterministic fallback for local demos; never presented as patient data."""
    rng = np.random.default_rng(284)
    result: dict[str, Any] = {"GENDER": rng.integers(0, 2, rows), "AGE": rng.integers(25, 87, rows)}
    binary_features = [name for name in FEATURES if name not in {"GENDER", "AGE"}]
    for name in binary_features:
        result[name] = rng.binomial(1, 0.34 if name in {"SMOKING", "ALCOHOL_CONSUMING"} else 0.38, rows)
    weights = {
        "SMOKING": 0.72, "YELLOW_FINGERS": 0.5, "ANXIETY": 0.25, "PEER_PRESSURE": 0.38,
        "CHRONIC_DISEASE": 0.4, "FATIGUE": 0.42, "ALLERGY": 0.25, "WHEEZING": 0.5,
        "ALCOHOL_CONSUMING": 0.32, "COUGHING": 0.48, "SHORTNESS_OF_BREATH": 0.5,
        "SWALLOWING_DIFFICULTY": 0.35, "CHEST_PAIN": 0.45,
    }
    linear = -3.9 + (result["AGE"] - 25) * 0.035 + result["GENDER"] * 0.12
    for name, weight in weights.items():
        linear = linear + np.asarray(result[name]) * weight
    probability = 1 / (1 + np.exp(-linear))
    result["_TARGET"] = rng.binomial(1, probability)
    return pd.DataFrame(result)[FEATURES + ["_TARGET"]]


def _build_models() -> dict[str, Any]:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.75, solver="liblinear", max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, max_depth=10, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        ),
    }


def _metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predicted = model.predict(X)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def _feature_importances(model: Any) -> list[dict[str, Any]]:
    estimator = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        scores = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        scores = np.abs(np.asarray(estimator.coef_[0], dtype=float))
        total = float(scores.sum())
        scores = scores / total if total else scores
    else:
        scores = np.zeros(len(FEATURES), dtype=float)
    ranked = sorted(zip(FEATURES, scores), key=lambda item: item[1], reverse=True)[:5]
    return [{"name": name, "label": LABELS[name], "score": round(float(score), 3)} for name, score in ranked]


def train_and_save() -> tuple[dict[str, Any], dict[str, Any]]:
    data, synthetic, source = _ensure_dataset()
    X, y = data[FEATURES], data["_TARGET"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates: dict[str, Any] = {}
    results: dict[str, dict[str, float]] = {}
    for name, estimator in _build_models().items():
        scores = cross_validate(
            estimator, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
            n_jobs=1, error_score="raise",
        )
        fitted = estimator.fit(X_train, y_train)
        test_metrics = _metrics(fitted, X_test, y_test)
        results[name] = {
            **test_metrics,
            "cv_accuracy": float(np.mean(scores["test_accuracy"])),
            "cv_roc_auc": float(np.mean(scores["test_roc_auc"])),
        }
        candidates[name] = estimator
    best_name = max(results, key=lambda name: (results[name]["cv_accuracy"], results[name]["cv_roc_auc"]))
    final_model = candidates[best_name].fit(X, y)
    metadata = {
        "best_model": best_name,
        "dataset_rows": int(len(data)),
        "dataset_source": source,
        "synthetic_fallback": bool(synthetic),
        "selection_method": "Highest 5-fold cross-validation accuracy, with ROC-AUC as the tie-breaker",
        "metrics": results,
        "feature_importances": _feature_importances(final_model),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "features": [{"name": name, "label": LABELS[name]} for name in FEATURES],
        "screening_notice": "Educational survey classifier only. Its score is not a medical diagnosis or calibrated clinical probability.",
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps({"model": final_model, "features": FEATURES, "best_model": best_name}, protocol=pickle.HIGHEST_PROTOCOL))
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"model": final_model, "features": FEATURES, "best_model": best_name}, metadata


