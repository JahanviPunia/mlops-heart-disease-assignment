import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

ARTIFACT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load the cleaned dataset saved during the Colab exploration."""
    df = pd.read_csv(DATA_DIR / "heart_disease_cleaned.csv")
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    return X, y

def build_pipeline(model):
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", model),
    ])

def main():
    X, y = load_data()
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2,
            class_weight="balanced", random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart-Disease-Classification")

    results = []
    fitted_pipelines = {}

    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

        row = {"model": name}
        for metric in scoring:
            row[metric] = float(np.mean(scores[f"test_{metric}"]))
        results.append(row)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", 42)
            for metric in scoring:
                mlflow.log_metric(
                    metric, float(np.mean(scores[f"test_{metric}"]))
                )

        fitted_pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_name]
    best_pipeline.fit(X, y)

    import joblib
    joblib.dump({
        "pipeline": best_pipeline,
        "feature_columns": X.columns.tolist(),
        "model_name": best_name,
    }, ARTIFACT_DIR / "model.joblib")

    summary = {
        "selected_model": best_name,
        "selection_metric": "roc_auc",
        "cross_validation": results,
        "training_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
    }
    (REPORT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("Saved model to artifacts/model.joblib")

if __name__ == "__main__":
    main()
