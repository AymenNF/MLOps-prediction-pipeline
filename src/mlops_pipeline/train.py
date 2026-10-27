"""Train, evaluate and register the model.

Run with:  python -m mlops_pipeline.train
"""

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_pipeline.config import ROOT, load_params
from mlops_pipeline.data import load_data, split, validate


def build_model(C: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter)),
        ]
    )


def evaluate(model: Pipeline, X, y) -> dict[str, float]:
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def run(params: dict | None = None) -> dict[str, float]:
    params = params or load_params()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(params["mlflow"]["experiment"])

    X, y = load_data()
    validate(X, y)
    X_train, X_test, y_train, y_test = split(X, y, **params["data"])

    with mlflow.start_run():
        mlflow.log_params({**params["data"], **params["model"]})

        model = build_model(**params["model"])
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        print(f"Test metrics: {metrics}")

        if metrics["accuracy"] < params["min_accuracy"]:
            raise RuntimeError(
                f"Quality gate failed: accuracy {metrics['accuracy']:.3f} "
                f"< {params['min_accuracy']}"
            )

        mlflow.sklearn.log_model(model, name="model", input_example=X_test.head(2))

    model_path = ROOT / params["paths"]["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    Path(ROOT / params["paths"]["metrics"]).write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {model_path}")
    return metrics


if __name__ == "__main__":
    run()
