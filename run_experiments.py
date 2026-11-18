from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from clearml import Dataset, OutputModel, Task
from mlflow.models import get_model_info, infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_WINE_ID = "998f9aad4e5245509a421e596a01ebeb"
DEFAULT_WINE_SHA256 = "724d80b05e39c7a72fe7631de59583c764049137f9d7089b01b5460e212c75c0"
DEFAULT_EXPERIMENT_NAME = "miit-mlops-practice-4-koptilin"
DEFAULT_CLEARML_PROJECT = "MIIT/MLOps/practice_4/koptilin"
DEFAULT_CLEARML_TASK = "wine-best-model"
RANDOM_STATE = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_wine(dataset_id: str, expected_sha256: str) -> tuple[pd.DataFrame, str]:
    dataset = Dataset.get(dataset_id=dataset_id)
    if not dataset.is_final():
        raise RuntimeError(f"ClearML Dataset {dataset_id} is not finalized.")

    local_dir = Path(dataset.get_local_copy())
    wine_path = local_dir / "wine.csv"
    if not wine_path.exists():
        raise FileNotFoundError(f"wine.csv not found in dataset directory: {local_dir}")

    actual_sha256 = sha256_file(wine_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for wine.csv! Expected {expected_sha256}, got {actual_sha256}"
        )

    df = pd.read_csv(wine_path)
    if "target" not in df.columns:
        raise ValueError("Column 'target' is missing in wine.csv")
    if any(col.startswith("Unnamed:") for col in df.columns):
        raise ValueError("Unexpected index column found in wine.csv")

    return df, actual_sha256


def compute_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision_weighted": float(precision_score(y, preds, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y, preds, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
    }


def save_detailed_evaluation(
    model, X: pd.DataFrame, y: pd.Series, out_dir: Path, prefix: str
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = model.predict(X)
    labels = sorted(y.unique().tolist())

    cm = confusion_matrix(y, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix ({prefix})")
    cm_png_path = out_dir / f"confusion_matrix_{prefix}.png"
    fig.tight_layout()
    fig.savefig(cm_png_path, dpi=150)
    plt.close(fig)

    report = classification_report(y, preds, labels=labels, output_dict=True, zero_division=0)
    report["confusion_matrix"] = cm.tolist()
    report["labels"] = labels
    report_path = out_dir / f"per_class_metrics_{prefix}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return cm_png_path, report_path


def get_configurations() -> list[dict]:
    return [
        {
            "run_name": "logreg-c0.01-lbfgs",
            "model_family": "logistic_regression",
            "params": {
                "model_type": "LogisticRegression",
                "C": 0.01,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=0.01, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
        },
        {
            "run_name": "logreg-c0.1-newton-cg",
            "model_family": "logistic_regression",
            "params": {
                "model_type": "LogisticRegression",
                "C": 0.1,
                "solver": "newton-cg",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
            },
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=0.1, solver="newton-cg", max_iter=1000, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
        },
        {
            "run_name": "logreg-c1.0-saga",
            "model_family": "logistic_regression",
            "params": {
                "model_type": "LogisticRegression",
                "C": 1.0,
                "solver": "saga",
                "max_iter": 2000,
                "random_state": RANDOM_STATE,
            },
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=1.0, solver="saga", max_iter=2000, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
        },
        {
            "run_name": "rf-n50-depth2",
            "model_family": "random_forest",
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 50,
                "max_depth": 2,
                "min_samples_split": 5,
                "random_state": RANDOM_STATE,
            },
            "estimator": RandomForestClassifier(
                n_estimators=50,
                max_depth=2,
                min_samples_split=5,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "run_name": "rf-n100-depth4",
            "model_family": "random_forest",
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 4,
                "min_samples_split": 2,
                "random_state": RANDOM_STATE,
            },
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_split=2,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "run_name": "rf-n200-depthNone",
            "model_family": "random_forest",
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 200,
                "max_depth": "None",
                "min_samples_split": 2,
                "random_state": RANDOM_STATE,
            },
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_STATE,
            ),
        },
    ]


def validate_run_completeness(
    client: MlflowClient, run_id: str, required_params: list[str], required_metrics: list[str]
) -> dict[str, bool]:
    run = client.get_run(run_id)
    logged_params = run.data.params
    logged_metrics = run.data.metrics
    logged_tags = run.data.tags

    params_ok = all(k in logged_params for k in required_params)
    metrics_ok = all(k in logged_metrics for k in required_metrics)
    tags_ok = all(k in logged_tags for k in ["model_family", "dataset_id", "dataset_sha256"])

    try:
        info = get_model_info(f"runs:/{run_id}/model")
        model_ok = info is not None and info.signature is not None
    except Exception:
        model_ok = False

    return {
        "params_complete": params_ok,
        "metrics_complete": metrics_ok,
        "tags_complete": tags_ok,
        "model_artifact_present": model_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MLflow & ClearML experiment tracking for Wine")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_WINE_ID,
        help="ClearML Dataset ID for Wine 1.1.0",
    )
    parser.add_argument(
        "--expected-sha256",
        default=DEFAULT_WINE_SHA256,
        help="Expected SHA-256 of wine.csv",
    )
    parser.add_argument(
        "--clearml-project",
        default=DEFAULT_CLEARML_PROJECT,
        help="ClearML project name",
    )
    parser.add_argument(
        "--clearml-task-name",
        default=DEFAULT_CLEARML_TASK,
        help="ClearML task name",
    )
    parser.add_argument(
        "--skip-clearml-task",
        action="store_true",
        help="Skip ClearML Task and OutputModel creation",
    )
    args = parser.parse_args()

    df, actual_sha256 = load_verified_wine(args.dataset_id, args.expected_sha256)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_remain, X_test, y_remain, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_remain, y_remain, test_size=0.25, random_state=RANDOM_STATE, stratify=y_remain
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    existing_exp = mlflow.get_experiment_by_name(args.experiment_name)
    if existing_exp is None:
        experiment_id = mlflow.create_experiment(args.experiment_name)
    else:
        experiment_id = existing_exp.experiment_id

    client = MlflowClient(tracking_uri=args.tracking_uri)
    session_id = uuid.uuid4().hex[:12]
    artifacts_root = Path("artifacts").resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    configs = get_configurations()
    trained_models: dict[str, object] = {}
    config_by_run: dict[str, dict] = {}

    for cfg in configs:
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=cfg["run_name"],
            tags={
                "model_family": cfg["model_family"],
                "dataset_id": args.dataset_id,
                "dataset_sha256": actual_sha256,
                "session_id": session_id,
                "split_random_state": str(RANDOM_STATE),
            },
        ) as active_run:
            run_id = active_run.info.run_id
            mlflow.log_params(cfg["params"])

            model = cfg["estimator"]
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            training_time = time.perf_counter() - t0

            val_metrics = compute_metrics(model, X_val, y_val)
            val_metrics["training_time"] = float(training_time)
            mlflow.log_metrics(val_metrics)

            run_art_dir = artifacts_root / cfg["run_name"]
            cm_png, report_json = save_detailed_evaluation(
                model, X_val, y_val, run_art_dir, prefix="val"
            )
            mlflow.log_artifact(str(cm_png), artifact_path="evaluation")
            mlflow.log_artifact(str(report_json), artifact_path="evaluation")

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train.iloc[:5],
                serialization_format="cloudpickle",
            )

            trained_models[run_id] = model
            config_by_run[run_id] = cfg

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.session_id = '{session_id}'",
        order_by=["metrics.f1_weighted DESC", "metrics.training_time ASC"],
    )
    if len(runs_df) != len(configs):
        print(f"ERROR: Expected {len(configs)} runs, found {len(runs_df)}", file=sys.stderr)
        return 1

    best_row = runs_df.iloc[0]
    best_run_id = str(best_row["run_id"])
    best_cfg = config_by_run[best_run_id]

    best_model_uri = f"runs:/{best_run_id}/model"
    loaded_model = mlflow.sklearn.load_model(best_model_uri)
    model_info = get_model_info(best_model_uri)
    signature_present = model_info.signature is not None

    orig_preds = trained_models[best_run_id].predict(X_test)
    loaded_preds = loaded_model.predict(X_test)
    predictions_match = bool(np.array_equal(orig_preds, loaded_preds))

    test_metrics = compute_metrics(loaded_model, X_test, y_test)
    test_cm_png, test_report_json = save_detailed_evaluation(
        loaded_model, X_test, y_test, artifacts_root / "best_test", prefix="test"
    )

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(test_cm_png), artifact_path="test_evaluation")
        mlflow.log_artifact(str(test_report_json), artifact_path="test_evaluation")

    required_val_metrics = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "training_time",
    ]
    required_test_metrics = [f"test_{m}" for m in required_val_metrics if m != "training_time"]
    completeness = validate_run_completeness(
        client,
        best_run_id,
        required_params=list(best_cfg["params"].keys()),
        required_metrics=required_val_metrics + required_test_metrics,
    )

    all_checks_ok = (
        predictions_match
        and signature_present
        and all(completeness.values())
    )

    verification = {
        "mlflow_tracking_uri": args.tracking_uri,
        "experiment_id": str(experiment_id),
        "experiment_name": args.experiment_name,
        "session_id": session_id,
        "dataset_id": args.dataset_id,
        "wine_sha256": actual_sha256,
        "split": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "best_run_id": best_run_id,
        "best_run_name": best_cfg["run_name"],
        "best_model_family": best_cfg["model_family"],
        "best_params": best_cfg["params"],
        "validation_metrics": {
            m: float(best_row[f"metrics.{m}"]) for m in required_val_metrics
        },
        "test_metrics": test_metrics,
        "checks": {
            "accepts_dataframe_with_columns": True,
            "predictions_match": predictions_match,
            "signature_present": signature_present,
            **completeness,
        },
        "clearml_task_id": None,
        "clearml_model_id": None,
    }

    verification_path = Path("verification.json").resolve()
    verification_path.write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_artifact(str(verification_path), artifact_path="verification")

    if not args.skip_clearml_task:
        task = Task.init(
            project_name=args.clearml_project,
            task_name=args.clearml_task_name,
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
            output_uri=True,
        )
        task.connect(best_cfg["params"], name="best_model_params")
        task.connect(
            {
                "mlflow_tracking_uri": args.tracking_uri,
                "mlflow_experiment_id": str(experiment_id),
                "mlflow_run_id": best_run_id,
                "clearml_dataset_id": args.dataset_id,
                "wine_sha256": actual_sha256,
                "split_ratios": "60/20/20",
                "random_state": RANDOM_STATE,
            },
            name="provenance",
        )

        logger = task.get_logger()
        for m in required_val_metrics:
            logger.report_single_value(f"val_{m}", float(best_row[f"metrics.{m}"]))
        for k, v in test_metrics.items():
            logger.report_single_value(f"test_{k}", float(v))

        comparison_cols = [
            "run_id",
            "tags.mlflow.runName",
            "tags.model_family",
            "metrics.accuracy",
            "metrics.precision_weighted",
            "metrics.recall_weighted",
            "metrics.f1_weighted",
            "metrics.training_time",
        ]
        param_cols = [c for c in runs_df.columns if c.startswith("params.")]
        comparison_df = runs_df[comparison_cols + sorted(param_cols)].copy()
        comparison_path = artifacts_root / "runs_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)

        task.upload_artifact("verification", artifact_object=str(verification_path))
        task.upload_artifact("runs_comparison", artifact_object=str(comparison_path))
        task.upload_artifact("test_confusion_matrix", artifact_object=str(test_cm_png))
        task.upload_artifact("test_per_class_metrics", artifact_object=str(test_report_json))

        model_file_path = artifacts_root / "best_wine_model.joblib"
        joblib.dump(loaded_model, model_file_path)

        output_model = OutputModel(
            task=task,
            framework="scikit-learn",
            name="wine-best-classifier",
        )
        output_model.update_labels({"class_0": 0, "class_1": 1, "class_2": 2})
        output_model.set_metadata(key="mlflow_run_id", value=best_run_id, v_type="str")
        output_model.set_metadata(key="clearml_dataset_id", value=args.dataset_id, v_type="str")
        output_model.set_metadata(key="wine_sha256", value=actual_sha256, v_type="str")
        output_model.set_metadata(key="model_family", value=best_cfg["model_family"], v_type="str")
        output_model.update_weights(weights_filename=str(model_file_path))
        output_model.publish()

        verification["clearml_task_id"] = str(task.id)
        verification["clearml_model_id"] = str(output_model.id)
        verification_path.write_text(
            json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with mlflow.start_run(run_id=best_run_id):
            mlflow.log_artifact(str(verification_path), artifact_path="verification")

        task.close()

    if not all_checks_ok:
        print("ERROR: Verification checks failed!", file=sys.stderr)
        return 1

    print(f"Experiment ID: {experiment_id}")
    print(f"Best run ID: {best_run_id}")
    print(f"Test F1: {test_metrics['f1_weighted']:.4f}")
    if verification["clearml_task_id"]:
        print(f"ClearML Task ID: {verification['clearml_task_id']}")
        print(f"ClearML Model ID: {verification['clearml_model_id']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
