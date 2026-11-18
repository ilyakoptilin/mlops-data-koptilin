from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient


def test_mlflow_runs_structure():
    verification_path = Path("verification.json")
    assert verification_path.exists(), "verification.json must exist"
    ver = json.loads(verification_path.read_text(encoding="utf-8"))

    tracking_uri = ver["mlflow_tracking_uri"]
    experiment_id = ver["experiment_id"]
    session_id = ver["session_id"]
    best_run_id = ver["best_run_id"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.session_id = '{session_id}'",
    )
    assert len(runs) >= 6, f"Expected at least 6 runs, got {len(runs)}"

    families = runs["tags.model_family"].value_counts().to_dict()
    assert families.get("logistic_regression", 0) >= 3
    assert families.get("random_forest", 0) >= 3

    required_val_metrics = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "training_time",
    ]
    required_test_metrics = [
        "test_accuracy",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
    ]

    for _, row in runs.iterrows():
        rid = row["run_id"]
        run = client.get_run(rid)
        assert "dataset_id" in run.data.tags
        assert "dataset_sha256" in run.data.tags
        assert "model_family" in run.data.tags
        assert "model_type" in run.data.params
        assert "random_state" in run.data.params

        for m in required_val_metrics:
            assert m in run.data.metrics, f"Missing {m} in run {rid}"

        artifacts = [a.path for a in client.list_artifacts(rid)]
        assert "evaluation" in artifacts
        run_model_info = get_model_info(f"runs:/{rid}/model")
        assert run_model_info.signature is not None

        if rid == best_run_id:
            for tm in required_test_metrics:
                assert tm in run.data.metrics, f"Missing {tm} in best run {rid}"
            info = get_model_info(f"runs:/{rid}/model")
            assert info.signature is not None
        else:
            for tm in required_test_metrics:
                assert tm not in run.data.metrics, f"Test metric {tm} leaked to non-best run {rid}"

    assert ver["checks"]["predictions_match"] is True
    assert ver["checks"]["signature_present"] is True
