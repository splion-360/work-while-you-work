from mlflow.tracking import MlflowClient

from scripts.train_classifier import configure_mlflow


def test_configure_mlflow_restores_a_soft_deleted_experiment(tmp_path):
    tracking_db = tmp_path / "tracking.db"
    artifact_root = tmp_path / "artifacts"
    experiment_name = "deleted-experiment"
    experiment_id = configure_mlflow(tracking_db, artifact_root, experiment_name)
    client = MlflowClient()
    client.delete_experiment(experiment_id)

    restored_id = configure_mlflow(tracking_db, artifact_root, experiment_name)

    assert restored_id == experiment_id
    assert client.get_experiment(experiment_id).lifecycle_stage == "active"
