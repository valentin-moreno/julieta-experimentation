import pytest

from julieta.tracking.mlflow_config import TRACKING_URI_ENV_VAR, configure_mlflow, log_run


def test_raises_when_tracking_uri_is_missing(monkeypatch):
    monkeypatch.delenv(TRACKING_URI_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=TRACKING_URI_ENV_VAR):
        configure_mlflow(experiment_id="ID01", experiment_name="some_experiment")


def test_configures_tracking_uri_and_experiment_name(monkeypatch):
    monkeypatch.setenv(TRACKING_URI_ENV_VAR, "azureml://fake-tracking-uri")

    calls = {}
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config.mlflow.set_tracking_uri",
        lambda uri: calls.setdefault("tracking_uri", uri),
    )
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config.mlflow.set_experiment",
        lambda name: calls.setdefault("experiment_name", name),
    )

    result = configure_mlflow(experiment_id="ID01", experiment_name="some_experiment")

    assert calls["tracking_uri"] == "azureml://fake-tracking-uri"
    assert calls["experiment_name"] == "julieta/ID01-some_experiment"
    assert result == "julieta/ID01-some_experiment"


class _FakeRunInfo:
    run_id = "fake-run-id"


class _FakeRun:
    info = _FakeRunInfo()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _patch_mlflow_calls(monkeypatch, calls):
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config.mlflow.set_tracking_uri",
        lambda uri: calls.setdefault("tracking_uri", uri),
    )
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config.mlflow.set_experiment",
        lambda name: calls.setdefault("experiment_name", name),
    )

    def _fake_start_run(run_name=None):
        calls["run_name"] = run_name
        return _FakeRun()

    monkeypatch.setattr("julieta.tracking.mlflow_config.mlflow.start_run", _fake_start_run)
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config.mlflow.set_tags",
        lambda tags: calls.setdefault("tags", tags),
    )
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config.mlflow.log_params",
        lambda params: calls.setdefault("params", params),
    )
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config.mlflow.log_metric",
        lambda key, value: calls.setdefault("metrics", {}).update({key: value}),
    )
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config.mlflow.log_artifact",
        lambda path: calls.setdefault("artifacts", []).append(path),
    )
    monkeypatch.setattr(
        "julieta.tracking.mlflow_config._current_git_commit",
        lambda: "abc123",
    )


def test_log_run_logs_all_metrics_params_tags_and_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv(TRACKING_URI_ENV_VAR, "azureml://fake-tracking-uri")
    calls = {}
    _patch_mlflow_calls(monkeypatch, calls)

    fake_artifact = tmp_path / "plot.png"
    fake_artifact.write_text("fake", encoding="utf-8")

    run_id = log_run(
        experiment_id="ID01",
        experiment_name="some_experiment",
        author="valentin",
        config={"max_depth": 5, "n_estimators": 100},
        metrics={"roc_auc": 0.87, "precision": 0.81, "recall": 0.79},
        artifacts=[str(fake_artifact)],
        tags={"dataset_version": "v2.1_active_users"},
        run_name="baseline",
    )

    assert run_id == "fake-run-id"
    assert calls["run_name"] == "baseline"
    assert calls["params"] == {"max_depth": 5, "n_estimators": 100}
    assert calls["metrics"] == {"roc_auc": 0.87, "precision": 0.81, "recall": 0.79}
    assert calls["artifacts"] == [str(fake_artifact)]
    assert calls["tags"] == {
        "author": "valentin",
        "git_commit": "abc123",
        "dataset_version": "v2.1_active_users",
    }


def test_log_run_works_with_no_optional_fields(monkeypatch):
    monkeypatch.setenv(TRACKING_URI_ENV_VAR, "azureml://fake-tracking-uri")
    calls = {}
    _patch_mlflow_calls(monkeypatch, calls)

    run_id = log_run(experiment_id="ID01", experiment_name="some_experiment", author="valentin")

    assert run_id == "fake-run-id"
    assert calls["tags"]["author"] == "valentin"
    assert "params" not in calls
    assert "metrics" not in calls
    assert "artifacts" not in calls
