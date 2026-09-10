import pytest

import julieta.tracking.mlflow_config as mlflow_config_module
from julieta.tracking.mlflow_config import TRACKING_URI_ENV_VAR, configure_mlflow, log_run


def test_does_not_pollute_os_environ_with_dotenv_values():
    # Regresión: antes usaba load_dotenv(), que vuelca TODO .env al proceso --
    # eso se filtraba a cualquier subprocess llamado despues (ej. dvc push
    # heredaba AZURE_STORAGE_CONNECTION_STRING de otra cuenta de Azure y DVC
    # la priorizaba sobre az login). No debe volver a pasar.
    assert not hasattr(mlflow_config_module, "load_dotenv")


def test_raises_when_tracking_uri_is_missing(monkeypatch):
    monkeypatch.delenv(TRACKING_URI_ENV_VAR, raising=False)
    monkeypatch.setattr("julieta.tracking.mlflow_config._DOTENV_VALUES", {})

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


def test_log_run_logs_model_when_provided(monkeypatch):
    # Regresión: log_run() usaba mlflow.sklearn.log_model(), que en mlflow>=3
    # siempre intenta crear una entidad "Logged Model" via
    # POST /api/2.0/mlflow/logged-models -- el tracking server de Azure ML no
    # implementa ese endpoint (404 real, confirmado corriendo esto). Ahora usa
    # joblib.dump() + mlflow.log_artifact(), que sí funciona contra ese server.
    import joblib

    monkeypatch.setenv(TRACKING_URI_ENV_VAR, "azureml://fake-tracking-uri")
    calls = {}
    _patch_mlflow_calls(monkeypatch, calls)
    fake_model = {"coef": [1, 2, 3]}

    # El path que se le pasa a mlflow.log_artifact vive en un directorio
    # temporal que log_run() borra al salir de su `with` -- se lee el
    # contenido acá, dentro del mock, mientras el archivo todavía existe.
    def _fake_log_artifact(path):
        calls.setdefault("artifacts", []).append(path)
        if path.endswith("model.joblib"):
            calls["logged_model_content"] = joblib.load(path)

    monkeypatch.setattr("julieta.tracking.mlflow_config.mlflow.log_artifact", _fake_log_artifact)

    run_id = log_run(
        experiment_id="ID01",
        experiment_name="some_experiment",
        author="valentin",
        model=fake_model,
    )

    assert run_id == "fake-run-id"
    assert len(calls["artifacts"]) == 1
    assert calls["artifacts"][0].endswith("model.joblib")
    assert calls["logged_model_content"] == fake_model
