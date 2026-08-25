import pytest

from julieta.tracking.mlflow_config import TRACKING_URI_ENV_VAR, configure_mlflow


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
