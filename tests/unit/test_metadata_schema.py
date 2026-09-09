import pytest
from pydantic import ValidationError

from julieta.utils.metadata_schema import ExperimentMetadata

BASE_FIELDS = {
    "name": "some_experiment",
    "author": "someone",
    "domain": "some_domain",
    "dataset_version": "v1",
    "start_date": "2026-08-24",
    "status": "planned",
    "type": "experiment",
}


def test_pending_id_is_valid():
    ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "IDXX"})


def test_assigned_id_is_valid():
    ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "ID07"})


def test_arbitrary_id_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "experiment-1"})


def test_type_is_required():
    fields = {k: v for k, v in BASE_FIELDS.items() if k != "type"}
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**fields, "id": "ID01"})


def test_data_report_type_is_valid():
    metadata = ExperimentMetadata.model_validate(
        {**BASE_FIELDS, "id": "ID01", "type": "data_report"}
    )
    assert metadata.type == "data_report"


def test_arbitrary_type_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "ID01", "type": "notebook"})


def test_author_is_required():
    fields = {k: v for k, v in BASE_FIELDS.items() if k != "author"}
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**fields, "id": "ID01"})


def test_author_rejects_empty_string():
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "ID01", "author": ""})


def test_domain_is_required():
    fields = {k: v for k, v in BASE_FIELDS.items() if k != "domain"}
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**fields, "id": "ID01"})


def test_domain_rejects_empty_string():
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "ID01", "domain": ""})


def test_dataset_version_is_required():
    fields = {k: v for k, v in BASE_FIELDS.items() if k != "dataset_version"}
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**fields, "id": "ID01"})


def test_dataset_version_rejects_empty_string():
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "ID01", "dataset_version": ""})


def test_objective_defaults_to_empty_string():
    metadata = ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "ID01"})
    assert metadata.objective == ""


def test_objective_can_be_set():
    metadata = ExperimentMetadata.model_validate(
        {**BASE_FIELDS, "id": "ID01", "objective": "Reducir el tiempo de carga en un 20%"}
    )
    assert metadata.objective == "Reducir el tiempo de carga en un 20%"
