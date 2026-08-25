import pytest
from pydantic import ValidationError

from julieta.utils.metadata_schema import ExperimentMetadata

BASE_FIELDS = {
    "name": "some_experiment",
    "author": "someone",
    "start_date": "2026-08-24",
    "status": "planned",
}


def test_pending_id_is_valid():
    ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "IDXX"})


def test_assigned_id_is_valid():
    ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "ID07"})


def test_arbitrary_id_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate({**BASE_FIELDS, "id": "experiment-1"})
