import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

PENDING_ID = "IDXX"
ASSIGNED_ID_PATTERN = re.compile(r"^ID\d+$")


class ExperimentStatus(StrEnum):
    PLANNED = "planned"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class ExperimentMetrics(BaseModel):
    primary_metric_name: str = ""
    target_value: float | None = None
    final_value: float | None = None


class ExperimentMetadata(BaseModel):
    id: str

    @field_validator("id")
    @classmethod
    def id_is_pending_or_assigned(cls, value: str) -> str:
        if value == PENDING_ID or ASSIGNED_ID_PATTERN.match(value):
            return value
        raise ValueError(
            f'id debe ser "{PENDING_ID}" (pendiente de asignar) o seguir el patrón "ID<numero>", '
            f"recibido: {value!r}"
        )

    name: str
    author: str = ""
    colaborators: list[str] = Field(default_factory=list)
    start_date: date
    end_date: date | None = None

    status: ExperimentStatus
    domain: str = ""
    tags: list[str] = Field(default_factory=list)

    dataset_version: str = ""
    model_type: str = ""

    metrics: ExperimentMetrics = Field(default_factory=ExperimentMetrics)
    conclusion: str = ""
