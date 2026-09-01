"""Data lineage tracking."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DataLineage(BaseModel):
    dataset_id: str
    dataset_version: str
    license_id: str
    commercial_use_status: str
    source_url: str | None = None
    source_commit: str | None = None
    download_timestamp: datetime | None = None
    preprocessing_steps: list[str] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataLineage":
        return cls.model_validate(d)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
