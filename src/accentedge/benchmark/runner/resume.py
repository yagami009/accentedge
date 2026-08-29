"""Resume logic for benchmark runs."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Any

from accentedge.benchmark.schemas import DatasetItem


def load_completed_items(run_dir: Path) -> list[DatasetItem]:
    """Load list of completed DatasetItems from a previous run."""
    state_path = run_dir / "completed_items.jsonl"
    if not state_path.exists():
        return []
    completed: list[DatasetItem] = []
    with open(state_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if "utterance_id" in entry:
                    completed.append(DatasetItem.model_validate(entry))
            except Exception:
                pass
    return completed


def save_completed_item(run_dir: Path, item: DatasetItem | str, metadata: dict[str, Any]) -> None:
    """Append a completed item to the run state."""
    state_path = run_dir / "completed_items.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(item, DatasetItem):
        entry = item.model_dump()
        entry.update(metadata)
    else:
        entry = {"utterance_id": item, **metadata}
    with open(state_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
