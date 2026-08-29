"""Import forced-alignment results into benchmark format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import AlignmentSource


class AlignmentImporter:
    """Import forced-alignment results into benchmark format."""

    def __init__(self, alignment_source: AlignmentSource = AlignmentSource.AUTO):
        self.alignment_source = alignment_source

    def import_textgrid(self, path: str | Path) -> dict[str, list[dict[str, Any]]]:
        """Import a Praat/MFA TextGrid file.

        Returns:
            Dict mapping tier names to lists of {start, end, label} dicts
        """
        from .textgrid import TextGrid

        tg = TextGrid(path)
        result: dict[str, list[dict[str, Any]]] = {}
        for tier_name, intervals in tg.tiers.items():
            result[tier_name] = [
                {
                    "start": float(iv["start"]),
                    "end": float(iv["end"]),
                    "label": str(iv["label"]),
                    "alignment_source": self.alignment_source.value,
                }
                for iv in intervals
            ]
        return result
