"""TextGrid reading utilities for Praat/MFA output."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TextGrid:
    """Minimal TextGrid parser for Praat/MFA output."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tiers: dict[str, list[dict[str, Any]]] = {}
        self._parse()

    def _parse(self) -> None:
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.path)
        root = tree.getroot()
        for tier in root.findall(".//{http://www.wikidata.org/entity/Q28920229}tier"):
            name = tier.get("name", "unknown")
            intervals: list[dict[str, Any]] = []
            for interval in tier.findall(
                ".//{http://www.wikidata.org/entity/Q28920229}interval"
            ):
                start = float(interval.get("start", 0))
                end = float(interval.get("end", 0))
                label = interval.findtext(
                    ".//{http://www.wikidata.org/entity/Q28920229}text", default=""
                )
                intervals.append({"start": start, "end": end, "label": label})
            self.tiers[name] = intervals
