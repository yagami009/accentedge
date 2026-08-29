"""Input/output timeline mapping for streaming."""

from __future__ import annotations


class TimingMapping:
    def __init__(
        self,
        input_start: int,
        input_end: int,
        output_start: int,
        output_end: int,
    ) -> None:
        self.input_start = input_start
        self.input_end = input_end
        self.output_start = output_start
        self.output_end = output_end

    def contains_output(self, sample: int) -> bool:
        return self.output_start <= sample < self.output_end


class Timeline:
    def __init__(self) -> None:
        self.mappings: list[TimingMapping] = []

    def add(
        self,
        input_start: int,
        input_end: int,
        output_start: int,
        output_end: int,
    ) -> None:
        self.mappings.append(
            TimingMapping(input_start, input_end, output_start, output_end)
        )

    def get_output_for_input(self, input_sample: int) -> int | None:
        for m in self.mappings:
            if m.input_start <= input_sample < m.input_end:
                frac = (input_sample - m.input_start) / max(
                    m.input_end - m.input_start, 1
                )
                return m.output_start + int(frac * (m.output_end - m.output_start))
        return None

    def get_input_for_output(self, output_sample: int) -> int | None:
        for m in self.mappings:
            if m.contains_output(output_sample):
                frac = (output_sample - m.output_start) / max(
                    m.output_end - m.output_start, 1
                )
                return m.input_start + int(frac * (m.input_end - m.input_start))
        return None

    def total_input_samples(self) -> int:
        if not self.mappings:
            return 0
        return max(m.input_end for m in self.mappings)

    def total_output_samples(self) -> int:
        if not self.mappings:
            return 0
        return max(m.output_end for m in self.mappings)

    def drift_samples(self) -> float:
        total_in = self.total_input_samples()
        total_out = self.total_output_samples()
        return float(total_out - total_in)
