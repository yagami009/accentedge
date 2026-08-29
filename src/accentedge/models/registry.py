"""Model registry for streaming candidates."""

from __future__ import annotations

from typing import Any

from accentedge.models.models.interfaces import StreamingCandidate


class ModelRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[StreamingCandidate]] = {}

    def register(self, arch_id: str, cls: type[StreamingCandidate]) -> None:
        if arch_id in self._registry:
            raise ValueError(f"Architecture {arch_id!r} already registered")
        self._registry[arch_id] = cls

    def get(self, arch_id: str) -> type[StreamingCandidate]:
        if arch_id not in self._registry:
            raise KeyError(f"Architecture {arch_id!r} not registered")
        return self._registry[arch_id]

    def list_available(self) -> list[str]:
        return list(self._registry.keys())

    def create(self, arch_id: str, config: dict[str, Any]) -> StreamingCandidate:
        cls = self.get(arch_id)
        instance = cls.__new__(cls)
        if hasattr(instance, "metadata") and not isinstance(
            instance.metadata, type
        ):
            instance.metadata.architecture_id = arch_id
        return instance

    def sweep_configs(
        self,
        arch_id: str,
        base_config: dict[str, Any],
        chunk_sizes: list[int] | None = None,
        lookahead_values: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate config dicts for each (chunk_size, lookahead) pair.

        Useful for iterating over the sweep space when using a registered
        architecture.
        """
        from accentedge.benchmark.sweeps import CHUNK_SIZES_MS, LOOKAHEAD_SIZES_MS

        chunk_sizes = chunk_sizes if chunk_sizes is not None else list(CHUNK_SIZES_MS)
        lookahead_values = (
            lookahead_values
            if lookahead_values is not None
            else list(LOOKAHEAD_SIZES_MS)
        )
        configs: list[dict[str, Any]] = []
        for chunk_size in chunk_sizes:
            for lookahead in lookahead_values:
                cfg = dict(base_config)
                cfg["chunk_size_ms"] = chunk_size
                cfg["lookahead_ms"] = lookahead
                configs.append(cfg)
        return configs

    def create_for_configs(
        self,
        arch_id: str,
        configs: list[dict[str, Any]],
    ) -> list[StreamingCandidate]:
        """Instantiate the registered architecture for each config dict."""
        return [self.create(arch_id, cfg) for cfg in configs]


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
