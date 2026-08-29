"""Articulatory/DDSP candidate for architecture bake-off."""

from accentedge.models.articulatory_ddsp.interfaces import (
    ArticulatoryAccentMapper,
    ArticulatoryEncoder,
    ArticulatoryFrame,
    ArticulatoryFrameSequence,
    DDSPSynthesizer,
)
from accentedge.models.articulatory_ddsp.streaming_config import ArticulatoryStreamingConfig

__all__ = [
    "ArticulatoryFrame",
    "ArticulatoryFrameSequence",
    "ArticulatoryEncoder",
    "ArticulatoryAccentMapper",
    "DDSPSynthesizer",
    "ArticulatoryStreamingConfig",
]
