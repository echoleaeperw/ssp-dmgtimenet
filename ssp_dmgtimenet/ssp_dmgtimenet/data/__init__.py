"""HighD / NGSIM / OpenACC data ingestion and platoon construction."""

from .highd import (
    HighDRecording,
    HIGHD_TRACKS_COLUMNS,
    HIGHD_TRACKS_META_COLUMNS,
    HIGHD_RECORDING_META_COLUMNS,
    discover_highd_recordings,
    load_highd_recording,
)
from .ngsim import (
    NGSIMRecording,
    NGSIM_REQUIRED_COLUMNS,
    NGSIM_CLASS_MAP,
    NGSIM_FREEWAY_PERIODS,
    FOOT_TO_METER,
    discover_ngsim_recordings,
    load_ngsim_recording,
)
from .platoons import (
    PlatoonSegment,
    PlatoonExtractionConfig,
    PlatoonStationarityThresholds,
    extract_platoon_segments,
    audit_platoon_counts,
    leader_stationarity_features,
)
from .windowing import (
    WindowingConfig,
    PlatoonSample,
    resample_segment_to_target_hz,
    slice_windows,
)
from .dataset import PlatoonDataset, build_platoon_loaders

__all__ = [
    "HighDRecording",
    "HIGHD_TRACKS_COLUMNS",
    "HIGHD_TRACKS_META_COLUMNS",
    "HIGHD_RECORDING_META_COLUMNS",
    "discover_highd_recordings",
    "load_highd_recording",
    "NGSIMRecording",
    "NGSIM_REQUIRED_COLUMNS",
    "NGSIM_CLASS_MAP",
    "NGSIM_FREEWAY_PERIODS",
    "FOOT_TO_METER",
    "discover_ngsim_recordings",
    "load_ngsim_recording",
    "PlatoonSegment",
    "PlatoonExtractionConfig",
    "PlatoonStationarityThresholds",
    "extract_platoon_segments",
    "audit_platoon_counts",
    "leader_stationarity_features",
    "WindowingConfig",
    "PlatoonSample",
    "resample_segment_to_target_hz",
    "slice_windows",
    "PlatoonDataset",
    "build_platoon_loaders",
]
