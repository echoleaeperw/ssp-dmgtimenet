"""YAML config loader producing nested namespace objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


class Config(Mapping[str, Any]):
    """Read-only nested view that exposes both attribute and item access.

    The class wraps an arbitrary nested ``dict`` (typically loaded from YAML).
    Sub-dicts are recursively wrapped, so callers can do
    ``cfg.model.sp_daca.tau_min`` while still being able to iterate keys for
    serialisation. Lists keep their primitive element ordering, but inner
    dicts inside lists are wrapped on demand.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise TypeError(f"Config expects a mapping, got {type(data)!r}")
        self._data: dict[str, Any] = dict(data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return self._wrap(value)

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self._wrap(self._data[key])
        return default

    def to_dict(self) -> dict[str, Any]:
        return _deep_unwrap(self._data)

    def merged(self, override: Mapping[str, Any]) -> "Config":
        merged = _deep_merge(self._data, override)
        return Config(merged)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping):
            return Config(value)
        if isinstance(value, list):
            return [Config(v) if isinstance(v, Mapping) else v for v in value]
        return value


def _deep_unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return _deep_unwrap(value._data)  # type: ignore[attr-defined]
    if isinstance(value, Mapping):
        return {k: _deep_unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_unwrap(v) for v in value]
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(slots=True)
class ResolvedPaths:
    config_path: Path
    project_root: Path


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load a YAML config from disk and apply optional dictionary overrides.

    The loader supports an ``include`` key whose value is a path (relative to
    the current file) of another YAML file to merge before the current one.
    This lets ``ssp_dmgtimenet.yaml`` re-use everything in ``default.yaml``
    while only overriding specific fields.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Top level of {path} must be a mapping, got {type(raw)!r}")

    include = raw.pop("include", None) if isinstance(raw, dict) else None
    if include:
        if isinstance(include, str):
            include_paths = [include]
        elif isinstance(include, list):
            include_paths = list(include)
        else:
            raise TypeError(f"include directive in {path} must be string or list, got {type(include)!r}")
        merged: dict[str, Any] = {}
        for inc in include_paths:
            inc_path = (path.parent / inc).resolve()
            inc_cfg = load_config(inc_path).to_dict()
            merged = _deep_merge(merged, inc_cfg)
        raw = _deep_merge(merged, raw)

    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw)
