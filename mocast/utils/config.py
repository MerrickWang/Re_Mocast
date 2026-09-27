"""Configuration system.

The reproduction document requires every non-public or uncertain quantity to be
*parameterised* (risk R1-R10).  All hyper-parameters therefore live in YAML and
are merged at run time; the fully resolved configuration is dumped next to the
run artefacts so that every experiment stays traceable (section 1.1 "结果可追溯").
"""

from __future__ import annotations

import ast
import copy
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import yaml

__all__ = ["Config", "load_config", "parse_cli_overrides", "deep_update"]


class Config(dict):
    """Nested mapping with attribute access, dotted get/set and YAML/JSON IO."""

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        data: Dict[str, Any] = {}
        if mapping:
            data.update(dict(mapping))
        data.update(kwargs)
        for key, value in data.items():
            dict.__setitem__(self, key, Config.wrap(value))

    # ------------------------------------------------------------------ wrap
    @staticmethod
    def wrap(value: Any) -> Any:
        if isinstance(value, Config):
            return value
        if isinstance(value, Mapping):
            return Config(value)
        if isinstance(value, tuple):
            return tuple(Config.wrap(v) for v in value)
        if isinstance(value, list):
            return [Config.wrap(v) for v in value]
        return value

    # -------------------------------------------------------------- protocol
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"Config has no key '{item}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Config.wrap(value)

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, Config.wrap(value))

    def update(self, other: Mapping[str, Any] = (), **kwargs: Any) -> None:  # type: ignore[override]
        for key, value in dict(other, **kwargs).items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key not in self:
            self[key] = default
        return self[key]

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out

    def save(self, path: str, fmt: Optional[str] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fmt = fmt or os.path.splitext(path)[1].lstrip(".")
        with open(path, "w", encoding="utf-8") as handle:
            if fmt in ("json",):
                json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            else:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        return path

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            if path.endswith(".json"):
                data = json.load(handle)
            else:
                data = yaml.safe_load(handle) or {}
        return cls(data)

    # --------------------------------------------------------------- helpers
    def clone(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))

    def merge(self, other: Mapping[str, Any], inplace: bool = False) -> "Config":
        target = self if inplace else self.clone()
        deep_update(target, other)
        return target

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: MutableMapping[str, Any] = self
        for part in parts[:-1]:
            if not isinstance(node.get(part), Mapping):
                node[part] = Config()
            node = node[part]  # type: ignore[assignment]
        node[parts[-1]] = Config.wrap(value)


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``override`` into ``base`` (in place)."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_update(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = Config.wrap(value)
    return base


def _coerce(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        lowered = text.strip().lower()
        if lowered in ("true", "yes"):
            return True
        if lowered in ("false", "no"):
            return False
        if lowered in ("none", "null"):
            return None
        return text


def parse_cli_overrides(items: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Turn ``["model.msm.num_experts=2", ...]`` into a nested dict."""
    overrides: Dict[str, Any] = {}
    if not items:
        return overrides
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key.subkey=value")
        key, value = item.split("=", 1)
        node = overrides
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value.strip())
    return overrides


def load_config(
    paths: Any,
    overrides: Optional[Any] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Config:
    """Load and deep-merge one or more YAML configs, then apply CLI overrides.

    A config may declare ``defaults: [other.yaml, ...]`` (paths relative to the
    declaring file); those are loaded first so that the declaring file only lists
    the differences.  This keeps the experiment files small and makes the
    "all uncertain quantities live in the config" requirement practical.

    Args:
        paths: path or list of paths.  Later files override earlier ones.
        overrides: list of ``"a.b=value"`` strings or a nested mapping.
        extra: nested mapping merged last (highest priority).
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    cfg = Config()
    for path in paths:
        deep_update(cfg, _load_with_defaults(str(path)))
    if isinstance(overrides, Mapping):
        deep_update(cfg, overrides)
    elif overrides:
        deep_update(cfg, parse_cli_overrides(list(overrides)))
    if extra:
        deep_update(cfg, extra)
    return cfg


def _load_with_defaults(path: str, _seen: Optional[List[str]] = None) -> Config:
    _seen = _seen if _seen is not None else []
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"Circular config defaults detected at '{path}'")
    _seen.append(path)
    try:
        cfg = Config.load(path)
        defaults = cfg.pop("defaults", []) or []
        if isinstance(defaults, str):
            defaults = [defaults]
        base = Config()
        resolved: List[str] = []
        for entry in defaults:
            target = str(entry)
            if not os.path.isabs(target):
                target = os.path.join(os.path.dirname(path), target)
            resolved.append(os.path.relpath(target))
            deep_update(base, _load_with_defaults(target, _seen))
        cfg["_loaded_from"] = os.path.relpath(path)
        cfg["_defaults"] = resolved
        merged = Config()
        deep_update(merged, base)
        deep_update(merged, cfg)
        return merged
    finally:
        _seen.pop()
