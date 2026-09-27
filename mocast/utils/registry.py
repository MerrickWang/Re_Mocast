"""Tiny decorator based registries (models / datasets / losses / metrics)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

T = TypeVar("T")

__all__ = [
    "Registry",
    "register_model",
    "register_dataset",
    "register_loss",
    "register_metric",
    "MODELS",
    "DATASETS",
    "LOSSES",
    "METRICS",
]


class Registry:
    """Name -> object registry with helpful error messages."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, obj: T, name: Optional[str] = None) -> T:
        key = (name or getattr(obj, "registered_name", None) or obj.__name__).lower()
        if key in self._items and self._items[key] is not obj:
            raise KeyError(f"{self.name} registry already contains '{key}'")
        self._items[key] = obj
        return obj

    def __call__(self, obj: Optional[T] = None, name: Optional[str] = None):
        if obj is None:  # used as @REGISTRY(name="x")
            def decorator(inner: T) -> T:
                return self.register(inner, name)

            return decorator
        return self.register(obj, name)

    def get(self, key: str) -> Any:
        try:
            return self._items[key.lower()]
        except KeyError as exc:
            raise KeyError(
                f"Unknown {self.name} '{key}'. Available: {sorted(self._items)}"
            ) from exc

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._items

    def keys(self) -> List[str]:
        return sorted(self._items)

    def items(self) -> Iterable[Any]:
        return self._items.items()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Registry({self.name}, {sorted(self._items)})"


MODELS = Registry("model")
DATASETS = Registry("dataset")
LOSSES = Registry("loss")
METRICS = Registry("metric")

register_model = MODELS
register_dataset = DATASETS
register_loss = LOSSES
register_metric = METRICS
