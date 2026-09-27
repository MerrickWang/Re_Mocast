"""Frame stores: uniform, lazily loaded access to raw precipitation frames.

Every dataset adapter describes *where the frames live* by producing a list of
:class:`FrameStore` objects.  A store is a logical 1-D sequence of 2-D frames,
which keeps the window/split logic in :mod:`mocast.datasets.base` dataset
agnostic (requirement FR-DATA-01).
"""

from __future__ import annotations

import abc
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "FrameStore",
    "ArrayStore",
    "NpyStore",
    "NpzStore",
    "H5Store",
    "MultiFrameStore",
    "PngSequenceStore",
    "open_store",
]


class FrameStore(abc.ABC):
    """A logical sequence of ``len(store)`` frames of shape ``[H, W]``."""

    kind = "base"

    def __init__(self, name: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.meta: Dict[str, Any] = dict(meta or {})

    @abc.abstractmethod
    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, index: int) -> np.ndarray:  # pragma: no cover - interface
        """Return frame ``index`` as ``float32 [H, W]`` *without* normalisation."""
        raise NotImplementedError

    def read_many(self, start: int, length: int) -> np.ndarray:
        return np.stack([self.read(start + i) for i in range(length)], axis=0)

    @property
    def shape(self) -> tuple:
        return (len(self), int(self.meta.get("height", 0)), int(self.meta.get("width", 0)))

    def close(self) -> None:  # pragma: no cover - optional
        return None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}(name={self.name!r}, len={len(self)})"


class ArrayStore(FrameStore):
    """In-memory store (synthetic data, tests, cached subsets)."""

    kind = "array"

    def __init__(self, array: np.ndarray, name: str = "array", meta: Optional[Dict[str, Any]] = None) -> None:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3:
            raise ValueError(f"ArrayStore expects [T,H,W] got {array.shape}")
        info = dict(meta or {})
        info.update(height=array.shape[1], width=array.shape[2])
        super().__init__(name, info)
        self._array = np.ascontiguousarray(array, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def read(self, index: int) -> np.ndarray:
        return self._array[index]


class NpyStore(FrameStore):
    """Memory-mapped ``.npy`` file with ``[T,H,W]`` (or ``[T,1,H,W]``) layout."""

    kind = "npy"

    def __init__(self, path: str, name: Optional[str] = None, key: Optional[int] = None,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if key is not None:
            array = array[key]
        self._array = array
        info = dict(meta or {})
        info.update(path=os.path.abspath(path))
        info["height"] = int(array.shape[-2])
        info["width"] = int(array.shape[-1])
        info["frames"] = int(np.prod(array.shape[:-2])) if array.ndim > 2 else 1
        super().__init__(name or os.path.basename(path), info)
        self._flat = array.reshape(-1, array.shape[-2], array.shape[-1])

    def __len__(self) -> int:
        return int(self._flat.shape[0])

    def read(self, index: int) -> np.ndarray:
        return np.asarray(self._flat[index], dtype=np.float32)


class NpzStore(FrameStore):
    """``.npz`` archive (MeteoNet style: one key holds the frame)."""

    kind = "npz"

    def __init__(self, path: str, key: Optional[str] = None, name: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        with np.load(path, allow_pickle=False) as data:
            if key is None:
                keys = [k for k in data.files if k != "metadata"]
                if not keys:
                    raise ValueError(f"No array found in {path}")
                key = keys[0]
            array = np.asarray(data[key])
        info = dict(meta or {})
        info.update(path=os.path.abspath(path), npz_key=key)
        info["height"] = int(array.shape[-2])
        info["width"] = int(array.shape[-1])
        super().__init__(name or os.path.basename(path), info)
        self._flat = array.reshape(-1, array.shape[-2], array.shape[-1])

    def __len__(self) -> int:
        return int(self._flat.shape[0])

    def read(self, index: int) -> np.ndarray:
        return np.asarray(self._flat[index], dtype=np.float32)


class H5Store(FrameStore):
    """HDF5 store, e.g. the official SEVIR VIL event files (``vil`` dataset).

    Any leading dimension combination is flattened to a frame index
    (``[N,49,384,384] -> N*49 frames``), which lets the SEVIR adapter treat every
    event as an independent contiguous sequence via ``frames_per_event``.
    """

    kind = "h5"

    def __init__(self, path: str, key: str = "vil", name: Optional[str] = None,
                 frame_offset: int = 0, frame_count: Optional[int] = None,
                 frames_per_group: Optional[int] = None,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        import h5py  # lazy: keeps h5py optional for synthetic / test paths

        self.path = os.path.abspath(path)
        self.key = key
        self._handle = None
        self._offset = int(frame_offset)
        with h5py.File(self.path, "r") as handle:
            data = handle[key]
            self._raw_shape = tuple(int(s) for s in data.shape)
            height, width = int(data.shape[-2]), int(data.shape[-1])
        total = int(np.prod(self._raw_shape[:-2])) if len(self._raw_shape) > 2 else 1
        self._count = total - self._offset
        if frame_count is not None:
            self._count = min(self._count, int(frame_count))
        info = dict(meta or {})
        info.update(path=self.path, h5_key=key, height=height, width=width)
        if frames_per_group:
            info["frames_per_group"] = int(frames_per_group)
        super().__init__(name or os.path.basename(path), info)
        self._flat_shape = (total, height, width)

    # ------------------------------------------------------------------ io
    def _file(self):
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return int(self._count)

    def read(self, index: int) -> np.ndarray:
        if index < 0 or index >= self._count:
            raise IndexError(index)
        flat = self._offset + index
        row = flat % self._flat_shape[1]
        group = flat // self._flat_shape[1]
        data = self._file()[self.key]
        frame = data if len(self._raw_shape) == 3 else data[group, row]
        return np.asarray(frame, dtype=np.float32)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self) -> Dict[str, Any]:  # pragma: no cover - dataloader workers
        state = self.__dict__.copy()
        state["_handle"] = None
        return state


class MultiFrameStore(FrameStore):
    """Chain of single-frame files (one ``.npy``/``.npz`` per timestep)."""

    kind = "multi"

    def __init__(self, paths: Sequence[str], name: str = "multi", npz_key: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        if not paths:
            raise ValueError("MultiFrameStore requires at least one path")
        self.paths: List[str] = [os.path.abspath(p) for p in paths]
        self.npz_key = npz_key
        first = self._read_path(self.paths[0])
        info = dict(meta or {})
        info.update(height=int(first.shape[-2]), width=int(first.shape[-1]), n_files=len(self.paths))
        super().__init__(name, info)

    def _read_path(self, path: str) -> np.ndarray:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as data:
                key = self.npz_key or [k for k in data.files if k != "metadata"][0]
                array = np.asarray(data[key])
        elif path.endswith(".npy"):
            array = np.load(path, allow_pickle=False)
        else:
            raise ValueError(f"Unsupported frame file '{path}'")
        return array.reshape(-1, array.shape[-2], array.shape[-1])

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, index: int) -> np.ndarray:
        array = self._read_path(self.paths[index])
        return np.asarray(array[0], dtype=np.float32)


class H5SequenceStore(FrameStore):
    """One [T,H,W] dataset in a grouped archive; no persistent worker handles."""

    kind = "h5_sequence"

    def __init__(self, path: str, key: str, shape: Sequence[int],
                 scale: float, offset: float = 0.0,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        self.path = os.path.abspath(path)
        self.key = key
        self.scale, self.offset = float(scale), float(offset)
        self.frames = int(shape[0])
        stat = os.stat(self.path)
        info = dict(meta or {})
        info.update(path=self.path, h5_key=key, height=int(shape[1]),
                    width=int(shape[2]), value_scale=self.scale, value_offset=self.offset,
                    file_size=stat.st_size, file_mtime_ns=stat.st_mtime_ns)
        super().__init__(key, info)

    def __len__(self) -> int:
        return self.frames

    def read_many(self, start: int, length: int) -> np.ndarray:
        import h5py

        if start < 0 or length < 0 or start + length > self.frames:
            raise IndexError((start, length))
        with h5py.File(self.path, "r") as handle:
            array = np.asarray(handle[self.key][start:start + length], dtype=np.float32)
        return array * self.scale + self.offset

    def read(self, index: int) -> np.ndarray:
        return self.read_many(index, 1)[0]


class PngSequenceStore(FrameStore):
    """A time-ordered sequence of single-channel PNG radar frames.

    Shanghai radar frames store reflectivity as an 8-bit image.  ``scale`` and
    ``offset`` convert the stored pixel value into the physical unit returned by
    :meth:`read` (dBZ for the default Shanghai configuration).
    """

    kind = "png_sequence"

    def __init__(self, paths: Sequence[str], name: str = "png_sequence",
                 scale: float = 1.0, offset: float = 0.0,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        if not paths:
            raise ValueError("PngSequenceStore requires at least one path")
        self.paths: List[str] = [os.path.abspath(p) for p in paths]
        self.scale = float(scale)
        self.offset = float(offset)
        first = self._read_png(self.paths[0])
        info = dict(meta or {})
        info.update(height=int(first.shape[0]), width=int(first.shape[1]),
                    n_files=len(self.paths), value_scale=self.scale,
                    value_offset=self.offset)
        super().__init__(name, info)

    @staticmethod
    def _read_png(path: str) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as image:
            # The archive is grayscale; conversion also makes palette/RGB input
            # deterministic if a differently encoded frame is encountered.
            array = np.asarray(image.convert("L"), dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"PNG frame must be 2-D after grayscale conversion: {path}")
        return array

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, index: int) -> np.ndarray:
        frame = self._read_png(self.paths[index])
        return frame * self.scale + self.offset


def open_store(path: str, **kwargs: Any) -> FrameStore:
    """Open ``path`` picking the store implementation from the extension."""
    lowered = path.lower()
    if lowered.endswith(".npy"):
        return NpyStore(path, **kwargs)
    if lowered.endswith(".npz"):
        return NpzStore(path, **kwargs)
    if lowered.endswith((".h5", ".hdf5")):
        return H5Store(path, **kwargs)
    raise ValueError(f"Unsupported store format: {path}")
