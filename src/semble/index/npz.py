from __future__ import annotations

import zipfile
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def read_npz(path: Path) -> dict[str, npt.NDArray[Any]]:
    """Read every array in an .npz file, raising ValueError if the file is unreadable."""
    try:
        with np.load(path) as arrays:
            return dict(arrays)
    except zipfile.BadZipFile as exc:  # e.g. truncated by an interrupted save
        raise ValueError(f"Persisted index file {path} is unreadable") from exc


def pack_strings(strings: Collection[str]) -> npt.NDArray[np.uint8]:
    """Encode strings as NUL-terminated UTF-8 in one byte array, so they can be saved in an .npz."""
    packed = "\0".join([*strings, ""])
    if packed.count("\0") != len(strings):
        raise ValueError("Cannot save strings that contain a NUL character")
    return np.frombuffer(packed.encode(), dtype=np.uint8)


def unpack_strings(packed: npt.NDArray[np.uint8]) -> list[str]:
    """Decode strings encoded by pack_strings."""
    return packed.tobytes().decode().split("\0")[:-1]
