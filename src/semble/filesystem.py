import os
from collections.abc import Iterator
from pathlib import Path


def walk_files(root: Path, extensions: frozenset[str], ignore: frozenset[str]) -> Iterator[Path]:
    """Yield files under root matching extensions, skipping ignored directories."""
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = sorted(d for d in dirnames if d not in ignore)
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in extensions:
                yield file_path
