from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cached_property
from pathlib import Path
from typing import overload

import numpy as np
import numpy.typing as npt

from semble.index.npz import pack_strings, read_npz, unpack_strings
from semble.types import Chunk


class ChunkTable(Sequence[Chunk]):
    """Chunks stored as arrays, with all content in one UTF-8 buffer. A Chunk is only built when it is accessed."""

    def __init__(
        self,
        files: list[str],
        languages: list[str | None],
        file_ids: npt.NDArray[np.int32],
        start_lines: npt.NDArray[np.int32],
        end_lines: npt.NDArray[np.int32],
        offsets: npt.NDArray[np.int64],
        content: npt.NDArray[np.uint8],
    ) -> None:
        """Create a table; chunk i is in files[file_ids[i]], and its text is content[offsets[i] : offsets[i + 1]]."""
        self.files = files
        self.languages = languages  # one per file
        self.file_ids = file_ids
        self.start_lines = start_lines
        self.end_lines = end_lines
        self._offsets = offsets
        self._content = content

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk]) -> ChunkTable:
        """Store chunks as a table."""
        files: dict[str, int] = {}
        languages: list[str | None] = []
        file_ids, start_lines, end_lines, contents = [], [], [], []
        for chunk in chunks:
            if chunk.file_path not in files:
                files[chunk.file_path] = len(files)
                languages.append(chunk.language)
            file_ids.append(files[chunk.file_path])
            start_lines.append(chunk.start_line)
            end_lines.append(chunk.end_line)
            contents.append(chunk.content.encode())
        return cls(
            list(files),
            languages,
            np.array(file_ids, dtype=np.int32),
            np.array(start_lines, dtype=np.int32),
            np.array(end_lines, dtype=np.int32),
            np.cumsum([0] + [len(content) for content in contents], dtype=np.int64),
            np.frombuffer(b"".join(contents), dtype=np.uint8),
        )

    def save(self, path: Path) -> None:
        """Persist the table to path (an .npz file)."""
        with path.open("wb") as f:
            np.savez(
                f,
                files=pack_strings(self.files),
                languages=pack_strings([language or "" for language in self.languages]),
                file_ids=self.file_ids,
                start_lines=self.start_lines,
                end_lines=self.end_lines,
                offsets=self._offsets,
                content=self._content,
            )

    @classmethod
    def load(cls, path: Path) -> ChunkTable:
        """Load a table saved by save."""
        arrays = read_npz(path)
        files = unpack_strings(arrays["files"])
        languages = [language or None for language in unpack_strings(arrays["languages"])]
        file_ids, start_lines, end_lines, offsets, content = (
            arrays[key] for key in ("file_ids", "start_lines", "end_lines", "offsets", "content")
        )
        count = len(file_ids)
        if (
            len(languages) != len(files)
            or not len(start_lines) == len(end_lines) == len(offsets) - 1 == count
            or offsets[0] != 0
            or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(content)
            or (count and (file_ids.min() < 0 or file_ids.max() >= len(files)))
        ):
            raise ValueError("Persisted chunks are inconsistent")
        return cls(files, languages, file_ids, start_lines, end_lines, offsets, content)

    def __len__(self) -> int:
        """Return the number of chunks."""
        return len(self.file_ids)

    @overload
    def __getitem__(self, index: int) -> Chunk: ...

    @overload
    def __getitem__(self, index: slice) -> list[Chunk]: ...

    def __getitem__(self, index: int | slice) -> Chunk | list[Chunk]:
        """Build the chunk at index, or a list of chunks for a slice."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = range(len(self))[index]  # resolves negative indices and raises IndexError when out of range
        start, end = self._offsets[index], self._offsets[index + 1]
        return Chunk(
            content=self._content[start:end].tobytes().decode(),
            file_path=self.file_path(index),
            start_line=int(self.start_lines[index]),
            end_line=int(self.end_lines[index]),
            language=self.languages[self.file_ids[index]],
        )

    def file_path(self, index: int) -> str:
        """Return the file path of the chunk at index, without building the chunk."""
        return self.files[self.file_ids[index]]

    @cached_property
    def indices_by_file(self) -> dict[str, list[int]]:
        """Return each file's chunk indices, in chunk order."""
        by_file: dict[str, list[int]] = {}
        for index, file_id in enumerate(self.file_ids.tolist()):
            by_file.setdefault(self.files[file_id], []).append(index)
        return by_file

    @cached_property
    def indices_by_language(self) -> dict[str, list[int]]:
        """Return each language's chunk indices, skipping files without a language."""
        by_language: dict[str, list[int]] = {}
        for file_path, language in zip(self.files, self.languages):
            if language:
                by_language.setdefault(language, []).extend(self.indices_by_file[file_path])
        return by_language
