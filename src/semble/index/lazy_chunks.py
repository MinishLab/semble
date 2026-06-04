from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import overload

from semble.index.chunk_store import LmdbChunkStore
from semble.types import Chunk

_StemLookup = tuple[dict[str, list[int]], dict[str, list[int]], dict[int, int]]


def _stem_lookup_keys(file_path: str) -> tuple[set[str], set[str]]:
    stem = Path(file_path).stem.lower()
    stem_norm = stem.replace("_", "")
    exact_keys = {stem, stem_norm}
    symbol_keys = {*exact_keys, stem.rstrip("s"), stem_norm.rstrip("s")}
    return exact_keys, symbol_keys


def _build_stem_lookup(chunk_ids: Sequence[int], file_paths: Sequence[str]) -> _StemLookup:
    exact_stems: dict[str, list[int]] = defaultdict(list)
    symbol_stems: dict[str, list[int]] = defaultdict(list)
    position_by_id: dict[int, int] = {}
    for position, (chunk_id, file_path) in enumerate(zip(chunk_ids, file_paths)):
        position_by_id[chunk_id] = position
        exact_keys, symbol_keys = _stem_lookup_keys(file_path)
        for key in exact_keys:
            exact_stems[key].append(chunk_id)
        for key in symbol_keys:
            symbol_stems[key].append(chunk_id)
    return dict(exact_stems), dict(symbol_stems), position_by_id


def _ordered_unique_chunk_ids(chunk_ids: Sequence[int], position_by_id: dict[int, int]) -> list[int]:
    return sorted(set(chunk_ids), key=position_by_id.__getitem__)


class BaseLazyChunkList(Sequence[Chunk]):
    """Shared stable-ID chunk lookup and sidecar indexes for lazy chunk sequences."""

    def __init__(
        self,
        chunk_ids: Sequence[int],
        store_path: Path,
        file_paths: Sequence[str] = (),
        languages: Sequence[str | None] = (),
        cached_chunks_by_id: dict[int, Chunk] | None = None,
    ) -> None:
        """Create a lazy chunk sequence backed by stable chunk IDs."""
        normalized_chunk_ids = tuple(int(chunk_id) for chunk_id in chunk_ids)
        file_paths_tuple = tuple(file_paths)
        languages_tuple = tuple(languages)
        if len(file_paths_tuple) != len(normalized_chunk_ids) or len(languages_tuple) != len(normalized_chunk_ids):
            raise ValueError("Chunk sidecar metadata must have one file path and language per chunk ID")
        if len(set(normalized_chunk_ids)) != len(normalized_chunk_ids):
            raise ValueError("Chunk IDs must be unique")

        self._chunk_ids = normalized_chunk_ids
        self._store_path = store_path
        self.chunk_file_paths = file_paths_tuple
        self.chunk_languages = languages_tuple
        self._cache: dict[int, Chunk] = {
            int(chunk_id): chunk for chunk_id, chunk in (cached_chunks_by_id or {}).items()
        }
        self._file_mapping = self._build_mapping(self.chunk_file_paths)
        self._language_mapping = self._build_language_mapping(self.chunk_languages)
        self._exact_stems, self._symbol_stems, self._position_by_id = _build_stem_lookup(
            self._chunk_ids,
            self.chunk_file_paths,
        )

    def __len__(self) -> int:
        """Return total chunk count."""
        return len(self._chunk_ids)

    def __iter__(self) -> Iterator[Chunk]:
        """Yield chunks in persisted order."""
        for chunk_id in self._chunk_ids:
            yield self.chunk_by_id(chunk_id)

    @overload
    def __getitem__(self, index: int) -> Chunk: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Chunk]: ...

    def __getitem__(self, index: int | slice) -> Chunk | Sequence[Chunk]:
        """Return chunks by persisted position."""
        if isinstance(index, slice):
            return [self.chunk_by_id(chunk_id) for chunk_id in self._chunk_ids[index]]
        return self.chunk_by_id(self._chunk_ids[index])

    def chunk_by_id(self, chunk_id: int) -> Chunk:
        """Return one chunk by stable ID, loading it from LMDB on first access."""
        chunk = self._cache.get(chunk_id)
        if chunk is not None:
            return chunk
        store = LmdbChunkStore.open(self._store_path, readonly=True)
        try:
            loaded = store.get_chunk(chunk_id)
        finally:
            store.close()
        if loaded is None:
            raise FileNotFoundError(f"Index chunk store is missing chunk payload for id {chunk_id}")
        self._cache[chunk_id] = loaded
        return loaded

    def chunks_by_id(self, chunk_ids: Sequence[int]) -> list[Chunk]:
        """Return chunks for stable IDs, preserving requested order."""
        missing_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in self._cache]
        if missing_ids:
            store = LmdbChunkStore.open(self._store_path, readonly=True)
            try:
                loaded_chunks = store.get_chunks(missing_ids)
            finally:
                store.close()
            if len(loaded_chunks) != len(missing_ids):
                raise FileNotFoundError("Index chunk store is missing chunk payloads")
            self._cache.update(zip(missing_ids, loaded_chunks))
        return [self._cache[chunk_id] for chunk_id in chunk_ids]

    def chunk_ids_for_paths(self, file_paths: frozenset[str]) -> list[int]:
        """Return stable IDs for persisted chunks under selected file paths."""
        return [chunk_id for path in file_paths for chunk_id in self._file_mapping.get(path, [])]

    def chunk_ids_for_languages(self, languages: frozenset[str]) -> list[int]:
        """Return stable IDs for persisted chunks in selected languages."""
        return [chunk_id for language in languages for chunk_id in self._language_mapping.get(language, [])]

    def chunk_ids_for_symbol_stem(self, symbol_name: str) -> list[int]:
        """Return stable IDs whose file stem matches a queried symbol name."""
        name = symbol_name.lower()
        keys = {name, name.replace("_", ""), name.rstrip("s"), name.replace("_", "").rstrip("s")}
        chunk_ids = [chunk_id for key in keys for chunk_id in self._symbol_stems.get(key, [])]
        return _ordered_unique_chunk_ids(chunk_ids, self._position_by_id)

    def chunk_ids_for_embedded_symbol_stem(self, symbol_name: str, min_prefix_len: int) -> list[int]:
        """Return stable IDs whose file stem exactly or prefix-matches an embedded symbol."""
        symbol = symbol_name.lower()
        symbol_norm = symbol.replace("_", "")
        keys = {symbol, symbol_norm}
        for size in range(min_prefix_len, len(symbol) + 1):
            keys.add(symbol[:size])
        for size in range(min_prefix_len, len(symbol_norm) + 1):
            keys.add(symbol_norm[:size])
        chunk_ids = [chunk_id for key in keys for chunk_id in self._exact_stems.get(key, [])]
        return _ordered_unique_chunk_ids(chunk_ids, self._position_by_id)

    def order_chunk_ids(self, chunk_ids: Sequence[int]) -> list[int]:
        """Return stable IDs in persisted sequence order."""
        return _ordered_unique_chunk_ids(chunk_ids, self._position_by_id)

    def file_mapping(self) -> dict[str, list[int]]:
        """Return persisted file-to-chunk-ID mapping."""
        return {path: list(chunk_ids) for path, chunk_ids in self._file_mapping.items()}

    def language_mapping(self) -> dict[str, list[int]]:
        """Return persisted language-to-chunk-ID mapping."""
        return {language: list(chunk_ids) for language, chunk_ids in self._language_mapping.items()}

    def chunk_ids(self) -> list[int]:
        """Return stable chunk IDs in sequence order."""
        return list(self._chunk_ids)

    def _build_mapping(self, file_paths: Sequence[str]) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = defaultdict(list)
        for chunk_id, file_path in zip(self._chunk_ids, file_paths):
            mapping[file_path].append(chunk_id)
        return dict(mapping)

    def _build_language_mapping(self, languages: Sequence[str | None]) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = defaultdict(list)
        for chunk_id, language in zip(self._chunk_ids, languages):
            if language:
                mapping[language].append(chunk_id)
        return dict(mapping)


class LazyChunkList(BaseLazyChunkList):
    """Sequence that loads persisted chunks by stable ID only when requested."""

    def __eq__(self, other: object) -> bool:
        """Compare with another chunk sequence by loaded chunk values."""
        if not isinstance(other, Sequence):
            return False
        return list(self) == list(other)


class MergedLazyChunkList(BaseLazyChunkList):
    """Lazy chunk sequence that overlays changed chunks on a persisted chunk store."""

    def __init__(
        self,
        chunk_ids: Sequence[int],
        store_path: Path,
        file_paths: Sequence[str],
        languages: Sequence[str | None],
        changed_chunks_by_id: dict[int, Chunk],
        deleted_chunk_ids: set[int],
    ) -> None:
        """Create a sequence for incremental rebuilds without loading unchanged payloads."""
        chunk_id_set = {int(chunk_id) for chunk_id in chunk_ids}
        changed_chunks = {int(chunk_id): chunk for chunk_id, chunk in changed_chunks_by_id.items()}
        changed_ids = set(changed_chunks)
        deleted_ids = {int(chunk_id) for chunk_id in deleted_chunk_ids}
        if changed_ids - chunk_id_set:
            raise ValueError("Changed chunk IDs must exist in the rebuilt chunk sequence")
        if not deleted_ids.isdisjoint(chunk_id_set):
            raise ValueError("Deleted chunk IDs must not remain in the rebuilt chunk sequence")
        for chunk_id, chunk in changed_chunks.items():
            if chunk.chunk_id is not None and chunk.chunk_id != chunk_id:
                raise ValueError("Changed chunk payload ID must match its stable chunk ID")

        super().__init__(chunk_ids, store_path, file_paths, languages, changed_chunks)
        self._changed_chunks_by_id = changed_chunks
        self._deleted_chunk_ids = deleted_ids

    def copy_chunk_store_to(self, target_path: Path, chunk_ids: list[int]) -> None:
        """Copy the old LMDB store wholesale, then patch changed/deleted chunk payloads."""
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(self._store_path, target_path)
        store = LmdbChunkStore.open(target_path)
        try:
            store.delete_chunks(sorted(self._deleted_chunk_ids))
            changed_ids = [chunk_id for chunk_id in chunk_ids if chunk_id in self._changed_chunks_by_id]
            if changed_ids:
                store.write_chunks_with_ids(
                    [self._changed_chunks_by_id[chunk_id] for chunk_id in changed_ids],
                    changed_ids,
                )
        finally:
            store.close()
