from benchmarks.swe.backends.base import Backend
from benchmarks.swe.backends.claude import ClaudeBackend
from benchmarks.swe.backends.codex import CodexBackend
from benchmarks.swe.backends.opencode import OpencodeBackend

_BACKENDS: dict[str, type[Backend]] = {
    "claude": ClaudeBackend,
    "codex": CodexBackend,
    "opencode": OpencodeBackend,
}

__all__ = [
    "Backend",
    "ClaudeBackend",
    "CodexBackend",
    "OpencodeBackend",
    "_BACKENDS",
]
