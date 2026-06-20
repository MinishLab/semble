from .base import Backend, RunResult
from .claude import ClaudeBackend
from .codex import CodexBackend
from .opencode import OpencodeBackend

_BACKENDS: dict[str, type[Backend]] = {
    "claude": ClaudeBackend,
    "codex": CodexBackend,
    "opencode": OpencodeBackend,
}

__all__ = [
    "Backend",
    "RunResult",
    "ClaudeBackend",
    "CodexBackend",
    "OpencodeBackend",
    "_BACKENDS",
]
