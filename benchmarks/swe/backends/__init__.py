from benchmarks.swe.backends.base import Backend, RunResult, is_semble_tool_call
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
    "RunResult",
    "ClaudeBackend",
    "CodexBackend",
    "OpencodeBackend",
    "is_semble_tool_call",
    "_BACKENDS",
]
