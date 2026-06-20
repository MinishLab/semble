from benchmarks.swe.backends.base import (
    WITH_SEMBLE,
    WITHOUT_SEMBLE,
    Backend,
    ParsedRun,
    RunResult,
    is_semble_tool_call,
    variant_name,
)
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
    "ParsedRun",
    "RunResult",
    "WITH_SEMBLE",
    "WITHOUT_SEMBLE",
    "_BACKENDS",
    "is_semble_tool_call",
    "variant_name",
]
