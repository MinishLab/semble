# Spec: Fix bytes-vs-str TypeError in tree-sitter Parser.parse()

## Feedback ID
bug-20260521-26ea06

## Problem
`semble chunking.core.chunk()` passes `bytes` (via `text.encode("utf-8")`) to `parser.parse()`.
Starting with tree-sitter 0.24+, `Parser.parse()` requires `str`, not `bytes`.
This causes a `TypeError: argument 'source': 'bytes' object is not an instance of 'str'`
when running `semble --include-text-files` on any repo.

## Root Cause
In `src/semble/chunking/core.py`, line:
```python
as_bytes = text.encode("utf-8")
root = parser.parse(as_bytes).root_node  # BUG: bytes not accepted by tree-sitter>=0.24
```

## Fix
Pass `text` (str) directly to `parser.parse()` instead of `as_bytes` (bytes).
Keep `as_bytes` for the byte-to-character offset conversion loop since tree-sitter
nodes still expose `start_byte`/`end_byte` attributes.

## Test
Add a regression test that directly calls `chunk()` with a Python code string
and verifies it returns valid ChunkBoundary objects without TypeError.

## Verification
- `uv run pytest tests/test_chunker.py` passes
- `semble --include-text-files /root/gitrepos/demos` no longer crashes with TypeError
