import json
import subprocess
from pathlib import Path


def run_ripgrep_count(
    query: str,
    benchmark_dir: Path,
    *,
    top_k: int,
    fixed_strings: bool = True,
    timeout: int = 30,
) -> list[str]:
    """Return file paths sorted by ripgrep match count."""
    cmd = ["rg", "--count", "--no-heading", "--ignore-case", "--hidden", "--glob", "!.git"]
    if fixed_strings:
        cmd.append("--fixed-strings")
    cmd += [query, str(benchmark_dir)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode not in (0, 1):
        return []

    entries: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        *path_parts, count_str = line.split(":")
        try:
            entries.append((":".join(path_parts), int(count_str)))
        except ValueError:
            continue
    entries.sort(key=lambda x: -x[1])
    return [path for path, _ in entries[:top_k]]


def run_cs(
    query: str,
    benchmark_dir: Path,
    *,
    top_k: int,
    timeout: int = 30,
) -> list[str]:
    """Return file paths from cs (Code Spelunker) JSON output, ranked by its structural BM25 score."""
    cmd = ["cs", query, "--format", "json", "--dir", str(benchmark_dir), "--result-limit", str(top_k)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if not data:
        return []
    seen: dict[str, None] = {}
    for item in data:
        location = item.get("location", "")
        if location:
            seen[location] = None
    return list(seen)[:top_k]


def run_colgrep_files(
    query: str,
    benchmark_dir: Path,
    *,
    top_k: int,
    code_only: bool = True,
    timeout: int = 30,
) -> list[str]:
    """Return file paths from ColGREP JSON output."""
    cmd = ["colgrep", "--force-cpu"]
    if code_only:
        cmd.append("--code-only")
    cmd += ["--json", "-k", str(top_k), query, str(benchmark_dir)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return [item["unit"]["file"] for item in data if "unit" in item and "file" in item["unit"]]
