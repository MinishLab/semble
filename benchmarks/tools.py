import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from semble.ranking.boosting import _STOPWORDS as _SEMBLE_STOPWORDS

_KW_MIN_LEN = 3
# Extend semble's code-ranking stopwords with broader natural-language query words.
_STOPWORDS: frozenset[str] = _SEMBLE_STOPWORDS | frozenset(
    """
    but its can not no nor so yet both either neither than then
    will would could should may might been being had did will
    they all any each few more most other some such only own same too very just
    about after also before between during into through under up down over
    """.split()
)


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful search keywords from a natural-language query.

    Strips stopwords, deduplicates, and requires a minimum token length.
    Preserves original casing so ripgrep can match camelCase/PascalCase identifiers.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", query)
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        lw = w.lower()
        if len(lw) >= _KW_MIN_LEN and lw not in _STOPWORDS and lw not in seen:
            seen.add(lw)
            result.append(w)
    return result


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


def run_ripgrep_keywords(
    query: str,
    benchmark_dir: Path,
    *,
    top_k: int,
    timeout: int = 30,
) -> list[str]:
    """Return file paths ranked by number of distinct query keywords matched by ripgrep.

    Splits the query into keywords (dropping stopwords), runs a separate rg search
    for each keyword, then ranks files by how many distinct keywords they contain.
    Falls back to a full-query fixed-string search when no keywords are extracted.
    """
    keywords = extract_keywords(query)
    if not keywords:
        return run_ripgrep_count(query, benchmark_dir, top_k=top_k, timeout=timeout)
    keyword_hits: Counter[str] = Counter()
    for kw in keywords:
        for path in run_ripgrep_count(kw, benchmark_dir, top_k=500, timeout=timeout):
            keyword_hits[path] += 1
    return [path for path, _ in keyword_hits.most_common(top_k)]


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
