from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from benchmarks.swe.backends.base import _PROJECT_ROOT, _TIMEOUT, Backend, ParsedRun, _run_with_timeout, _subprocess_env
from benchmarks.swe.gitutils import git_diff

_CODEX_CONFIG = Path.home() / ".codex" / "config.toml"


def _item_entry(item: dict) -> str | None:
    """Format a single Codex ``item.completed`` payload as a console/log-friendly entry string."""
    item_type = item.get("type", "")
    if item_type == "mcp_tool_call":
        server = item.get("server", "")
        tool = item.get("tool", "")
        params = item.get("params", {})
        snippet = params.get("snippet_lines", "default")
        query = params.get("query", "")
        entry = f"codex_mcp:{server}/{tool}(snippet={snippet})"
        if query:
            entry += f"[q={query[:80]}]"
        return entry
    if item_type == "command_execution":
        cmd = item.get("command", "")
        entry = f"codex_bash:{cmd[:120]}"
        if "semble" in cmd:
            entry += " [SEMBLE_BYPASS]"
        return entry
    if item_type == "file_change":
        return "codex_edit"
    return None


class CodexBackend(Backend):
    """Codex (OpenAI) backend using ``codex exec --json``."""

    name = "codex"
    default_model = "gpt-5.4-mini"
    _PRICE_INPUT = 0.15  # $/1M input tokens
    _PRICE_OUTPUT = 0.60  # $/1M output tokens

    def _strip_semble_section(self, text: str) -> str:
        """Return config content with ``[mcp_servers.semble]`` section removed."""
        lines = text.splitlines()
        out: list[str] = []
        skip = False
        for line in lines:
            if line.strip().startswith("[mcp_servers.semble"):
                skip = True
                continue
            if skip and line.strip().startswith("[") and not line.strip().startswith("[mcp_servers.semble"):
                skip = False
            if not skip:
                out.append(line)
        return "\n".join(out) + "\n"

    def _replace_semble_command(self, text: str) -> str:
        """Swap the semble MCP command to use local branch via ``uv run``."""
        lines = text.splitlines()
        out: list[str] = []
        in_semble = False
        for line in lines:
            if line.strip().startswith("[mcp_servers.semble") and "tools" not in line:
                in_semble = True
            elif in_semble and line.strip().startswith("[") and "tools" not in line:
                in_semble = False
            if in_semble:
                if line.strip().startswith("command"):
                    out.append('command = "uv"')
                    continue
                if line.strip().startswith("args"):
                    out.append(f'args = ["run", "--directory", "{_PROJECT_ROOT}", "semble"]')
                    continue
            out.append(line)
        return "\n".join(out) + "\n"

    def _make_temp_home(self, *, with_semble: bool) -> tuple[Path | None, dict[str, str]]:
        """Return ``(tempdir, env_overrides)``. *tempdir* is ``None`` when no temp is needed."""
        if with_semble and not self.local_semble:
            return None, {}

        temp_home = Path(tempfile.mkdtemp(prefix="codex_no_semble_"))
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))

        if _CODEX_CONFIG.exists():
            content = _CODEX_CONFIG.read_text()
            if not with_semble:
                content = self._strip_semble_section(content)
            elif self.local_semble:
                content = self._replace_semble_command(content)
            (temp_home / "config.toml").write_text(content)
        auth_src = codex_home / "auth.json"
        if auth_src.exists():
            shutil.copy2(auth_src, temp_home / "auth.json")

        return temp_home, {"CODEX_HOME": str(temp_home)}

    def _parse(self, raw: str) -> ParsedRun:
        """Aggregate tool calls, cost, and token usage out of Codex's ``exec --json`` output."""
        tool_calls: list[str] = []
        total_input = 0
        total_output = 0
        total_cached = 0
        total_reasoning = 0
        num_turns = 0
        rate_limited = False

        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                if "429" in line or "rate" in line.lower():
                    rate_limited = True
                continue

            t = d.get("type")
            if t == "item.completed":
                entry = _item_entry(d.get("item", {}))
                if entry:
                    tool_calls.append(entry)
            elif t == "turn.completed":
                num_turns += 1
                usage = d.get("usage", {})
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                total_cached += usage.get("cached_input_tokens", 0)
                total_reasoning += usage.get("reasoning_output_tokens", 0)

        input_tokens = total_input + total_cached
        output_tokens = total_output + total_reasoning
        cost_usd = (input_tokens / 1_000_000) * self._PRICE_INPUT + (output_tokens / 1_000_000) * self._PRICE_OUTPUT

        return ParsedRun(
            tool_calls=tool_calls,
            cost_usd=round(cost_usd, 6),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_turns=num_turns,
            rate_limited=rate_limited,
        )

    def _run_once(self, prompt: str, repo: Path, *, with_semble: bool) -> tuple[ParsedRun, str]:
        temp_home, env_overrides = self._make_temp_home(with_semble=with_semble)
        try:
            with _subprocess_env(env_overrides, with_semble=with_semble) as env:
                proc = _run_with_timeout(
                    [
                        "codex",
                        "exec",
                        "--json",
                        "--model",
                        self.model,
                        "--sandbox",
                        "danger-full-access",
                        "--dangerously-bypass-approvals-and-sandbox",
                        prompt,
                    ],
                    cwd=repo,
                    env=env,
                    timeout=_TIMEOUT,
                )
            parsed = self._parse(proc.stdout + proc.stderr)
            diff = git_diff(repo)
            return parsed, diff
        finally:
            if temp_home is not None:
                shutil.rmtree(temp_home, ignore_errors=True)
