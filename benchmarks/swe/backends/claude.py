from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from benchmarks.swe.backends.base import _TIMEOUT, Backend, ParsedRun, _run_with_timeout, _subprocess_env
from benchmarks.swe.gitutils import git_diff

_TOOLS = "Bash,Read,Glob,Grep,Edit,Write"


def _tool_call_entry(name: str, inp: dict) -> str:
    """Format a single Claude ``tool_use`` block as a console/log-friendly entry string."""
    if name == "Bash":
        cmd = inp.get("command", "")
        return "semble_bash [SEMBLE_BYPASS]" if "semble" in cmd else "Bash"
    if name.startswith("mcp__semble__"):
        tool = name.rsplit("__", 1)[-1]
        snippet = inp.get("snippet_lines", "default")
        query = inp.get("query", "")
        entry = f"claude_mcp:semble/{tool}(snippet={snippet})"
        if query:
            entry += f"[q={query[:80]}]"
        return entry
    return name


class ClaudeBackend(Backend):
    """Claude Code backend, semble wired in via an ephemeral ``--mcp-config``."""

    name = "claude"
    default_model = "claude-haiku-4-5-20251001"
    _rate_limit_msg = "rate limited (5-hour limit)"

    def _attempt_succeeded(self, parsed: ParsedRun) -> bool:
        return parsed.cost_usd > 0 or parsed.num_turns > 1

    def _make_mcp_config(self, *, with_semble: bool, repo: Path) -> Path:
        """Write a temp ``--mcp-config`` file (empty ``mcpServers`` when *with_semble* is False)."""
        tmp = Path(tempfile.mkdtemp(prefix="claude_mcp_"))
        config_path = tmp / "mcp.json"
        servers: dict[str, object] = {}
        if with_semble:
            cmd, *args = self._semble_cmd
            servers["semble"] = {"command": cmd, "args": [*args, str(repo)]}
        config_path.write_text(json.dumps({"mcpServers": servers}))
        return config_path

    def _parse(self, raw: str) -> ParsedRun:
        """Aggregate tool calls, cost, and token usage out of Claude's ``stream-json`` output."""
        tool_calls: list[str] = []
        cost_usd = 0.0
        input_tokens = 0
        output_tokens = 0
        num_turns = 0
        rate_limited = False

        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            if t == "rate_limit_event":
                if d.get("rate_limit_info", {}).get("status") != "allowed":
                    rate_limited = True
            elif t == "assistant":
                for block in d.get("message", {}).get("content", []):
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool_calls.append(_tool_call_entry(block["name"], block.get("input", {})))
            elif t == "result":
                cost_usd = d.get("total_cost_usd", 0.0)
                num_turns = d.get("num_turns", 0)
                usage = d.get("usage", {})
                input_tokens = (
                    usage.get("input_tokens", 0)
                    + usage.get("cache_creation_input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0)
                )
                output_tokens = usage.get("output_tokens", 0)

        return ParsedRun(
            tool_calls=tool_calls,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_turns=num_turns,
            rate_limited=rate_limited,
        )

    def _run_once(self, prompt: str, repo: Path, *, with_semble: bool) -> tuple[ParsedRun, str]:
        mcp_config = self._make_mcp_config(with_semble=with_semble, repo=repo)
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--model",
            self.model,
            "--allowed-tools",
            _TOOLS,
            "--strict-mcp-config",
            "--mcp-config",
            str(mcp_config),
        ]
        try:
            with _subprocess_env({}, with_semble=with_semble) as env:
                proc = _run_with_timeout(cmd, cwd=repo, env=env, timeout=_TIMEOUT)
            parsed = self._parse(proc.stdout + proc.stderr)
            diff = git_diff(repo)
            return parsed, diff
        finally:
            shutil.rmtree(mcp_config.parent, ignore_errors=True)
