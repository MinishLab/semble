from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from ..gitutils import _git_diff
from .base import _PROJECT_ROOT, _TIMEOUT, Backend, _run_with_timeout, _subprocess_env

_OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"


def _tool_use_entry(tool: str, inp: dict) -> str:
    """Format a single opencode tool_use part as a console/log-friendly entry string."""
    if tool in ("semble_search", "semble_find_related"):
        query = inp.get("query", "")
        snippet = inp.get("snippet_lines", "default")
        entry = f"opencode_mcp:semble/{tool}(snippet={snippet})"
        if query:
            entry += f"[q={query[:80]}]"
        return entry
    if tool == "bash":
        cmd = inp.get("command", "")
        entry = f"opencode_bash:{cmd[:120]}"
        if "semble" in cmd:
            entry += " [SEMBLE_BYPASS]"
        return entry
    return f"opencode_{tool}"


class OpencodeBackend(Backend):
    """opencode (opencode go) backend using `opencode run --format json`."""

    name = "opencode"
    default_model = "opencode-go/deepseek-v4-pro"

    def label(self) -> str:
        """Human-readable backend/model identifier, without a redundant "opencode/" prefix."""
        # Model names like "opencode-go/deepseek-v4-pro" already contain the backend namespace.
        if self.model.startswith(("opencode/", "opencode-go/")):
            return self.model
        return f"{self.name}/{self.model}"

    def _disable_semble_in_config(self, text: str) -> str:
        """Flip enabled: true → false in the semble MCP block.

        opencode.json is JSONC (allows // comments) so we use text replacement
        rather than json.loads. We operate only within a window after "semble"
        to avoid hitting an unrelated "enabled" field elsewhere in the config.
        """
        start = text.find('"semble"')
        if start == -1:
            return text
        window_end = min(start + 2000, len(text))
        patched = re.sub(r'("enabled"\s*:\s*)true', r"\1false", text[start:window_end], count=1)
        return text[:start] + patched + text[window_end:]

    def _replace_semble_command(self, text: str) -> str:
        """Swap the semble MCP command to use local branch via uv run.

        opencode.json is JSONC so we use targeted text replacement.
        Old: ["uvx", "--from", "semble[mcp]", "semble"]
        New: ["uv", "run", "--directory", "<project_root>", "semble"]
        """
        old_cmd = '"command": ["uvx", "--from", "semble[mcp]", "semble"]'
        new_cmd = f'"command": ["uv", "run", "--directory", "{_PROJECT_ROOT}", "semble"]'
        if old_cmd in text:
            return text.replace(old_cmd, new_cmd, 1)
        start = text.find('"semble"')
        if start == -1:
            return text
        window_end = min(start + 2000, len(text))
        patched = re.sub(
            r'"command"\s*:\s*\[[^\]]*"semble"\]',
            new_cmd,
            text[start:window_end],
            count=1,
        )
        return text[:start] + patched + text[window_end:]

    def _make_temp_home(self, *, with_semble: bool) -> tuple[Path | None, dict[str, str]]:
        """Returns (tempdir, env_overrides). tempdir is None when no temp needed."""
        if with_semble and not self.local_semble:
            return None, {}
        if not _OPENCODE_CONFIG.exists():
            return None, {}

        temp_home = Path(tempfile.mkdtemp(prefix="opencode_no_semble_"))
        config_dir = temp_home / "opencode"
        config_dir.mkdir(parents=True)
        text = _OPENCODE_CONFIG.read_text()
        if not with_semble:
            text = self._disable_semble_in_config(text)
        elif self.local_semble:
            text = self._replace_semble_command(text)
        (config_dir / "opencode.json").write_text(text)
        return temp_home, {"XDG_CONFIG_HOME": str(temp_home)}

    def _parse(self, raw: str) -> dict:
        """Aggregate tool calls, cost, and token usage out of opencode's `run --format json` output."""
        tool_calls: list[str] = []
        total_cost = 0.0
        total_input = 0
        total_output = 0
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
            if t == "tool_use":
                part = d.get("part", {})
                tool = part.get("tool", "")
                inp = part.get("state", {}).get("input", {})
                tool_calls.append(_tool_use_entry(tool, inp))
            elif t == "step_finish":
                num_turns += 1
                part = d.get("part", {})
                tokens = part.get("tokens", {})
                cache = tokens.get("cache", {})
                total_input += tokens.get("input", 0) + cache.get("read", 0)
                total_output += tokens.get("output", 0) + tokens.get("reasoning", 0)
                total_cost += part.get("cost", 0.0)

        return {
            "tool_calls": tool_calls,
            "cost_usd": round(total_cost, 6),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "num_turns": num_turns,
            "rate_limited": rate_limited,
        }

    def _run_once(self, prompt: str, repo: Path, *, with_semble: bool) -> tuple[dict, str]:
        temp_home, env_overrides = self._make_temp_home(with_semble=with_semble)
        try:
            with _subprocess_env(env_overrides, with_semble=with_semble) as env:
                proc = _run_with_timeout(
                    [
                        "opencode",
                        "run",
                        "--format",
                        "json",
                        "--model",
                        self.model,
                        "--dir",
                        str(repo),
                        "--dangerously-skip-permissions",
                        prompt,
                    ],
                    cwd=repo,
                    env=env,
                    timeout=_TIMEOUT,
                )
            parsed = self._parse(proc.stdout + proc.stderr)
            diff = _git_diff(repo)
            return parsed, diff
        finally:
            if temp_home is not None:
                shutil.rmtree(temp_home, ignore_errors=True)
