import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from model2vec import StaticModel

from semble.cli import _cli_main
from semble.index import SembleIndex
from semble.index.bm25 import BM25
from semble.mcp import _IndexCache, create_server
from semble.types import SearchResult
from tests.conftest import make_chunk


@pytest.fixture
def two_repos(tmp_path: Path, mock_model: StaticModel) -> list[tuple[str, SembleIndex]]:
    """Indexes for two sibling projects where repo B defines an endpoint that repo A calls."""
    repo_a = tmp_path / "service-a"
    repo_b = tmp_path / "service-b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "client.py").write_text(
        'def fetch_invoice(invoice_id):\n    return http_get(f"/invoices/{invoice_id}")\n'
    )
    (repo_b / "api.py").write_text(
        textwrap.dedent(
            """\
            def get_invoice(invoice_id):
                \"\"\"Return the invoice endpoint payload.\"\"\"
                return load_invoice(invoice_id)
            """
        )
    )
    with (
        patch("semble.index.index.load_model", return_value=(mock_model, "/fake/model")),
        patch("semble.index.index.get_validated_cache", return_value=None),
    ):
        return [(str(repo), SembleIndex.from_path(repo)) for repo in (repo_a, repo_b)]


def test_merge_search_and_find_related(two_repos: list[tuple[str, SembleIndex]]) -> None:
    """A merged index prefixes paths, searches both repos, and find_related crosses repo boundaries."""
    merged = SembleIndex.merge(two_repos)

    assert merged.sources == {"service-a": two_repos[0][0], "service-b": two_repos[1][0]}
    assert {c.file_path for c in merged.chunks} == {"service-a/client.py", "service-b/api.py"}
    assert {r.chunk.file_path for r in merged.search("invoice", top_k=5)} == {"service-a/client.py", "service-b/api.py"}

    seed = next(c for c in merged.chunks if c.file_path.startswith("service-a/"))
    assert [r.chunk.file_path for r in merged.find_related(seed, top_k=5)] == ["service-b/api.py"]


def test_merge_labels(two_repos: list[tuple[str, SembleIndex]]) -> None:
    """Labels come from the path basename or git repo name, with duplicates suffixed."""
    index = two_repos[0][1]
    merged = SembleIndex.merge([("https://github.com/org/repo.git", index), ("/x/repo", index), ("other", index)])
    assert merged.sources == {
        "repo": "https://github.com/org/repo.git",
        "repo-2": "/x/repo",
        "other": str(Path.cwd() / "other"),
    }
    assert [c.file_path for c in merged.chunks] == ["repo/client.py", "repo-2/client.py", "other/client.py"]


def test_bm25_merge_matches_single_corpus() -> None:
    """Merging two BM25 indexes scores identically to one index built over all documents."""
    docs = {"a": ["invoice", "client"], "b": ["invoice", "endpoint", "payload"], "c": ["config", "host"]}
    single, left, right = BM25(), BM25(), BM25()
    for chunk_id, tokens in docs.items():
        single.add_document(f"{'l' if chunk_id != 'c' else 'r'}/{chunk_id}", tokens)
        (left if chunk_id != "c" else right).add_document(chunk_id, tokens)
    single.set_doc_order(["l/a", "l/b", "r/c"])
    left.set_doc_order(["a", "b"])
    right.set_doc_order(["c"])
    merged = BM25.merge([("l", left), ("r", right)])
    assert merged.doc_order == single.doc_order
    np.testing.assert_allclose(merged.get_scores(["invoice", "host"]), single.get_scores(["invoice", "host"]))


def test_cli_search_multiple_paths(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Passing several paths builds one index per path, merges them, saves each, and reports the repo mapping."""
    fake_index, merged = MagicMock(), MagicMock(sources={"one": "/p/one", "two": "/p/two"})
    merged.search.return_value = [SearchResult(chunk=make_chunk("x = 1", "one/a.py"), score=0.9)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "x", "/p/one", "/p/two"])
    with (
        patch("semble.cli.SembleIndex.from_path", return_value=fake_index) as from_path,
        patch("semble.cli.SembleIndex.merge", return_value=merged) as merge,
        patch("semble.cli.save_index_to_cache") as save,
    ):
        _cli_main()
    assert [c.args[0] for c in from_path.call_args_list] == ["/p/one", "/p/two"]
    merge.assert_called_once_with([("/p/one", fake_index), ("/p/two", fake_index)])
    assert [c.args[1] for c in save.call_args_list] == ["/p/one", "/p/two"]
    out = json.loads(capsys.readouterr().out)
    assert out["repos"] == merged.sources
    assert out["results"][0]["file_path"] == "one/a.py"


@pytest.mark.anyio
async def test_mcp_search_multiple_repos(two_repos: list[tuple[str, SembleIndex]]) -> None:
    """The MCP search tool accepts a list of repos, returns prefixed paths plus the mapping, and reuses the merge."""
    repos = [source for source, _ in two_repos]
    cache = _IndexCache()
    cache._model_path = "/fake/model"
    cache._model_ready.set()
    with patch("semble.mcp.SembleIndex.from_path", side_effect=lambda path, **_: dict(two_repos)[path]):
        server = create_server(cache)
        result = await server.call_tool("search", {"query": "invoice", "repo": repos, "top_k": 5})
        with patch("semble.mcp.SembleIndex.merge") as merge:
            await server.call_tool("search", {"query": "invoice", "repo": repos})
    payload = json.loads(result[0][0].text)
    assert set(payload["repos"]) == {"service-a", "service-b"}
    assert {item["file_path"] for item in payload["results"]} == {"service-a/client.py", "service-b/api.py"}
    merge.assert_not_called()
