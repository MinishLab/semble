"""Shared fixtures for semble tests."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from semble.types import Chunk


@pytest.fixture
def tmp_py_file(tmp_path: Path) -> Path:
    """A simple Python file with two functions."""
    code = textwrap.dedent(
        """\
        def add(a, b):
            \"\"\"Add two numbers.\"\"\"
            return a + b

        def subtract(a, b):
            return a - b

        X = 42
        """
    )
    f = tmp_path / "math_utils.py"
    f.write_text(code)
    return f


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A small project with a few Python files."""
    (tmp_path / "auth.py").write_text(
        textwrap.dedent(
            """\
            def authenticate(token):
                \"\"\"Verify an auth token.\"\"\"
                return token == "secret"

            def login(username, password):
                return authenticate(password)
            """
        )
    )
    (tmp_path / "utils.py").write_text(
        textwrap.dedent(
            """\
            def format_name(first, last):
                return f"{first} {last}"

            class Config:
                debug = False
                host = "localhost"
            """
        )
    )
    (tmp_path / "README.md").write_text("# Test project\n")
    return tmp_path


@pytest.fixture
def sample_chunks(tmp_py_file: Path) -> list[Chunk]:
    from semble.chunker import chunk_file

    return chunk_file(tmp_py_file)


@pytest.fixture
def mock_model():
    """A model stub that returns deterministic random embeddings."""
    model = MagicMock()
    rng = np.random.default_rng(42)

    def _encode(texts):
        embs = rng.standard_normal((len(texts), 256)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / (norms + 1e-8)

    model.encode.side_effect = _encode
    return model
