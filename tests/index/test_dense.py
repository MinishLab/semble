from pathlib import Path

import numpy as np
from vicinity.backends.basic import BasicArgs

from semble.index.dense import SelectableBasicBackend


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Test save and load roundtrip."""
    vecs = np.random.default_rng(seed=42).normal(size=(10, 32))
    args = BasicArgs()
    selectable = SelectableBasicBackend(vecs, args)
    selectable.save(tmp_path)

    selectable_2 = SelectableBasicBackend.load(tmp_path)
    assert np.allclose(selectable.vectors, selectable_2.vectors)
