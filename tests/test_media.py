from pathlib import Path

import numpy as np
from PIL import Image

from depth_anything_mcp.media import collect_image_sources, stem_of


def test_collect_single_and_directory(tmp_path: Path) -> None:
    folder = tmp_path / "imgs"
    folder.mkdir()
    path = folder / "a.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    assert collect_image_sources(str(path)) == [str(path.resolve())]
    listed = collect_image_sources(str(folder))
    assert listed == [str(path.resolve())]
    assert stem_of(str(path)) == "a"
