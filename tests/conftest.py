from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """3 splits x 3 classes x 2 slides x 4 random 256x256 PNG patches."""
    rng = np.random.default_rng(0)
    root = tmp_path / "images"
    for split in ("train", "valid", "test"):
        for cls in "BHN":
            for s in range(2):
                d = root / split / f"{cls}-GTEX-{s:04d}"
                d.mkdir(parents=True)
                for i in range(4):
                    arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(d / f"{i:04d}.png")
    return root
