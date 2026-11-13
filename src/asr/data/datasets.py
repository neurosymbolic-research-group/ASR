"""Datasets and data loaders for the thyroid patch dataset.

Expected layout (one folder per whole-slide image, class encoded by the first
letter of the folder name: ``B`` = Benign, ``H`` = Hashimoto, ``N`` = Nodularity)::

    <root>/train/B-GTEX-XXXX-XXXX/0000.png
    <root>/valid/...
    <root>/test/...
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from asr.data.transforms import get_transform

SPLITS = ("train", "valid", "test")
CLASS_NAMES = ("B", "H", "N")
"""Class prefixes: Benign, Hashimoto, Nodularity."""
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def image_folder(
    root: Path, split: str, resize_shape: tuple[int, int], augment: bool = False
) -> datasets.ImageFolder:
    """All patches of one split as a torchvision ``ImageFolder``.

    The ``ImageFolder`` labels index slide folders; they are not used in Stage 1.
    """
    return datasets.ImageFolder(
        str(root / split), transform=get_transform(resize_shape, augment=augment)
    )


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool
) -> DataLoader:
    """Create a ``DataLoader`` with the repository defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def cycle_batches(loader: DataLoader) -> Iterator[Tensor]:
    """Yield image batches forever, starting a new (reshuffled) pass when exhausted.

    The iterator persists across epochs, so with a fixed number of steps per
    epoch each epoch continues where the previous one stopped.
    """
    while True:
        for images, *_ in loader:
            yield images


class SlidePatches(Dataset):
    """All PNG patches of one slide folder (used to form bags in Stage 2).

    Args:
        slide_dir: Folder containing the slide's ``*.png`` patches.
        resize_shape: Resize target ``(H, W)``.
        transform: Optional transform overriding the default.
    """

    def __init__(
        self,
        slide_dir: Path,
        resize_shape: tuple[int, int],
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self.paths = sorted(slide_dir.glob("*.png"))
        self.transform = transform or get_transform(resize_shape, augment=False)

    def __len__(self) -> int:
        """Number of patches."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        """Load patch ``index`` as a ``(3, H, W)`` float tensor in ``[0, 1]``."""
        with Image.open(self.paths[index]) as img:
            return self.transform(img.convert("RGB"))


def slide_dirs(root: Path, split: str) -> list[Path]:
    """Sorted slide folders of a split."""
    return sorted(p for p in (root / split).iterdir() if p.is_dir())


def slide_label(slide_dir: Path) -> int:
    """Class index of a slide folder from its first letter (B/H/N)."""
    return CLASS_TO_IDX[slide_dir.name[0]]


def default_workers(device: torch.device) -> int:
    """Default number of DataLoader workers (2 on GPU, 0 on CPU)."""
    return 2 if device.type == "cuda" else 0
