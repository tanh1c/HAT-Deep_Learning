"""
Utilities for loading the CIFAR-10 test split from local project assets.

The workspace keeps the archive at ``image/data/cifar-10-python.tar.gz``.
Reading the test batch directly from that archive avoids permission issues
with extracted files while keeping calibration fully offline.
"""

import os
import pickle
import tarfile
from functools import lru_cache
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


ASSIGNMENT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_DATA_DIR = os.path.join(ASSIGNMENT_ROOT, "image", "data")
DEFAULT_ARCHIVE_PATH = os.path.join(DEFAULT_DATA_DIR, "cifar-10-python.tar.gz")


@lru_cache(maxsize=1)
def load_cifar10_test_arrays(
    archive_path: str = DEFAULT_ARCHIVE_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 test images and labels from the local archive."""
    if not os.path.exists(archive_path):
        raise FileNotFoundError(
            f"CIFAR-10 archive not found at {archive_path}. "
            "Expected image/data/cifar-10-python.tar.gz to exist."
        )

    with tarfile.open(archive_path, "r:gz") as tar:
        member = tar.extractfile("cifar-10-batches-py/test_batch")
        if member is None:
            raise FileNotFoundError(
                "Could not find cifar-10-batches-py/test_batch inside the archive."
            )

        batch = pickle.load(member, encoding="bytes")

    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


class LocalCIFAR10TestDataset(Dataset):
    """Dataset wrapper that serves the CIFAR-10 test split from local files."""

    def __init__(self, transform=None, archive_path: str = DEFAULT_ARCHIVE_PATH):
        self.transform = transform
        self.images, self.labels = load_cifar10_test_arrays(archive_path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_cifar10_test_dataset(transform=None) -> LocalCIFAR10TestDataset:
    """Create the CIFAR-10 test dataset used by the calibration tab."""
    return LocalCIFAR10TestDataset(transform=transform)
