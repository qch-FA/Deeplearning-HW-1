from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class DatasetBundle:
    images: np.ndarray
    labels: np.ndarray
    paths: list[str]
    class_names: list[str]
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    image_size: int

    @property
    def input_dim(self) -> int:
        return int(self.image_size * self.image_size * 3)


def _split_indices(
    indices: list[int],
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = np.array(indices, dtype=np.int64)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return train, val, test


def load_eurosat(
    data_root: str | Path,
    image_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetBundle:
    data_root = Path(data_root)
    class_names = sorted([path.name for path in data_root.iterdir() if path.is_dir()])
    if not class_names:
        raise FileNotFoundError(f"No class folders found under {data_root}.")

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    running_index = 0
    resample = getattr(Image, "Resampling", Image).BILINEAR

    for class_index, class_name in enumerate(class_names):
        class_files = sorted((data_root / class_name).glob("*.jpg"))
        local_indices = list(range(running_index, running_index + len(class_files)))
        train_idx, val_idx, test_idx = _split_indices(local_indices, train_ratio, val_ratio, rng)
        train_parts.append(train_idx)
        val_parts.append(val_idx)
        test_parts.append(test_idx)

        for file_path in class_files:
            with Image.open(file_path) as image:
                array = np.asarray(
                    image.convert("RGB").resize((image_size, image_size), resample),
                    dtype=np.uint8,
                )
            images.append(array)
            labels.append(class_index)
            paths.append(file_path.as_posix())
        running_index += len(class_files)

    image_array = np.stack(images, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)
    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    test_indices = np.concatenate(test_parts)

    train_pixels = image_array[train_indices].astype(np.float32) / 255.0
    mean = train_pixels.mean(axis=(0, 1, 2))
    std = train_pixels.std(axis=(0, 1, 2)) + 1e-6

    return DatasetBundle(
        images=image_array,
        labels=label_array,
        paths=paths,
        class_names=class_names,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        image_size=image_size,
    )


class BatchIterator:
    def __init__(
        self,
        bundle: DatasetBundle,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        if split == "train":
            self.indices = bundle.train_indices.copy()
        elif split == "val":
            self.indices = bundle.val_indices.copy()
        elif split == "test":
            self.indices = bundle.test_indices.copy()
        else:
            raise ValueError(f"Unknown split: {split}")

    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_images = self.bundle.images[batch_idx].astype(np.float32) / 255.0
            batch_images = (batch_images - self.bundle.mean) / self.bundle.std
            batch_images = batch_images.reshape(len(batch_idx), -1)
            batch_labels = self.bundle.labels[batch_idx]
            yield batch_images.astype(np.float32), batch_labels.astype(np.int64), batch_idx
