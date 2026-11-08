"""ImageNet-1k (ILSVRC 2012) loaders for the Hugging Face parquet copy in /data/imagenet-1k.

``ImageNet`` mirrors ``torchvision.datasets.ImageNet``: it yields ``(PIL.Image, int)``
before transforms and accepts ``transform`` / ``target_transform`` / ``transforms``, so any
torchvision augmentation pipeline (v1 or v2) can be used unchanged.
"""
import ast
import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

IMAGENET_ROOT = "/data/imagenet-1k"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 1000

# The official 50k ILSVRC validation images are partitioned into two disjoint, fixed splits:
#   "val"  -> a class-balanced holdout of VAL_SIZE images (5 per class by default)
#   "test" -> all remaining images (45k by default)
VAL_SIZE = 5000
VAL_SPLIT_SEED = 0

# split name -> HF parquet file prefix. Note "test" is carved from the official validation set,
# NOT HF's unlabeled ``test-*.parquet``.
_SPLITS = {"train": "train", "val": "validation", "validation": "validation", "test": "validation"}


def _canonical_split(split: str) -> str:
    _hf_split_name(split)  # validates
    return "val" if split == "validation" else split


def val_test_indices(
    labels: np.ndarray, val_size: int = VAL_SIZE, seed: int = VAL_SPLIT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Partition the official validation rows into ``(val, test)`` index arrays: a class-balanced
    holdout of ``val_size`` images and its complement. Sorted, disjoint, deterministic in ``seed``."""
    val = _sample_indices(labels, val_size, class_balanced=True, seed=seed)
    test = np.setdiff1d(np.arange(len(labels)), val, assume_unique=True)
    return val, test


def _hf_split_name(split: str) -> str:
    if split not in _SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(_SPLITS)}")
    return _SPLITS[split]


def load_hf_split(split: str, root: str = IMAGENET_ROOT, cache_dir: Optional[str] = None):
    """Load one split as a ``datasets.Dataset`` (memory-mapped Arrow, built once per cache_dir)."""
    from datasets import load_dataset

    prefix = _hf_split_name(split)
    return load_dataset(
        "parquet",
        data_files=os.path.join(root, "data", f"{prefix}-*.parquet"),
        split="train",  # HF's name for the single data_files group, not the ImageNet split
        cache_dir=cache_dir,
    )


def load_class_info(root: str = IMAGENET_ROOT) -> dict[str, str]:
    """Return the ordered ``{wnid: "name1, name2, ..."}`` mapping from ``{root}/classes.py``
    without executing that file."""
    with open(os.path.join(root, "classes.py")) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "IMAGENET2012_CLASSES" for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            return ast.literal_eval(node.value.args[0])
    raise RuntimeError(f"IMAGENET2012_CLASSES not found in {root}/classes.py")


def _sample_indices(labels: np.ndarray, num_samples: int, class_balanced: bool, seed: int) -> np.ndarray:
    """Pick ``num_samples`` row indices, either uniformly or stratified by class. Returned sorted
    (row order doesn't matter for training and sorted reads are friendlier to the Arrow cache).

    Subsets are nested: for a fixed ``seed``, the subset for a smaller ``num_samples`` is contained
    in the subset for any larger one. This holds because every random draw below depends only on
    ``seed`` (never on ``num_samples``), and the subset is always a prefix of those fixed orderings.
    """
    n = len(labels)
    if not 0 < num_samples <= n:
        raise ValueError(f"num_samples must be in [1, {n}], got {num_samples}")
    rng = np.random.default_rng(seed)
    if not class_balanced:
        return np.sort(rng.permutation(n)[:num_samples])

    classes, class_sizes = np.unique(labels, return_counts=True)
    # Fixed class order deciding who gets the remainder: the first `remainder` classes get +1.
    # Since per-class quotas only grow with num_samples, every class's quota is monotone.
    class_order = rng.permutation(len(classes))
    per_class, remainder = divmod(num_samples, len(classes))
    counts = np.full(len(classes), per_class)
    counts[class_order[:remainder]] += 1
    if np.any(counts > class_sizes):
        c = int(np.argmax(counts > class_sizes))
        raise ValueError(f"Class {classes[c]} has only {class_sizes[c]} images, cannot draw {counts[c]}")

    # Fixed random order of images within each class; take the first counts[c] of each.
    pools = np.split(np.argsort(labels, kind="stable"), np.cumsum(class_sizes)[:-1])
    chosen = [rng.permutation(pool)[:k] for pool, k in zip(pools, counts)]
    return np.sort(np.concatenate(chosen))


class ImageNet(VisionDataset):
    """Drop-in replacement for ``torchvision.datasets.ImageNet`` backed by HF parquet shards.

    Args:
        root: directory containing ``data/*.parquet`` and ``classes.py``.
        split: ``"train"``, ``"val"`` (alias ``"validation"``) or ``"test"``. ``"val"`` is a
            class-balanced holdout of ``val_size`` images from the official ILSVRC validation set
            and ``"test"`` is the rest of it; the two are disjoint.
        transform: applied to the PIL image.
        target_transform: applied to the integer target.
        transforms: joint ``(image, target)`` transform (mutually exclusive with the two above).
        cache_dir: where HF ``datasets`` stores its Arrow cache (default: ~/.cache/huggingface).
        num_samples: if given, keep only a random subset of this many images (``None`` = full split).
        class_balanced: when subsetting, draw ``num_samples // NUM_CLASSES`` images from every
            class (the remainder goes to randomly chosen classes, one extra image each). If False,
            draw uniformly at random from the whole split.
        subset_seed: seed for the subset draw, so the same subset is reproduced across runs.
            Subsetting is applied after the val/test partition.
        val_size: number of official validation images in ``"val"`` (class-balanced); the rest
            form ``"test"``. Use the same ``val_size`` / ``val_split_seed`` when loading both
            splits so they stay disjoint.
        val_split_seed: seed of the val/test partition (independent of ``subset_seed``).

    Attributes match torchvision's ImageNet (``split``, ``wnids``, ``wnid_to_idx``, ``classes``,
    ``class_to_idx``, ``targets``), except ``samples``/``imgs`` since there are no file paths.
    ``subset_indices`` holds the selected row indices into the full HF split (train or the full
    50k validation set), or ``None`` if the full split is used.
    """

    def __init__(
        self,
        root: str = IMAGENET_ROOT,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        num_samples: Optional[int] = None,
        class_balanced: bool = True,
        subset_seed: int = 0,
        val_size: int = VAL_SIZE,
        val_split_seed: int = VAL_SPLIT_SEED,
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.split = _canonical_split(split)
        self.data = load_hf_split(self.split, root=self.root, cache_dir=cache_dir)
        self.subset_indices: Optional[np.ndarray] = None

        indices: Optional[np.ndarray] = None
        if self.split in ("val", "test") or num_samples is not None:
            labels = np.asarray(self.data.data.column("label").to_numpy())
            indices = np.arange(len(labels))
            if self.split in ("val", "test"):
                val, test = val_test_indices(labels, val_size, val_split_seed)
                indices = val if self.split == "val" else test
            if num_samples is not None:
                indices = indices[_sample_indices(labels[indices], num_samples, class_balanced, subset_seed)]
        if indices is not None:
            self.subset_indices = indices
            self.data = self.data.select(indices)

        class_info = load_class_info(self.root)
        self.wnids = list(class_info)
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.classes = [tuple(name.strip() for name in names.split(",")) for names in class_info.values()]
        self.class_to_idx = {name: i for i, names in enumerate(self.classes) for name in names}
        self._targets: Optional[list[int]] = None

    @property
    def targets(self) -> list[int]:
        if self._targets is None:
            self._targets = self.data["label"]
        return self._targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.data[index]
        image: Image.Image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        target = row["label"]
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def extra_repr(self) -> str:
        return f"Split: {self.split}"


def get_transforms(
    train: bool,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    size: int = 224,
    resize_size: int = 256,
) -> v2.Compose:
    """Standard ImageNet train / eval transforms."""
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    if train:
        return v2.Compose([v2.RandomResizedCrop(size), v2.RandomHorizontalFlip(), *to_tensor])
    return v2.Compose([v2.Resize(resize_size), v2.CenterCrop(size), *to_tensor])


def get_dataloader(
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 2,
    transform: Optional[Callable] = None,
    shuffle: Optional[bool] = None,
    seed: int = 42,
    root: str = IMAGENET_ROOT,
    cache_dir: Optional[str] = None,
    train_size: Optional[int] = None,
    class_balanced: bool = True,
    val_size: int = VAL_SIZE,
    val_split_seed: int = VAL_SPLIT_SEED,
    **loader_kwargs: Any,
) -> DataLoader:
    """DataLoader over ``ImageNet(split)``. Defaults to standard train/eval transforms and to
    shuffling only the train split. Extra kwargs (e.g. ``collate_fn`` for MixUp/CutMix,
    ``drop_last``, ``sampler``) are forwarded to ``DataLoader``.

    ``train_size`` restricts the train split to a random subset of that many images, drawn with
    ``seed`` (e.g. ``train_size=10_000`` -> 10 images per class when ``class_balanced``). The
    selected indices are available as ``loader.dataset.subset_indices``.

    ``split="val"`` / ``split="test"`` give the two disjoint parts of the official validation set
    (``val_size`` class-balanced images vs. the rest), fixed by ``val_split_seed`` — deliberately
    separate from ``seed`` so changing the training seed never moves images between val and test."""
    is_train = _hf_split_name(split) == "train"
    if train_size is not None and not is_train:
        raise ValueError("train_size only applies to the train split")
    if transform is None:
        transform = get_transforms(train=is_train)
    if shuffle is None:
        shuffle = is_train if "sampler" not in loader_kwargs else False
    dataset = ImageNet(
        root=root,
        split=split,
        transform=transform,
        cache_dir=cache_dir,
        num_samples=train_size,
        class_balanced=class_balanced,
        subset_seed=seed,
        val_size=val_size,
        val_split_seed=val_split_seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    loader_kwargs.setdefault("pin_memory", True)
    if num_workers > 0:
        loader_kwargs.setdefault("persistent_workers", True)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=g, **loader_kwargs
    )
