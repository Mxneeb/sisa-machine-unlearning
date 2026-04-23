"""
Data sharding and slicing logic for SISA training.

Sharding: split dataset into S disjoint shards.
Slicing:  within each shard, checkpoint after every R/S/Q data points.
"""
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class ShardedDataset:
    """
    Wraps a PyTorch Dataset and partitions it into shards and slices.

    Args:
        dataset:    Full training dataset.
        num_shards: Number of shards (S).
        num_slices: Number of slices per shard (Q).
        shuffle:    Whether to shuffle indices before sharding.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_shards: int = 5,
        num_slices: int = 5,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_shards = num_shards
        self.num_slices = num_slices

        indices = np.arange(len(dataset))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        # Split into shards
        self.shard_indices: List[np.ndarray] = np.array_split(indices, num_shards)

        # Split each shard into slices
        self.slice_indices: List[List[np.ndarray]] = [
            np.array_split(shard, num_slices) for shard in self.shard_indices
        ]

        # Map from dataset index -> (shard_id, slice_id, position_within_slice)
        self._index_map: dict = {}
        for s_id, shard in enumerate(self.shard_indices):
            for q_id, slc in enumerate(self.slice_indices[s_id]):
                for pos, idx in enumerate(slc):
                    self._index_map[int(idx)] = (s_id, q_id, pos)

    def get_shard_slice_subset(self, shard_id: int, up_to_slice: int) -> Subset:
        """Return a Subset covering slices [0 .. up_to_slice] of the given shard."""
        combined = np.concatenate(self.slice_indices[shard_id][: up_to_slice + 1])
        return Subset(self.dataset, combined.tolist())

    def locate(self, dataset_index: int) -> Tuple[int, int, int]:
        """Return (shard_id, slice_id, position) for a given dataset index."""
        return self._index_map[dataset_index]

    def save_mapping(self, path: str) -> None:
        mapping = {
            "num_shards": self.num_shards,
            "num_slices": self.num_slices,
            "shard_indices": [s.tolist() for s in self.shard_indices],
            "slice_indices": [
                [sl.tolist() for sl in shard_slices]
                for shard_slices in self.slice_indices
            ],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(mapping, f)

    @classmethod
    def load_mapping(cls, path: str, dataset: Dataset) -> "ShardedDataset":
        with open(path) as f:
            mapping = json.load(f)
        obj = cls.__new__(cls)
        obj.dataset = dataset
        obj.num_shards = mapping["num_shards"]
        obj.num_slices = mapping["num_slices"]
        obj.shard_indices = [np.array(s) for s in mapping["shard_indices"]]
        obj.slice_indices = [
            [np.array(sl) for sl in shard_slices]
            for shard_slices in mapping["slice_indices"]
        ]
        obj._index_map = {}
        for s_id, shard in enumerate(obj.shard_indices):
            for q_id, slc in enumerate(obj.slice_indices[s_id]):
                for pos, idx in enumerate(slc):
                    obj._index_map[int(idx)] = (s_id, q_id, pos)
        return obj
