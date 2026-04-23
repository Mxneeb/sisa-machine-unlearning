"""
SISA Training engine.

Trains each shard model slice-by-slice and saves checkpoints so that
an unlearn request only requires replaying slices after the affected one.
"""
import os
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .model import get_model
from .sharding import ShardedDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SISATrainer:
    """
    Trains S constituent models using the SISA protocol.

    Args:
        sharded_dataset: ShardedDataset wrapping the training data.
        checkpoint_dir:  Directory to store shard checkpoints.
        dataset_name:    'purchase' or 'mnist'.
        epochs_per_slice: Training epochs for each slice increment.
        batch_size:      Mini-batch size.
        lr:              Learning rate.
        progress_callback: Optional callable(shard_id, slice_id, metrics) for UI updates.
    """

    def __init__(
        self,
        sharded_dataset: ShardedDataset,
        checkpoint_dir: str = "checkpoints",
        dataset_name: str = "purchase",
        epochs_per_slice: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        progress_callback: Optional[Callable] = None,
    ):
        self.sd = sharded_dataset
        self.checkpoint_dir = checkpoint_dir
        self.dataset_name = dataset_name
        self.epochs_per_slice = epochs_per_slice
        self.batch_size = batch_size
        self.lr = lr
        self.progress_callback = progress_callback
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _checkpoint_path(self, shard_id: int, slice_id: int) -> str:
        return os.path.join(self.checkpoint_dir, f"shard{shard_id}_slice{slice_id}.pt")

    def _final_path(self, shard_id: int) -> str:
        return os.path.join(self.checkpoint_dir, f"shard{shard_id}_final.pt")

    def train_all(self) -> Dict:
        """Train all shards from scratch. Returns timing/accuracy dict."""
        results = {}
        total_start = time.time()
        for shard_id in range(self.sd.num_shards):
            results[shard_id] = self._train_shard(shard_id, start_slice=0)
        results["total_time"] = time.time() - total_start
        return results

    def _train_shard(self, shard_id: int, start_slice: int = 0) -> Dict:
        model = get_model(self.dataset_name).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        shard_start = time.time()

        # Load checkpoint from the slice before start_slice if we are resuming
        if start_slice > 0:
            ckpt_path = self._checkpoint_path(shard_id, start_slice - 1)
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        for slice_id in range(start_slice, self.sd.num_slices):
            subset = self.sd.get_shard_slice_subset(shard_id, up_to_slice=slice_id)
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)

            for _ in range(self.epochs_per_slice):
                model.train()
                for X, y in loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    optimizer.step()

            # Save checkpoint after each slice
            torch.save(model.state_dict(), self._checkpoint_path(shard_id, slice_id))
            acc = self._evaluate(model, subset)

            if self.progress_callback:
                self.progress_callback(shard_id, slice_id, {"accuracy": acc})

        torch.save(model.state_dict(), self._final_path(shard_id))
        return {"shard_id": shard_id, "time": time.time() - shard_start}

    def unlearn(self, dataset_index: int) -> Dict:
        """
        Execute a forget request for dataset_index.
        Only the affected shard is partially retrained.
        Returns dict with unlearn_time and affected_shard.
        """
        shard_id, slice_id, _ = self.sd.locate(dataset_index)
        start = time.time()

        # Temporarily remove the target index from the dataset
        # (in a real system the dataset itself would be filtered)
        self.sd.shard_indices[shard_id] = self.sd.shard_indices[shard_id][
            self.sd.shard_indices[shard_id] != dataset_index
        ]
        # Rebuild slice_indices for this shard
        self.sd.slice_indices[shard_id] = [
            slc[slc != dataset_index] for slc in self.sd.slice_indices[shard_id]
        ]
        del self.sd._index_map[dataset_index]

        self._train_shard(shard_id, start_slice=slice_id)
        elapsed = time.time() - start
        return {
            "affected_shard": shard_id,
            "replay_from_slice": slice_id,
            "unlearn_time_seconds": round(elapsed, 3),
        }

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, subset: Subset) -> float:
        model.eval()
        loader = DataLoader(subset, batch_size=512)
        correct = total = 0
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return round(correct / total, 4) if total > 0 else 0.0

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Aggregate predictions across all shard models."""
        X = X.to(DEVICE)
        logits_sum = None
        for shard_id in range(self.sd.num_shards):
            model = get_model(self.dataset_name).to(DEVICE)
            model.load_state_dict(
                torch.load(self._final_path(shard_id), map_location=DEVICE)
            )
            model.eval()
            logits = model(X)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        probs = torch.softmax(logits_sum / self.sd.num_shards, dim=1)
        return probs
