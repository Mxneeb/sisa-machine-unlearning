"""
Simple neural network for Purchase-100 / MNIST classification.
"""
import torch
import torch.nn as nn


class PurchaseClassifier(nn.Module):
    """MLP for Purchase-100 dataset (600-dim binary feature vector, 100 classes)."""

    def __init__(self, input_dim: int = 600, num_classes: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MNISTClassifier(nn.Module):
    """Small CNN for MNIST (for visual demo purposes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_model(dataset: str = "purchase") -> nn.Module:
    if dataset == "purchase":
        return PurchaseClassifier()
    elif dataset == "mnist":
        return MNISTClassifier()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
