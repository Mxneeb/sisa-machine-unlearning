"""
Membership Inference Attack (MIA) — verification that unlearning worked.

Uses a simple threshold attack: if the model's confidence on a sample is
above a threshold, it is classified as a member. After unlearning, the
confidence on the forgotten sample should drop, indicating successful removal.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def confidence_score(model: nn.Module, X: torch.Tensor, y: int, device="cpu") -> float:
    """Return model's softmax confidence on the true class for a single sample."""
    model.eval()
    X = X.unsqueeze(0).to(device)
    probs = torch.softmax(model(X), dim=1)
    return probs[0, y].item()


@torch.no_grad()
def membership_inference_attack(
    model: nn.Module,
    member_subset: Subset,
    non_member_subset: Subset,
    threshold: float = 0.5,
    device="cpu",
) -> dict:
    """
    Simple threshold-based MIA.

    Returns accuracy of the attack (how often it correctly distinguishes
    members from non-members). Higher = model memorises training data more.
    After unlearning, this score on the forgotten sample should be ~0.5 (random).
    """
    model.eval()

    def get_confidences(subset):
        loader = DataLoader(subset, batch_size=256)
        confs = []
        for X, y in loader:
            X = X.to(device)
            probs = torch.softmax(model(X), dim=1)
            true_class_conf = probs[torch.arange(len(y)), y]
            confs.extend(true_class_conf.cpu().tolist())
        return confs

    member_confs = get_confidences(member_subset)
    non_member_confs = get_confidences(non_member_subset)

    tp = sum(c >= threshold for c in member_confs)
    tn = sum(c < threshold for c in non_member_confs)
    total = len(member_confs) + len(non_member_confs)
    attack_accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "attack_accuracy": round(attack_accuracy, 4),
        "mean_member_confidence": round(sum(member_confs) / len(member_confs), 4) if member_confs else 0,
        "mean_non_member_confidence": round(sum(non_member_confs) / len(non_member_confs), 4) if non_member_confs else 0,
        "threshold": threshold,
    }
