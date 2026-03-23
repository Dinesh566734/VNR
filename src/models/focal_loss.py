"""Binary focal loss for highly imbalanced fraud classification."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Reduction = Literal["none", "mean", "sum"]


def binary_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: Reduction = "mean",
    from_logits: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute binary focal loss.

    `alpha` is applied to the positive fraud class and `1 - alpha` to the
    negative class. Inputs may be probabilities in [0, 1] or raw logits.
    """

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    targets = targets.float()
    if inputs.shape != targets.shape:
        inputs = inputs.reshape_as(targets)

    if from_logits:
        probabilities = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    else:
        probabilities = inputs.clamp(min=eps, max=1.0 - eps)
        bce_loss = F.binary_cross_entropy(probabilities, targets, reduction="none")

    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_factor = (1.0 - p_t).pow(gamma)
    loss = alpha_t * focal_factor * bce_loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    """Binary focal loss module with the paper-specified defaults."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Reduction = "mean",
        from_logits: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return binary_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
            from_logits=self.from_logits,
            eps=self.eps,
        )
