from __future__ import annotations

import numpy as np

from .autodiff import Tensor


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    sample_indices = np.arange(targets.shape[0])
    loss_value = -np.log(probs[sample_indices, targets] + 1e-12).mean()
    out = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), _op="cross_entropy")

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        grad = probs.copy()
        grad[sample_indices, targets] -= 1.0
        grad /= targets.shape[0]
        logits.grad += grad * out.grad

    out._backward = _backward
    return out


def l2_penalty(parameters: list[Tensor]) -> Tensor:
    if not parameters:
        return Tensor(0.0, requires_grad=False)
    penalty = (parameters[0] * parameters[0]).sum()
    for param in parameters[1:]:
        penalty = penalty + (param * param).sum()
    return penalty
