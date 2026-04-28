from __future__ import annotations

import numpy as np

from .autodiff import Tensor
from .layers import Linear, Module


def _apply_activation_tensor(x: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return x.relu()
    if activation == "sigmoid":
        return x.sigmoid()
    if activation == "tanh":
        return x.tanh()
    raise ValueError(f"Unsupported activation: {activation}")


def _apply_activation_array(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if activation == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unsupported activation: {activation}")


class MLPClassifier(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] | list[int] | int,
        num_classes: int,
        activation: str = "relu",
    ) -> None:
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims, hidden_dims)
        elif len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0])
        elif len(hidden_dims) != 2:
            raise ValueError("hidden_dims must be an int or a pair of ints.")

        self.input_dim = input_dim
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.num_classes = num_classes
        self.activation = activation

        self.fc1 = Linear(input_dim, self.hidden_dims[0], nonlinearity=activation)
        self.fc2 = Linear(self.hidden_dims[0], self.hidden_dims[1], nonlinearity=activation)
        self.fc3 = Linear(self.hidden_dims[1], num_classes, nonlinearity="linear")

    def forward(self, x: Tensor) -> Tensor:
        x = _apply_activation_tensor(self.fc1(x), self.activation)
        x = _apply_activation_tensor(self.fc2(x), self.activation)
        return self.fc3(x)

    def forward_array(self, x: np.ndarray) -> np.ndarray:
        x = _apply_activation_array(self.fc1.forward_array(x), self.activation)
        x = _apply_activation_array(self.fc2.forward_array(x), self.activation)
        return self.fc3.forward_array(x)
