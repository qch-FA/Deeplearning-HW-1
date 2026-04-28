from __future__ import annotations

import numpy as np

from .autodiff import Parameter, Tensor


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        params: list[tuple[str, Parameter]] = []
        for name, value in self.__dict__.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Parameter):
                params.append((full_name, value))
            elif isinstance(value, Module):
                params.extend(value.named_parameters(full_name))
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    item_name = f"{full_name}.{index}"
                    if isinstance(item, Parameter):
                        params.append((item_name, item))
                    elif isinstance(item, Module):
                        params.extend(item.named_parameters(item_name))
        return params

    def parameters(self) -> list[Parameter]:
        return [param for _, param in self.named_parameters()]

    def weight_parameters(self) -> list[Parameter]:
        return [param for name, param in self.named_parameters() if name.endswith("weight")]

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: param.data.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        for name, param in self.named_parameters():
            if name not in state_dict:
                raise KeyError(f"Missing parameter '{name}' in state dict.")
            param.data[...] = state_dict[name]


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, nonlinearity: str = "relu") -> None:
        scale = np.sqrt(2.0 / in_features) if nonlinearity == "relu" else np.sqrt(1.0 / in_features)
        weights = np.random.randn(in_features, out_features).astype(np.float32) * scale
        bias = np.zeros((1, out_features), dtype=np.float32)
        self.weight = Parameter(weights)
        self.bias = Parameter(bias)

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias

    def forward_array(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.data + self.bias.data
