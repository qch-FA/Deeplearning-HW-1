from __future__ import annotations

from .autodiff import Parameter


class SGD:
    def __init__(self, parameters: list[Parameter], lr: float) -> None:
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.zero_grad()

    def step(self) -> None:
        for param in self.parameters:
            if param.grad is None:
                continue
            param.data -= self.lr * param.grad


class ExponentialDecay:
    def __init__(self, optimizer: SGD, gamma: float) -> None:
        self.optimizer = optimizer
        self.gamma = gamma
        self.epoch = 0

    def step(self) -> float:
        self.epoch += 1
        self.optimizer.lr *= self.gamma
        return self.optimizer.lr
