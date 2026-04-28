from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_float_array(data: np.ndarray | float | int) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return array


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __hash__ = object.__hash__

    def __init__(
        self,
        data: np.ndarray | float | int,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = _as_float_array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Non-scalar tensor requires an explicit gradient.")
            grad = np.ones_like(self.data, dtype=np.float32)
        else:
            grad = _as_float_array(grad)

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float32)
        self.grad += grad

        for node in reversed(topo):
            node._backward()

    def __add__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self + other

    def __sub__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return other + (-self)

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self * other

    def __truediv__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="div",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication expects another Tensor.")
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                if not keepdims:
                    for ax in sorted(ax % self.data.ndim for ax in axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = 1
            for ax in axes:
                denom *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        values = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(values, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += values * (1.0 - values) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        values = np.tanh(self.data)
        out = Tensor(values, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += (1.0 - values ** 2) * out.grad

        out._backward = _backward
        return out


class Parameter(Tensor):
    def __init__(self, data: np.ndarray | float | int) -> None:
        super().__init__(data=data, requires_grad=True)
