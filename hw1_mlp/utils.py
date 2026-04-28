from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def cross_entropy_np(logits: np.ndarray, targets: np.ndarray) -> float:
    probs = softmax_np(logits)
    losses = -np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)
    return float(losses.mean())


def accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def to_serializable(payload: Any) -> Any:
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (np.floating,)):
        return float(payload)
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: to_serializable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [to_serializable(item) for item in payload]
    return payload


def save_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)


def parse_hidden_dims(text: str) -> tuple[int, int]:
    parts = text.lower().replace(" ", "").split("x")
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError(f"Invalid hidden dimension specification: {text}")


def format_hidden_dims(hidden_dims: tuple[int, int]) -> str:
    return f"{hidden_dims[0]}x{hidden_dims[1]}"
