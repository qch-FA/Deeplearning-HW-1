from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .autodiff import Tensor
from .data import BatchIterator, DatasetBundle
from .losses import cross_entropy, l2_penalty
from .model import MLPClassifier
from .optim import ExponentialDecay, SGD
from .utils import accuracy_np, cross_entropy_np, save_json, softmax_np


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def save_checkpoint(
    path: str | Path,
    model: MLPClassifier,
    metadata: dict,
) -> None:
    state = model.state_dict()
    payload = {f"param__{name}": value for name, value in state.items()}
    np.savez(Path(path), **payload)
    save_json(Path(path).with_suffix(".json"), metadata)


def load_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    raw = np.load(Path(path))
    state = {}
    for name in raw.files:
        if name.startswith("param__"):
            state[name.replace("param__", "", 1)] = raw[name]
    return state


def train_one_epoch(
    model: MLPClassifier,
    bundle: DatasetBundle,
    optimizer: SGD,
    batch_size: int,
    weight_decay: float,
    seed: int,
) -> EpochMetrics:
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_x, batch_y, _ in BatchIterator(bundle, "train", batch_size, shuffle=True, seed=seed):
        optimizer.zero_grad()
        logits = model(Tensor(batch_x, requires_grad=False))
        data_loss = cross_entropy(logits, batch_y)
        if weight_decay > 0.0:
            penalty = 0.5 * weight_decay * l2_penalty(model.weight_parameters())
            loss = data_loss + penalty
        else:
            loss = data_loss
        loss.backward()
        optimizer.step()

        predictions = logits.data.argmax(axis=1)
        total_correct += int((predictions == batch_y).sum())
        total_loss += float(loss.data) * batch_y.shape[0]
        total_count += batch_y.shape[0]

    return EpochMetrics(loss=total_loss / total_count, accuracy=total_correct / total_count)


def evaluate_split(
    model: MLPClassifier,
    bundle: DatasetBundle,
    split: str,
    batch_size: int,
) -> tuple[EpochMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []

    for batch_x, batch_y, batch_indices in BatchIterator(bundle, split, batch_size, shuffle=False):
        logits = model.forward_array(batch_x)
        all_logits.append(logits.astype(np.float32))
        all_labels.append(batch_y)
        all_indices.append(batch_indices)

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)
    metrics = EpochMetrics(loss=cross_entropy_np(logits, labels), accuracy=accuracy_np(labels, preds))
    return metrics, labels, preds, probs, indices


def train_with_validation(
    model: MLPClassifier,
    bundle: DatasetBundle,
    output_dir: str | Path,
    learning_rate: float,
    lr_decay: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    seed: int,
    checkpoint_name: str,
    config: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = SGD(model.parameters(), lr=learning_rate)
    scheduler = ExponentialDecay(optimizer, gamma=lr_decay)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    best_val_accuracy = -1.0
    best_epoch = -1
    checkpoint_path = output_dir / checkpoint_name

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            bundle=bundle,
            optimizer=optimizer,
            batch_size=batch_size,
            weight_decay=weight_decay,
            seed=seed + epoch,
        )
        val_metrics, _, _, _, _ = evaluate_split(model, bundle, "val", batch_size=batch_size)

        history["train_loss"].append(train_metrics.loss)
        history["train_accuracy"].append(train_metrics.accuracy)
        history["val_loss"].append(val_metrics.loss)
        history["val_accuracy"].append(val_metrics.accuracy)
        history["learning_rate"].append(optimizer.lr)

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            best_epoch = epoch
            metadata = {
                "config": config,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "image_size": bundle.image_size,
                "mean": bundle.mean.tolist(),
                "std": bundle.std.tolist(),
                "class_names": bundle.class_names,
                "train_ratio": config.get("train_ratio"),
                "val_ratio": config.get("val_ratio"),
                "seed": config.get("seed"),
            }
            save_checkpoint(checkpoint_path, model, metadata)

        scheduler.step()

    best_state = load_checkpoint(checkpoint_path)
    model.load_state_dict(best_state)

    return {
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path.as_posix(),
    }
