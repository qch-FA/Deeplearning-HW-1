from __future__ import annotations

import argparse
import json
from pathlib import Path

from hw1_mlp.data import load_eurosat
from hw1_mlp.model import MLPClassifier
from hw1_mlp.reporting import compute_confusion_matrix
from hw1_mlp.trainer import evaluate_split, load_checkpoint
from hw1_mlp.utils import parse_hidden_dims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved NumPy MLP checkpoint on EuroSAT.")
    parser.add_argument("--data-root", type=str, default="EuroSAT_RGB")
    parser.add_argument("--checkpoint", type=str, default="outputs/final/best_model.npz")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=str, default=None)
    parser.add_argument("--activation", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    config = metadata.get("config", {})
    hidden_dims = args.hidden_dims or (
        f"{config['hidden_dims'][0]}x{config['hidden_dims'][1]}" if "hidden_dims" in config else "128x64"
    )
    activation = args.activation or config.get("activation", "relu")
    image_size = args.image_size if args.image_size is not None else metadata.get("image_size", 32)
    train_ratio = args.train_ratio if args.train_ratio is not None else metadata.get("train_ratio", 0.7)
    val_ratio = args.val_ratio if args.val_ratio is not None else metadata.get("val_ratio", 0.15)
    seed = args.seed if args.seed is not None else metadata.get("seed", 42)

    bundle = load_eurosat(
        data_root=args.data_root,
        image_size=image_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    model = MLPClassifier(
        input_dim=bundle.input_dim,
        hidden_dims=parse_hidden_dims(hidden_dims),
        num_classes=len(bundle.class_names),
        activation=activation,
    )
    state = load_checkpoint(checkpoint_path)
    model.load_state_dict(state)
    metrics, y_true, y_pred, _, _ = evaluate_split(model, bundle, split="test", batch_size=args.batch_size)
    confusion = compute_confusion_matrix(y_true, y_pred, len(bundle.class_names))

    print("Test loss:", metrics.loss)
    print("Test accuracy:", metrics.accuracy)
    print("Confusion matrix:")
    for row in confusion:
        print(" ".join(f"{int(value):4d}" for value in row))


if __name__ == "__main__":
    main()
