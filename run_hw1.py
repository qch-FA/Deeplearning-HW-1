from __future__ import annotations

import argparse

from hw1_mlp.data import load_eurosat
from hw1_mlp.model import MLPClassifier
from hw1_mlp.reporting import (
    build_error_analysis,
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_search_results,
    plot_training_curves,
    save_confusion_json,
    visualize_first_layer_weights,
    write_weight_analysis,
)
from hw1_mlp.search import random_search
from hw1_mlp.trainer import evaluate_split, load_checkpoint, train_with_validation
from hw1_mlp.utils import ensure_dir, format_hidden_dims, parse_hidden_dims, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a custom NumPy MLP on EuroSAT.")
    parser.add_argument("--data-root", type=str, default="EuroSAT_RGB")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--search-trials", type=int, default=4)
    parser.add_argument("--search-epochs", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=12)
    parser.add_argument("--lr-decay", type=float, default=0.92)
    parser.add_argument("--hidden-space", nargs="+", default=["128x64", "192x96", "256x128"])
    parser.add_argument("--lr-space", nargs="+", type=float, default=[0.03, 0.05])
    parser.add_argument("--weight-decay-space", nargs="+", type=float, default=[1e-4, 5e-4])
    parser.add_argument("--activation-space", nargs="+", default=["relu", "tanh"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    search_dir = ensure_dir(output_dir / "search")
    final_dir = ensure_dir(output_dir / "final")
    report_dir = ensure_dir(output_dir / "reports")

    bundle = load_eurosat(
        data_root=args.data_root,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_json(
        output_dir / "dataset_summary.json",
        {
            "image_size": bundle.image_size,
            "num_samples": int(bundle.images.shape[0]),
            "num_classes": len(bundle.class_names),
            "class_names": bundle.class_names,
            "train_size": int(bundle.train_indices.shape[0]),
            "val_size": int(bundle.val_indices.shape[0]),
            "test_size": int(bundle.test_indices.shape[0]),
            "mean": bundle.mean,
            "std": bundle.std,
        },
    )

    hidden_candidates = [parse_hidden_dims(spec) for spec in args.hidden_space]
    best_config, search_results = random_search(
        bundle=bundle,
        output_dir=search_dir,
        hidden_candidates=hidden_candidates,
        learning_rates=list(args.lr_space),
        weight_decays=list(args.weight_decay_space),
        activations=list(args.activation_space),
        batch_size=args.batch_size,
        epochs=args.search_epochs,
        lr_decay=args.lr_decay,
        num_trials=args.search_trials,
        seed=args.seed,
    )
    plot_search_results(search_results, report_dir / "search_results.png")

    final_config = {
        "hidden_dims": tuple(best_config["hidden_dims"]),
        "learning_rate": float(best_config["learning_rate"]),
        "weight_decay": float(best_config["weight_decay"]),
        "activation": best_config["activation"],
        "image_size": args.image_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    final_model = MLPClassifier(
        input_dim=bundle.input_dim,
        hidden_dims=final_config["hidden_dims"],
        num_classes=len(bundle.class_names),
        activation=final_config["activation"],
    )
    final_run = train_with_validation(
        model=final_model,
        bundle=bundle,
        output_dir=final_dir,
        learning_rate=final_config["learning_rate"],
        lr_decay=args.lr_decay,
        weight_decay=final_config["weight_decay"],
        batch_size=args.batch_size,
        epochs=args.final_epochs,
        seed=args.seed + 999,
        checkpoint_name="best_model.npz",
        config=final_config,
    )

    plot_training_curves(final_run["history"], report_dir / "training_curves.png")

    best_state = load_checkpoint(final_dir / "best_model.npz")
    final_model.load_state_dict(best_state)
    test_metrics, y_true, y_pred, probs, indices = evaluate_split(
        final_model,
        bundle=bundle,
        split="test",
        batch_size=args.batch_size,
    )

    confusion = compute_confusion_matrix(y_true, y_pred, len(bundle.class_names))
    plot_confusion_matrix(confusion, bundle.class_names, report_dir / "confusion_matrix.png")
    save_confusion_json(confusion, bundle.class_names, report_dir / "confusion_matrix.json")

    weight_summaries = visualize_first_layer_weights(
        final_model,
        image_size=bundle.image_size,
        output_path=report_dir / "first_layer_weights.png",
    )
    write_weight_analysis(weight_summaries, report_dir / "weight_analysis.md")

    error_examples = build_error_analysis(
        bundle=bundle,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        indices=indices,
        output_image_path=report_dir / "error_cases.png",
        output_text_path=report_dir / "error_analysis.md",
    )

    summary = {
        "best_search_config": {
            **best_config,
            "hidden_dims_label": format_hidden_dims(tuple(best_config["hidden_dims"])),
        },
        "final_training": {
            "best_epoch": final_run["best_epoch"],
            "best_val_accuracy": final_run["best_val_accuracy"],
        },
        "test_metrics": {
            "loss": test_metrics.loss,
            "accuracy": test_metrics.accuracy,
        },
        "paths": {
            "training_curves": (report_dir / "training_curves.png").as_posix(),
            "search_results": (report_dir / "search_results.png").as_posix(),
            "confusion_matrix": (report_dir / "confusion_matrix.png").as_posix(),
            "first_layer_weights": (report_dir / "first_layer_weights.png").as_posix(),
            "error_cases": (report_dir / "error_cases.png").as_posix(),
            "best_model": (final_dir / "best_model.npz").as_posix(),
        },
        "error_examples": error_examples,
        "confusion_matrix": confusion,
    }
    save_json(output_dir / "summary.json", summary)

    print("Best config:", summary["best_search_config"])
    print("Final validation accuracy:", final_run["best_val_accuracy"])
    print("Test accuracy:", test_metrics.accuracy)
    print("Confusion matrix:")
    for row in confusion:
        print(" ".join(f"{int(value):4d}" for value in row))


if __name__ == "__main__":
    main()
