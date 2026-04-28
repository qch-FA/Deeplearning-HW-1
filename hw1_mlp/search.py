from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np

from .data import DatasetBundle
from .model import MLPClassifier
from .trainer import train_with_validation
from .utils import format_hidden_dims, save_json


def random_search(
    bundle: DatasetBundle,
    output_dir: str | Path,
    hidden_candidates: list[tuple[int, int]],
    learning_rates: list[float],
    weight_decays: list[float],
    activations: list[str],
    batch_size: int,
    epochs: int,
    lr_decay: float,
    num_trials: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_configs = [
        {
            "hidden_dims": tuple(hidden_dims),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "activation": activation,
        }
        for hidden_dims, lr, weight_decay, activation in product(
            hidden_candidates,
            learning_rates,
            weight_decays,
            activations,
        )
    ]

    rng = np.random.default_rng(seed)
    if num_trials >= len(all_configs):
        selected_configs = all_configs.copy()
        rng.shuffle(selected_configs)
    else:
        chosen = rng.choice(len(all_configs), size=num_trials, replace=False)
        selected_configs = [all_configs[index] for index in chosen]

    results: list[dict] = []
    best_result: dict | None = None

    for trial_id, config in enumerate(selected_configs, start=1):
        model = MLPClassifier(
            input_dim=bundle.input_dim,
            hidden_dims=config["hidden_dims"],
            num_classes=len(bundle.class_names),
            activation=config["activation"],
        )
        run = train_with_validation(
            model=model,
            bundle=bundle,
            output_dir=output_dir / f"trial_{trial_id:02d}",
            learning_rate=config["learning_rate"],
            lr_decay=lr_decay,
            weight_decay=config["weight_decay"],
            batch_size=batch_size,
            epochs=epochs,
            seed=seed + trial_id * 13,
            checkpoint_name="best_weights.npz",
            config=config,
        )
        result = {
            "trial_id": trial_id,
            **config,
            "hidden_dims_label": format_hidden_dims(config["hidden_dims"]),
            "best_val_accuracy": run["best_val_accuracy"],
            "best_epoch": run["best_epoch"],
            "checkpoint_path": run["checkpoint_path"],
        }
        results.append(result)

        if best_result is None or result["best_val_accuracy"] > best_result["best_val_accuracy"]:
            best_result = result

    csv_path = output_dir / "search_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial_id",
                "hidden_dims_label",
                "learning_rate",
                "weight_decay",
                "activation",
                "best_val_accuracy",
                "best_epoch",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in writer.fieldnames})

    save_json(output_dir / "search_results.json", results)
    return best_result, results
