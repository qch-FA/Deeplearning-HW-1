from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import DatasetBundle
from .model import MLPClassifier
from .utils import ensure_dir, save_json

def plot_training_curves(history: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Acc", marker="o")
    plt.plot(epochs, history["val_accuracy"], label="Val Acc", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["val_accuracy"], color="tab:green", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[truth, pred] += 1
    return matrix


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(9, 7))
    plt.imshow(confusion, cmap="Blues")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    max_value = confusion.max() if confusion.size else 0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            color = "white" if confusion[row, col] > max_value / 2 else "black"
            plt.text(col, row, int(confusion[row, col]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def visualize_first_layer_weights(
    model: MLPClassifier,
    image_size: int,
    output_path: str | Path,
    max_filters: int = 16,
) -> list[dict]:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    weights = model.fc1.weight.data.T
    norms = np.linalg.norm(weights, axis=1)
    top_indices = np.argsort(norms)[::-1][:max_filters]

    cols = 4
    rows = int(np.ceil(len(top_indices) / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    summaries: list[dict] = []
    channel_labels = ["red", "green", "blue"]

    for plot_index, unit_index in enumerate(top_indices, start=1):
        kernel = weights[unit_index].reshape(image_size, image_size, 3)
        normalized = kernel - kernel.min()
        if normalized.max() > 0:
            normalized = normalized / normalized.max()

        channel_energy = np.abs(kernel).mean(axis=(0, 1))
        dominant_channel = channel_labels[int(channel_energy.argmax())]
        horizontal_energy = np.abs(np.diff(kernel, axis=1)).mean()
        vertical_energy = np.abs(np.diff(kernel, axis=0)).mean()

        summaries.append(
            {
                "unit_index": int(unit_index),
                "dominant_channel": dominant_channel,
                "horizontal_energy": float(horizontal_energy),
                "vertical_energy": float(vertical_energy),
                "l2_norm": float(norms[unit_index]),
            }
        )

        plt.subplot(rows, cols, plot_index)
        plt.imshow(normalized)
        plt.axis("off")
        plt.title(f"Unit {unit_index}\n{dominant_channel}")

    plt.suptitle("First-Layer Weight Visualization", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return summaries


PAIR_REASONS = {
    ("Highway", "River"): "两类都可能表现为细长带状结构，且常与植被或裸地区域相邻，低分辨率下边界纹理接近。",
    ("River", "Highway"): "河流和高速公路都可能以长条状贯穿画面，颜色与周围地物混合后容易混淆。",
    ("Industrial", "Residential"): "工业区与住宅区都包含规则建筑块和道路网络，密集人工纹理较相似。",
    ("Residential", "Industrial"): "住宅区与工业区都包含大量屋顶和道路，局部块状纹理相近。",
    ("Forest", "Pasture"): "森林与草地都以绿色植被为主，当树冠纹理不明显时容易被看作更均匀的草地区域。",
    ("Pasture", "Forest"): "草地与森林共享绿色主色调，阴影或地块边缘会让草地看起来更像森林。",
    ("SeaLake", "River"): "水体类别共享明显蓝色通道响应，若水面形状较窄或岸线复杂，湖泊容易误判为河流。",
    ("River", "SeaLake"): "宽河段或河道弯曲区域在局部视野下可能与湖泊十分接近。",
}

def write_weight_analysis(weight_summaries: list[dict], output_path: str | Path) -> None:
    lines = [
        "# Weight Pattern Notes",
        "",
        "下列结论基于第一层权重中范数最大的若干隐藏单元，属于对空间模式的定性观察。",
        "",
    ]
    for summary in weight_summaries:
        orientation = "横向纹理更强" if summary["horizontal_energy"] > summary["vertical_energy"] else "纵向纹理更强"
        dominant_channel = summary["dominant_channel"]
        if dominant_channel == "blue":
            semantic = "更可能响应水体或岸线相关区域"
        elif dominant_channel == "green":
            semantic = "更可能响应植被覆盖区域"
        else:
            semantic = "更可能响应建筑、裸地或高反差人工地物"
        lines.append(
            f"- 隐藏单元 {summary['unit_index']}: {dominant_channel} 通道响应最强，{orientation}，{semantic}。"
        )

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _denormalize_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def build_error_analysis(
    bundle: DatasetBundle,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    indices: np.ndarray,
    output_image_path: str | Path,
    output_text_path: str | Path,
    max_examples: int = 9,
) -> list[dict]:
    output_image_path = Path(output_image_path)
    output_text_path = Path(output_text_path)
    ensure_dir(output_image_path.parent)

    wrong_mask = y_true != y_pred
    wrong_indices = np.where(wrong_mask)[0]
    if wrong_indices.size == 0:
        output_text_path.write_text("# Error Analysis\n\n测试集没有误分类样本。", encoding="utf-8")
        return []

    pair_counter = Counter(zip(y_true[wrong_indices], y_pred[wrong_indices]))
    ranked_pairs = pair_counter.most_common()

    selected_positions: list[int] = []
    for (truth, pred), _ in ranked_pairs:
        pair_positions = wrong_indices[(y_true[wrong_indices] == truth) & (y_pred[wrong_indices] == pred)]
        if pair_positions.size > 0:
            selected_positions.append(int(pair_positions[0]))
        if len(selected_positions) >= max_examples:
            break

    if len(selected_positions) < max_examples:
        remaining = [int(pos) for pos in wrong_indices if int(pos) not in selected_positions]
        selected_positions.extend(remaining[: max_examples - len(selected_positions)])

    cols = 3
    rows = int(np.ceil(len(selected_positions) / cols))
    plt.figure(figsize=(cols * 4, rows * 4))

    examples: list[dict] = []
    lines = [
        "# Error Analysis",
        "",
        "以下错例来自测试集误分类样本，用于报告中的现象分析。",
        "",
    ]

    for plot_idx, position in enumerate(selected_positions, start=1):
        dataset_index = int(indices[position])
        truth_name = bundle.class_names[int(y_true[position])]
        pred_name = bundle.class_names[int(y_pred[position])]
        confidence = float(probs[position, y_pred[position]])
        reason = PAIR_REASONS.get((truth_name, pred_name), "该样本在颜色分布、纹理密度或局部几何结构上与预测类别存在重合。")
        image = _denormalize_image(bundle.images[dataset_index])

        plt.subplot(rows, cols, plot_idx)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"T:{truth_name}\nP:{pred_name} ({confidence:.2f})")

        example = {
            "dataset_index": dataset_index,
            "path": bundle.paths[dataset_index],
            "true_label": truth_name,
            "pred_label": pred_name,
            "pred_confidence": confidence,
            "reason": reason,
        }
        examples.append(example)
        lines.append(
            f"- `{bundle.paths[dataset_index]}`: 真实类别为 {truth_name}，预测为 {pred_name}，预测置信度 {confidence:.3f}。可能原因：{reason}"
        )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=220)
    plt.close()
    output_text_path.write_text("\n".join(lines), encoding="utf-8")
    return examples


def plot_search_results(results: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    labels = [
        f"T{row['trial_id']}\n{row['hidden_dims_label']}\n{row['activation']}\nlr={row['learning_rate']}"
        for row in results
    ]
    values = [row["best_val_accuracy"] for row in results]

    plt.figure(figsize=(max(8, len(labels) * 1.4), 4.8))
    plt.bar(range(len(labels)), values, color="tab:orange")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Hyperparameter Search Results")
    plt.ylim(0.0, min(1.0, max(values) + 0.1) if values else 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_confusion_json(confusion: np.ndarray, class_names: list[str], output_path: str | Path) -> None:
    save_json(
        output_path,
        {
            "class_names": class_names,
            "confusion_matrix": confusion,
        },
    )
