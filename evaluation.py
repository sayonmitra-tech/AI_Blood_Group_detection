"""
Evaluation Module – AI Blood Group Detection System
====================================================
Computes accuracy, precision, recall, F1-score, and a confusion matrix
for the full end-to-end prediction pipeline (region extraction → blood group).

Usage
-----
    python evaluation.py \\
        --model-path  blood_model.keras \\
        --annotations data/test/_annotations.coco.json \\
        --image-dir   data/test

The ground-truth blood group is inferred from the image *filename prefix*
(e.g. ``AB-1-_jpg…`` → ``AB``).  The Rh factor (+ / -) **cannot** be
inferred from filenames alone and is therefore excluded from the filename-
based label; instead an explicit ``--ground-truth-csv`` may be supplied
for full Rh evaluation (see ``--help``).
"""

import os
import re
import csv
import argparse
import logging
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend – works without a display
import matplotlib.pyplot as plt

# Local module
from blood_group_detection import (
    load_model,
    load_coco_annotations,
    predict_blood_group,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Label extraction from filename
# ──────────────────────────────────────────────

# Supported ABO groups (without Rh) extracted from filename prefix
ABO_GROUPS   = {"A", "B", "AB", "O"}
ABO_FULL     = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}


def extract_label_from_filename(filename: str) -> str | None:
    """
    Infer ABO+Rh ground-truth label from an image filename convention.

    Expected filename format:  ``<GROUP>-<number>-_<ext>.rf.<hash>.jpg``
    where ``<GROUP>`` is one of ``A``, ``B``, ``AB``, ``O``.

    Since the Rh factor is not encoded in the filename, the returned label
    is ABO-only (e.g. ``"AB"``, ``"O"``).

    Parameters
    ----------
    filename : str
        Bare filename (no directory component).

    Returns
    -------
    str or None
        ABO group string, or ``None`` if the pattern is not recognised.
    """
    basename = os.path.basename(filename)
    # Match filenames like AB-143-_jpg... or O-91-_jpg...
    m = re.match(r"^(A|B|AB|O)-\d+", basename)
    if m:
        return m.group(1)
    return None


# ──────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────

def evaluate(
    model,
    coco: dict,
    image_dir: str,
    ground_truth_csv: str | None = None,
    output_dir: str = "evaluation_results",
) -> dict:
    """
    Evaluate model performance across all images in the COCO annotation set.

    Two evaluation modes are supported:

    * **ABO-only** (default): ground truth derived from filename prefix.
      Rh factor is stripped from predictions for fair comparison.
    * **Full ABO+Rh**: ground truth loaded from ``ground_truth_csv``
      (columns: ``filename``, ``blood_group``).

    Parameters
    ----------
    model : tf.keras.Model
        Loaded classification model.
    coco : dict
        COCO annotation dict for the test split.
    image_dir : str
        Directory containing the test images.
    ground_truth_csv : str or None
        Optional path to a CSV with columns ``filename,blood_group``
        (e.g. ``"AB-1-...jpg,AB+"``).
    output_dir : str
        Directory where confusion-matrix plot and CSV results are saved.

    Returns
    -------
    dict
        Dictionary with keys ``accuracy``, ``per_class`` (precision / recall /
        f1 per class), ``confusion_matrix``, and ``results`` (per-image rows).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build ground-truth lookup
    gt_lookup: dict[str, str] = {}

    if ground_truth_csv:
        with open(ground_truth_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_lookup[row["filename"].strip()] = row["blood_group"].strip().upper()
        logger.info("Loaded %d ground-truth labels from %s", len(gt_lookup), ground_truth_csv)
        use_full_rh = True
    else:
        use_full_rh = False

    # Collect unique test image names from COCO
    image_names = [img["file_name"] for img in coco["images"]]
    logger.info("Evaluating %d images …", len(image_names))

    # Determine the set of classes actually present
    if use_full_rh:
        all_classes = sorted(ABO_FULL)
    else:
        all_classes = sorted(ABO_GROUPS)

    class_idx = {c: i for i, c in enumerate(all_classes)}
    n_classes  = len(all_classes)

    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    results_rows = []          # [(filename, true_label, pred_label, correct)]
    skipped = 0

    for name in image_names:
        # ── Ground truth ──────────────────────────────────
        if use_full_rh:
            true_label = gt_lookup.get(name)
        else:
            true_label = extract_label_from_filename(name)

        if true_label is None:
            logger.warning("Could not determine ground truth for '%s' – skipping.", name)
            skipped += 1
            continue

        if true_label not in class_idx:
            logger.warning("Unknown label '%s' for '%s' – skipping.", true_label, name)
            skipped += 1
            continue

        # ── Prediction ────────────────────────────────────
        pred_full = predict_blood_group(name, model, coco, image_dir, verbose=False)

        if pred_full is None:
            logger.warning("Prediction returned None for '%s' – skipping.", name)
            skipped += 1
            continue

        pred_label = pred_full if use_full_rh else pred_full.rstrip("+-")

        if pred_label not in class_idx:
            logger.warning("Predicted label '%s' not in class set – skipping.", pred_label)
            skipped += 1
            continue

        # ── Record ────────────────────────────────────────
        ti = class_idx[true_label]
        pi = class_idx[pred_label]
        conf_matrix[ti, pi] += 1

        correct = true_label == pred_label
        results_rows.append((name, true_label, pred_label, correct))

    # ── Aggregate metrics ─────────────────────────────────
    total   = len(results_rows)
    correct = sum(1 for _, _, _, c in results_rows if c)
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    for cls in all_classes:
        i  = class_idx[cls]
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1,
                          "support": int(conf_matrix[i, :].sum())}

    # ── Print summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS  ({total} images, {skipped} skipped)")
    print("=" * 60)
    print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
    print()
    print(f"  {'Class':<8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print("  " + "-" * 52)
    for cls in all_classes:
        m = per_class[cls]
        print(f"  {cls:<8}  {m['precision']:>10.4f}  {m['recall']:>8.4f}"
              f"  {m['f1']:>8.4f}  {m['support']:>8d}")
    print()

    # ── Save per-image CSV ────────────────────────────────
    csv_path = os.path.join(output_dir, "per_image_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "predicted_label", "correct"])
        writer.writerows(results_rows)
    logger.info("Per-image results saved to %s", csv_path)

    # ── Confusion matrix plot ─────────────────────────────
    _plot_confusion_matrix(conf_matrix, all_classes, output_dir)

    return {
        "accuracy":        accuracy,
        "per_class":       per_class,
        "confusion_matrix": conf_matrix,
        "results":         results_rows,
        "total":           total,
        "skipped":         skipped,
    }


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_dir: str,
) -> None:
    """
    Render and save a normalised confusion matrix as a PNG figure.

    Parameters
    ----------
    matrix : np.ndarray
        Raw (non-normalised) confusion matrix of shape ``(n, n)``.
    class_names : list[str]
        Ordered list of class label strings.
    output_dir : str
        Directory to write ``confusion_matrix.png``.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm     = np.divide(matrix.astype(float), row_sums,
                         out=np.zeros_like(matrix, dtype=float),
                         where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, pad=14)

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if norm[i, j] > thresh else "black"
            ax.text(j, i, f"{norm[i,j]:.2f}\n({matrix[i,j]})",
                    ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", out_path)


# ──────────────────────────────────────────────
# Misclassification analysis
# ──────────────────────────────────────────────

def analyse_errors(results: list[tuple]) -> None:
    """
    Print a summary of misclassified images grouped by (true → predicted) pair.

    Parameters
    ----------
    results : list[tuple]
        As returned in ``evaluate()["results"]``:
        ``[(filename, true_label, pred_label, correct), ...]``
    """
    errors = [(fn, t, p) for fn, t, p, c in results if not c]

    if not errors:
        print("\nNo misclassifications found!")
        return

    grouped: dict[tuple, list[str]] = defaultdict(list)
    for fn, true, pred in errors:
        grouped[(true, pred)].append(fn)

    print(f"\n{'='*60}")
    print(f"  MISCLASSIFICATION ANALYSIS  ({len(errors)} errors)")
    print(f"{'='*60}")
    for (true, pred), fns in sorted(grouped.items()):
        print(f"\n  True={true}  →  Predicted={pred}  (×{len(fns)})")
        for fn in fns:
            print(f"    • {fn}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AI Blood Group Detection System"
    )
    parser.add_argument(
        "--model-path",
        default="blood_model.keras",
        help="Path to saved Keras model (default: blood_model.keras)",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to test set _annotations.coco.json",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing the test images",
    )
    parser.add_argument(
        "--ground-truth-csv",
        default=None,
        help="Optional CSV with columns 'filename,blood_group' for full ABO+Rh evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory to save confusion matrix and CSV results",
    )
    args = parser.parse_args()

    loaded_model = load_model(args.model_path)
    coco_data    = load_coco_annotations(args.annotations)

    metrics = evaluate(
        loaded_model,
        coco_data,
        args.image_dir,
        ground_truth_csv=args.ground_truth_csv,
        output_dir=args.output_dir,
    )

    analyse_errors(metrics["results"])
