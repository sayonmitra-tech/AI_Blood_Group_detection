"""
AI-Based Blood Group Detection System
======================================
Modular pipeline for automated blood group detection from blood card images
using COCO-annotated region extraction and MobileNetV2 transfer learning.

Pipeline:
  Full Blood Card Image
  → Extract regions (A, B, D) via COCO bounding-box annotations
  → Classify clumping level per region (Strong / Medium / No Clumping)
  → Map clumping → reaction (Positive / Negative)
  → Apply ABO + Rh logic → Blood group (A/B/AB/O  ±  +/-)
"""

import os
import json
import logging

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMAGE_SIZE = (224, 224)   # MobileNetV2 input resolution
NUM_CLASSES = 3           # Strong, Medium, No Clumping
BATCH_SIZE = 32
EPOCHS = 10

# Category IDs in the COCO annotation file that correspond to A, B, D reagent zones
CATEGORY_A = 1
CATEGORY_B = 2
CATEGORY_D = 3

# Class indices returned by the model
CLASS_STRONG = 0   # Strong clumping  → Positive
CLASS_MEDIUM = 1   # Medium clumping  → Positive
CLASS_NONE   = 2   # No clumping      → Negative

# Confidence threshold: if "No Clumping" score < this, treat as weak positive
WEAK_POSITIVE_THRESHOLD = 0.90


# ══════════════════════════════════════════════
# 1. DATA PREPROCESSING
# ══════════════════════════════════════════════

def load_coco_annotations(annotation_path: str) -> dict:
    """
    Load and return a COCO annotation JSON file.

    Parameters
    ----------
    annotation_path : str
        Absolute path to ``_annotations.coco.json``.

    Returns
    -------
    dict
        Parsed COCO JSON with keys ``images``, ``annotations``, ``categories``.

    Raises
    ------
    FileNotFoundError
        If the annotation file does not exist.
    """
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    with open(annotation_path, "r") as f:
        coco = json.load(f)
    logger.info("Loaded %d images, %d annotations from %s",
                len(coco["images"]), len(coco["annotations"]), annotation_path)
    return coco


def extract_and_save_crops(
    folder: str,
    data_root: str = "data",
    output_root: str = "dataset",
) -> None:
    """
    Extract bounding-box crops from all annotated images in *folder* and save
    them into ``output_root/<folder>/<category_id>/`` for use with
    ``image_dataset_from_directory``.

    Each annotation's ``category_id`` becomes the sub-directory name so that
    Keras can infer class labels automatically.

    Parameters
    ----------
    folder : str
        Sub-folder name, e.g. ``"train"``, ``"valid"``, or ``"test"``.
    data_root : str
        Root directory that contains the raw dataset folders.
    output_root : str
        Root directory where cropped images will be saved.

    Raises
    ------
    FileNotFoundError
        If the annotation file or an image file cannot be found.
    """
    annotation_path = os.path.join(data_root, folder, "_annotations.coco.json")
    coco = load_coco_annotations(annotation_path)

    # Build a quick id → image-info lookup
    images_by_id = {img["id"]: img for img in coco["images"]}

    saved = 0
    skipped = 0

    for ann in coco["annotations"]:
        img_id  = ann["image_id"]
        bbox    = ann["bbox"]          # [x, y, width, height]  (COCO format)
        cat_id  = str(ann["category_id"])

        img_info = images_by_id.get(img_id)
        if img_info is None:
            logger.warning("Annotation references unknown image_id=%d – skipping.", img_id)
            skipped += 1
            continue

        img_path = os.path.join(data_root, folder, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            logger.warning("Could not read image: %s – skipping.", img_path)
            skipped += 1
            continue

        x, y, w, h = map(int, bbox)

        # Guard against degenerate or out-of-bounds boxes
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            logger.warning("Degenerate bbox %s in %s – skipping.", bbox, img_path)
            skipped += 1
            continue

        crop = img[y1:y2, x1:x2]

        save_dir = os.path.join(output_root, folder, cat_id)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{img_id}_{ann['id']}.jpg")
        cv2.imwrite(out_path, crop)
        saved += 1

    logger.info(
        "process_folder('%s'): saved=%d  skipped=%d", folder, saved, skipped
    )


# ══════════════════════════════════════════════
# 2. DATASET LOADING
# ══════════════════════════════════════════════

def load_datasets(
    dataset_root: str = "dataset",
    image_size: tuple = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """
    Create ``tf.data.Dataset`` objects for train and validation splits.

    Images are normalised to [0, 1] via a rescaling layer applied at
    dataset-pipeline level so the same preprocessing is automatically used
    during inference when the saved model is loaded.

    Parameters
    ----------
    dataset_root : str
        Directory that contains ``train/`` and ``valid/`` sub-folders
        (as produced by :func:`extract_and_save_crops`).
    image_size : tuple
        ``(height, width)`` used for resizing.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset]
        ``(train_dataset, validation_dataset)``
    """
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_root, "train"),
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_root, "valid"),
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    # Normalise pixel values from [0, 255] → [0, 1]
    normalise = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    train_data = train_data.map(normalise, num_parallel_calls=tf.data.AUTOTUNE)
    val_data   = val_data.map(normalise,   num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch for performance
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    val_data   = val_data.prefetch(tf.data.AUTOTUNE)

    logger.info("Loaded train and validation datasets from %s", dataset_root)
    return train_data, val_data


# ══════════════════════════════════════════════
# 3. MODEL ARCHITECTURE
# ══════════════════════════════════════════════

def build_model(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = (*IMAGE_SIZE, 3),
    fine_tune_layers: int = 0,
) -> tf.keras.Model:
    """
    Build a transfer-learning model based on MobileNetV2.

    Architecture
    ------------
    MobileNetV2 (ImageNet, frozen)
      → GlobalAveragePooling2D
      → Dropout(0.3)
      → Dense(128, relu)
      → Dense(num_classes, softmax)

    When ``fine_tune_layers > 0`` the last *n* layers of the MobileNetV2
    backbone are unfrozen for fine-tuning.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 3: Strong, Medium, None).
    input_shape : tuple
        ``(H, W, C)`` expected by the backbone.
    fine_tune_layers : int
        Number of trailing backbone layers to unfreeze for fine-tuning.
        Set to ``0`` (default) to keep the entire backbone frozen.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    if fine_tune_layers > 0:
        # Unfreeze the last `fine_tune_layers` layers
        for layer in base_model.layers[-fine_tune_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        logger.info("Fine-tuning last %d backbone layers.", fine_tune_layers)

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Model built: %d trainable params", model.count_params())
    return model


# ══════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════

def train_model(
    model: tf.keras.Model,
    train_data,
    val_data,
    epochs: int = EPOCHS,
    model_save_path: str = "blood_model.keras",
) -> tf.keras.callbacks.History:
    """
    Train the model with early stopping and model checkpointing.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled model returned by :func:`build_model`.
    train_data : tf.data.Dataset
        Training dataset.
    val_data : tf.data.Dataset
        Validation dataset.
    epochs : int
        Maximum training epochs.
    model_save_path : str
        Path to save the best checkpoint (``*.keras`` format).

    Returns
    -------
    tf.keras.callbacks.History
        Training history object (loss / accuracy curves).
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
    )

    logger.info(
        "Training complete. Best val_accuracy: %.4f",
        max(history.history.get("val_accuracy", [0])),
    )
    return history


# ══════════════════════════════════════════════
# 5. INFERENCE – SINGLE CROP
# ══════════════════════════════════════════════

def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """
    Resize and normalise a single BGR crop for model inference.

    Parameters
    ----------
    crop : np.ndarray
        Raw BGR image array as returned by ``cv2.imread`` / slicing.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, 224, 224, 3)`` with values in [0, 1].
    """
    resized = cv2.resize(crop, IMAGE_SIZE)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr     = rgb.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_crop(
    model: tf.keras.Model,
    crop: np.ndarray,
) -> np.ndarray:
    """
    Run inference on a single pre-cropped region image.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded / trained classification model.
    crop : np.ndarray
        Raw BGR crop from a blood card image.

    Returns
    -------
    np.ndarray
        Softmax probability vector of shape ``(1, num_classes)``.
    """
    tensor = preprocess_crop(crop)
    pred   = model.predict(tensor, verbose=0)
    return pred


def get_reaction(pred: np.ndarray) -> int:
    """
    Convert a softmax probability vector to a binary reaction value.

    Decision rules
    --------------
    * Class ``CLASS_STRONG`` (0) → Positive (1)
    * Class ``CLASS_MEDIUM`` (1) → Positive (1)
    * Class ``CLASS_NONE``   (2) → Negative (0)
      Unless the "No Clumping" score < ``WEAK_POSITIVE_THRESHOLD``,
      in which case it is treated as a weak positive (1).

    Parameters
    ----------
    pred : np.ndarray
        Softmax output of shape ``(1, 3)``.

    Returns
    -------
    int
        ``1`` for Positive, ``0`` for Negative.
    """
    pred = np.array(pred).flatten()
    c    = int(pred.argmax())

    if c in (CLASS_STRONG, CLASS_MEDIUM):
        return 1
    # Weak positive: "No Clumping" is not highly confident
    if c == CLASS_NONE and pred[CLASS_NONE] < WEAK_POSITIVE_THRESHOLD:
        return 1
    return 0


def get_reaction_label(pred: np.ndarray) -> str:
    """
    Return a human-readable clumping label for a prediction vector.

    Parameters
    ----------
    pred : np.ndarray
        Softmax output of shape ``(1, 3)``.

    Returns
    -------
    str
        One of ``"Strong Clumping (Positive)"``,
        ``"Medium Clumping (Positive)"``, or ``"No Clumping (Negative)"``.
    """
    pred = np.array(pred).flatten()
    c    = int(pred.argmax())
    labels = {
        CLASS_STRONG: "Strong Clumping (Positive)",
        CLASS_MEDIUM: "Medium Clumping (Positive)",
        CLASS_NONE  : "No Clumping (Negative)",
    }
    return labels.get(c, "Unknown")


# ══════════════════════════════════════════════
# 6. BLOOD GROUP DECISION LOGIC
# ══════════════════════════════════════════════

def determine_blood_group(A_reaction: int, B_reaction: int, D_reaction: int) -> str:
    """
    Apply ABO + Rh factor logic to derive the final blood group.

    ABO logic
    ---------
    +A, -B  →  A
    -A, +B  →  B
    +A, +B  →  AB
    -A, -B  →  O

    Rh factor
    ---------
    +D  →  Positive (+)
    -D  →  Negative (-)

    Parameters
    ----------
    A_reaction : int
        Anti-A reagent reaction (1 = Positive, 0 = Negative).
    B_reaction : int
        Anti-B reagent reaction.
    D_reaction : int
        Anti-D (Rh) reagent reaction.

    Returns
    -------
    str
        Blood group string, e.g. ``"A+"``, ``"O-"``, ``"AB+"``.
    """
    # ABO determination
    if A_reaction and not B_reaction:
        abo = "A"
    elif B_reaction and not A_reaction:
        abo = "B"
    elif A_reaction and B_reaction:
        abo = "AB"
    else:
        abo = "O"

    rh = "+" if D_reaction else "-"
    return abo + rh


# ══════════════════════════════════════════════
# 7. REGION EXTRACTION FROM FULL BLOOD CARD
# ══════════════════════════════════════════════

def extract_regions(
    image_name: str,
    coco: dict,
    image_dir: str,
) -> dict:
    """
    Extract reagent-zone crops (A, B, D) from a full blood card image using
    COCO bounding-box annotations.

    Parameters
    ----------
    image_name : str
        Filename (not full path) of the blood card image to process.
    coco : dict
        Loaded COCO annotation dictionary (from :func:`load_coco_annotations`).
    image_dir : str
        Directory that contains ``image_name``.

    Returns
    -------
    dict
        Mapping ``{category_id: crop_ndarray}`` for every annotated region
        found in the image.  Returns an empty dict if the image is not in the
        COCO manifest or cannot be read.
    """
    # Find image metadata
    img_info = next(
        (img for img in coco["images"] if img["file_name"] == image_name),
        None,
    )
    if img_info is None:
        logger.warning("'%s' not found in COCO annotations.", image_name)
        return {}

    img_path = os.path.join(image_dir, image_name)
    img = cv2.imread(img_path)
    if img is None:
        logger.error("Could not read image: %s", img_path)
        return {}

    regions = {}
    for ann in coco["annotations"]:
        if ann["image_id"] != img_info["id"]:
            continue
        x, y, w, h = map(int, ann["bbox"])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Degenerate bbox [%d,%d,%d,%d] for image '%s' – skipping.",
                x, y, w, h, image_name,
            )
            continue
        crop = img[y1:y2, x1:x2]
        regions[ann["category_id"]] = crop

    if not regions:
        logger.warning("No valid regions found for '%s'.", image_name)

    return regions


# ══════════════════════════════════════════════
# 8. FULL IMAGE PREDICTION PIPELINE
# ══════════════════════════════════════════════

def predict_blood_group(
    image_name: str,
    model: tf.keras.Model,
    coco: dict,
    image_dir: str,
    verbose: bool = True,
) -> str | None:
    """
    End-to-end prediction: full blood card image → blood group string.

    Steps
    -----
    1. Extract A, B, D reagent regions via COCO annotations.
    2. Run clumping-level classification on each region.
    3. Convert predictions to binary reactions.
    4. Apply ABO + Rh logic.
    5. Return the blood group label.

    Parameters
    ----------
    image_name : str
        Filename of the blood card image (must exist in ``image_dir``).
    model : tf.keras.Model
        Loaded classification model.
    coco : dict
        COCO annotation dict for the dataset split that contains ``image_name``.
    image_dir : str
        Directory containing the raw blood card images.
    verbose : bool
        If ``True``, log per-region predictions and the final result.

    Returns
    -------
    str or None
        Blood group string (e.g. ``"AB+"``) or ``None`` if prediction fails.
    """
    regions = extract_regions(image_name, coco, image_dir)

    if not regions:
        logger.error("Prediction aborted: no regions extracted for '%s'.", image_name)
        return None

    preds = {}
    for cat_id, crop in regions.items():
        pred = predict_crop(model, crop)
        preds[cat_id] = pred
        if verbose:
            logger.info(
                "  Region cat_id=%d  →  %s  (conf=%.3f)",
                cat_id,
                get_reaction_label(pred),
                float(np.max(pred)),
            )

    # Default to "No Clumping" (Negative) if a zone annotation is missing
    default_neg = np.array([[0.0, 0.0, 1.0]])
    A_reaction = get_reaction(preds.get(CATEGORY_A, default_neg))
    B_reaction = get_reaction(preds.get(CATEGORY_B, default_neg))
    D_reaction = get_reaction(preds.get(CATEGORY_D, default_neg))

    if verbose:
        logger.info("  A=%d  B=%d  D=%d", A_reaction, B_reaction, D_reaction)

    blood_group = determine_blood_group(A_reaction, B_reaction, D_reaction)

    if verbose:
        logger.info("  Predicted Blood Group: %s", blood_group)

    return blood_group


# ══════════════════════════════════════════════
# 9. MODEL I/O
# ══════════════════════════════════════════════

def save_model(model: tf.keras.Model, path: str = "blood_model.keras") -> None:
    """
    Save the trained model to disk.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model to persist.
    path : str
        Output file path.  Use ``.keras`` (recommended) or ``.h5``.
    """
    model.save(path)
    logger.info("Model saved to %s", path)


def load_model(path: str = "blood_model.keras") -> tf.keras.Model:
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    path : str
        File path to a saved Keras model.

    Returns
    -------
    tf.keras.Model
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = tf.keras.models.load_model(path)
    logger.info("Model loaded from %s", path)
    return model


# ══════════════════════════════════════════════
# 10. CONVENIENCE: BATCH PREDICTION
# ══════════════════════════════════════════════

def batch_predict(
    image_names: list[str],
    model: tf.keras.Model,
    coco: dict,
    image_dir: str,
    verbose: bool = False,
) -> dict[str, str | None]:
    """
    Run :func:`predict_blood_group` over a list of images.

    Parameters
    ----------
    image_names : list[str]
        List of filenames (not full paths) to process.
    model : tf.keras.Model
        Loaded classification model.
    coco : dict
        COCO annotation dict covering the images.
    image_dir : str
        Directory containing the images.
    verbose : bool
        Passed to :func:`predict_blood_group`.

    Returns
    -------
    dict[str, str | None]
        ``{image_name: predicted_blood_group}`` mapping.
    """
    results = {}
    for name in image_names:
        logger.info("Processing: %s", name)
        results[name] = predict_blood_group(name, model, coco, image_dir, verbose=verbose)
    return results


# ══════════════════════════════════════════════
# MAIN – Example Usage
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blood Group Detection CLI")
    subparsers = parser.add_subparsers(dest="command")

    # ------ train ------
    train_parser = subparsers.add_parser("train", help="Train the model from scratch")
    train_parser.add_argument("--data-root",    default="data",          help="Raw data root")
    train_parser.add_argument("--dataset-root", default="dataset",       help="Cropped dataset root")
    train_parser.add_argument("--save-path",    default="blood_model.keras")
    train_parser.add_argument("--epochs",       type=int, default=EPOCHS)
    train_parser.add_argument("--fine-tune",    type=int, default=0,     help="Last N backbone layers to unfreeze")

    # ------ predict ------
    pred_parser = subparsers.add_parser("predict", help="Predict blood group for an image")
    pred_parser.add_argument("image",        help="Image filename (must be in --image-dir)")
    pred_parser.add_argument("--model-path", default="blood_model.keras")
    pred_parser.add_argument("--annotations",required=True, help="Path to _annotations.coco.json")
    pred_parser.add_argument("--image-dir",  required=True, help="Directory containing the image")

    args = parser.parse_args()

    if args.command == "train":
        # Step 1: crop extraction
        for split in ("train", "valid"):
            extract_and_save_crops(split, data_root=args.data_root, output_root=args.dataset_root)

        # Step 2: load datasets
        train_ds, val_ds = load_datasets(args.dataset_root)

        # Step 3: build model
        m = build_model(fine_tune_layers=args.fine_tune)

        # Step 4: train
        train_model(m, train_ds, val_ds, epochs=args.epochs, model_save_path=args.save_path)

    elif args.command == "predict":
        m    = load_model(args.model_path)
        coco = load_coco_annotations(args.annotations)
        result = predict_blood_group(args.image, m, coco, args.image_dir, verbose=True)
        print(f"\nBlood Group: {result}")

    else:
        parser.print_help()
