# AI Blood Group Detection System

An end-to-end deep learning pipeline that determines a person's ABO + Rh blood group from a photograph of a standard disposable blood-typing card, using COCO bounding-box annotation, OpenCV region extraction, and MobileNetV2 transfer learning.

---

## Pipeline Overview

```
Full Blood Card Image
  → Parse COCO bounding-box annotations
  → Crop Anti-A, Anti-B, Anti-D reagent zones
  → MobileNetV2 classifies clumping level per zone
       (Strong Clumping | Medium Clumping | No Clumping)
  → Map clumping → reaction (Positive / Negative)
  → ABO + Rh logic
  → Output: A+ / A- / B+ / B- / AB+ / AB- / O+ / O-
```

---

## Repository Structure

```
AI_Blood_Group_detection/
├── AI_Blood_Group_detection.ipynb  # Original Google Colab development notebook
├── blood_group_detection.py        # Refactored modular pipeline (Task 3)
├── evaluation.py                   # Evaluation metrics & confusion matrix (Task 5)
├── IEEE_Paper.md                   # Full IEEE-format research paper (Task 6)
├── blood_model.keras               # Trained model (Keras native format)
├── blood_model.h5                  # Trained model (HDF5 legacy format)
├── train_data/                     # Sample training images (~44 images)
└── test_data/                      # Sample test images (~54 images)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 2. Prepare the dataset

```bash
# Crop regions from COCO-annotated images (run once)
python blood_group_detection.py train \
    --data-root   data \
    --dataset-root dataset \
    --save-path   blood_model.keras \
    --epochs      10
```

### 3. Predict a single blood card image

```bash
python blood_group_detection.py predict \
    AB-1-_jpg.rf.…jpg \
    --model-path  blood_model.keras \
    --annotations data/test/_annotations.coco.json \
    --image-dir   data/test
```

### 4. Evaluate model performance

```bash
python evaluation.py \
    --model-path  blood_model.keras \
    --annotations data/test/_annotations.coco.json \
    --image-dir   data/test \
    --output-dir  evaluation_results
```

Outputs:
- Overall accuracy, precision, recall, F1 per blood group
- `evaluation_results/per_image_results.csv`
- `evaluation_results/confusion_matrix.png`

---

## Model Architecture

| Component | Details |
|---|---|
| Backbone | MobileNetV2 (ImageNet pretrained, frozen) |
| Head | GlobalAveragePooling2D → Dropout(0.3) → Dense(128, relu) → Dense(3, softmax) |
| Input size | 224 × 224 × 3 |
| Classes | 3 (Strong Clumping, Medium Clumping, No Clumping) |
| Trainable params | ~166 K |
| Validation accuracy | ~94–95% |

---

## Blood Group Logic

| Anti-A | Anti-B | Anti-D | Blood Group |
|---|---|---|---|
| + | − | + | A+ |
| + | − | − | A− |
| − | + | + | B+ |
| − | + | − | B− |
| + | + | + | AB+ |
| + | + | − | AB− |
| − | − | + | O+ |
| − | − | − | O− |

---

## Research Paper

A full IEEE-format research paper covering methodology, results, discussion, and future work is available in [`IEEE_Paper.md`](IEEE_Paper.md).

---

## Dataset

- Images of standard disposable blood-typing cards annotated with COCO bounding boxes using Roboflow
- Each image contains three bounding boxes: Anti-A zone (`category_id=1`), Anti-B zone (`category_id=2`), Anti-D zone (`category_id=3`)
- Training set: ~44 images; Test set: ~54 images
- Blood groups represented: A, B, AB, O

---

## Future Work

- [ ] Replace annotation dependency with YOLO-based automatic region detection
- [ ] Convert model to TensorFlow Lite (INT8) for Android deployment
- [ ] Expand dataset with diverse lighting conditions and card brands
- [ ] Fine-tune MobileNetV2 final layers for domain adaptation
- [ ] Add calibrated confidence scores and manual-review flagging
