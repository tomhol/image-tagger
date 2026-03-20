# Clip Tagger: Automated Photo Tagging for Dogs

Clip Tagger is a project designed to automate the tagging of large photo collections using machine learning. It specializes in detecting and identifying specific subjects—currently focused on two dogs: **Saga** (Holland Shepherd) and **Raff** (Czechoslovakian Wolfdog).

The pipeline uses **YOLO** for robust object detection and a custom **Keras 3** classifier (running on the **PyTorch** backend) for fine-grained identification.

## Workflow

The project follows a modular workflow: data preparation, model training, and inference.

### 1. Data Preparation
**Script:** `crop-the-training-data.py`

This script builds a high-quality training dataset by automatically extracting subject crops from raw photos.

- **Detection**: Uses a large YOLO model (`yolo26l.pt`) to find dogs in source directories.
- **Cropping**: Identifies the largest dog in each image to ensure the best training sample.
- **Standardization**: Resizes crops to a fixed height (720px) while maintaining aspect ratio and organizes them into a structured `training-set/` directory based on class names.

### 2. Model Training
**Script:** `train_dog_classifier.py`

Trains a specialist image classifier using Transfer Learning.

- **Architecture**: Employs **EfficientNetV2B0** (pretrained on ImageNet) for its high efficiency and strong feature extraction capabilities.
- **Backend**: Utilizes **Keras 3** with the **PyTorch** backend, optimized for CPU execution.
- **Output**: Generates `dog_classifier.keras` (model weights) and `dog_classes.txt` (label mapping).

### 3. Inference & Analysis
**Script:** `analyze-dogs-keras.py`

The main analysis script that processes collections of images and generates annotated previews.

- **Hybrid Pipeline**: Combines YOLO for general detection with the custom Keras specialist for identification.
- **Robustness**: Detects both 'dog' and 'bear' (COCO classes) to catch potential misclassifications by the object detector before passing them to the Keras model.
- **Annotation**: Draws bounding boxes and labels with confidence scores on detected subjects.
- **Output**: Saves annotated images to a review directory for manual validation.

## Installation & Setup

1.  **Environment**: Always operate within the provided virtual environment.
    ```bash
    source .venv/bin/activate
    ```
2.  **Dependencies**:
    ```bash
    pip install torch torchvision keras ultralytics opencv-python pillow
    ```
3.  **Keras Backend**: The scripts automatically configure `KERAS_BACKEND="torch"`.

## Sample Usage

### Step 1: Prepare Training Data
Organize raw photos of specific dogs into folders (e.g., `Raff/`, `Saga/`, `Cakun/`) and update the `INPUT_ROOT` in `crop-the-training-data.py`. Then run:
```bash
python crop-the-training-data.py
```

### Step 2: Train the Model
Once the `training-set/` folder is populated with crops:
```bash
python train_dog_classifier.py
```

### Step 3: Analyze a Collection
Run the full detection and identification pipeline on a directory of new photos:
```bash
python analyze-dogs-keras.py --collection-root collection/ --output-dir dog-detection-results/ --threshold 0.7
```

## Directory Structure
- `training-set/`: Specialist training data (auto-generated crops).
- `dog-detection-keras/`: Default output for annotated Keras analysis.
- `anchors/`: Reference images for known subjects used for validation.
- `tricky-images/`: Sample collection for testing edge cases.
