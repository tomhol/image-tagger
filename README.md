# Dog Tagger: Automated Photo Tagging for Dogs

Dog Tagger is a project designed to automate the tagging of large photo collections using machine learning. It specializes in detecting and identifying specific subjects — currently focused on two dogs: **Saga** (Holland Shepherd) and **Raff** (Czechoslovakian Wolfdog).

The pipeline uses **RF-DETR** or **YOLO** for robust object detection and a custom **Keras 3** classifier (running on the **PyTorch** backend) for fine-grained identification.

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
**Script:** `analyze-dogs.py`

The main analysis script that processes collections of images, generates annotated previews, and applies metadata tags.

- **Modes of Operation**:
  - **`analyze`**: Generates annotated images with bounding boxes and classification scores. Previews are automatically scaled to a maximum of 1600px for easy review.
  - **`dry-run`**: Performs full analysis and prints detections to the console without modifying any files.
  - **`tag-images`**: Adds identified dog names (e.g., "Saga", "Raff") as **IPTC keywords** directly to the original image files while preserving original file timestamps.
- **Hybrid Pipeline**: Combines state-of-the-art detectors (**RF-DETR** or **YOLO**) with a custom Keras specialist for identification.
- **Filtering**: Supports glob patterns (`--filter`) to process specific subsets of a collection.
- **Tag Validation**: Use `--valid-tags` to restrict identification to a specific list of subjects.
- **Summary Statistics**: Provides a detailed report at the end of each run, including detection counts and average confidence.

## Installation & Setup

1.  **Environment**: Always operate within the provided virtual environment.
    ```bash
    source .venv/bin/activate
    ```
2.  **Dependencies**:
    ```bash
    pip install torch torchvision keras ultralytics rfdetr opencv-python pillow iptcinfo3
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

### Step 3: Analyze and Tag a Collection
Run the full detection and identification pipeline.

**Generate annotated previews for review:**
```bash
python analyze-dogs.py --mode analyze --collection-root collection/ --output-dir review-results/
```

**Perform a dry-run for specific images:**
```bash
python analyze-dogs.py --mode dry-run --collection-root collection/ --filter "2024-05-*.jpg"
```

**Apply IPTC tags to images (only for Saga and Raff):**
```bash
python analyze-dogs.py --mode tag-images --collection-root collection/ --valid-tags "Saga,Raff"
```

## Directory Structure
- `training-set/`: Reference images for known subjects used for training and validation.
- `dog-detection-keras/`: Default output for annotated Keras analysis.
- `tricky-images/`: Sample collection for testing edge cases.
