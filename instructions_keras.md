# Project: Dog Classification with Keras 3 (PyTorch Backend)

## Objective
Create a new analysis workflow based on `analyze-dogs-visual.py`, utilizing YOLO for object detection and a custom Keras 3 model for specific dog classification.

## Tasks

### 1. Model Training Script
Create a script to train a classifier for the 3 specific dog classes using **Keras 3**.
*   **Strategy:** Use Transfer Learning. Select a pre-trained model optimized for animal/dog features (e.g., MobileNetV3, EfficientNet, or similar light-weight models).
*   **Data Source:** `training-set` folder (subfolders contain the class images).
*   **Data Split:** 80% Training / 20% Validation.
*   **Configuration:**
    *   **Backend:** Prefer **PyTorch** over TensorFlow.
    *   **Hardware:** Optimize for **CPU** execution (CUDA is not available).
    *   **Parameters:** Use sensible defaults for batch size and learning rate.

### 2. Inference Script (`analyze-dogs-keras.py`)
Develop a script to analyze images using the trained components.
*   **Detection:** Use YOLO to locate objects. Detect both `dog` and `bear` classes to account for potential misclassification by the object detector.
*   **Classification:** Pass the detected image regions (crops) to the new Keras 3 model to identify the specific dog class.