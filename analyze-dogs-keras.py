import os
# Configure Keras to use PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import cv2
import numpy as np
import keras
from pathlib import Path
from ultralytics import YOLO

# Configuration matching training script
IMG_SIZE = (224, 224)
MODEL_PATH = "dog_classifier.keras"
CLASSES_PATH = "dog_classes.txt"

# Default constants
YOLO_MODEL_PATH = "yolo26l.pt"  # Maintaining filename from original script
DEFAULT_COLLECTION_ROOT = "tricky-images"
DEFAULT_REVIEW_OUTPUT_DIR = "dog-detection-keras"
DEFAULT_DOG_CLASS_ID = 16  # COCO class ID for 'dog'
BEAR_CLASS_ID = 21         # COCO class ID for 'bear'
DEFAULT_YOLO_CONFIDENCE = 0.25
DEFAULT_CLASS_THRESHOLD = 0.5

def load_class_names(path):
    if not Path(path).exists():
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def analyze_collection(
    collection_root: str,
    output_dir: str,
    yolo_confidence: float,
    threshold: float,
):
    collection_root = Path(collection_root)
    output_dir = Path(output_dir)

    # Check for trained model
    if not Path(MODEL_PATH).exists() or not Path(CLASSES_PATH).exists():
        print(f"Error: Model '{MODEL_PATH}' or classes file '{CLASSES_PATH}' not found.")
        print("Please run 'train_dog_classifier.py' first.")
        return

    print(f"Loading YOLO model {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print(f"Loading Keras model {MODEL_PATH}...")
    keras_model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASSES_PATH)
    print(f"Known classes: {class_names}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    if not collection_root.exists():
        print(f"Collection root '{collection_root}' does not exist")
        return
        
    image_paths = sorted([p for p in collection_root.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])
    print(f"Analyzing {len(image_paths)} images in {collection_root}...\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        cv2_img = cv2.imread(str(img_path))
        if cv2_img is None:
            continue

        annotated_img = cv2_img.copy()
        has_dog = False
        detected_tags = []

        # Run YOLO inference
        results = yolo_model(cv2_img, conf=yolo_confidence, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                # Check for dog or bear (to catch misclassified dogs)
                if cls_id not in [DEFAULT_DOG_CLASS_ID, BEAR_CLASS_ID]:
                    continue

                # Get bbox coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_orig, w_orig = cv2_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_orig, x2), min(h_orig, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop and preprocess for Keras
                crop_bgr = cv2_img[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(crop_rgb, IMG_SIZE)
                input_array = np.expand_dims(resized_img, axis=0)

                # Predict class
                preds = keras_model.predict(input_array, verbose=0)
                score = np.max(preds[0])
                class_idx = np.argmax(preds[0])
                
                label = class_names[class_idx]
                if score < threshold:
                    label = "Unknown"

                label_text = f"{label} ({score:.2f})"
                detected_tags.append(label_text)
                has_dog = True

                # Annotation
                color = (0, 255, 0)
                text_color = (0, 0, 0)
                font_scale = max(0.5, w_orig / 2000.0)
                thickness = max(1, int(w_orig / 1000.0))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(annotated_img, (x1, y1), (x1 + text_w + 10, y1 + text_h + baseline + 10), color, -1)
                cv2.putText(annotated_img, label_text, (x1 + 5, y1 + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        if has_dog:
            cv2.imwrite(str(output_dir / img_path.name), annotated_img)
            print(f"{img_path.name:<30} | {', '.join(detected_tags)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dogs with YOLO detection and Keras classification.")
    parser.add_argument("--collection-root", type=str, default=DEFAULT_COLLECTION_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_REVIEW_OUTPUT_DIR)
    parser.add_argument("--yolo-confidence", type=float, default=DEFAULT_YOLO_CONFIDENCE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CLASS_THRESHOLD, help="Classification confidence threshold")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    analyze_collection(
        collection_root=args.collection_root,
        output_dir=args.output_dir,
        yolo_confidence=args.yolo_confidence,
        threshold=args.threshold,
    )